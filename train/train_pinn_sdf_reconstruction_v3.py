#!/usr/bin/env python3
"""
train_pinn_sdf_reconstruction_v2.py  

Goal
----
Train a network to reconstruct INNER and OUTER signed distance fields (SDFs)
for 3D coronary artery geometry, robust for thin and wide arteries, using:

1) PDE (physics): Eikonal regularization (|grad(phi)| = 1) near surfaces
2) Boundary conditions (BC): Dirichlet anchoring from binary-derived iso-surfaces
3) Region-consistent constraints from binaries: lumen, wall, outside
4) Case-adaptive thickness priors: per-case t_min and thin-threshold derived from GT SDFs
5) Thin-wall hard sampling: explicit sampling bucket targeting thin wall regions
6) Curriculum: geometry -> strict physics -> refine
7) Safe export: predicted grids + marching cubes meshes + per-case QC metrics

Inputs per case folder (required)
--------------------------------
INNER_SDF.nrrd
OUTER_SDF.nrrd
INNER_trimmed.nrrd
OUTER_contained.nrrd

Outputs
-------
<out_root>/<case_name>/
  checkpoints/
    stage*_last.pt
    stage*_best.pt
    best.pt
  train_metrics.csv
  qc_metrics.json
  pred_inner_sdf.npy
  pred_outer_sdf.npy
  inner_mesh.ply
  outer_mesh.ply

Notes
-----
- INNER and OUTER volumes may have different spacing/origin/direction. This code does NOT assume they match.
  All sampling is performed in physical space (mm), and each volume is sampled with its own transform.
- Predicted grids/meshes are exported on the INNER volume grid (canonical grid).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time  # [ADDED] for training time measurement
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

from skimage.measure import marching_cubes

# ============================================================
# Curriculum (you can tune weights if needed)
# ============================================================
CURRICULUM = [
    {
        "name": "stage1_geometry",
        "steps": 3500,
        "lr": 1e-4,
        "band_mm": 2.0,
        "n_total": 45000,

        "r_inner": 0.28,
        "r_outer": 0.28,
        "r_wall":  0.18,
        "r_thin":  0.10,
        "r_lumen": 0.12,

        "w_sdf": 1.0,
        "w_eik": 0.05,
        "w_bc":  2.5,
        "w_enc": 1.2,
        "w_order": 1.8,
        "w_minT": 0.10,
        "w_smoothT": 0.04,
        "w_lumen_in": 2.0,
    },
    {
        "name": "stage2_strict_physics",
        "steps": 6000,
        "lr": 5e-5,
        "band_mm": 1.5,
        "n_total": 45000,

        "r_inner": 0.24,
        "r_outer": 0.24,
        "r_wall":  0.22,
        "r_thin":  0.12,
        "r_lumen": 0.14,

        "w_sdf": 1.0,
        "w_eik": 0.10,
        "w_bc":  1.8,
        "w_enc": 4.0,
        "w_order": 2.2,
        "w_minT": 0.28,
        "w_smoothT": 0.07,
        "w_lumen_in": 2.5,
    },
    {
        "name": "stage3_refine",
        "steps": 4000,
        "lr": 3e-5,
        "band_mm": 1.0,
        "n_total": 50000,

        "r_inner": 0.28,
        "r_outer": 0.28,
        "r_wall":  0.16,
        "r_thin":  0.10,
        "r_lumen": 0.12,

        "w_sdf": 1.0,
        "w_eik": 0.12,
        "w_bc":  1.0,
        "w_enc": 4.5,
        "w_order": 2.5,
        "w_minT": 0.22,
        "w_smoothT": 0.08,
        "w_lumen_in": 2.5,
    }
]

GRID_CHUNK = 200000
OOM_FALLBACK_NTOTAL = 25000


# ============================================================
# Logging / determinism
# ============================================================
def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# [ADDED] Device/GPU info + memory helpers
# ============================================================
def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def get_device_info(device: torch.device) -> Dict[str, object]:
    """
    JSON-serializable device info. Keeps code independent of external libs.
    """
    info: Dict[str, object] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = int(torch.cuda.current_device())
        except Exception:
            idx = 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "type": "cuda",
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "total_memory_bytes": int(props.total_memory),
            "capability": [int(props.major), int(props.minor)],
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version(),
        })
    else:
        info.update({"type": "cpu"})
    return info


# ============================================================
# Volume wrappers (SDF + Binary)
# ============================================================
@dataclass
class Volume:
    tensor: torch.Tensor           # (1,1,D,H,W) float32
    origin: np.ndarray             # (3,)
    spacing: np.ndarray            # (3,)
    direction: np.ndarray          # (3,3)
    size_zyx: Tuple[int, int, int] # (D,H,W)
    center_mm: torch.Tensor        # (3,)
    half_mm: torch.Tensor          # (3,)
    device: torch.device
    is_binary: bool

    @staticmethod
    def from_nrrd(path: Path, device: torch.device, is_binary: bool) -> "Volume":
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z,y,x)

        origin = np.array(img.GetOrigin(), dtype=np.float64)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

        D, H, W = arr.shape
        t = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        corners_ijk = np.array([
            [0, 0, 0],
            [W - 1, 0, 0],
            [0, H - 1, 0],
            [0, 0, D - 1],
            [W - 1, H - 1, 0],
            [W - 1, 0, D - 1],
            [0, H - 1, D - 1],
            [W - 1, H - 1, D - 1],
        ], dtype=np.float64)  # (x,y,z)

        phys = []
        for x, y, z in corners_ijk:
            v = np.array([x * spacing[0], y * spacing[1], z * spacing[2]], dtype=np.float64)
            p = origin + direction @ v
            phys.append(p)
        phys = np.vstack(phys)
        pmin = phys.min(axis=0)
        pmax = phys.max(axis=0)

        center = torch.tensor((pmin + pmax) * 0.5, device=device, dtype=torch.float32)
        half = torch.tensor((pmax - pmin) * 0.5, device=device, dtype=torch.float32).clamp(min=1e-6)

        return Volume(
            tensor=t,
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_zyx=(D, H, W),
            center_mm=center,
            half_mm=half,
            device=device,
            is_binary=is_binary
        )

    def normalize_input(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        return (xyz_mm - self.center_mm) / self.half_mm

    def grad_to_physical(self, grad_wrt_norm: torch.Tensor) -> torch.Tensor:
        return grad_wrt_norm / self.half_mm

    def sample_trilinear(self, xyz_mm: torch.Tensor, threshold_binary: bool = True) -> torch.Tensor:
        origin = torch.tensor(self.origin, device=self.device, dtype=torch.float32)
        spacing = torch.tensor(self.spacing, device=self.device, dtype=torch.float32)
        direction = torch.tensor(self.direction, device=self.device, dtype=torch.float32)
        inv_dir = torch.inverse(direction)

        v = (xyz_mm - origin)
        ijk = (v @ inv_dir.T) / spacing  # (N,3) -> (x,y,z)

        D, H, W = self.size_zyx
        x = ijk[:, 0]
        y = ijk[:, 1]
        z = ijk[:, 2]

        gx = 2.0 * (x / (W - 1)) - 1.0
        gy = 2.0 * (y / (H - 1)) - 1.0
        gz = 2.0 * (z / (D - 1)) - 1.0

        grid = torch.stack([gx, gy, gz], dim=-1).view(1, 1, -1, 1, 3)
        val = F.grid_sample(
            self.tensor, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).view(-1)

        if self.is_binary and threshold_binary:
            return (val > 0.5).float()
        return val


# ============================================================
# Model
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs))
        self.num_freqs = num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x[..., None] * self.freq_bands  # (N,3,F)
        sin = torch.sin(2.0 * math.pi * xb)
        cos = torch.cos(2.0 * math.pi * xb)
        pe = torch.cat([sin, cos], dim=-1)  # (N,3,2F)
        return pe.view(x.shape[0], -1)      # (N,3*2F)


class MLP2SDF(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 6, pe_freqs: int = 10):
        super().__init__()
        self.pe = PositionalEncoding(pe_freqs)
        in_dim = 3 + 3 * 2 * pe_freqs

        blocks: List[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_norm, self.pe(x_norm)], dim=-1)
        h = self.backbone(h)
        return self.head(h)


# ============================================================
# Sampling utilities (physical space)
# ============================================================
def sample_uniform_physical(center_mm: torch.Tensor, half_mm: torch.Tensor, device: torch.device, n: int) -> torch.Tensor:
    x_norm = (torch.rand((n, 3), device=device) * 2.0 - 1.0).float()
    return x_norm * half_mm + center_mm


def rejection_sample_physical(center_mm: torch.Tensor, half_mm: torch.Tensor, device: torch.device,
                              n: int, predicate_fn,
                              max_tries: int = 120, oversample: float = 4.0) -> torch.Tensor:
    collected = []
    remaining = n
    tries = 0
    while remaining > 0 and tries < max_tries:
        tries += 1
        m = int(math.ceil(remaining * oversample))
        x = sample_uniform_physical(center_mm, half_mm, device, m)
        keep = predicate_fn(x)
        if keep.any():
            xk = x[keep]
            if xk.shape[0] > remaining:
                xk = xk[:remaining]
            collected.append(xk)
            remaining -= xk.shape[0]

    if remaining > 0:
        collected.append(sample_uniform_physical(center_mm, half_mm, device, remaining))

    return torch.cat(collected, dim=0)


# ============================================================
# BC surface point extraction from binaries
# ============================================================
def binary_surface_points_mm(bin_vol: Volume, n_points: int, seed: int = 0) -> np.ndarray:
    arr = bin_vol.tensor.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    if float(arr.min()) == float(arr.max()):
        return np.zeros((0, 3), dtype=np.float32)

    verts_zyx, faces, _, _ = marching_cubes(arr, level=0.5)

    origin = np.array(bin_vol.origin, dtype=np.float64)
    spacing = np.array(bin_vol.spacing, dtype=np.float64)
    direction = np.array(bin_vol.direction, dtype=np.float64)

    z = verts_zyx[:, 0]
    y = verts_zyx[:, 1]
    x = verts_zyx[:, 2]
    v = np.stack([x * spacing[0], y * spacing[1], z * spacing[2]], axis=1)
    pts = origin + (v @ direction.T)

    if pts.shape[0] <= n_points:
        return pts.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=n_points, replace=False)
    return pts[idx].astype(np.float32)


# ============================================================
# Loss utilities
# ============================================================
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None or (not mask.any()):
        return torch.tensor(0.0, device=x.device)
    return x[mask].mean()


def eikonal_loss_masked(phi: torch.Tensor, x_norm: torch.Tensor, ref_vol: Volume, mask: torch.Tensor) -> torch.Tensor:
    if mask is None or (not mask.any()):
        return torch.tensor(0.0, device=x_norm.device)

    grad = torch.autograd.grad(
        outputs=phi,
        inputs=x_norm,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False,
    )[0]

    grad_mm = ref_vol.grad_to_physical(grad)
    gnorm = torch.linalg.norm(grad_mm, dim=-1)
    return ((gnorm[mask] - 1.0) ** 2).mean()


def smooth_thickness_loss_masked(phi_in: torch.Tensor, phi_out: torch.Tensor,
                                 x_norm: torch.Tensor, ref_vol: Volume, mask: torch.Tensor) -> torch.Tensor:
    if mask is None or (not mask.any()):
        return torch.tensor(0.0, device=x_norm.device)

    t = phi_in - phi_out
    grad = torch.autograd.grad(
        outputs=t,
        inputs=x_norm,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False,
    )[0]

    grad_mm = ref_vol.grad_to_physical(grad)
    g2 = torch.sum(grad_mm ** 2, dim=-1)
    return g2[mask].mean()


# ============================================================
# Thickness prior estimation (case-adaptive)  
# ============================================================
def _physical_bbox_from_binary_mask(vol_bin: Volume, pad_mm: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a tight physical bounding box (center, half) from a binary mask volume.
    Falls back to vol_bin.center_mm/half_mm if mask is empty/degenerate.
    """
    arr = vol_bin.tensor.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    m = arr > 0.5
    if not np.any(m):
        return vol_bin.center_mm, vol_bin.half_mm

    zz, yy, xx = np.where(m)  # z,y,x indices
    z0, z1 = int(zz.min()), int(zz.max())
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())

    origin = np.array(vol_bin.origin, dtype=np.float64)
    spacing = np.array(vol_bin.spacing, dtype=np.float64)
    direction = np.array(vol_bin.direction, dtype=np.float64)

    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y1, z0],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
    ], dtype=np.float64)

    phys = []
    for x, y, z in corners:
        v = np.array([x * spacing[0], y * spacing[1], z * spacing[2]], dtype=np.float64)
        p = origin + direction @ v
        phys.append(p)
    phys = np.vstack(phys)
    pmin = phys.min(axis=0) - pad_mm
    pmax = phys.max(axis=0) + pad_mm

    center = torch.tensor((pmin + pmax) * 0.5, device=vol_bin.device, dtype=torch.float32)
    half = torch.tensor((pmax - pmin) * 0.5, device=vol_bin.device, dtype=torch.float32).clamp(min=1e-6)
    return center, half


@torch.no_grad()
def estimate_thickness_priors(
    vol_in_sdf: Volume,
    vol_out_sdf: Volume,
    vol_in_bin: Volume,
    vol_out_bin: Volume,
    n_probe: int = 120000,
    seed: int = 0,
    floor_tmin_mm: float = 0.15,
    tmin_scale_p05: float = 0.90,
    thin_thr_p20: float = 1.10,
) -> Dict[str, float]:
    """
    Estimate thickness distribution in the wall using probing in physical space.
    Wall defined by: outer_bin==1 and inner_bin==0.

    FIX:
    - Instead of uniform probing over the full INNER bbox (often too sparse),
      we compute a tight bbox from OUTER_trimmed and then rejection-sample wall points.
    """
    device = vol_in_sdf.device

    # Tight bbox around the vessel (OUTER mask), padded a bit
    center, half = _physical_bbox_from_binary_mask(vol_out_bin, pad_mm=4.0)

    # Wall predicate in physical space
    def pred_wall(x_mm: torch.Tensor) -> torch.Tensor:
        in_lumen = vol_in_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        in_outer = vol_out_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        return in_outer & (~in_lumen)

    # Try to get enough wall points robustly
    # Use at most n_probe points, but ensure we aim for a meaningful wall sample
    n_wall_target = int(min(max(8000, n_probe // 10), 60000))

    # Rejection sampling directly in wall
    xw = rejection_sample_physical(
        center_mm=center,
        half_mm=half,
        device=device,
        n=n_wall_target,
        predicate_fn=pred_wall,
        max_tries=140,
        oversample=4.0
    )

    # Evaluate wall mask again to know how many truly landed in wall
    wall_mask = pred_wall(xw)
    if not wall_mask.any():
        logging.warning("Thickness priors fallback: no wall points found (after rejection sampling).")
        return {
            "t_p05": 0.30, "t_p10": 0.35, "t_p20": 0.40, "t_med": 0.60,
            "t_min_case": max(floor_tmin_mm, 0.20),
            "thin_threshold": 0.35
        }

    xw = xw[wall_mask]

    # Compute GT thickness at those wall points
    pin = vol_in_sdf.sample_trilinear(xw, threshold_binary=False)
    pout = vol_out_sdf.sample_trilinear(xw, threshold_binary=False)
    t = (pin - pout).detach().cpu().numpy()

    t = t[np.isfinite(t)]
    # Remove extreme outliers (robust trimming)
    if t.size >= 50:
        lo = np.percentile(t, 1)
        hi = np.percentile(t, 99)
        t = t[(t >= lo) & (t <= hi)]

    if t.size < 1000:
        logging.warning(
            f"Thickness priors fallback: insufficient valid wall samples (t.size={t.size})."
        )
        return {
            "t_p05": 0.30, "t_p10": 0.35, "t_p20": 0.40, "t_med": 0.60,
            "t_min_case": max(floor_tmin_mm, 0.20),
            "thin_threshold": 0.35
        }

    p05 = float(np.percentile(t, 5))
    p10 = float(np.percentile(t, 10))
    p20 = float(np.percentile(t, 20))
    med = float(np.percentile(t, 50))

    t_min_case = max(floor_tmin_mm, tmin_scale_p05 * p05)
    thin_threshold = max(0.0, thin_thr_p20 * p20)
    thin_threshold = max(thin_threshold, t_min_case)

    return {
        "t_p05": p05,
        "t_p10": p10,
        "t_p20": p20,
        "t_med": med,
        "t_min_case": float(t_min_case),
        "thin_threshold": float(thin_threshold),
    }


# ============================================================
# Export helpers (PLY + grid prediction)
# ============================================================
def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


@torch.no_grad()
def predict_on_inner_grid(model: nn.Module, vol_inner_ref: Volume, chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    D, H, W = vol_inner_ref.size_zyx

    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=vol_inner_ref.device),
        torch.arange(H, device=vol_inner_ref.device),
        torch.arange(W, device=vol_inner_ref.device),
        indexing="ij"
    )
    ijk = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1).float()

    origin = torch.tensor(vol_inner_ref.origin, device=vol_inner_ref.device, dtype=torch.float32)
    spacing = torch.tensor(vol_inner_ref.spacing, device=vol_inner_ref.device, dtype=torch.float32)
    direction = torch.tensor(vol_inner_ref.direction, device=vol_inner_ref.device, dtype=torch.float32)

    v = ijk * spacing
    xyz = origin + (v @ direction.T)
    x_norm = vol_inner_ref.normalize_input(xyz)

    pin_list, pout_list = [], []
    for i in range(0, x_norm.shape[0], chunk):
        y = model(x_norm[i:i + chunk])
        pin_list.append(y[:, 0].detach().cpu())
        pout_list.append(y[:, 1].detach().cpu())

    pin = torch.cat(pin_list).numpy().reshape(D, H, W)
    pout = torch.cat(pout_list).numpy().reshape(D, H, W)
    return pin, pout


def export_meshes(case_out: Path, vol_inner_ref: Volume, pin: np.ndarray, pout: np.ndarray, prefix: str = "") -> Dict[str, int]:
    origin = np.array(vol_inner_ref.origin, dtype=np.float64)
    spacing = np.array(vol_inner_ref.spacing, dtype=np.float64)
    direction = np.array(vol_inner_ref.direction, dtype=np.float64)

    def to_phys(verts_zyx: np.ndarray) -> np.ndarray:
        z = verts_zyx[:, 0]
        y = verts_zyx[:, 1]
        x = verts_zyx[:, 2]
        v = np.stack([x * spacing[0], y * spacing[1], z * spacing[2]], axis=1)
        return origin + (v @ direction.T)

    def mc_safe(field: np.ndarray, name: str):
        vmin = float(np.min(field))
        vmax = float(np.max(field))
        if not (vmin <= 0.0 <= vmax):
            logging.warning(
                f"Skipping marching_cubes for {name}: iso=0 not in range [min={vmin:.6f}, max={vmax:.6f}]."
            )
            return None, None
        verts, faces, _, _ = marching_cubes(field, level=0.0)
        return verts, faces

    out_flags = {"inner_mesh_ok": 0, "outer_mesh_ok": 0}

    vi, fi = mc_safe(pin, "pred_inner")
    if vi is not None:
        write_ply(case_out / f"{prefix}inner_mesh.ply", to_phys(vi), fi)
        out_flags["inner_mesh_ok"] = 1

    vo, fo = mc_safe(pout, "pred_outer")
    if vo is not None:
        write_ply(case_out / f"{prefix}outer_mesh.ply", to_phys(vo), fo)
        out_flags["outer_mesh_ok"] = 1

    return out_flags


# ============================================================
# Checkpointing
# ============================================================
def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer,
              global_step: int, best_loss: float, stage_name: str, stage_cfg: dict,
              thickness_priors: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "global_step": global_step,
        "best_loss": best_loss,
        "stage_name": stage_name,
        "stage_cfg": stage_cfg,
        "thickness_priors": thickness_priors,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }, path)


# ============================================================
# Training step
# ============================================================
def one_train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    vol_in_sdf: Volume,
    vol_out_sdf: Volume,
    vol_in_bin: Volume,
    vol_out_bin: Volume,
    bc_inner_pts_mm: torch.Tensor,
    bc_outer_pts_mm: torch.Tensor,
    band_mm: float,
    n_total: int,
    t_min_case_mm: float,
    thin_threshold_mm: float,
    ratios: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:

    device = vol_in_sdf.device
    center = vol_in_sdf.center_mm
    half = vol_in_sdf.half_mm

    def pred_inner_band(x_mm: torch.Tensor) -> torch.Tensor:
        pin = vol_in_sdf.sample_trilinear(x_mm, threshold_binary=False)
        return torch.abs(pin) < band_mm

    def pred_outer_band(x_mm: torch.Tensor) -> torch.Tensor:
        pout = vol_out_sdf.sample_trilinear(x_mm, threshold_binary=False)
        return torch.abs(pout) < band_mm

    def pred_lumen(x_mm: torch.Tensor) -> torch.Tensor:
        return vol_in_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5

    def pred_wall(x_mm: torch.Tensor) -> torch.Tensor:
        in_lumen = vol_in_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        in_outer = vol_out_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        return in_outer & (~in_lumen)

    def pred_thin_wall(x_mm: torch.Tensor) -> torch.Tensor:
        wall = pred_wall(x_mm)
        if not wall.any():
            return wall
        xw = x_mm[wall]
        pin = vol_in_sdf.sample_trilinear(xw, threshold_binary=False)
        pout = vol_out_sdf.sample_trilinear(xw, threshold_binary=False)
        t = pin - pout
        keep = t < thin_threshold_mm
        out = torch.zeros_like(wall)
        out[wall] = keep
        return out

    r_inner = float(ratios["inner"])
    r_outer = float(ratios["outer"])
    r_wall  = float(ratios["wall"])
    r_thin  = float(ratios["thin"])
    r_lumen = float(ratios["lumen"])

    r_sum = r_inner + r_outer + r_wall + r_thin + r_lumen
    if r_sum >= 1.0:
        raise ValueError(f"Sampling ratios sum to >= 1.0 (got {r_sum:.3f}). Reduce ratios.")

    n_inner = int(r_inner * n_total)
    n_outer = int(r_outer * n_total)
    n_wall  = int(r_wall  * n_total)
    n_thin  = int(r_thin  * n_total)
    n_lumen = int(r_lumen * n_total)
    n_bg = n_total - (n_inner + n_outer + n_wall + n_thin + n_lumen)

    x_inner = rejection_sample_physical(center, half, device, n_inner, pred_inner_band)
    x_outer = rejection_sample_physical(center, half, device, n_outer, pred_outer_band)
    x_wall  = rejection_sample_physical(center, half, device, n_wall,  pred_wall)
    x_thin  = rejection_sample_physical(center, half, device, n_thin,  pred_thin_wall)
    x_lumen = rejection_sample_physical(center, half, device, n_lumen, pred_lumen)
    x_bg    = sample_uniform_physical(center, half, device, n_bg)

    x_mm = torch.cat([x_inner, x_outer, x_wall, x_thin, x_lumen, x_bg], dim=0)
    x_norm = vol_in_sdf.normalize_input(x_mm).requires_grad_(True)

    phi_in_gt  = vol_in_sdf.sample_trilinear(x_mm, threshold_binary=False).detach()
    phi_out_gt = vol_out_sdf.sample_trilinear(x_mm, threshold_binary=False).detach()

    y = model(x_norm)
    phi_in  = y[:, 0]
    phi_out = y[:, 1]

    lumen_mask  = (vol_in_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5)
    outer_in    = (vol_out_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5)
    wall_mask   = outer_in & (~lumen_mask)
    outside_mask = ~outer_in

    band_mask = (torch.abs(phi_in_gt) < band_mm) | (torch.abs(phi_out_gt) < band_mm)

    loss_sdf = F.l1_loss(phi_in, phi_in_gt) + F.l1_loss(phi_out, phi_out_gt)

    loss_eik = 0.5 * eikonal_loss_masked(phi_in,  x_norm, vol_in_sdf, band_mask) + \
               0.5 * eikonal_loss_masked(phi_out, x_norm, vol_in_sdf, band_mask)

    loss_bc = torch.tensor(0.0, device=device)
    if bc_inner_pts_mm.numel() > 0:
        xbi = vol_in_sdf.normalize_input(bc_inner_pts_mm).requires_grad_(False)
        phi_in_bc = model(xbi)[:, 0]
        loss_bc = loss_bc + (phi_in_bc ** 2).mean()
    if bc_outer_pts_mm.numel() > 0:
        xbo = vol_in_sdf.normalize_input(bc_outer_pts_mm).requires_grad_(False)
        phi_out_bc = model(xbo)[:, 1]
        loss_bc = loss_bc + (phi_out_bc ** 2).mean()

    loss_enc = masked_mean(F.relu(phi_out), lumen_mask)
    loss_order = masked_mean(F.relu(-phi_in), outside_mask)

    t = phi_in - phi_out
    loss_minT = masked_mean(F.relu(t_min_case_mm - t), wall_mask)
    loss_smoothT = smooth_thickness_loss_masked(phi_in, phi_out, x_norm, vol_in_sdf, wall_mask)

    loss_lumen_in = masked_mean(F.relu(phi_in), lumen_mask)

    loss_total = (
        weights["sdf"] * loss_sdf +
        weights["eik"] * loss_eik +
        weights["bc"]  * loss_bc +
        weights["enc"] * loss_enc +
        weights["order"] * loss_order +
        weights["minT"] * loss_minT +
        weights["smoothT"] * loss_smoothT +
        weights["lumen_in"] * loss_lumen_in
    )

    loss_total.backward()
    opt.step()

    stats = {
        "sdf": float(loss_sdf.detach().cpu()),
        "eik": float(loss_eik.detach().cpu()),
        "bc": float(loss_bc.detach().cpu()),
        "enc": float(loss_enc.detach().cpu()),
        "order": float(loss_order.detach().cpu()),
        "minT": float(loss_minT.detach().cpu()),
        "smoothT": float(loss_smoothT.detach().cpu()),
        "lumen_in": float(loss_lumen_in.detach().cpu()),
    }
    return float(loss_total.detach().cpu()), stats


# ============================================================
# Post-training QC metrics on predicted grids
# ============================================================
@torch.no_grad()
def compute_predicted_thickness_metrics(
    vol_in_ref: Volume,
    vol_in_bin: Volume,
    vol_out_bin: Volume,
    pin_grid: np.ndarray,
    pout_grid: np.ndarray,
    t_min_case_mm: float
) -> Dict[str, float]:
    D, H, W = vol_in_ref.size_zyx

    origin = torch.tensor(vol_in_ref.origin, device=vol_in_ref.device, dtype=torch.float32)
    spacing = torch.tensor(vol_in_ref.spacing, device=vol_in_ref.device, dtype=torch.float32)
    direction = torch.tensor(vol_in_ref.direction, device=vol_in_ref.device, dtype=torch.float32)

    pin = torch.from_numpy(pin_grid.reshape(-1)).to(vol_in_ref.device)
    pout = torch.from_numpy(pout_grid.reshape(-1)).to(vol_in_ref.device)
    t_pred = (pin - pout)

    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=vol_in_ref.device),
        torch.arange(H, device=vol_in_ref.device),
        torch.arange(W, device=vol_in_ref.device),
        indexing="ij"
    )
    ijk = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1).float()
    xyz = origin + ((ijk * spacing) @ direction.T)

    in_lumen = (vol_in_bin.sample_trilinear(xyz, threshold_binary=True) > 0.5)
    in_outer = (vol_out_bin.sample_trilinear(xyz, threshold_binary=True) > 0.5)
    wall = in_outer & (~in_lumen)

    if not wall.any():
        return {
            "pred_wall_count": 0,
            "pred_t_mean": float("nan"),
            "pred_t_std": float("nan"),
            "pred_t_min": float("nan"),
            "pred_t_p05": float("nan"),
            "pred_t_p10": float("nan"),
            "pred_t_p50": float("nan"),
            "pred_t_violation_rate": float("nan"),
        }

    tw = t_pred[wall].detach().cpu().numpy()
    tw = tw[np.isfinite(tw)]
    if tw.size == 0:
        return {
            "pred_wall_count": int(wall.sum().item()),
            "pred_t_mean": float("nan"),
            "pred_t_std": float("nan"),
            "pred_t_min": float("nan"),
            "pred_t_p05": float("nan"),
            "pred_t_p10": float("nan"),
            "pred_t_p50": float("nan"),
            "pred_t_violation_rate": float("nan"),
        }

    p05 = float(np.percentile(tw, 5))
    p10 = float(np.percentile(tw, 10))
    p50 = float(np.percentile(tw, 50))
    viol = float(np.mean(tw < t_min_case_mm))

    return {
        "pred_wall_count": int(tw.size),
        "pred_t_mean": float(np.mean(tw)),
        "pred_t_std": float(np.std(tw)),
        "pred_t_min": float(np.min(tw)),
        "pred_t_p05": p05,
        "pred_t_p10": p10,
        "pred_t_p50": p50,
        "pred_t_violation_rate": viol,
    }


# ============================================================
# Main training loop
# ============================================================
def train_case(
    case_dir: Path,
    out_root: Path,
    device: torch.device,
    export_final: bool,
    export_each_stage: bool,
    seed: int,
    bc_points_per_step: int,
    bc_cache_points: int,
) -> None:
    inner_sdf_path = case_dir / "INNER_SDF.nrrd"
    outer_sdf_path = case_dir / "OUTER_SDF.nrrd"
    inner_bin_path = case_dir / "INNER_trimmed.nrrd"
    outer_bin_path = case_dir / "OUTER_contained.nrrd"

    for p in [inner_sdf_path, outer_sdf_path, inner_bin_path, outer_bin_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    vol_in_sdf  = Volume.from_nrrd(inner_sdf_path, device=device, is_binary=False)
    vol_out_sdf = Volume.from_nrrd(outer_sdf_path, device=device, is_binary=False)
    vol_in_bin  = Volume.from_nrrd(inner_bin_path, device=device, is_binary=True)
    vol_out_bin = Volume.from_nrrd(outer_bin_path, device=device, is_binary=True)

    priors = estimate_thickness_priors(
        vol_in_sdf=vol_in_sdf,
        vol_out_sdf=vol_out_sdf,
        vol_in_bin=vol_in_bin,
        vol_out_bin=vol_out_bin,
        n_probe=120000,
        seed=seed,
        floor_tmin_mm=0.15,
        tmin_scale_p05=0.90,
        thin_thr_p20=1.10,
    )
    t_min_case = float(priors["t_min_case"])
    thin_thr = float(priors["thin_threshold"])

    logging.info(
        f"{case_dir.name} | thickness priors: t_p05={priors['t_p05']:.4f} "
        f"t_p20={priors['t_p20']:.4f} t_min_case={t_min_case:.4f} thin_thr={thin_thr:.4f}"
    )

    inner_bc_pool = binary_surface_points_mm(vol_in_bin, n_points=bc_cache_points, seed=seed + 11)
    outer_bc_pool = binary_surface_points_mm(vol_out_bin, n_points=bc_cache_points, seed=seed + 17)

    inner_bc_pool_t = torch.from_numpy(inner_bc_pool).to(device=device, dtype=torch.float32)
    outer_bc_pool_t = torch.from_numpy(outer_bc_pool).to(device=device, dtype=torch.float32)

    model = MLP2SDF(hidden=256, layers=6, pe_freqs=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CURRICULUM[0]["lr"])

    case_out = out_root / case_dir.name
    ckpt_dir = case_out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = case_out / "train_metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "global_step", "stage", "stage_step",
                "loss_total", "loss_sdf", "loss_eik", "loss_bc", "loss_enc", "loss_order",
                "loss_minT", "loss_smoothT", "loss_lumen_in",
                "t_min_case_mm", "thin_threshold_mm"
            ])

    best_loss = float("inf")
    global_step = 0

    logging.info(f"Training case: {case_dir.name} on {device} | stages={len(CURRICULUM)}")

    # ============================================================
    # [ADDED] runtime + GPU accounting (kept separate, does not affect training)
    # ============================================================
    run_start = time.perf_counter()
    stage_times_sec: Dict[str, float] = {}
    device_info = get_device_info(device)

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    # ============================================================

    def sample_bc(pool: torch.Tensor, n: int) -> torch.Tensor:
        if pool.numel() == 0:
            return torch.zeros((0, 3), device=device, dtype=torch.float32)
        if pool.shape[0] <= n:
            return pool
        idx = torch.randint(0, pool.shape[0], (n,), device=device)
        return pool[idx]

    for cfg in CURRICULUM:
        stage_name = cfg["name"]
        # [ADDED] stage timer start
        stage_start = time.perf_counter()

        steps = int(cfg["steps"])
        lr = float(cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        band_mm = float(cfg["band_mm"])
        n_total = int(cfg["n_total"])

        ratios = {
            "inner": float(cfg["r_inner"]),
            "outer": float(cfg["r_outer"]),
            "wall":  float(cfg["r_wall"]),
            "thin":  float(cfg["r_thin"]),
            "lumen": float(cfg["r_lumen"]),
        }

        weights = {
            "sdf": float(cfg["w_sdf"]),
            "eik": float(cfg["w_eik"]),
            "bc":  float(cfg["w_bc"]),
            "enc": float(cfg["w_enc"]),
            "order": float(cfg["w_order"]),
            "minT": float(cfg["w_minT"]),
            "smoothT": float(cfg["w_smoothT"]),
            "lumen_in": float(cfg["w_lumen_in"]),
        }

        logging.info(
            f"{case_dir.name} | {stage_name} | steps={steps} lr={lr} n_total={n_total} band={band_mm}mm "
            f"| w_sdf={weights['sdf']} w_eik={weights['eik']} w_bc={weights['bc']} w_enc={weights['enc']} "
            f"w_order={weights['order']} w_minT={weights['minT']} w_smoothT={weights['smoothT']} "
            f"w_lumen_in={weights['lumen_in']} | t_min_case={t_min_case:.4f} thin_thr={thin_thr:.4f}"
        )

        stage_best = float("inf")

        for stage_step in range(1, steps + 1):
            model.train()
            opt.zero_grad(set_to_none=True)

            bc_in = sample_bc(inner_bc_pool_t, bc_points_per_step)
            bc_out = sample_bc(outer_bc_pool_t, bc_points_per_step)

            try:
                loss_total, stats = one_train_step(
                    model=model, opt=opt,
                    vol_in_sdf=vol_in_sdf, vol_out_sdf=vol_out_sdf,
                    vol_in_bin=vol_in_bin, vol_out_bin=vol_out_bin,
                    bc_inner_pts_mm=bc_in, bc_outer_pts_mm=bc_out,
                    band_mm=band_mm,
                    n_total=n_total,
                    t_min_case_mm=t_min_case,
                    thin_threshold_mm=thin_thr,
                    ratios=ratios,
                    weights=weights,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logging.warning(
                    f"{case_dir.name} | {stage_name} | CUDA OOM at n_total={n_total}. "
                    f"Retrying with n_total={OOM_FALLBACK_NTOTAL}."
                )
                loss_total, stats = one_train_step(
                    model=model, opt=opt,
                    vol_in_sdf=vol_in_sdf, vol_out_sdf=vol_out_sdf,
                    vol_in_bin=vol_in_bin, vol_out_bin=vol_out_bin,
                    bc_inner_pts_mm=bc_in, bc_outer_pts_mm=bc_out,
                    band_mm=band_mm,
                    n_total=OOM_FALLBACK_NTOTAL,
                    t_min_case_mm=t_min_case,
                    thin_threshold_mm=thin_thr,
                    ratios=ratios,
                    weights=weights,
                )

            global_step += 1

            if stage_step == 1 or stage_step % 50 == 0 or stage_step == steps:
                lt = float(loss_total)
                logging.info(
                    f"{case_dir.name} | {stage_name} | step {stage_step:05d}/{steps} | gstep {global_step:06d} | "
                    f"total={lt:.6f} sdf={stats['sdf']:.6f} eik={stats['eik']:.6f} bc={stats['bc']:.6f} "
                    f"enc={stats['enc']:.6f} order={stats['order']:.6f} minT={stats['minT']:.6f} "
                    f"smoothT={stats['smoothT']:.6f} lumen_in={stats['lumen_in']:.6f}"
                )

                with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        global_step, stage_name, stage_step,
                        lt,
                        stats["sdf"], stats["eik"], stats["bc"], stats["enc"], stats["order"],
                        stats["minT"], stats["smoothT"], stats["lumen_in"],
                        t_min_case, thin_thr
                    ])

                stage_last = ckpt_dir / f"{stage_name}_last.pt"
                save_ckpt(stage_last, model, opt, global_step, best_loss, stage_name, cfg, priors)

                if lt < stage_best:
                    stage_best = lt
                    stage_best_path = ckpt_dir / f"{stage_name}_best.pt"
                    save_ckpt(stage_best_path, model, opt, global_step, best_loss, stage_name, cfg, priors)

                if lt < best_loss:
                    best_loss = lt
                    best_path = ckpt_dir / "best.pt"
                    save_ckpt(best_path, model, opt, global_step, best_loss, stage_name, cfg, priors)

        # [ADDED] stage timer end
        stage_times_sec[stage_name] = float(time.perf_counter() - stage_start)

        if export_each_stage:
            model.eval()
            pin_grid, pout_grid = predict_on_inner_grid(model, vol_in_sdf, chunk=GRID_CHUNK)
            np.save(case_out / f"{stage_name}_pred_inner_sdf.npy", pin_grid)
            np.save(case_out / f"{stage_name}_pred_outer_sdf.npy", pout_grid)
            export_meshes(case_out, vol_in_sdf, pin_grid, pout_grid, prefix=f"{stage_name}_")
            logging.info(f"{case_dir.name} | {stage_name} | Exported stage grids/meshes to: {case_out}")

    # [ADDED] finalize overall runtime and GPU peak memory
    train_time_sec = float(time.perf_counter() - run_start)

    gpu_peak_mem_mb = None
    gpu_peak_reserved_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            gpu_peak_mem_mb = _bytes_to_mb(int(torch.cuda.max_memory_allocated()))
            gpu_peak_reserved_mb = _bytes_to_mb(int(torch.cuda.max_memory_reserved()))
        except Exception:
            gpu_peak_mem_mb = None
            gpu_peak_reserved_mb = None

    qc = {
        "case": case_dir.name,
        "device": str(device),
        "device_info": device_info,                     # [ADDED]
        "train_time_sec": train_time_sec,               # [ADDED]
        "stage_times_sec": stage_times_sec,             # [ADDED]
        "gpu_peak_mem_mb": gpu_peak_mem_mb,             # [ADDED]
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,   # [ADDED]
        "thickness_priors": priors,
    }

    if export_final:
        model.eval()
        pin_grid, pout_grid = predict_on_inner_grid(model, vol_in_sdf, chunk=GRID_CHUNK)

        np.save(case_out / "pred_inner_sdf.npy", pin_grid)
        np.save(case_out / "pred_outer_sdf.npy", pout_grid)

        mesh_flags = export_meshes(case_out, vol_in_sdf, pin_grid, pout_grid)
        qc.update(mesh_flags)

        pred_t = compute_predicted_thickness_metrics(
            vol_in_ref=vol_in_sdf,
            vol_in_bin=vol_in_bin,
            vol_out_bin=vol_out_bin,
            pin_grid=pin_grid,
            pout_grid=pout_grid,
            t_min_case_mm=t_min_case
        )
        qc.update(pred_t)

        qc_path = case_out / "qc_metrics.json"
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2)

        logging.info(f"{case_dir.name} | Exported FINAL outputs to: {case_out}")
        logging.info(
            f"{case_dir.name} | QC: inner_mesh_ok={qc.get('inner_mesh_ok')} outer_mesh_ok={qc.get('outer_mesh_ok')} "
            f"pred_t_violation_rate={qc.get('pred_t_violation_rate')}"
        )
    else:
        qc_path = case_out / "qc_metrics.json"
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-SDF PINN reconstruction v2 (PDE + BC + binary constraints + thin sampling).")
    ap.add_argument("--case-dir", required=True, type=str, help="Path to a single case folder.")
    ap.add_argument("--out-root", required=True, type=str, help="Output root folder.")
    ap.add_argument("--export-final", action="store_true", help="Export final grids/meshes/QC at the end.")
    ap.add_argument("--export-each-stage", action="store_true", help="Also export stage grids/meshes.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bc-points-per-step", type=int, default=4096, help="BC points sampled per step per surface.")
    ap.add_argument("--bc-cache-points", type=int, default=200000, help="BC pool size per surface (marching cubes sampled once).")
    args = ap.parse_args()

    setup_logger(args.log_level)
    set_determinism(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    case_dir = Path(args.case_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_case(
        case_dir=case_dir,
        out_root=out_root,
        device=device,
        export_final=args.export_final,
        export_each_stage=args.export_each_stage,
        seed=args.seed,
        bc_points_per_step=args.bc_points_per_step,
        bc_cache_points=args.bc_cache_points,
    )


if __name__ == "__main__":
    main()