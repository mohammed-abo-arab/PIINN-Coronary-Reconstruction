#!/usr/bin/env python3
"""
train_vanilla_inr_sdf_reconstruction_corrected.py

Vanilla INR baseline for dual-SDF coronary artery reconstruction.

Design goals
------------
This script is the *controlled ablation* of the PIINN implementation:
- same coordinate-based dual-output MLP backbone
- same physical-space sampling framework
- same canonical export on the INNER grid
- NO physics-informed losses
- NO boundary losses
- NO enclosure / thickness / lumen losses
- NO staged PINN curriculum
- training objective = supervised SDF regression only
- DeepSDF-style refinements preserved within Vanilla INR: stronger near-surface
  sampling, clamped SDF regression, and slight inner-surface weighting

Required inputs per case folder
-------------------------------
INNER_SDF.nrrd
OUTER_SDF.nrrd

Optional inputs per case folder (used only for final QC thickness metrics)
-------------------------------------------------------------------------
INNER_trimmed.nrrd
OUTER_contained.nrrd

Outputs
-------
<out_root>/<case_name>/
  checkpoints/
    best.pt
    last.pt
    iter_XXXXXX.pt              (optional, if --save-every > 0)
  train_metrics.csv
  qc_metrics.json
  pred_inner_sdf.npy            (if --export-final)
  pred_outer_sdf.npy            (if --export-final)
  inner_mesh.ply                (if --export-final)
  outer_mesh.ply                (if --export-final)

Notes
-----
- INNER and OUTER volumes may have different spacing/origin/direction.
  All training samples are generated in physical space (mm), then each GT SDF
  volume is sampled with its own physical transform.
- The network input normalization and exported prediction grid follow the INNER
  SDF reference grid, exactly as in the PIINN implementation.
- This is a DeepSDF-style vanilla coordinate-based INR baseline, but not a
  literal reimplementation of the DeepSDF paper. It preserves your backbone and
  data pipeline for a fair comparison against PIINN.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import marching_cubes


# ============================================================
# Vanilla INR training configuration
# ============================================================
DEFAULT_OUT_ROOT = r"D:\Mo_PINNs\Processed\outputs_Vanilla_INR"
GRID_CHUNK = 200000
OOM_FALLBACK_NTOTAL = 25000

TRAIN_CONFIG = {
    # Match the total PIINN step budget: 3500 + 6000 + 4000 = 13500
    "steps": 13500,
    "lr": 1e-4,
    "band_mm": 1.0,
    "n_total": 45000,
    "clamp_delta_mm": 2.0,
    "w_inner": 1.5,
    "w_outer": 1.0,
    # Pure SDF sampling with stronger near-surface emphasis.
    # Sum must be <= 1.0
    "r_inner": 0.42,
    "r_outer": 0.42,
    "r_bg": 0.16,
}


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
# Device helpers
# ============================================================
def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def get_device_info(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = int(torch.cuda.current_device())
        except Exception:
            idx = 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "type": "cuda",
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_bytes": int(props.total_memory),
                "capability": [int(props.major), int(props.minor)],
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    else:
        info.update({"type": "cpu"})
    return info


# ============================================================
# Volume wrapper
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

        d, h, w = arr.shape
        t = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        corners_ijk = np.array(
            [
                [0, 0, 0],
                [w - 1, 0, 0],
                [0, h - 1, 0],
                [0, 0, d - 1],
                [w - 1, h - 1, 0],
                [w - 1, 0, d - 1],
                [0, h - 1, d - 1],
                [w - 1, h - 1, d - 1],
            ],
            dtype=np.float64,
        )  # (x,y,z)

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
            size_zyx=(d, h, w),
            center_mm=center,
            half_mm=half,
            device=device,
            is_binary=is_binary,
        )

    def normalize_input(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        return (xyz_mm - self.center_mm) / self.half_mm

    def sample_trilinear(self, xyz_mm: torch.Tensor, threshold_binary: bool = True) -> torch.Tensor:
        origin = torch.tensor(self.origin, device=self.device, dtype=torch.float32)
        spacing = torch.tensor(self.spacing, device=self.device, dtype=torch.float32)
        direction = torch.tensor(self.direction, device=self.device, dtype=torch.float32)
        inv_dir = torch.inverse(direction)

        v = xyz_mm - origin
        ijk = (v @ inv_dir.T) / spacing  # (N,3) -> (x,y,z)

        d, h, w = self.size_zyx
        x = ijk[:, 0]
        y = ijk[:, 1]
        z = ijk[:, 2]

        gx = 2.0 * (x / (w - 1)) - 1.0
        gy = 2.0 * (y / (h - 1)) - 1.0
        gz = 2.0 * (z / (d - 1)) - 1.0

        grid = torch.stack([gx, gy, gz], dim=-1).view(1, 1, -1, 1, 3)
        val = F.grid_sample(
            self.tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
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
        return pe.view(x.shape[0], -1)


class MLP2SDF(nn.Module):
    """
    Same backbone as the PIINN implementation.
    The baseline differs from PIINN only in the objective.
    """

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
# Sampling utilities
# ============================================================
def sample_uniform_physical(
    center_mm: torch.Tensor,
    half_mm: torch.Tensor,
    device: torch.device,
    n: int,
) -> torch.Tensor:
    x_norm = (torch.rand((n, 3), device=device) * 2.0 - 1.0).float()
    return x_norm * half_mm + center_mm


def rejection_sample_physical(
    center_mm: torch.Tensor,
    half_mm: torch.Tensor,
    device: torch.device,
    n: int,
    predicate_fn,
    max_tries: int = 120,
    oversample: float = 4.0,
) -> torch.Tensor:
    collected: List[torch.Tensor] = []
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
# Optional thickness priors / QC helpers
# ============================================================
def _physical_bbox_from_binary_mask(vol_bin: Volume, pad_mm: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = vol_bin.tensor.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    m = arr > 0.5
    if not np.any(m):
        return vol_bin.center_mm, vol_bin.half_mm

    zz, yy, xx = np.where(m)
    z0, z1 = int(zz.min()), int(zz.max())
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())

    origin = np.array(vol_bin.origin, dtype=np.float64)
    spacing = np.array(vol_bin.spacing, dtype=np.float64)
    direction = np.array(vol_bin.direction, dtype=np.float64)

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y1, z0],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    )

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
    floor_tmin_mm: float = 0.15,
    tmin_scale_p05: float = 0.90,
    thin_thr_p20: float = 1.10,
) -> Dict[str, float]:
    device = vol_in_sdf.device
    center, half = _physical_bbox_from_binary_mask(vol_out_bin, pad_mm=4.0)

    def pred_wall(x_mm: torch.Tensor) -> torch.Tensor:
        in_lumen = vol_in_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        in_outer = vol_out_bin.sample_trilinear(x_mm, threshold_binary=True) > 0.5
        return in_outer & (~in_lumen)

    n_wall_target = int(min(max(8000, n_probe // 10), 60000))
    xw = rejection_sample_physical(
        center_mm=center,
        half_mm=half,
        device=device,
        n=n_wall_target,
        predicate_fn=pred_wall,
        max_tries=140,
        oversample=4.0,
    )

    wall_mask = pred_wall(xw)
    if not wall_mask.any():
        logging.warning("Thickness priors fallback: no wall points found.")
        return {
            "t_p05": 0.30,
            "t_p10": 0.35,
            "t_p20": 0.40,
            "t_med": 0.60,
            "t_min_case": max(floor_tmin_mm, 0.20),
            "thin_threshold": 0.35,
        }

    xw = xw[wall_mask]
    pin = vol_in_sdf.sample_trilinear(xw, threshold_binary=False)
    pout = vol_out_sdf.sample_trilinear(xw, threshold_binary=False)
    t = (pin - pout).detach().cpu().numpy()
    t = t[np.isfinite(t)]

    if t.size >= 50:
        lo = np.percentile(t, 1)
        hi = np.percentile(t, 99)
        t = t[(t >= lo) & (t <= hi)]

    if t.size < 1000:
        logging.warning(f"Thickness priors fallback: insufficient valid wall samples (t.size={t.size}).")
        return {
            "t_p05": 0.30,
            "t_p10": 0.35,
            "t_p20": 0.40,
            "t_med": 0.60,
            "t_min_case": max(floor_tmin_mm, 0.20),
            "thin_threshold": 0.35,
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
# Export helpers
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
    d, h, w = vol_inner_ref.size_zyx

    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=vol_inner_ref.device),
        torch.arange(h, device=vol_inner_ref.device),
        torch.arange(w, device=vol_inner_ref.device),
        indexing="ij",
    )
    ijk = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1).float()

    origin = torch.tensor(vol_inner_ref.origin, device=vol_inner_ref.device, dtype=torch.float32)
    spacing = torch.tensor(vol_inner_ref.spacing, device=vol_inner_ref.device, dtype=torch.float32)
    direction = torch.tensor(vol_inner_ref.direction, device=vol_inner_ref.device, dtype=torch.float32)

    v = ijk * spacing
    xyz = origin + (v @ direction.T)
    x_norm = vol_inner_ref.normalize_input(xyz)

    pin_list: List[torch.Tensor] = []
    pout_list: List[torch.Tensor] = []
    for i in range(0, x_norm.shape[0], chunk):
        y = model(x_norm[i:i + chunk])
        pin_list.append(y[:, 0].detach().cpu())
        pout_list.append(y[:, 1].detach().cpu())

    pin = torch.cat(pin_list).numpy().reshape(d, h, w)
    pout = torch.cat(pout_list).numpy().reshape(d, h, w)
    return pin, pout


def export_meshes(case_out: Path, vol_inner_ref: Volume, pin: np.ndarray, pout: np.ndarray) -> Dict[str, int]:
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
        write_ply(case_out / "inner_mesh.ply", to_phys(vi), fi)
        out_flags["inner_mesh_ok"] = 1

    vo, fo = mc_safe(pout, "pred_outer")
    if vo is not None:
        write_ply(case_out / "outer_mesh.ply", to_phys(vo), fo)
        out_flags["outer_mesh_ok"] = 1

    return out_flags


# ============================================================
# Checkpointing
# ============================================================
def save_ckpt(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    global_step: int,
    best_loss: float,
    train_cfg: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "best_loss": best_loss,
            "train_cfg": train_cfg,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        },
        path,
    )


# ============================================================
# Training step
# ============================================================
def one_train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    vol_in_sdf: Volume,
    vol_out_sdf: Volume,
    band_mm: float,
    n_total: int,
    ratios: Dict[str, float],
    clamp_delta_mm: float,
    w_inner: float,
    w_outer: float,
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

    r_inner = float(ratios["inner"])
    r_outer = float(ratios["outer"])
    r_bg = float(ratios["bg"])

    r_sum = r_inner + r_outer + r_bg
    if r_sum > 1.0 + 1e-8:
        raise ValueError(f"Sampling ratios sum to > 1.0 (got {r_sum:.3f}).")

    n_inner = int(r_inner * n_total)
    n_outer = int(r_outer * n_total)
    n_bg = n_total - (n_inner + n_outer)
    if n_bg < 0:
        raise ValueError("n_bg became negative; check n_total and ratios.")

    x_inner = rejection_sample_physical(center, half, device, n_inner, pred_inner_band)
    x_outer = rejection_sample_physical(center, half, device, n_outer, pred_outer_band)
    x_bg = sample_uniform_physical(center, half, device, n_bg)

    x_mm = torch.cat([x_inner, x_outer, x_bg], dim=0)
    x_norm = vol_in_sdf.normalize_input(x_mm)

    phi_in_gt = vol_in_sdf.sample_trilinear(x_mm, threshold_binary=False).detach()
    phi_out_gt = vol_out_sdf.sample_trilinear(x_mm, threshold_binary=False).detach()

    y = model(x_norm)
    phi_in = y[:, 0]
    phi_out = y[:, 1]

    phi_in_gt_c = torch.clamp(phi_in_gt, -clamp_delta_mm, clamp_delta_mm)
    phi_out_gt_c = torch.clamp(phi_out_gt, -clamp_delta_mm, clamp_delta_mm)
    phi_in_c = torch.clamp(phi_in, -clamp_delta_mm, clamp_delta_mm)
    phi_out_c = torch.clamp(phi_out, -clamp_delta_mm, clamp_delta_mm)

    loss_in = F.l1_loss(phi_in_c, phi_in_gt_c)
    loss_out = F.l1_loss(phi_out_c, phi_out_gt_c)
    loss_sdf = (w_inner * loss_in) + (w_outer * loss_out)

    loss_total = loss_sdf
    loss_total.backward()
    opt.step()

    stats = {
        "sdf": float(loss_sdf.detach().cpu()),
        "sdf_inner": float(loss_in.detach().cpu()),
        "sdf_outer": float(loss_out.detach().cpu()),
    }
    return float(loss_total.detach().cpu()), stats


# ============================================================
# QC metrics on predicted grids (optional, needs binaries)
# ============================================================
@torch.no_grad()
def compute_predicted_thickness_metrics(
    vol_in_ref: Volume,
    vol_in_bin: Volume,
    vol_out_bin: Volume,
    pin_grid: np.ndarray,
    pout_grid: np.ndarray,
    t_min_case_mm: float,
) -> Dict[str, float]:
    d, h, w = vol_in_ref.size_zyx

    origin = torch.tensor(vol_in_ref.origin, device=vol_in_ref.device, dtype=torch.float32)
    spacing = torch.tensor(vol_in_ref.spacing, device=vol_in_ref.device, dtype=torch.float32)
    direction = torch.tensor(vol_in_ref.direction, device=vol_in_ref.device, dtype=torch.float32)

    pin = torch.from_numpy(pin_grid.reshape(-1)).to(vol_in_ref.device)
    pout = torch.from_numpy(pout_grid.reshape(-1)).to(vol_in_ref.device)
    t_pred = pin - pout

    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=vol_in_ref.device),
        torch.arange(h, device=vol_in_ref.device),
        torch.arange(w, device=vol_in_ref.device),
        indexing="ij",
    )
    ijk = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1).float()
    xyz = origin + ((ijk * spacing) @ direction.T)

    in_lumen = vol_in_bin.sample_trilinear(xyz, threshold_binary=True) > 0.5
    in_outer = vol_out_bin.sample_trilinear(xyz, threshold_binary=True) > 0.5
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
    seed: int,
    save_every: int,
) -> None:
    inner_sdf_path = case_dir / "INNER_SDF.nrrd"
    outer_sdf_path = case_dir / "OUTER_SDF.nrrd"
    inner_bin_path = case_dir / "INNER_trimmed.nrrd"
    outer_bin_path = case_dir / "OUTER_contained.nrrd"

    for p in [inner_sdf_path, outer_sdf_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    has_binary_qc = inner_bin_path.exists() and outer_bin_path.exists()
    if not has_binary_qc:
        logging.warning(
            "Optional binary masks not found. Training will proceed normally, "
            "but thickness-based QC metrics will be skipped."
        )

    vol_in_sdf = Volume.from_nrrd(inner_sdf_path, device=device, is_binary=False)
    vol_out_sdf = Volume.from_nrrd(outer_sdf_path, device=device, is_binary=False)

    vol_in_bin: Optional[Volume] = None
    vol_out_bin: Optional[Volume] = None
    priors: Optional[Dict[str, float]] = None
    t_min_case: Optional[float] = None

    if has_binary_qc:
        vol_in_bin = Volume.from_nrrd(inner_bin_path, device=device, is_binary=True)
        vol_out_bin = Volume.from_nrrd(outer_bin_path, device=device, is_binary=True)
        priors = estimate_thickness_priors(
            vol_in_sdf=vol_in_sdf,
            vol_out_sdf=vol_out_sdf,
            vol_in_bin=vol_in_bin,
            vol_out_bin=vol_out_bin,
            n_probe=120000,
            floor_tmin_mm=0.15,
            tmin_scale_p05=0.90,
            thin_thr_p20=1.10,
        )
        t_min_case = float(priors["t_min_case"])
        logging.info(
            f"{case_dir.name} | QC priors: t_p05={priors['t_p05']:.4f} "
            f"t_p20={priors['t_p20']:.4f} t_min_case={t_min_case:.4f}"
        )

    model = MLP2SDF(hidden=256, layers=6, pe_freqs=10).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_nontrainable = n_params - n_trainable
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0)

    logging.info("==== Model Complexity ====")
    logging.info(f"Total parameters: {n_params:,}")
    logging.info(f"Trainable parameters: {n_trainable:,}")
    logging.info(f"Frozen parameters: {n_nontrainable:,}")
    logging.info(f"Model size (parameters only): {model_size_mb:.2f} MB")

    opt = torch.optim.Adam(model.parameters(), lr=float(TRAIN_CONFIG["lr"]))

    case_out = out_root / case_dir.name
    ckpt_dir = case_out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = case_out / "train_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["global_step", "loss_total", "loss_sdf", "loss_sdf_inner", "loss_sdf_outer", "lr", "n_total", "band_mm", "clamp_delta_mm", "w_inner", "w_outer"])

    best_loss = float("inf")
    global_step = 0

    steps = int(TRAIN_CONFIG["steps"])
    lr = float(TRAIN_CONFIG["lr"])
    band_mm = float(TRAIN_CONFIG["band_mm"])
    n_total = int(TRAIN_CONFIG["n_total"])
    ratios = {
        "inner": float(TRAIN_CONFIG["r_inner"]),
        "outer": float(TRAIN_CONFIG["r_outer"]),
        "bg": float(TRAIN_CONFIG["r_bg"]),
    }
    clamp_delta_mm = float(TRAIN_CONFIG["clamp_delta_mm"])
    w_inner = float(TRAIN_CONFIG["w_inner"])
    w_outer = float(TRAIN_CONFIG["w_outer"])

    logging.info(
        f"Training Vanilla INR case: {case_dir.name} on {device} | "
        f"steps={steps} lr={lr} n_total={n_total} band={band_mm}mm clamp={clamp_delta_mm}mm "
        f"w_inner={w_inner:.2f} w_outer={w_outer:.2f} | loss=clamped weighted SDF only"
    )

    run_start = time.perf_counter()
    device_info = get_device_info(device)

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        try:
            loss_total, stats = one_train_step(
                model=model,
                opt=opt,
                vol_in_sdf=vol_in_sdf,
                vol_out_sdf=vol_out_sdf,
                band_mm=band_mm,
                n_total=n_total,
                ratios=ratios,
                clamp_delta_mm=clamp_delta_mm,
                w_inner=w_inner,
                w_outer=w_outer,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logging.warning(
                f"{case_dir.name} | CUDA OOM at n_total={n_total}. "
                f"Retrying with n_total={OOM_FALLBACK_NTOTAL}."
            )
            opt.zero_grad(set_to_none=True)
            loss_total, stats = one_train_step(
                model=model,
                opt=opt,
                vol_in_sdf=vol_in_sdf,
                vol_out_sdf=vol_out_sdf,
                band_mm=band_mm,
                n_total=OOM_FALLBACK_NTOTAL,
                ratios=ratios,
                clamp_delta_mm=clamp_delta_mm,
                w_inner=w_inner,
                w_outer=w_outer,
            )

        global_step += 1

        if step == 1 or step % 50 == 0 or step == steps:
            logging.info(
                f"{case_dir.name} | step {step:05d}/{steps} | "
                f"loss={loss_total:.6f} sdf={stats['sdf']:.6f} in={stats['sdf_inner']:.6f} out={stats['sdf_outer']:.6f}"
            )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([global_step, loss_total, stats["sdf"], stats["sdf_inner"], stats["sdf_outer"], lr, n_total, band_mm, clamp_delta_mm, w_inner, w_outer])

        if loss_total < best_loss:
            best_loss = float(loss_total)
            save_ckpt(ckpt_dir / "best.pt", model, opt, global_step, best_loss, TRAIN_CONFIG)

        if save_every > 0 and (step % save_every == 0):
            save_ckpt(ckpt_dir / f"iter_{global_step:06d}.pt", model, opt, global_step, best_loss, TRAIN_CONFIG)

    save_ckpt(ckpt_dir / "last.pt", model, opt, global_step, best_loss, TRAIN_CONFIG)

    train_time_sec = float(time.perf_counter() - run_start)
    gpu_peak_mem_mb = None
    gpu_peak_reserved_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            gpu_peak_mem_mb = _bytes_to_mb(int(torch.cuda.max_memory_allocated()))
            gpu_peak_reserved_mb = _bytes_to_mb(int(torch.cuda.max_memory_reserved()))
        except Exception:
            pass

    time_per_iter_ms = (train_time_sec / max(1, steps)) * 1000.0

    qc: Dict[str, object] = {
        "case": case_dir.name,
        "device": str(device),
        "device_info": device_info,
        "train_time_sec": train_time_sec,
        "time_per_iter_ms": time_per_iter_ms,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "method": "Vanilla INR",
        "architecture": {
            "model": "MLP2SDF",
            "hidden": 256,
            "layers": 6,
            "pe_freqs": 10,
            "outputs": 2,
            "activation": "SiLU",
            "parameters_total": int(n_params),
            "parameters_trainable": int(n_trainable),
            "parameters_frozen": int(n_nontrainable),
            "model_size_mb": float(model_size_mb),
        },
        "training_config": TRAIN_CONFIG,
        "training_objective": "Weighted clamped L1 SDF regression only",
        "uses_binary_inputs_for_training": False,
        "uses_binary_inputs_for_qc_only": bool(has_binary_qc),
        "best_loss": best_loss,
    }
    if priors is not None:
        qc["thickness_priors"] = priors

    if export_final:
        model.eval()
        pin_grid, pout_grid = predict_on_inner_grid(model, vol_in_sdf, chunk=GRID_CHUNK)

        np.save(case_out / "pred_inner_sdf.npy", pin_grid)
        np.save(case_out / "pred_outer_sdf.npy", pout_grid)

        mesh_flags = export_meshes(case_out, vol_in_sdf, pin_grid, pout_grid)
        qc.update(mesh_flags)

        if has_binary_qc and vol_in_bin is not None and vol_out_bin is not None and t_min_case is not None:
            pred_t = compute_predicted_thickness_metrics(
                vol_in_ref=vol_in_sdf,
                vol_in_bin=vol_in_bin,
                vol_out_bin=vol_out_bin,
                pin_grid=pin_grid,
                pout_grid=pout_grid,
                t_min_case_mm=t_min_case,
            )
            qc.update(pred_t)
        else:
            qc["pred_wall_count"] = None
            qc["pred_t_mean"] = None
            qc["pred_t_std"] = None
            qc["pred_t_min"] = None
            qc["pred_t_p05"] = None
            qc["pred_t_p10"] = None
            qc["pred_t_p50"] = None
            qc["pred_t_violation_rate"] = None

        logging.info(f"{case_dir.name} | Exported FINAL Vanilla INR outputs to: {case_out}")
    else:
        logging.info(f"{case_dir.name} | Training completed. Final export skipped (--export-final not set).")

    with open(case_out / "qc_metrics.json", "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train updated Vanilla INR for dual-SDF coronary reconstruction (single-stage, clamped weighted SDF loss only)."
    )
    ap.add_argument("--case-dir", required=True, type=str, help="Path to a single case folder.")
    ap.add_argument(
        "--out-root",
        type=str,
        default=DEFAULT_OUT_ROOT,
        help=f"Output root folder. Default: {DEFAULT_OUT_ROOT}",
    )
    ap.add_argument("--export-final", action="store_true", help="Export final grids/meshes/QC at the end.")
    ap.add_argument("--save-every", type=int, default=0, help="Save iter_XXXXXX.pt every N steps. 0 disables periodic saves.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--seed", type=int, default=42)
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
        seed=args.seed,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
