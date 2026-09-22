#!/usr/bin/env python3
"""
validate_wall_thickness_revision.py
==================================
Revision-grade validation of wall-thickness fidelity and lumen/outer-wall
geometric consistency for the PIINN coronary reconstruction study.

WHY THIS SCRIPT EXISTS
----------------------
This analysis is designed to answer the reviewer concern that the outer wall
could be driven predominantly by geometric priors (e.g., a dilation of the
lumen) rather than by the case-specific CCTA-derived reference anatomy.

The script deliberately evaluates TWO complementary notions:

1) Dual-SDF separation-field fidelity (implementation-level validation)
   --------------------------------------------------------------------
   On the canonical INNER SDF grid, it compares

       t_ref_field  = phi_inner_ref - phi_outer_ref
       t_pred_field = phi_inner_pred - phi_outer_pred

   inside the REFERENCE wall region.  This is the same implicit separation
   quantity used by the PIINN minimum-thickness/smoothness formulation.  It is
   useful for testing whether the learned dual field follows the reference
   inner/outer SDF relationship.  It should NOT be presented as the only or
   exact anatomical wall-thickness definition.

2) Mesh-based wall-separation fidelity (paper/post-processing validation)
   -----------------------------------------------------------------------
   The manuscript defines thickness using shortest Euclidean distance from
   one surface to the opposite surface, symmetrically.  The script therefore
   samples the GT and predicted INNER/OUTER meshes area-weightedly and compares
   local wall separation using nearest-opposite-surface distances.

   To compare local values at approximately corresponding anatomical
   locations, GT surface anchors are mapped to the nearest point on the
   corresponding predicted surface; thickness is then measured from that
   predicted anchor to the opposite predicted surface.  Anchor-shift is
   reported as a quality-control measure for this correspondence.

3) Geometric consistency / nesting validation
   -------------------------------------------
   Reports voxel-level and surface-level violations of the intended nesting:
      predicted lumen subset of predicted outer vessel.
   It also reports minimum-thickness violation rate when a case-specific
   t_min is available from qc_metrics.json.

OUTPUTS
-------
<out_root>/
  run_manifest.json
  per_case_summary.csv
  aggregate_summary.csv
  cohort_statistics.json
  revision_validation.xlsx
  REPORT.md
  plots/
      case_mean_thickness_reference_vs_predicted.png
      case_thickness_mae.png
      case_nesting_violation.png
      cohort_mean_thickness_scatter.png
      cohort_mean_thickness_bland_altman.png
  per_case/<CASE>/
      metrics.json
      field_pointwise_sample.csv          (optional, controlled by --save-pointwise)
      mesh_pointwise_sample.csv           (optional, controlled by --save-pointwise)
      pointwise_data.npz                  (optional, controlled by --save-pointwise)

REQUIRED CASE LAYOUT
--------------------
Reference root (<gt_root>/<CASE>/):
  INNER_SDF.nrrd
  OUTER_SDF.nrrd
  INNER_trimmed.nrrd       [preferred; otherwise derived from INNER_SDF sign]
  OUTER_contained.nrrd     [preferred; otherwise derived from OUTER_SDF sign]

Prediction root (<pred_root>/<CASE>/):
  pred_inner_sdf.npy
  pred_outer_sdf.npy
  inner_mesh.ply           [preferred; generated from predicted SDF if absent]
  outer_mesh.ply           [preferred; generated from predicted SDF if absent]
  qc_metrics.json          [optional; used for t_min_case and saved QC metadata]

Optional GT mesh root (<gt_mesh_root>/<CASE>/):
  inner_gt_from_sdf.ply
  outer_gt_from_sdf.ply
If omitted or missing, GT meshes are generated directly from the native GT SDFs.

DEPENDENCIES
------------
  pip install numpy pandas scipy SimpleITK scikit-image trimesh matplotlib openpyxl

EXAMPLE
-------
python validate_wall_thickness_revision.py ^
  --gt-root "D:\\Mo_PINNs\\Processed" ^
  --pred-root "D:\\Mo_PINNs\\Processed\\outputs_pinn_v3" ^
  --gt-mesh-root "D:\\Mo_PINNs\\Processed\\gt_sdf_meshes" ^
  --out-root "D:\\Mo_PINNs\\Processed\\revision_wall_thickness_validation" ^
  --mesh-points 100000 ^
  --seed 42 ^
  --save-pointwise

IMPORTANT INTERPRETATION NOTE
-----------------------------
This script validates agreement with the CCTA-DERIVED REFERENCE ANNOTATIONS.
It does not establish that the model directly infers wall anatomy from raw CTA
intensities, because the submitted PIINN operates on pre-segmented reference
masks/SDFs rather than raw intensity volumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import trimesh
from scipy import __version__ as scipy_version
from scipy.ndimage import label as cc_label
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, wilcoxon
from skimage import __version__ as skimage_version
from skimage.measure import marching_cubes


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ImageGeometry:
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray
    direction: np.ndarray
    size_xyz: Tuple[int, int, int]
    shape_zyx: Tuple[int, int, int]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def stable_case_seed(base_seed: int, case_name: str) -> int:
    """Return a process-independent deterministic seed for one case.

    Python's built-in hash() is salted per interpreter process, so it is not
    appropriate for revision-grade reproducibility.  A SHA-256 digest gives a
    stable case-specific offset while preserving the user-supplied base seed.
    """
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    case_offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + case_offset) % (2**32 - 1))


def safe_float(x: object) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def percentile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def describe(arr: np.ndarray, prefix: str = "") -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        keys = ["n", "mean", "std", "median", "p05", "p10", "p20", "p25", "p75", "p90", "p95", "min", "max"]
        return {prefix + k: (0 if k == "n" else float("nan")) for k in keys}
    out = {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p05": percentile(a, 5),
        "p10": percentile(a, 10),
        "p20": percentile(a, 20),
        "p25": percentile(a, 25),
        "p75": percentile(a, 75),
        "p90": percentile(a, 90),
        "p95": percentile(a, 95),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }
    return {prefix + k: v for k, v in out.items()}


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    mx = np.mean(x)
    my = np.mean(y)
    cov = np.cov(x, y, ddof=1)[0, 1]
    den = vx + vy + (mx - my) ** 2
    return float(2.0 * cov / den) if den > 0 else float("nan")


def paired_agreement_metrics(ref: np.ndarray, pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[m], pred[m]
    if ref.size == 0:
        return {prefix + k: float("nan") for k in [
            "n", "mae_mm", "rmse_mm", "bias_mm", "median_abs_error_mm",
            "p95_abs_error_mm", "relative_mae_pct", "pearson_r", "spearman_rho",
            "ccc", "wasserstein_mm", "ba_loa_lower_mm", "ba_loa_upper_mm",
        ]}

    diff = pred - ref
    absdiff = np.abs(diff)
    mae = float(np.mean(absdiff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    denom = float(np.mean(np.abs(ref)))

    if ref.size >= 2 and np.std(ref) > 0 and np.std(pred) > 0:
        pr = float(pearsonr(ref, pred).statistic)
        sr = float(spearmanr(ref, pred).statistic)
    else:
        pr = sr = float("nan")

    out = {
        "n": int(ref.size),
        "mae_mm": mae,
        "rmse_mm": rmse,
        "bias_mm": bias,
        "median_abs_error_mm": float(np.median(absdiff)),
        "p95_abs_error_mm": percentile(absdiff, 95),
        "relative_mae_pct": float(100.0 * mae / denom) if denom > 0 else float("nan"),
        "pearson_r": pr,
        "spearman_rho": sr,
        "ccc": lin_ccc(ref, pred),
        "wasserstein_mm": float(wasserstein_distance(ref, pred)),
        "ba_loa_lower_mm": bias - 1.96 * sd_diff,
        "ba_loa_upper_mm": bias + 1.96 * sd_diff,
    }
    return {prefix + k: v for k, v in out.items()}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta, matching the effect-size convention used in the manuscript."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    return float((gt - lt) / (a.size * b.size))


def paired_method_comparison(
    df: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str, str, str]],
    seed: int,
) -> Dict[str, object]:
    """
    Paired case-level comparison between PIINN and the constant-offset
    comparator, using the same conventions as the manuscript: Wilcoxon signed
    rank, Holm correction across the family, and Cliff's delta.

    The case is the unit of analysis. Pointwise samples within a case are not
    independent, so no test is run on them.
    """
    results: Dict[str, object] = {}
    raw: List[Tuple[str, float]] = []
    for key, col_a, col_b, better in comparisons:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        pair = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            continue
        a = pair[col_a].to_numpy(dtype=np.float64)
        b = pair[col_b].to_numpy(dtype=np.float64)
        diff = a - b
        wins = int(np.sum(diff < 0)) if better == "lower" else int(np.sum(diff > 0))
        try:
            p = float(wilcoxon(a, b).pvalue)
        except Exception:
            p = float("nan")
        lo, hi = bootstrap_mean_ci(diff.tolist(), seed)
        results[key] = {
            "metric_piinn": col_a,
            "metric_comparator": col_b,
            "better_is": better,
            "n_cases": int(len(pair)),
            "piinn_mean": float(np.mean(a)),
            "piinn_sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "comparator_mean": float(np.mean(b)),
            "comparator_sd": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
            "mean_difference": float(np.mean(diff)),
            "mean_difference_bootstrap_95ci": [lo, hi],
            "median_difference": float(np.median(diff)),
            "cases_favouring_piinn": wins,
            "cliffs_delta": cliffs_delta(a, b),
            "wilcoxon_p": p,
        }
        raw.append((key, p))

    # Holm-Bonferroni across the family, as in the manuscript.
    valid = [(k, p) for k, p in raw if np.isfinite(p)]
    m = len(valid)
    for rank, (k, p) in enumerate(sorted(valid, key=lambda t: t[1])):
        results[k]["wilcoxon_p_holm"] = float(min(1.0, p * (m - rank)))
    for k, p in raw:
        results[k].setdefault("wilcoxon_p_holm", float("nan"))
    results["_family_size"] = m
    return results


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_boot: int = 10000) -> Tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = np.mean(a[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# -----------------------------------------------------------------------------
# Image IO / physical-space alignment
# -----------------------------------------------------------------------------
def image_geometry(img: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        origin_xyz=np.asarray(img.GetOrigin(), dtype=np.float64),
        spacing_xyz=np.asarray(img.GetSpacing(), dtype=np.float64),
        direction=np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3),
        size_xyz=tuple(int(v) for v in img.GetSize()),
        shape_zyx=tuple(int(v) for v in reversed(img.GetSize())),
    )


def read_float_image(path: Path) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat32)


def read_binary_image(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    arr = (sitk.GetArrayFromImage(img) > 0.5).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def array_from_image(img: sitk.Image, dtype=np.float32) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(dtype, copy=False)


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_binary: bool, default_value: float) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_binary else sitk.sitkLinear
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interp,
        float(default_value),
        moving.GetPixelID(),
    )


def derive_binary_from_sdf(sdf_img: sitk.Image) -> sitk.Image:
    arr = (array_from_image(sdf_img) <= 0.0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(sdf_img)
    return out


def load_reference_case(case_dir: Path) -> Dict[str, object]:
    in_sdf_path = case_dir / "INNER_SDF.nrrd"
    out_sdf_path = case_dir / "OUTER_SDF.nrrd"
    if not in_sdf_path.exists() or not out_sdf_path.exists():
        raise FileNotFoundError(f"Missing INNER_SDF.nrrd or OUTER_SDF.nrrd in {case_dir}")

    in_sdf_img = read_float_image(in_sdf_path)
    out_sdf_native_img = read_float_image(out_sdf_path)
    in_geom = image_geometry(in_sdf_img)

    # Resample OUTER GT SDF into the canonical INNER physical grid.
    outer_native_arr = array_from_image(out_sdf_native_img)
    finite_abs = np.abs(outer_native_arr[np.isfinite(outer_native_arr)])
    default_outer = float(np.max(finite_abs) + 10.0) if finite_abs.size else 100.0
    out_sdf_img = resample_to_reference(out_sdf_native_img, in_sdf_img, False, default_outer)

    in_bin_path = case_dir / "INNER_trimmed.nrrd"
    out_bin_path = case_dir / "OUTER_contained.nrrd"
    bin_source = "submitted_preprocessing_binaries"

    if in_bin_path.exists():
        in_bin_img = read_binary_image(in_bin_path)
        if image_geometry(in_bin_img).shape_zyx != in_geom.shape_zyx or in_bin_img.GetOrigin() != in_sdf_img.GetOrigin() or in_bin_img.GetSpacing() != in_sdf_img.GetSpacing() or in_bin_img.GetDirection() != in_sdf_img.GetDirection():
            in_bin_img = resample_to_reference(in_bin_img, in_sdf_img, True, 0)
    else:
        in_bin_img = derive_binary_from_sdf(in_sdf_img)
        bin_source = "derived_from_sdf_sign"

    if out_bin_path.exists():
        out_bin_native = read_binary_image(out_bin_path)
        out_bin_img = resample_to_reference(out_bin_native, in_sdf_img, True, 0)
    else:
        out_bin_img = derive_binary_from_sdf(out_sdf_img)
        bin_source = "derived_from_sdf_sign" if bin_source == "submitted_preprocessing_binaries" else bin_source

    return {
        "inner_sdf_img": in_sdf_img,
        "outer_sdf_img_native": out_sdf_native_img,
        "outer_sdf_img": out_sdf_img,
        "inner_bin_img": in_bin_img,
        "outer_bin_img": out_bin_img,
        "geometry": in_geom,
        "binary_source": bin_source,
    }


def load_predictions(case_dir: Path, expected_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    pin_path = case_dir / "pred_inner_sdf.npy"
    pout_path = case_dir / "pred_outer_sdf.npy"
    if not pin_path.exists() or not pout_path.exists():
        raise FileNotFoundError(f"Missing pred_inner_sdf.npy or pred_outer_sdf.npy in {case_dir}")
    pin = np.load(pin_path).astype(np.float32)
    pout = np.load(pout_path).astype(np.float32)
    if pin.shape != expected_shape or pout.shape != expected_shape:
        raise ValueError(
            f"Predicted grid shape mismatch in {case_dir.name}: "
            f"expected canonical INNER grid {expected_shape}, got inner={pin.shape}, outer={pout.shape}. "
            "This validation intentionally refuses shape-only zoom/resizing because physical-space alignment is required."
        )
    return pin, pout


# -----------------------------------------------------------------------------
# Mesh IO / generation / sampling
# -----------------------------------------------------------------------------
def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load_mesh(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        geoms = tuple(m.geometry.values())
        if not geoms:
            raise ValueError(f"Empty trimesh scene: {path}")
        m = trimesh.util.concatenate(geoms)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(m)}")
    if len(m.vertices) == 0 or len(m.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return m


def mesh_from_sdf_image(sdf_img: sitk.Image) -> trimesh.Trimesh:
    arr = array_from_image(sdf_img, np.float32)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (vmin <= 0.0 <= vmax):
        raise ValueError(f"SDF zero level not present: min={vmin}, max={vmax}")
    verts_zyx, faces, _, _ = marching_cubes(arr, level=0.0)
    geom = image_geometry(sdf_img)
    z, y, x = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]
    index_xyz = np.stack([x, y, z], axis=1)
    scaled = index_xyz * geom.spacing_xyz[None, :]
    verts_xyz = geom.origin_xyz[None, :] + scaled @ geom.direction.T
    return trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)


def mesh_from_sdf_array(arr_zyx: np.ndarray, reference_img: sitk.Image) -> trimesh.Trimesh:
    arr = np.asarray(arr_zyx, dtype=np.float32)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (vmin <= 0.0 <= vmax):
        raise ValueError(f"Predicted SDF zero level not present: min={vmin}, max={vmax}")
    verts_zyx, faces, _, _ = marching_cubes(arr, level=0.0)
    geom = image_geometry(reference_img)
    z, y, x = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]
    index_xyz = np.stack([x, y, z], axis=1)
    scaled = index_xyz * geom.spacing_xyz[None, :]
    verts_xyz = geom.origin_xyz[None, :] + scaled @ geom.direction.T
    return trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)


def choose_gt_mesh(case: str, gt_mesh_root: Optional[Path], which: str, sdf_img: sitk.Image) -> Tuple[trimesh.Trimesh, str]:
    if gt_mesh_root is not None:
        fname = "inner_gt_from_sdf.ply" if which == "inner" else "outer_gt_from_sdf.ply"
        p = gt_mesh_root / case / fname
        if p.exists():
            return load_mesh(p), str(p)
    return mesh_from_sdf_image(sdf_img), "generated_from_native_reference_sdf"


def choose_pred_mesh(case: str, pred_root: Path, which: str, pred_arr: np.ndarray, canonical_img: sitk.Image) -> Tuple[trimesh.Trimesh, str]:
    fname = "inner_mesh.ply" if which == "inner" else "outer_mesh.ply"
    p = pred_root / case / fname
    if p.exists():
        return load_mesh(p), str(p)
    return mesh_from_sdf_array(pred_arr, canonical_img), "generated_from_predicted_sdf_grid"


def sample_surface_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(np.sum(areas))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has non-positive total area")
    p = areas / total
    fidx = rng.choice(len(faces), size=int(n), replace=True, p=p)
    tri_sel = tri[fidx]
    u = rng.random(int(n))
    v = rng.random(int(n))
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    pts = w0[:, None] * tri_sel[:, 0] + w1[:, None] * tri_sel[:, 1] + w2[:, None] * tri_sel[:, 2]
    return pts.astype(np.float64)


def nearest_dist(query: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(target)
    d, idx = tree.query(query, k=1, workers=-1)
    return d.astype(np.float64), idx.astype(np.int64)


def surface_distance_metrics(gt_pts: np.ndarray, pred_pts: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Symmetric surface-distance metrics using the same definitions as the
    submitted evaluation_mesh.py, so numbers are directly comparable with the
    ASSD/HD95 values already reported in the manuscript.
    """
    if gt_pts.size == 0 or pred_pts.size == 0:
        return {f"{prefix}assd_mm": float("nan"), f"{prefix}hd95_mm": float("nan"),
                f"{prefix}gt_to_pred_mean_mm": float("nan"), f"{prefix}pred_to_gt_mean_mm": float("nan")}
    d_gp, _ = nearest_dist(gt_pts, pred_pts)
    d_pg, _ = nearest_dist(pred_pts, gt_pts)
    return {
        f"{prefix}assd_mm": float(0.5 * (np.mean(d_gp) + np.mean(d_pg))),
        f"{prefix}hd95_mm": float(max(np.percentile(d_gp, 95), np.percentile(d_pg, 95))),
        f"{prefix}gt_to_pred_mean_mm": float(np.mean(d_gp)),
        f"{prefix}pred_to_gt_mean_mm": float(np.mean(d_pg)),
    }


def point_cloud_resolution(pts: np.ndarray, rng: np.random.Generator, n_probe: int = 5000) -> float:
    """
    Median nearest-neighbour spacing within a sampled surface point cloud.

    Distances measured against a point cloud are biased upward by roughly this
    spacing, because the true closest point on the surface generally lies
    between samples. With a wall of ~0.37 mm this is not a negligible fraction
    of the measurand, so it is reported as an explicit noise floor rather than
    left implicit.
    """
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return float("nan")
    idx = rng.choice(len(p), size=int(min(n_probe, len(p))), replace=False)
    tree = cKDTree(p)
    d, _ = tree.query(p[idx], k=2, workers=-1)
    return float(np.median(d[:, 1]))


def distance_to_surface(
    query_xyz: np.ndarray,
    target_pts: np.ndarray,
    target_mesh: Optional[trimesh.Trimesh] = None,
    exact: bool = False,
) -> np.ndarray:
    """
    Distance from query points to a target surface.

    Default is nearest-neighbour against an area-weighted point sample, which is
    what the submitted evaluation_mesh.py does and therefore keeps the new
    numbers comparable with the manuscript. With exact=True and a mesh supplied,
    exact point-to-triangle distance is used instead, removing the sampling
    bias quantified by point_cloud_resolution().
    """
    if exact and target_mesh is not None:
        _, dist, _ = trimesh.proximity.closest_point(target_mesh, np.asarray(query_xyz, dtype=np.float64))
        return np.asarray(dist, dtype=np.float64)
    d, _ = nearest_dist(query_xyz, target_pts)
    return d


def coefficient_of_variation(x: np.ndarray) -> float:
    """
    Spatial variability of a measured thickness sample.  A nominal constant
    offset suppresses spatial variation, but nearest-opposite-surface distances
    on curved anatomy need not have exactly zero CV.  The useful comparison is
    therefore how closely each method preserves the reference spatial
    variability.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    m = float(np.mean(a))
    return float(np.std(a, ddof=1) / m) if m > 0 else float("nan")


def principal_axis_coordinate(points_xyz: np.ndarray, axis_reference_xyz: np.ndarray) -> np.ndarray:
    """
    Project points onto the dominant axis of the reference geometry, using the
    same PCA convention as the submitted check_extent_pca.py. Used only to
    identify end-cap regions.
    """
    ref = np.asarray(axis_reference_xyz, dtype=np.float64)
    centre = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - centre, full_matrices=False)
    axis = vt[0]
    return (np.asarray(points_xyz, dtype=np.float64) - centre) @ axis


def end_margin_mask(points_xyz: np.ndarray, axis_reference_xyz: np.ndarray, margin_mm: float) -> np.ndarray:
    """
    True for points at least margin_mm from either longitudinal end of the
    segment. A constant offset extends the lumen surface beyond the segment
    ends, so end caps are excluded identically for every method compared.
    """
    if margin_mm <= 0.0:
        return np.ones(len(points_xyz), dtype=bool)
    t_ref = principal_axis_coordinate(axis_reference_xyz, axis_reference_xyz)
    t_pts = principal_axis_coordinate(points_xyz, axis_reference_xyz)
    lo, hi = float(np.min(t_ref)) + margin_mm, float(np.max(t_ref)) - margin_mm
    if hi <= lo:
        return np.ones(len(points_xyz), dtype=bool)
    return (t_pts >= lo) & (t_pts <= hi)


# -----------------------------------------------------------------------------
# Continuous sampling from canonical predicted SDF grid
# -----------------------------------------------------------------------------
def physical_to_continuous_index_zyx(points_xyz: np.ndarray, geom: ImageGeometry) -> np.ndarray:
    p = np.asarray(points_xyz, dtype=np.float64)
    local = (p - geom.origin_xyz[None, :]) @ np.linalg.inv(geom.direction).T
    idx_xyz = local / geom.spacing_xyz[None, :]
    return np.stack([idx_xyz[:, 2], idx_xyz[:, 1], idx_xyz[:, 0]], axis=0)


def sample_grid_physical(arr_zyx: np.ndarray, points_xyz: np.ndarray, geom: ImageGeometry) -> np.ndarray:
    coords = physical_to_continuous_index_zyx(points_xyz, geom)
    vals = map_coordinates(
        np.asarray(arr_zyx, dtype=np.float64),
        coords,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return vals.astype(np.float64)


# -----------------------------------------------------------------------------
# Component / topology QC
# -----------------------------------------------------------------------------
def connected_component_stats(mask: np.ndarray) -> Dict[str, float]:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return {"components": 0, "largest_fraction": float("nan")}
    lab, n = cc_label(m)
    counts = np.bincount(lab.ravel())[1:]
    largest_fraction = float(np.max(counts) / np.sum(counts)) if counts.size else float("nan")
    return {"components": int(n), "largest_fraction": largest_fraction}


# -----------------------------------------------------------------------------
# Field-based analysis
# -----------------------------------------------------------------------------
def field_analysis(
    ref: Dict[str, object],
    pin: np.ndarray,
    pout: np.ndarray,
    t_min_case: float,
    thin_threshold_from_qc: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    gt_in = array_from_image(ref["inner_sdf_img"], np.float32)
    gt_out = array_from_image(ref["outer_sdf_img"], np.float32)
    bin_in = array_from_image(ref["inner_bin_img"], np.uint8) > 0
    bin_out = array_from_image(ref["outer_bin_img"], np.uint8) > 0

    ref_wall = bin_out & (~bin_in)
    finite = np.isfinite(gt_in) & np.isfinite(gt_out) & np.isfinite(pin) & np.isfinite(pout)
    mask = ref_wall & finite
    if not np.any(mask):
        raise ValueError("Reference wall mask is empty after physical-space alignment")

    t_ref = (gt_in - gt_out)[mask].astype(np.float64)
    t_pred = (pin - pout)[mask].astype(np.float64)

    # Remove non-positive reference field values from fidelity metrics only; retain rate as QC.
    positive_ref = t_ref > 0
    if np.sum(positive_ref) < 100:
        raise ValueError("Too few positive reference dual-SDF separation samples in wall")
    t_ref_valid = t_ref[positive_ref]
    t_pred_valid = t_pred[positive_ref]

    # Match the training code when QC value exists; otherwise derive from reference field.
    if np.isfinite(thin_threshold_from_qc):
        thin_thr = float(thin_threshold_from_qc)
        thin_source = 1.0  # encoded flag for table
    else:
        p20 = percentile(t_ref_valid, 20)
        thin_thr = max(1.10 * p20, t_min_case if np.isfinite(t_min_case) else 0.0)
        thin_source = 0.0

    thin_mask = t_ref_valid < thin_thr

    out: Dict[str, float] = {}
    out.update(describe(t_ref_valid, "field_ref_"))
    out.update(describe(t_pred_valid, "field_pred_"))
    out.update(paired_agreement_metrics(t_ref_valid, t_pred_valid, "field_"))
    if np.any(thin_mask):
        out.update(paired_agreement_metrics(t_ref_valid[thin_mask], t_pred_valid[thin_mask], "field_thin_"))
        out["field_thin_fraction"] = float(np.mean(thin_mask))
    else:
        out.update(paired_agreement_metrics(np.array([]), np.array([]), "field_thin_"))
        out["field_thin_fraction"] = float("nan")

    out["field_thin_threshold_mm"] = float(thin_thr)
    out["field_thin_threshold_from_qc"] = thin_source
    out["field_ref_nonpositive_rate_in_reference_wall"] = float(np.mean(t_ref <= 0))
    out["field_pred_nonpositive_rate_in_reference_wall"] = float(np.mean(t_pred <= 0))

    # Voxel-level nesting based only on predictions.
    pred_lumen = pin <= 0.0
    pred_outer = pout <= 0.0
    pred_lumen_count = int(np.sum(pred_lumen))
    enclosure_violation = pred_lumen & (~pred_outer)
    out["voxel_pred_lumen_count"] = pred_lumen_count
    out["voxel_enclosure_violation_count"] = int(np.sum(enclosure_violation))
    out["voxel_enclosure_violation_rate"] = float(np.sum(enclosure_violation) / pred_lumen_count) if pred_lumen_count else float("nan")

    # Minimum-thickness soft-constraint violation on the reference wall.
    if np.isfinite(t_min_case):
        out["field_minT_case_mm"] = float(t_min_case)
        out["field_minT_violation_rate_reference_wall"] = float(np.mean(t_pred < t_min_case))
    else:
        out["field_minT_case_mm"] = float("nan")
        out["field_minT_violation_rate_reference_wall"] = float("nan")

    ci = connected_component_stats(pred_lumen)
    co = connected_component_stats(pred_outer)
    out["pred_inner_components"] = ci["components"]
    out["pred_inner_largest_component_fraction"] = ci["largest_fraction"]
    out["pred_outer_components"] = co["components"]
    out["pred_outer_largest_component_fraction"] = co["largest_fraction"]

    pointwise = {
        "t_ref_field_mm": t_ref_valid,
        "t_pred_field_mm": t_pred_valid,
        "is_thin_reference": thin_mask.astype(np.uint8),
    }
    return out, pointwise


# -----------------------------------------------------------------------------
# Mesh-based local thickness analysis
# -----------------------------------------------------------------------------
def mesh_analysis(
    gt_inner: trimesh.Trimesh,
    gt_outer: trimesh.Trimesh,
    pred_inner: trimesh.Trimesh,
    pred_outer: trimesh.Trimesh,
    n_points: int,
    seed: int,
    correspondence_max_mm: float,
    exact: bool = False,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Independent deterministic streams.
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(4)
    gi = sample_surface_points(gt_inner, n_points, np.random.default_rng(child[0]))
    go = sample_surface_points(gt_outer, n_points, np.random.default_rng(child[1]))
    pi = sample_surface_points(pred_inner, n_points, np.random.default_rng(child[2]))
    po = sample_surface_points(pred_outer, n_points, np.random.default_rng(child[3]))

    # Inner-anchored reference local thickness: GT inner -> GT outer.
    tref_i = distance_to_surface(gi, go, gt_outer, exact)
    shift_i, idx_pi = nearest_dist(gi, pi)
    pi_match = pi[idx_pi]
    tpred_i = distance_to_surface(pi_match, po, pred_outer, exact)

    # Outer-anchored reverse local thickness: GT outer -> GT inner.
    tref_o = distance_to_surface(go, gi, gt_inner, exact)
    shift_o, idx_po = nearest_dist(go, po)
    po_match = po[idx_po]
    tpred_o = distance_to_surface(po_match, pi, pred_inner, exact)

    # Symmetric pointwise thickness vectors, consistent with manuscript's
    # shortest-distance-in-both-directions post-processing definition.
    tref = np.concatenate([tref_i, tref_o])
    tpred = np.concatenate([tpred_i, tpred_o])
    shifts = np.concatenate([shift_i, shift_o])
    dirs = np.concatenate([np.zeros_like(tref_i, dtype=np.uint8), np.ones_like(tref_o, dtype=np.uint8)])

    out: Dict[str, float] = {}
    out.update(describe(tref, "mesh_ref_"))
    out.update(describe(tpred, "mesh_pred_"))
    out.update(paired_agreement_metrics(tref, tpred, "mesh_"))
    out.update(paired_agreement_metrics(tref_i, tpred_i, "mesh_inner_anchor_"))
    out.update(paired_agreement_metrics(tref_o, tpred_o, "mesh_outer_anchor_"))
    out.update(describe(shifts, "mesh_correspondence_shift_"))

    high_conf = shifts <= float(correspondence_max_mm)
    out["mesh_correspondence_max_mm"] = float(correspondence_max_mm)
    out["mesh_high_confidence_fraction"] = float(np.mean(high_conf))
    if np.any(high_conf):
        out.update(paired_agreement_metrics(tref[high_conf], tpred[high_conf], "mesh_highconf_"))
    else:
        out.update(paired_agreement_metrics(np.array([]), np.array([]), "mesh_highconf_"))

    thin_thr = percentile(tref, 20)
    thin = tref <= thin_thr
    out["mesh_thin_threshold_p20_mm"] = thin_thr
    out["mesh_thin_fraction"] = float(np.mean(thin))
    out.update(paired_agreement_metrics(tref[thin], tpred[thin], "mesh_thin_"))

    # Distribution-only difference is useful because the reviewer's dilation
    # example would tend to erase case-specific/local thickness variation.
    out["mesh_ref_pred_mean_difference_mm"] = float(np.mean(tpred) - np.mean(tref))
    out["mesh_ref_pred_median_difference_mm"] = float(np.median(tpred) - np.median(tref))

    # Retention of spatial variation. A nominal constant-offset construction
    # suppresses thickness variation; with nearest-surface measurement on curved
    # anatomy its measured CV need not be exactly zero.
    # Measurement noise floor of the point-cloud distance backend.
    probe = np.random.default_rng(seed + 7717)
    out["mesh_sample_spacing_gt_outer_mm"] = point_cloud_resolution(go, probe)
    out["mesh_sample_spacing_pred_outer_mm"] = point_cloud_resolution(po, probe)
    out["mesh_sample_spacing_max_mm"] = float(
        np.nanmax([out["mesh_sample_spacing_gt_outer_mm"], out["mesh_sample_spacing_pred_outer_mm"]])
    )
    out["mesh_distance_backend_exact"] = 1.0 if exact else 0.0

    out["mesh_ref_cv"] = coefficient_of_variation(tref)
    out["mesh_pred_cv"] = coefficient_of_variation(tpred)
    out["mesh_ref_cv_inner_anchor"] = coefficient_of_variation(tref_i)
    out["mesh_pred_cv_inner_anchor"] = coefficient_of_variation(tpred_i)

    pointwise = {
        "t_ref_mesh_mm": tref,
        "t_pred_mesh_mm": tpred,
        "correspondence_shift_mm": shifts,
        "direction": dirs,  # 0 = inner anchored, 1 = outer anchored
        "high_confidence": high_conf.astype(np.uint8),
        "is_thin_reference": thin.astype(np.uint8),
    }
    surfaces = {
        "gt_inner_pts": gi,
        "gt_outer_pts": go,
        "pred_inner_pts": pi,
        "pred_outer_pts": po,
        "tref_inner_anchor": tref_i,
        "tpred_inner_anchor": tpred_i,
    }
    return out, pointwise, surfaces


# -----------------------------------------------------------------------------
# Constant-offset dilation baseline (Reviewer #2, Comments 2-3)
# -----------------------------------------------------------------------------
def dilated_mesh_from_inner_sdf(inner_sdf_img: sitk.Image, offset_mm: float) -> trimesh.Trimesh:
    """
    Outer surface obtained by offsetting the reference lumen by a constant
    distance, extracted as the +offset_mm level set of the reference inner SDF.

    The level set is used rather than explicit vertex-normal mesh displacement.
    This provides a clean Euclidean constant-offset construction in SDF space
    and avoids artifacts specific to naively moving mesh vertices along normals.
    As with any offset surface, topological changes can still occur near the
    medial axis or in highly complex geometry.
    """
    arr = array_from_image(inner_sdf_img, np.float32)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (vmin < offset_mm < vmax):
        raise ValueError(f"Dilation level {offset_mm:.4f} mm outside reference SDF range [{vmin:.4f}, {vmax:.4f}]")
    verts_zyx, faces, _, _ = marching_cubes(arr, level=float(offset_mm))
    geom = image_geometry(inner_sdf_img)
    z, y, x = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]
    index_xyz = np.stack([x, y, z], axis=1)
    scaled = index_xyz * geom.spacing_xyz[None, :]
    verts_xyz = geom.origin_xyz[None, :] + scaled @ geom.direction.T
    return trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)


def dilation_baseline_analysis(
    inner_sdf_img: sitk.Image,
    gt_inner_pts: np.ndarray,
    gt_outer_pts: np.ndarray,
    tref_inner_anchor: np.ndarray,
    tpred_inner_anchor: np.ndarray,
    n_points: int,
    seed: int,
    end_margin_mm: float,
    extra_offsets_mm: Optional[Sequence[float]] = None,
    exact: bool = False,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate the construction described by Reviewer #2: given the reference
    lumen surface, generate an outer surface by constant dilation.

    The comparator intentionally uses the exact reference lumen and an oracle
    case-specific constant derived from the same case's reference thickness
    distribution (mean and median variants).  It is therefore a deliberately
    favorable illustrative constant-offset construction, not an independent
    learning baseline. PIINN, by contrast, was optimized against both reference
    SDFs during case-wise fitting; the purpose of this comparison is only to
    test whether correct enclosure plus a constant offset can reproduce the
    observed outer-wall geometry and spatial thickness variation.  Both methods
    are evaluated with identical anchors and metrics.
    """
    out: Dict[str, float] = {}
    pw: Dict[str, np.ndarray] = {}

    finite = np.isfinite(tref_inner_anchor)
    if np.sum(finite) < 100:
        raise ValueError("Too few finite reference inner-anchored thickness samples")
    t_ref = np.asarray(tref_inner_anchor, dtype=np.float64)[finite]
    t_piinn = np.asarray(tpred_inner_anchor, dtype=np.float64)[finite]
    anchor_pts = np.asarray(gt_inner_pts, dtype=np.float64)[finite]

    interior = end_margin_mask(anchor_pts, gt_inner_pts, end_margin_mm)
    out["dil_end_margin_mm"] = float(end_margin_mm)
    out["dil_interior_fraction"] = float(np.mean(interior))

    # PIINN measured on the identical anchor subset, for a like-for-like row.
    out.update(paired_agreement_metrics(t_ref, t_piinn, "cmp_piinn_"))
    if np.any(interior):
        out.update(paired_agreement_metrics(t_ref[interior], t_piinn[interior], "cmp_piinn_interior_"))
    out["cmp_piinn_cv"] = coefficient_of_variation(t_piinn)
    out["cmp_ref_cv"] = coefficient_of_variation(t_ref)

    variants: Dict[str, float] = {
        "case_mean": float(np.mean(t_ref)),
        "case_median": float(np.median(t_ref)),
    }
    for i, v in enumerate(extra_offsets_mm or []):
        variants[f"fixed_{i}"] = float(v)

    rng_root = np.random.SeedSequence(seed)
    variant_seeds = rng_root.spawn(len(variants))
    for k, (name, offset) in enumerate(variants.items()):
        p = f"dil_{name}_"
        out[f"{p}offset_mm"] = offset
        try:
            mesh = dilated_mesh_from_inner_sdf(inner_sdf_img, offset)
            pts = sample_surface_points(mesh, n_points, np.random.default_rng(variant_seeds[k]))

            # Surface agreement with the reference outer wall.
            out.update(surface_distance_metrics(gt_outer_pts, pts, p))

            # Thickness realised by the construction, measured exactly as for PIINN.
            t_dil = distance_to_surface(anchor_pts, pts, mesh, exact)
            out.update(paired_agreement_metrics(t_ref, t_dil, p))
            if np.any(interior):
                out.update(paired_agreement_metrics(t_ref[interior], t_dil[interior], f"{p}interior_"))
            out[f"{p}cv"] = coefficient_of_variation(t_dil)
            out[f"{p}status"] = 1.0
            if name == "case_mean":
                pw["t_dilation_case_mean_mm"] = t_dil
                pw["t_reference_inner_anchor_mm"] = t_ref
                pw["t_piinn_inner_anchor_mm"] = t_piinn
                pw["is_interior"] = interior.astype(np.uint8)
        except Exception as exc:  # a failed variant must not void the case
            out[f"{p}status"] = 0.0
            out[f"{p}error"] = repr(exc)

    return out, pw


# -----------------------------------------------------------------------------
# Surface-level nesting analysis using predicted continuous SDF signs
# -----------------------------------------------------------------------------
def surface_nesting_analysis(
    pin: np.ndarray,
    pout: np.ndarray,
    geom: ImageGeometry,
    pred_inner_pts: np.ndarray,
    pred_outer_pts: np.ndarray,
    tol_mm: float,
) -> Dict[str, float]:
    # On predicted INNER surface, predicted OUTER SDF should be <= 0 (inside outer vessel).
    out_on_inner = sample_grid_physical(pout, pred_inner_pts, geom)
    # On predicted OUTER surface, predicted INNER SDF should be >= 0 (outside lumen).
    in_on_outer = sample_grid_physical(pin, pred_outer_pts, geom)

    vi = np.isfinite(out_on_inner)
    vo = np.isfinite(in_on_outer)
    out: Dict[str, float] = {
        "surface_tol_mm": float(tol_mm),
        "surface_inner_valid_fraction": float(np.mean(vi)),
        "surface_outer_valid_fraction": float(np.mean(vo)),
    }
    if np.any(vi):
        vals = out_on_inner[vi]
        out["surface_inner_outside_outer_strict_rate"] = float(np.mean(vals > 0.0))
        out["surface_inner_outside_outer_tol_rate"] = float(np.mean(vals > tol_mm))
        pos = vals[vals > 0]
        out["surface_inner_outside_outer_positive_mean_mm"] = float(np.mean(pos)) if pos.size else 0.0
        out["surface_inner_outside_outer_positive_p95_mm"] = percentile(pos, 95) if pos.size else 0.0
    else:
        out["surface_inner_outside_outer_strict_rate"] = float("nan")
        out["surface_inner_outside_outer_tol_rate"] = float("nan")
        out["surface_inner_outside_outer_positive_mean_mm"] = float("nan")
        out["surface_inner_outside_outer_positive_p95_mm"] = float("nan")

    if np.any(vo):
        vals = in_on_outer[vo]
        out["surface_outer_inside_lumen_strict_rate"] = float(np.mean(vals < 0.0))
        out["surface_outer_inside_lumen_tol_rate"] = float(np.mean(vals < -tol_mm))
        neg = -vals[vals < 0]
        out["surface_outer_inside_lumen_negative_mean_mm"] = float(np.mean(neg)) if neg.size else 0.0
        out["surface_outer_inside_lumen_negative_p95_mm"] = percentile(neg, 95) if neg.size else 0.0
    else:
        out["surface_outer_inside_lumen_strict_rate"] = float("nan")
        out["surface_outer_inside_lumen_tol_rate"] = float("nan")
        out["surface_outer_inside_lumen_negative_mean_mm"] = float("nan")
        out["surface_outer_inside_lumen_negative_p95_mm"] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def summarize_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in {"case", "status", "error", "binary_source", "gt_inner_mesh_source", "gt_outer_mesh_source", "pred_inner_mesh_source", "pred_outer_mesh_source"}:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "metric": col,
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "median": float(s.median()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def cohort_statistics_paired(df: pd.DataFrame, seed: int) -> Dict[str, object]:
    ok = df[df["status"] == "OK"]
    if ok.empty:
        return {}
    return paired_method_comparison(ok, PIINN_VS_COMPARATOR, seed)


def cohort_statistics(df: pd.DataFrame, seed: int) -> Dict[str, object]:
    ok = df[df["status"] == "OK"].copy()
    out: Dict[str, object] = {"n_ok": int(len(ok))}
    if len(ok) == 0:
        return out

    def vals(name: str) -> np.ndarray:
        return pd.to_numeric(ok[name], errors="coerce").to_numpy(dtype=np.float64)

    # Primary reviewer-facing mesh-based case summaries.
    ref_mean = vals("mesh_ref_mean")
    pred_mean = vals("mesh_pred_mean")
    ref_med = vals("mesh_ref_median")
    pred_med = vals("mesh_pred_median")
    mask = np.isfinite(ref_mean) & np.isfinite(pred_mean)

    if np.sum(mask) >= 2:
        diff = pred_mean[mask] - ref_mean[mask]
        lo, hi = bootstrap_mean_ci(diff, seed=seed, n_boot=10000)
        out["mesh_case_mean_thickness"] = {
            "reference_mean_mm": float(np.mean(ref_mean[mask])),
            "predicted_mean_mm": float(np.mean(pred_mean[mask])),
            "mean_bias_mm": float(np.mean(diff)),
            "mean_bias_bootstrap_95ci_mm": [lo, hi],
            "bias_sd_mm": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
            "bland_altman_loa_mm": [float(np.mean(diff) - 1.96 * np.std(diff, ddof=1)), float(np.mean(diff) + 1.96 * np.std(diff, ddof=1))] if diff.size > 1 else [float(diff[0]), float(diff[0])],
            "pearson_r": float(pearsonr(ref_mean[mask], pred_mean[mask]).statistic) if np.std(ref_mean[mask]) > 0 and np.std(pred_mean[mask]) > 0 else float("nan"),
            "spearman_rho": float(spearmanr(ref_mean[mask], pred_mean[mask]).statistic) if np.std(ref_mean[mask]) > 0 and np.std(pred_mean[mask]) > 0 else float("nan"),
            "ccc": lin_ccc(ref_mean[mask], pred_mean[mask]),
        }
        try:
            w = wilcoxon(pred_mean[mask], ref_mean[mask], zero_method="wilcox", alternative="two-sided", method="auto")
            out["mesh_case_mean_thickness"]["wilcoxon_p"] = float(w.pvalue)
        except Exception:
            out["mesh_case_mean_thickness"]["wilcoxon_p"] = float("nan")

    maskm = np.isfinite(ref_med) & np.isfinite(pred_med)
    if np.sum(maskm) >= 2:
        out["mesh_case_median_thickness"] = {
            "reference_median_of_cases_mm": float(np.median(ref_med[maskm])),
            "predicted_median_of_cases_mm": float(np.median(pred_med[maskm])),
            "pearson_r": float(pearsonr(ref_med[maskm], pred_med[maskm]).statistic) if np.std(ref_med[maskm]) > 0 and np.std(pred_med[maskm]) > 0 else float("nan"),
            "spearman_rho": float(spearmanr(ref_med[maskm], pred_med[maskm]).statistic) if np.std(ref_med[maskm]) > 0 and np.std(pred_med[maskm]) > 0 else float("nan"),
            "ccc": lin_ccc(ref_med[maskm], pred_med[maskm]),
        }

    # Direct summary of primary errors across cases.
    for metric in [
        "mesh_mae_mm", "mesh_rmse_mm", "mesh_bias_mm", "mesh_pearson_r", "mesh_spearman_rho", "mesh_ccc",
        "mesh_thin_mae_mm", "field_mae_mm", "field_rmse_mm", "field_bias_mm", "field_pearson_r", "field_spearman_rho", "field_ccc",
        "voxel_enclosure_violation_rate", "surface_inner_outside_outer_tol_rate", "surface_outer_inside_lumen_tol_rate",
        "field_minT_violation_rate_reference_wall",
    ]:
        if metric not in ok.columns:
            continue
        a = vals(metric)
        a = a[np.isfinite(a)]
        if a.size:
            out[metric] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
                "median": float(np.median(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "n": int(a.size),
            }
    return out


PIINN_VS_COMPARATOR = (
    ("outer_assd_mm", "piinn_outer_assd_mm", "dil_case_mean_assd_mm", "lower"),
    ("outer_hd95_mm", "piinn_outer_hd95_mm", "dil_case_mean_hd95_mm", "lower"),
    ("thickness_mae_mm", "cmp_piinn_mae_mm", "dil_case_mean_mae_mm", "lower"),
    ("thickness_mae_interior_mm", "cmp_piinn_interior_mae_mm", "dil_case_mean_interior_mae_mm", "lower"),
    ("thickness_rmse_mm", "cmp_piinn_rmse_mm", "dil_case_mean_rmse_mm", "lower"),
    ("thickness_pearson_r", "cmp_piinn_pearson_r", "dil_case_mean_pearson_r", "higher"),
    ("thickness_spearman_rho", "cmp_piinn_spearman_rho", "dil_case_mean_spearman_rho", "higher"),
    ("thickness_ccc", "cmp_piinn_ccc", "dil_case_mean_ccc", "higher"),
)


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return

    # 1) Reference vs predicted mean mesh thickness per case.
    cases = ok["case"].astype(str).tolist()
    x = np.arange(len(cases))
    ref = pd.to_numeric(ok["mesh_ref_mean"], errors="coerce").to_numpy()
    pred = pd.to_numeric(ok["mesh_pred_mean"], errors="coerce").to_numpy()
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(cases) * 0.7), 5))
    ax.bar(x - width / 2, ref, width, label="Reference")
    ax.bar(x + width / 2, pred, width, label="Predicted")
    ax.set_ylabel("Mean symmetric wall separation (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=60, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "case_mean_thickness_reference_vs_predicted.png", dpi=220)
    plt.close(fig)

    # 2) MAE per case.
    mae = pd.to_numeric(ok["mesh_mae_mm"], errors="coerce").to_numpy()
    fig, ax = plt.subplots(figsize=(max(10, len(cases) * 0.7), 5))
    ax.bar(x, mae)
    ax.set_ylabel("Local wall-thickness MAE (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "case_thickness_mae.png", dpi=220)
    plt.close(fig)

    # 3) Nesting violation per case.
    vio = pd.to_numeric(ok["voxel_enclosure_violation_rate"], errors="coerce").to_numpy() * 100.0
    fig, ax = plt.subplots(figsize=(max(10, len(cases) * 0.7), 5))
    ax.bar(x, vio)
    ax.set_ylabel("Predicted lumen outside predicted outer vessel (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "case_nesting_violation.png", dpi=220)
    plt.close(fig)

    # 4) Case-level scatter.
    m = np.isfinite(ref) & np.isfinite(pred)
    if np.sum(m) >= 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ref[m], pred[m])
        lo = min(float(np.min(ref[m])), float(np.min(pred[m])))
        hi = max(float(np.max(ref[m])), float(np.max(pred[m])))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Reference mean symmetric wall separation (mm)")
        ax.set_ylabel("Predicted mean symmetric wall separation (mm)")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(out_dir / "cohort_mean_thickness_scatter.png", dpi=220)
        plt.close(fig)

        # 5) Bland-Altman at independent case level.
        means = 0.5 * (ref[m] + pred[m])
        diffs = pred[m] - ref[m]
        bias = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(means, diffs)
        ax.axhline(bias, linestyle="-")
        ax.axhline(bias - 1.96 * sd, linestyle="--")
        ax.axhline(bias + 1.96 * sd, linestyle="--")
        ax.set_xlabel("Mean of reference and predicted case thickness (mm)")
        ax.set_ylabel("Predicted - reference (mm)")
        fig.tight_layout()
        fig.savefig(out_dir / "cohort_mean_thickness_bland_altman.png", dpi=220)
        plt.close(fig)


def _paired_table(cohort: Dict[str, object]) -> str:
    pv = cohort.get("piinn_vs_comparator", {}) if isinstance(cohort, dict) else {}
    if not pv:
        return ""
    lines = [
        "",
        "### Paired case-level comparison (Wilcoxon signed rank, Holm corrected)",
        "",
        "The case is the unit of analysis; pointwise samples within a case are not independent and are not tested.",
        "",
        "| Metric | PIINN | Comparator | Favouring PIINN | Cliff's delta | p (Holm) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k, v in pv.items():
        if not isinstance(v, dict):
            continue
        lines.append(
            f"| {k} | {v['piinn_mean']:.4f} ± {v['piinn_sd']:.4f} | "
            f"{v['comparator_mean']:.4f} ± {v['comparator_sd']:.4f} | "
            f"{v['cases_favouring_piinn']}/{v['n_cases']} | "
            f"{v['cliffs_delta']:.3f} | {v.get('wilcoxon_p_holm', float('nan')):.5f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, cohort: Dict[str, object], args: argparse.Namespace) -> str:
    ok = df[df["status"] == "OK"]
    n = len(ok)
    lines = [
        "# PIINN Wall-Thickness Fidelity and Geometric-Consistency Validation",
        "",
        f"Cases successfully evaluated: **{n}**",
        "",
        "## Interpretation",
        "",
        "This analysis distinguishes two complementary quantities. The dual-SDF separation field validates the same inner/outer field relationship used by the PIINN thickness regularizers. The mesh analysis validates the manuscript's post-processing definition of wall thickness as shortest Euclidean distance to the opposite surface, computed symmetrically. Agreement is assessed against the CCTA-derived reference annotations; this does not imply direct inference from raw CCTA intensities.",
        "",
        "## Primary reviewer-facing metrics",
        "",
    ]
    def stat(metric: str) -> str:
        if metric not in ok.columns or ok.empty:
            return "NA"
        s = pd.to_numeric(ok[metric], errors="coerce").dropna()
        if s.empty:
            return "NA"
        return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.iloc[0]:.4f}"

    lines += [
        f"- Mesh local-thickness MAE across cases (mean ± SD): **{stat('mesh_mae_mm')} mm**",
        f"- Mesh local-thickness RMSE across cases: **{stat('mesh_rmse_mm')} mm**",
        f"- Mesh local-thickness bias across cases: **{stat('mesh_bias_mm')} mm**",
        f"- Thin-region mesh thickness MAE: **{stat('mesh_thin_mae_mm')} mm**",
        f"- Dual-SDF separation-field MAE: **{stat('field_mae_mm')} mm**",
        f"- Voxel nesting violation rate: **{stat('voxel_enclosure_violation_rate')}** (fraction)",
        f"- Inner-surface outside-outer violation rate (tolerance {args.surface_tol_mm:g} mm): **{stat('surface_inner_outside_outer_tol_rate')}** (fraction)",
        f"- Minimum-thickness violation rate, separation field: **{stat('field_minT_violation_rate_reference_wall')}** (fraction; measured on the dual-SDF separation field, which is biased relative to mesh thickness - see below)",
        f"- Minimum-thickness violation rate, mesh thickness: **{stat('mesh_minT_violation_rate_predicted')}** (fraction)",
        f"- Minimum-thickness violation rate of the reference itself, same threshold: **{stat('mesh_minT_violation_rate_reference')}** (fraction)",
        f"- Non-positive predicted thickness rate: **{stat('mesh_nonpositive_thickness_rate_predicted')}** (fraction)",
        "",
        "## Measurement scale",
        "",
        f"- Mean reference wall thickness: **{stat('mesh_ref_mean')} mm**",
        f"- Mean voxel spacing: **{stat('mean_spacing_mm')} mm**; largest spacing: **{stat('max_spacing_mm')} mm**",
        f"- Reference wall thickness in mean voxels: **{stat('ref_thickness_in_mean_voxels')}**",
        f"- Thickness MAE as a fraction of mean voxel spacing: **{stat('mesh_mae_over_mean_spacing')}**",
        f"- Point-cloud sampling noise floor: **{stat('mesh_sample_spacing_max_mm')} mm**",
        "",
        "The reference wall is at or below the voxel scale, so thickness error must be reported relative to it. State this explicitly rather than reporting absolute millimetres alone.",
        "",
        "## Separation field versus mesh thickness",
        "",
        f"- Reference separation field minus reference mesh thickness: **{stat('field_minus_mesh_reference_mm')} mm**",
        f"- Predicted separation field minus predicted mesh thickness: **{stat('field_minus_mesh_predicted_mm')} mm**",
        "",
        "These two quantities are not interchangeable. The separation field phi_in - phi_out is distance-to-lumen plus distance-to-outer along generally different directions, and the gap widens on a sub-voxel wall. Report mesh thickness as thickness; treat the separation field as a diagnostic of the quantity the constraints operate on.",
        "",
    ]

    if "dil_case_mean_assd_mm" in ok.columns:
        lines += [
            "## Constant-offset dilation baseline (Reviewer #2, Comments 2-3)",
            "",
            "The comparator uses the exact reference lumen and an oracle case-specific constant offset derived from the same case's reference thickness distribution. It is therefore a deliberately favorable illustrative constant-offset construction, not an independent learning baseline. PIINN itself was optimized against both reference SDFs during case-wise fitting. Both methods are evaluated with identical anchors and identical metrics. Thickness rows are inner-anchored on the same points; end caps are excluded within "
            f"{args.dilation_end_margin_mm:g} mm of each segment end because a constant offset extends beyond the lumen.",
            "",
            "| Quantity | PIINN | Dilation (case mean) |",
            "| --- | --- | --- |",
            f"| Outer-surface ASSD (mm) | {stat('piinn_outer_assd_mm')} | {stat('dil_case_mean_assd_mm')} |",
            f"| Outer-surface HD95 (mm) | {stat('piinn_outer_hd95_mm')} | {stat('dil_case_mean_hd95_mm')} |",
            f"| Thickness MAE (mm) | {stat('cmp_piinn_mae_mm')} | {stat('dil_case_mean_mae_mm')} |",
            f"| Thickness MAE, interior (mm) | {stat('cmp_piinn_interior_mae_mm')} | {stat('dil_case_mean_interior_mae_mm')} |",
            f"| Thickness bias (mm) | {stat('cmp_piinn_bias_mm')} | {stat('dil_case_mean_bias_mm')} |",
            f"| Thickness Pearson r | {stat('cmp_piinn_pearson_r')} | {stat('dil_case_mean_pearson_r')} |",
            f"| Thickness Lin CCC | {stat('cmp_piinn_ccc')} | {stat('dil_case_mean_ccc')} |",
            f"| Thickness CV | {stat('cmp_piinn_cv')} | {stat('dil_case_mean_cv')} |",
            "",
            f"Reference thickness CV for the same points: **{stat('cmp_ref_cv')}**. A nominal constant-offset construction is expected to suppress spatial thickness variability, although nearest-surface measurements on curved geometry need not yield exactly zero CV. The comparison of interest is whether PIINN preserves the reference variability more closely.",
            "",
            _paired_table(cohort),
            "Interpretation. This comparator is supporting evidence for the reviewer's dilation thought experiment: it satisfies the intended nesting through a constant offset while using no spatially varying outer-wall information. Agreement of PIINN with the reference alone is not independent evidence because PIINN is supervised on those reference fields; the informative observation is whether PIINN preserves the reference outer-wall geometry and spatial thickness variation substantially better than the constant-offset construction. Constraint ablation remains complementary evidence for separating data fidelity from regularization effects.",
            "",
        ]

    c = cohort.get("mesh_case_mean_thickness", {}) if isinstance(cohort, dict) else {}
    if c:
        lines += [
            "## Case-level agreement of mean mesh thickness",
            "",
            f"- Reference cohort mean: {c.get('reference_mean_mm', float('nan')):.4f} mm",
            f"- Predicted cohort mean: {c.get('predicted_mean_mm', float('nan')):.4f} mm",
            f"- Mean bias (predicted - reference): {c.get('mean_bias_mm', float('nan')):.4f} mm",
            f"- Pearson r across cases: {c.get('pearson_r', float('nan')):.4f}",
            f"- Spearman rho across cases: {c.get('spearman_rho', float('nan')):.4f}",
            f"- Lin CCC across cases: {c.get('ccc', float('nan')):.4f}",
            "",
        ]

    lines += [
        "## Cautions for the manuscript/rebuttal",
        "",
        "1. Do not describe the geometric priors as direct raw-image evidence. The PIINN is supervised by CCTA-derived lumen and outer-wall annotations/SDFs.",
        "2. Do not claim a hard topological guarantee unless the measured violation rates are zero under a predefined tolerance. The current losses are soft penalties.",
        "3. The dual-SDF separation field and the mesh shortest-distance thickness are related but not identical quantities; report them separately.",
        "4. If thickness agreement is weak in one or more cases, report this transparently and treat the constraints as regularizers rather than as proof of anatomical wall-thickness recovery.",
        "5. Agreement with the reference is a fitting result, not independent evidence, because the model is supervised on those same reference fields. The comparative dilation-baseline rows and the constraint ablations carry the argument.",
        "6. The thickness smoothness penalty of Eq. (13) is the one two-sided term and does favour uniform thickness. State this and support it with the w_smoothT = 0 ablation rather than leaving it for a reviewer to find.",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Case evaluation
# -----------------------------------------------------------------------------
def load_qc(pred_case_dir: Path) -> Dict[str, object]:
    p = pred_case_dir / "qc_metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_cases(gt_root: Path, pred_root: Path) -> List[str]:
    g = {p.name for p in gt_root.iterdir() if p.is_dir()}
    p = {p.name for p in pred_root.iterdir() if p.is_dir()}
    return sorted(g & p)


def maybe_downsample_dataframe(data: Dict[str, np.ndarray], max_rows: int, seed: int) -> pd.DataFrame:
    n = len(next(iter(data.values()))) if data else 0
    if n == 0:
        return pd.DataFrame(data)
    if n <= max_rows:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return pd.DataFrame({k: np.asarray(v)[idx] for k, v in data.items()})


def evaluate_case(
    case: str,
    gt_root: Path,
    pred_root: Path,
    gt_mesh_root: Optional[Path],
    out_case_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    row: Dict[str, object] = {"case": case, "status": "OK", "error": ""}
    try:
        ref = load_reference_case(gt_root / case)
        canonical_img: sitk.Image = ref["inner_sdf_img"]
        geom: ImageGeometry = ref["geometry"]
        pin, pout = load_predictions(pred_root / case, geom.shape_zyx)
        qc = load_qc(pred_root / case)
        priors = qc.get("thickness_priors", {}) if isinstance(qc, dict) else {}
        t_min_case = safe_float(priors.get("t_min_case", np.nan)) if isinstance(priors, dict) else float("nan")
        thin_thr_qc = safe_float(priors.get("thin_threshold", np.nan)) if isinstance(priors, dict) else float("nan")

        row["binary_source"] = ref["binary_source"]
        row["spacing_x_mm"], row["spacing_y_mm"], row["spacing_z_mm"] = [float(v) for v in geom.spacing_xyz]
        row["t_min_case_mm_from_qc"] = t_min_case
        row["thin_threshold_mm_from_qc"] = thin_thr_qc
        row["training_time_sec_from_qc"] = safe_float(qc.get("train_time_sec", np.nan)) if isinstance(qc, dict) else float("nan")
        row["gpu_peak_mem_mb_from_qc"] = safe_float(qc.get("gpu_peak_mem_mb", np.nan)) if isinstance(qc, dict) else float("nan")

        # Field fidelity + voxel nesting.
        fm, field_pw = field_analysis(ref, pin, pout, t_min_case, thin_thr_qc)
        row.update(fm)

        # Mesh sources.
        gt_in_mesh, gt_in_src = choose_gt_mesh(case, gt_mesh_root, "inner", ref["inner_sdf_img"])
        gt_out_mesh, gt_out_src = choose_gt_mesh(case, gt_mesh_root, "outer", ref["outer_sdf_img_native"])
        pr_in_mesh, pr_in_src = choose_pred_mesh(case, pred_root, "inner", pin, canonical_img)
        pr_out_mesh, pr_out_src = choose_pred_mesh(case, pred_root, "outer", pout, canonical_img)
        row["gt_inner_mesh_source"] = gt_in_src
        row["gt_outer_mesh_source"] = gt_out_src
        row["pred_inner_mesh_source"] = pr_in_src
        row["pred_outer_mesh_source"] = pr_out_src

        mm, mesh_pw, surfaces = mesh_analysis(
            gt_in_mesh, gt_out_mesh, pr_in_mesh, pr_out_mesh,
            n_points=args.mesh_points,
            seed=stable_case_seed(args.seed, case),
            correspondence_max_mm=args.correspondence_max_mm,
            exact=args.exact_point_to_mesh,
        )
        row.update(mm)

        # PIINN surface agreement, same definitions as the submitted evaluation_mesh.py.
        row.update(surface_distance_metrics(surfaces["gt_inner_pts"], surfaces["pred_inner_pts"], "piinn_inner_"))
        row.update(surface_distance_metrics(surfaces["gt_outer_pts"], surfaces["pred_outer_pts"], "piinn_outer_"))

        # Constant-offset dilation baseline (Reviewer #2, Comments 2-3).
        dil_pw: Dict[str, np.ndarray] = {}
        if not args.skip_dilation_baseline:
            try:
                dm, dil_pw = dilation_baseline_analysis(
                    inner_sdf_img=ref["inner_sdf_img"],
                    gt_inner_pts=surfaces["gt_inner_pts"],
                    gt_outer_pts=surfaces["gt_outer_pts"],
                    tref_inner_anchor=surfaces["tref_inner_anchor"],
                    tpred_inner_anchor=surfaces["tpred_inner_anchor"],
                    n_points=args.mesh_points,
                    seed=stable_case_seed(args.seed, case),
                    end_margin_mm=args.dilation_end_margin_mm,
                    extra_offsets_mm=args.dilation_fixed_offsets_mm,
                    exact=args.exact_point_to_mesh,
                )
                row.update(dm)
                row["dilation_baseline_status"] = "OK"
            except Exception as exc:
                row["dilation_baseline_status"] = f"FAILED: {exc!r}"

        # (C) Voxel-scale context. Every thickness number needs this denominator:
        # the run showed the reference wall is thinner than one voxel in all cases.
        sp = np.asarray(geom.spacing_xyz, dtype=np.float64)
        row["mean_spacing_mm"] = float(np.mean(sp))
        row["min_spacing_mm"] = float(np.min(sp))
        row["max_spacing_mm"] = float(np.max(sp))
        row["voxel_diagonal_mm"] = float(np.linalg.norm(sp))
        ref_mean_mesh = safe_float(row.get("mesh_ref_mean", np.nan))
        row["ref_thickness_in_mean_voxels"] = float(ref_mean_mesh / np.mean(sp)) if np.isfinite(ref_mean_mesh) else float("nan")
        row["ref_thickness_in_max_voxels"] = float(ref_mean_mesh / np.max(sp)) if np.isfinite(ref_mean_mesh) else float("nan")
        mae_mesh = safe_float(row.get("mesh_mae_mm", np.nan))
        row["mesh_mae_over_mean_spacing"] = float(mae_mesh / np.mean(sp)) if np.isfinite(mae_mesh) else float("nan")

        # (B) Make the separation-field vs mesh-thickness discrepancy explicit.
        f_ref, f_pred = safe_float(row.get("field_ref_mean", np.nan)), safe_float(row.get("field_pred_mean", np.nan))
        m_ref, m_pred = ref_mean_mesh, safe_float(row.get("mesh_pred_mean", np.nan))
        row["field_minus_mesh_reference_mm"] = float(f_ref - m_ref) if np.isfinite(f_ref) and np.isfinite(m_ref) else float("nan")
        row["field_minus_mesh_predicted_mm"] = float(f_pred - m_pred) if np.isfinite(f_pred) and np.isfinite(m_pred) else float("nan")

        # (A) Minimum-thickness violation on mesh thickness, not on the biased
        # separation field, together with the reference's own rate. t_min is set
        # to 0.90 x the reference 5th percentile, so the reference itself
        # violates it at a low but non-zero rate; only the comparison is
        # interpretable.
        if np.isfinite(t_min_case):
            t_ref_ia = np.asarray(surfaces["tref_inner_anchor"], dtype=np.float64)
            t_pred_ia = np.asarray(surfaces["tpred_inner_anchor"], dtype=np.float64)
            ok_ia = np.isfinite(t_ref_ia) & np.isfinite(t_pred_ia)
            if np.any(ok_ia):
                row["mesh_minT_violation_rate_predicted"] = float(np.mean(t_pred_ia[ok_ia] < t_min_case))
                row["mesh_minT_violation_rate_reference"] = float(np.mean(t_ref_ia[ok_ia] < t_min_case))
                row["mesh_minT_violation_excess_over_reference"] = float(
                    row["mesh_minT_violation_rate_predicted"] - row["mesh_minT_violation_rate_reference"]
                )
                row["mesh_nonpositive_thickness_rate_predicted"] = float(np.mean(t_pred_ia[ok_ia] <= 0.0))

        # Surface-level sign/nesting QC from predicted continuous SDFs.
        sm = surface_nesting_analysis(
            pin, pout, geom,
            pred_inner_pts=surfaces["pred_inner_pts"],
            pred_outer_pts=surfaces["pred_outer_pts"],
            tol_mm=args.surface_tol_mm,
        )
        row.update(sm)

        out_case_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_case_dir / "metrics.json", row)

        if args.save_pointwise:
            # Store complete numeric arrays compactly for audit/reanalysis.
            # Namespace arrays from each analysis block before merging them into
            # one NPZ archive. Several blocks intentionally use semantically
            # identical local key names (e.g. ``is_thin_reference``); expanding
            # the raw dictionaries together would pass duplicate keyword names
            # to numpy.savez_compressed and raise TypeError. Namespacing also
            # makes the audit archive self-describing and prevents future key
            # collisions as additional reviewer analyses are added.
            npz_payload = {}
            npz_payload.update({f"field__{k}": v for k, v in field_pw.items()})
            npz_payload.update({f"mesh__{k}": v for k, v in mesh_pw.items()})
            npz_payload.update({f"dilation__{k}": v for k, v in dil_pw.items()})
            np.savez_compressed(
                out_case_dir / "pointwise_data.npz",
                **npz_payload,
            )
            maybe_downsample_dataframe(field_pw, args.pointwise_csv_max_rows, args.seed).to_csv(
                out_case_dir / "field_pointwise_sample.csv", index=False
            )
            maybe_downsample_dataframe(mesh_pw, args.pointwise_csv_max_rows, args.seed).to_csv(
                out_case_dir / "mesh_pointwise_sample.csv", index=False
            )

        return row
    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = repr(e)
        out_case_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_case_dir / "metrics.json", row)
        return row


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Revision-grade PIINN wall-thickness fidelity and nesting validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gt-root", required=True, type=Path, help="Root containing per-case reference SDF/binary folders")
    ap.add_argument("--pred-root", required=True, type=Path, help="Root containing per-case PIINN outputs")
    ap.add_argument("--gt-mesh-root", type=Path, default=None, help="Optional root containing per-case inner_gt_from_sdf.ply and outer_gt_from_sdf.ply")
    ap.add_argument("--out-root", required=True, type=Path, help="Output directory for revision analysis")
    ap.add_argument("--mesh-points", type=int, default=100000, help="Area-weighted sampled points per surface")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    ap.add_argument("--correspondence-max-mm", type=float, default=1.0, help="QC threshold for GT-to-pred surface anchor correspondence")
    ap.add_argument("--surface-tol-mm", type=float, default=0.02, help="Tolerance for surface-level nesting sign violations")
    ap.add_argument("--save-pointwise", action="store_true", help="Save compressed pointwise arrays and audit CSV samples")
    ap.add_argument("--pointwise-csv-max-rows", type=int, default=20000, help="Maximum rows per audit pointwise CSV")
    ap.add_argument("--exact-point-to-mesh", action="store_true", help="Use exact point-to-triangle distance instead of nearest-neighbour against sampled points. Removes the sampling bias reported as the noise floor, at a substantial speed cost")
    ap.add_argument("--skip-dilation-baseline", action="store_true", help="Disable the constant-offset dilation baseline")
    ap.add_argument("--dilation-end-margin-mm", type=float, default=2.0, help="Longitudinal margin excluded at both segment ends when comparing methods, since a constant offset extends past the lumen end caps")
    ap.add_argument("--dilation-fixed-offsets-mm", nargs="*", type=float, default=None, help="Optional additional constant offsets, e.g. a cohort-level mean thickness")
    ap.add_argument("--cases", nargs="*", default=None, help="Optional explicit case names; default is intersection of GT and prediction roots")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.gt_root = args.gt_root.resolve()
    args.pred_root = args.pred_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.gt_mesh_root is not None:
        args.gt_mesh_root = args.gt_mesh_root.resolve()

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "per_case").mkdir(exist_ok=True)

    if not args.gt_root.exists():
        raise FileNotFoundError(args.gt_root)
    if not args.pred_root.exists():
        raise FileNotFoundError(args.pred_root)

    cases = list(args.cases) if args.cases else discover_cases(args.gt_root, args.pred_root)
    if not cases:
        raise RuntimeError("No common case directories found between gt-root and pred-root")

    manifest = {
        "script": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "SimpleITK": sitk.Version_VersionString(),
        "scikit_image": skimage_version,
        "trimesh": getattr(trimesh, "__version__", "unknown"),
        "args": vars(args),
        "cases": cases,
        "methodological_note": (
            "Field analysis validates the dual-SDF separation quantity used by the PIINN constraints. "
            "Mesh analysis validates the manuscript/post-processing shortest Euclidean opposite-surface thickness. "
            "Both use CCTA-derived reference annotations; neither implies direct raw-intensity inference."
        ),
    }
    write_json(args.out_root / "run_manifest.json", manifest)

    rows: List[Dict[str, object]] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:02d}/{len(cases):02d}] Evaluating {case} ...", flush=True)
        row = evaluate_case(
            case=case,
            gt_root=args.gt_root,
            pred_root=args.pred_root,
            gt_mesh_root=args.gt_mesh_root,
            out_case_dir=args.out_root / "per_case" / case,
            args=args,
        )
        rows.append(row)
        if row["status"] == "OK":
            print(
                f"    OK | mesh MAE={safe_float(row.get('mesh_mae_mm')):.4f} mm "
                f"| field MAE={safe_float(row.get('field_mae_mm')):.4f} mm "
                f"| voxel nesting violation={100*safe_float(row.get('voxel_enclosure_violation_rate')):.4f}%",
                flush=True,
            )
        else:
            print(f"    FAILED | {row.get('error')}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_root / "per_case_summary.csv", index=False)

    agg = summarize_numeric_dataframe(df[df["status"] == "OK"])
    agg.to_csv(args.out_root / "aggregate_summary.csv", index=False)

    cohort = cohort_statistics(df, seed=args.seed)
    cohort["piinn_vs_comparator"] = cohort_statistics_paired(df, args.seed)
    write_json(args.out_root / "cohort_statistics.json", cohort)

    # Publication/revision-friendly workbook.
    with pd.ExcelWriter(args.out_root / "revision_validation.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per_Case", index=False)
        agg.to_excel(writer, sheet_name="Aggregate", index=False)
        cohort_rows = []
        def flatten(prefix: str, obj: object):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, list):
                cohort_rows.append({"metric": prefix, "value": json.dumps(json_safe(obj))})
            else:
                cohort_rows.append({"metric": prefix, "value": obj})
        flatten("", cohort)
        pd.DataFrame(cohort_rows).to_excel(writer, sheet_name="Cohort_Stats", index=False)
        pd.DataFrame([
            {"item": "gt_root", "value": str(args.gt_root)},
            {"item": "pred_root", "value": str(args.pred_root)},
            {"item": "gt_mesh_root", "value": str(args.gt_mesh_root) if args.gt_mesh_root else "generated from SDF if needed"},
            {"item": "mesh_points_per_surface", "value": args.mesh_points},
            {"item": "seed", "value": args.seed},
            {"item": "correspondence_max_mm", "value": args.correspondence_max_mm},
            {"item": "surface_tol_mm", "value": args.surface_tol_mm},
            {"item": "interpretation", "value": "Agreement with CCTA-derived reference annotations, not direct raw-intensity inference."},
        ]).to_excel(writer, sheet_name="Run_Info", index=False)

    save_plots(df, args.out_root / "plots")
    report = build_report(df, cohort, args)
    (args.out_root / "REPORT.md").write_text(report, encoding="utf-8")

    n_ok = int(np.sum(df["status"] == "OK"))
    n_fail = len(df) - n_ok
    print("\nValidation complete.")
    print(f"  OK: {n_ok}")
    print(f"  Failed: {n_fail}")
    print(f"  Outputs: {args.out_root}")
    if n_fail:
        print("  Review per_case_summary.csv and each per_case/<CASE>/metrics.json for failures.")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
