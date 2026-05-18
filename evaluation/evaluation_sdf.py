#!/usr/bin/env python3
"""
evaluation_sdf.py

SDF-field evaluation for PINN reconstruction.

GT SDFs (per case):
  <gt_root>/<CASE>/INNER_SDF.nrrd
  <gt_root>/<CASE>/OUTER_SDF.nrrd

Pred SDFs (per case):
  <pred_root>/<CASE>/pred_inner_sdf.npy
  <pred_root>/<CASE>/pred_outer_sdf.npy

Outputs (in <out_root>):
  metrics_sdf.csv
  per_case_sdf.json
  summary_sdf.json

Metrics (mm):
  - MAE / RMSE (whole grid)
  - Near-surface MAE / RMSE where |GT| < t
  - Sign accuracy (whole grid and near-surface)
  - Eikonal error near-surface: mean | ||∇pred|| - 1 |

Paths:
  D:\Mo_PINNs\Processed
  D:\Mo_PINNs\Processed\outputs_Vanilla_INR
  D:\Mo_PINNs\Processed\outputs_Vanilla_INR\evaluation

  run command:
  python evaluation_sdf.py


"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# deps:
#   pip install SimpleITK scipy
import SimpleITK as sitk
from scipy.ndimage import zoom


# -----------------------------
# IO helpers
# -----------------------------
def read_gt_sdf_nrrd(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      sdf_zyx: np.ndarray float32 with shape (z,y,x)
      spacing_xyz: (sx,sy,sz) from NRRD (physical mm per voxel)
    """
    img = sitk.ReadImage(str(path))
    sdf_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (z,y,x)
    spacing_xyz = img.GetSpacing()  # (sx,sy,sz)
    return sdf_zyx, spacing_xyz


def read_pred_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    return arr


def ensure_shape(pred: np.ndarray, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
    """
    Ensure pred matches target_shape. If mismatch, resample pred with trilinear interpolation.
    Returns:
      pred_resampled, note
    """
    if pred.shape == target_shape:
        return pred, "OK"

    # Resample pred to target shape using zoom
    zf = target_shape[0] / pred.shape[0]
    yf = target_shape[1] / pred.shape[1]
    xf = target_shape[2] / pred.shape[2]

    pred_rs = zoom(pred, zoom=(zf, yf, xf), order=1)  # trilinear
    # numeric safety: force exact shape if off by 1 due to rounding
    pred_rs = pred_rs[: target_shape[0], : target_shape[1], : target_shape[2]]
    if pred_rs.shape != target_shape:
        # pad if needed
        out = np.zeros(target_shape, dtype=np.float32)
        z = min(target_shape[0], pred_rs.shape[0])
        y = min(target_shape[1], pred_rs.shape[1])
        x = min(target_shape[2], pred_rs.shape[2])
        out[:z, :y, :x] = pred_rs[:z, :y, :x]
        pred_rs = out

    return pred_rs.astype(np.float32), f"RESAMPLED from {pred.shape} to {target_shape}"


# -----------------------------
# Metrics
# -----------------------------
def mae_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> Tuple[float, float]:
    if mask is None:
        diff = pred - gt
    else:
        diff = (pred - gt)[mask]
    if diff.size == 0:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


def sign_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        p = pred
        g = gt
    else:
        p = pred[mask]
        g = gt[mask]
    if p.size == 0:
        return float("nan")

    # treat 0 as positive (rare)
    sp = p >= 0
    sg = g >= 0
    return float(np.mean(sp == sg))


def eikonal_error_near_surface(pred: np.ndarray, spacing_xyz: Tuple[float, float, float], band_mask: np.ndarray) -> float:
    """
    Compute mean | ||∇pred|| - 1 | in near-surface band.
    pred is (z,y,x)
    spacing_xyz = (sx,sy,sz)
    Need spacings in array axis order: (sz,sy,sx)
    """
    if band_mask.sum() == 0:
        return float("nan")

    sx, sy, sz = spacing_xyz
    # np.gradient expects spacing per axis corresponding to array axes (z,y,x)
    dz, dy, dx = sz, sy, sx

    gz, gy, gx = np.gradient(pred, dz, dy, dx)
    grad_norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    err = np.abs(grad_norm - 1.0)
    return float(np.mean(err[band_mask]))


# -----------------------------
# Case discovery
# -----------------------------
def list_cases(gt_root: Path, pred_root: Path) -> List[str]:
    gt_cases = {p.name for p in gt_root.iterdir() if p.is_dir()}
    pred_cases = {p.name for p in pred_root.iterdir() if p.is_dir()}
    return sorted(list(gt_cases.intersection(pred_cases)))


def eval_one_case(
    case: str,
    gt_root: Path,
    pred_root: Path,
    band_mm: float,
    compute_eikonal: bool,
) -> Dict[str, object]:
    case_gt = gt_root / case
    case_pred = pred_root / case

    gt_inner_path = case_gt / "INNER_SDF.nrrd"
    gt_outer_path = case_gt / "OUTER_SDF.nrrd"
    pr_inner_path = case_pred / "pred_inner_sdf.npy"
    pr_outer_path = case_pred / "pred_outer_sdf.npy"

    row: Dict[str, object] = {
        "case": case,
        "status": "OK",
        "error": "",
        "gt_inner": str(gt_inner_path),
        "gt_outer": str(gt_outer_path),
        "pred_inner": str(pr_inner_path),
        "pred_outer": str(pr_outer_path),
        "band_mm": band_mm,
    }

    missing = [p.name for p in [gt_inner_path, gt_outer_path, pr_inner_path, pr_outer_path] if not p.exists()]
    if missing:
        row["status"] = "MISSING"
        row["error"] = "Missing: " + ", ".join(missing)
        return row

    try:
        gt_inner, spacing_xyz = read_gt_sdf_nrrd(gt_inner_path)
        gt_outer, spacing_xyz2 = read_gt_sdf_nrrd(gt_outer_path)
        # ensure same spacing for both GT fields (should be)
        if spacing_xyz2 != spacing_xyz:
            row["spacing_mismatch"] = f"INNER spacing={spacing_xyz}, OUTER spacing={spacing_xyz2}"

        pr_inner = read_pred_npy(pr_inner_path)
        pr_outer = read_pred_npy(pr_outer_path)

        # Ensure shapes match GT grids
        pr_inner, note_i = ensure_shape(pr_inner, gt_inner.shape)
        pr_outer, note_o = ensure_shape(pr_outer, gt_outer.shape)
        row["pred_inner_shape_note"] = note_i
        row["pred_outer_shape_note"] = note_o

        # Masks for near-surface band
        band_inner = np.abs(gt_inner) < float(band_mm)
        band_outer = np.abs(gt_outer) < float(band_mm)

        # Whole-grid metrics
        inner_mae, inner_rmse = mae_rmse(pr_inner, gt_inner)
        outer_mae, outer_rmse = mae_rmse(pr_outer, gt_outer)

        row["inner_mae_mm"] = inner_mae
        row["inner_rmse_mm"] = inner_rmse
        row["outer_mae_mm"] = outer_mae
        row["outer_rmse_mm"] = outer_rmse

        # Near-surface metrics
        inner_mae_b, inner_rmse_b = mae_rmse(pr_inner, gt_inner, mask=band_inner)
        outer_mae_b, outer_rmse_b = mae_rmse(pr_outer, gt_outer, mask=band_outer)

        row["inner_mae_band_mm"] = inner_mae_b
        row["inner_rmse_band_mm"] = inner_rmse_b
        row["outer_mae_band_mm"] = outer_mae_b
        row["outer_rmse_band_mm"] = outer_rmse_b

        # Sign accuracy
        row["inner_sign_acc"] = sign_accuracy(pr_inner, gt_inner)
        row["outer_sign_acc"] = sign_accuracy(pr_outer, gt_outer)
        row["inner_sign_acc_band"] = sign_accuracy(pr_inner, gt_inner, mask=band_inner)
        row["outer_sign_acc_band"] = sign_accuracy(pr_outer, gt_outer, mask=band_outer)

        # Optional eikonal error in near-surface band
        if compute_eikonal:
            row["inner_eikonal_band_mean_abs_err"] = eikonal_error_near_surface(pr_inner, spacing_xyz, band_inner)
            row["outer_eikonal_band_mean_abs_err"] = eikonal_error_near_surface(pr_outer, spacing_xyz, band_outer)
        else:
            row["inner_eikonal_band_mean_abs_err"] = ""
            row["outer_eikonal_band_mean_abs_err"] = ""

        # Basic info
        row["gt_shape_zyx"] = str(tuple(gt_inner.shape))
        row["spacing_xyz_mm"] = str(tuple(float(x) for x in spacing_xyz))
        row["band_inner_voxels"] = int(band_inner.sum())
        row["band_outer_voxels"] = int(band_outer.sum())

        return row

    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = repr(e)
        return row


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in rows if r.get("status") == "OK"]

    def collect(key: str) -> np.ndarray:
        vals = []
        for r in ok:
            v = r.get(key, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    def stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    summary = {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_failed_or_missing": len(rows) - len(ok),
        "inner": {
            "mae_mm": stats(collect("inner_mae_mm")),
            "rmse_mm": stats(collect("inner_rmse_mm")),
            "mae_band_mm": stats(collect("inner_mae_band_mm")),
            "rmse_band_mm": stats(collect("inner_rmse_band_mm")),
            "sign_acc": stats(collect("inner_sign_acc")),
            "sign_acc_band": stats(collect("inner_sign_acc_band")),
            "eikonal_band_mean_abs_err": stats(collect("inner_eikonal_band_mean_abs_err")),
        },
        "outer": {
            "mae_mm": stats(collect("outer_mae_mm")),
            "rmse_mm": stats(collect("outer_rmse_mm")),
            "mae_band_mm": stats(collect("outer_mae_band_mm")),
            "rmse_band_mm": stats(collect("outer_rmse_band_mm")),
            "sign_acc": stats(collect("outer_sign_acc")),
            "sign_acc_band": stats(collect("outer_sign_acc_band")),
            "eikonal_band_mean_abs_err": stats(collect("outer_eikonal_band_mean_abs_err")),
        },
    }
    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=str, default=r"D:\Mo_PINNs\Processed",
                    help="Root folder containing case folders with INNER_SDF.nrrd / OUTER_SDF.nrrd.")
    ap.add_argument("--pred-root", type=str, default=r"D:\Mo_PINNs\Processed\outputs_DeepSDF_v2",
                    help="Root folder containing per-case outputs with pred_inner_sdf.npy / pred_outer_sdf.npy.")
    ap.add_argument("--out-root", type=str, default=r"D:\Mo_PINNs\Processed\outputs_DeepSDF_v2\evaluation",
                    help="Output folder for evaluation results.")
    ap.add_argument("--band-mm", type=float, default=2.0,
                    help="Near-surface band threshold: evaluate where |GT| < band_mm.")
    ap.add_argument("--no-eikonal", action="store_true",
                    help="Disable eikonal error computation (faster).")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = list_cases(gt_root, pred_root)
    if not cases:
        raise RuntimeError(f"No overlapping case folders found between:\n  {gt_root}\n  {pred_root}")

    rows: List[Dict[str, object]] = []
    for case in cases:
        row = eval_one_case(
            case=case,
            gt_root=gt_root,
            pred_root=pred_root,
            band_mm=args.band_mm,
            compute_eikonal=not args.no_eikonal,
        )
        rows.append(row)
        print(f"[{row['status']}] {case}" + (f" | {row['error']}" if row["status"] != "OK" else ""))

    # Write CSV (must include _sdf)
    csv_path = out_root / "metrics_sdf.csv"
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write per-case JSON
    per_case_path = out_root / "per_case_sdf.json"
    with open(per_case_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Summary JSON
    summary_obj = summarize(rows)
    summary_path = out_root / "summary_sdf.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Per-case JSON: {per_case_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()