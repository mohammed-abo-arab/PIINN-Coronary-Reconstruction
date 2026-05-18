#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

# Corrected defaults:
# outputs are saved inside the existing Vanilla INR root, in a new subfolder for v2.
DEFAULT_GT_ROOT = Path(r"D:\Mo_PINNs\Processed")
DEFAULT_PRED_ROOT = Path(r"D:\Mo_PINNs\Processed\outputs_DeepSDF_v2")
DEFAULT_OUT_ROOT = DEFAULT_PRED_ROOT / "evaluation_sdf"

def read_gt_sdf_nrrd(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()

def read_pred_npy(path: Path) -> np.ndarray:
    return np.load(str(path)).astype(np.float32, copy=False)

def ensure_shape(pred: np.ndarray, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
    if pred.shape == target_shape:
        return pred, "OK"
    zf = target_shape[0] / pred.shape[0]
    yf = target_shape[1] / pred.shape[1]
    xf = target_shape[2] / pred.shape[2]
    pred_rs = zoom(pred, zoom=(zf, yf, xf), order=1)
    pred_rs = pred_rs[:target_shape[0], :target_shape[1], :target_shape[2]]
    if pred_rs.shape != target_shape:
        out = np.zeros(target_shape, dtype=np.float32)
        z = min(target_shape[0], pred_rs.shape[0])
        y = min(target_shape[1], pred_rs.shape[1])
        x = min(target_shape[2], pred_rs.shape[2])
        out[:z, :y, :x] = pred_rs[:z, :y, :x]
        pred_rs = out
    return pred_rs.astype(np.float32), f"RESAMPLED from {pred.shape} to {target_shape}"

def mae_rmse(pred, gt, mask=None):
    diff = pred - gt if mask is None else (pred - gt)[mask]
    if diff.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff)))

def sign_accuracy(pred, gt, mask=None):
    p = pred if mask is None else pred[mask]
    g = gt if mask is None else gt[mask]
    if p.size == 0:
        return float("nan")
    return float(np.mean((p >= 0) == (g >= 0)))

def eikonal_error_near_surface(pred, spacing_xyz, band_mask):
    if band_mask.sum() == 0:
        return float("nan")
    sx, sy, sz = spacing_xyz
    dz, dy, dx = sz, sy, sx
    gz, gy, gx = np.gradient(pred, dz, dy, dx)
    grad_norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    return float(np.mean(np.abs(grad_norm - 1.0)[band_mask]))

def list_cases(gt_root: Path, pred_root: Path) -> List[str]:
    gt_cases = {p.name for p in gt_root.iterdir() if p.is_dir()}
    pred_cases = {p.name for p in pred_root.iterdir() if p.is_dir()}
    return sorted(list(gt_cases.intersection(pred_cases)))

def eval_one_case(case, gt_root, pred_root, band_mm, compute_eikonal):
    case_gt = gt_root / case
    case_pred = pred_root / case
    gi = case_gt / "INNER_SDF.nrrd"
    go = case_gt / "OUTER_SDF.nrrd"
    pi = case_pred / "pred_inner_sdf.npy"
    po = case_pred / "pred_outer_sdf.npy"
    row = {
        "case": case,
        "status": "OK",
        "error": "",
        "gt_inner": str(gi),
        "gt_outer": str(go),
        "pred_inner": str(pi),
        "pred_outer": str(po),
        "band_mm": float(band_mm),
        "compute_eikonal": bool(compute_eikonal),
    }
    missing = [p.name for p in [gi, go, pi, po] if not p.exists()]
    if missing:
        row["status"] = "MISSING"
        row["error"] = "Missing: " + ", ".join(missing)
        return row
    try:
        gt_inner, spacing_xyz = read_gt_sdf_nrrd(gi)
        gt_outer, spacing_xyz2 = read_gt_sdf_nrrd(go)
        if spacing_xyz2 != spacing_xyz:
            row["spacing_mismatch"] = f"INNER spacing={spacing_xyz}, OUTER spacing={spacing_xyz2}"
        pr_inner, note_i = ensure_shape(read_pred_npy(pi), gt_inner.shape)
        pr_outer, note_o = ensure_shape(read_pred_npy(po), gt_outer.shape)
        row["pred_inner_shape_note"] = note_i
        row["pred_outer_shape_note"] = note_o
        band_inner = np.abs(gt_inner) < float(band_mm)
        band_outer = np.abs(gt_outer) < float(band_mm)
        row["inner_mae_mm"], row["inner_rmse_mm"] = mae_rmse(pr_inner, gt_inner)
        row["outer_mae_mm"], row["outer_rmse_mm"] = mae_rmse(pr_outer, gt_outer)
        row["inner_mae_band_mm"], row["inner_rmse_band_mm"] = mae_rmse(pr_inner, gt_inner, band_inner)
        row["outer_mae_band_mm"], row["outer_rmse_band_mm"] = mae_rmse(pr_outer, gt_outer, band_outer)
        row["inner_sign_acc"] = sign_accuracy(pr_inner, gt_inner)
        row["outer_sign_acc"] = sign_accuracy(pr_outer, gt_outer)
        row["inner_sign_acc_band"] = sign_accuracy(pr_inner, gt_inner, band_inner)
        row["outer_sign_acc_band"] = sign_accuracy(pr_outer, gt_outer, band_outer)
        if compute_eikonal:
            row["inner_eikonal_band_mean_abs_err"] = eikonal_error_near_surface(pr_inner, spacing_xyz, band_inner)
            row["outer_eikonal_band_mean_abs_err"] = eikonal_error_near_surface(pr_outer, spacing_xyz, band_outer)
        row["gt_shape_zyx"] = str(tuple(gt_inner.shape))
        row["spacing_xyz_mm"] = str(tuple(float(x) for x in spacing_xyz))
        row["band_inner_voxels"] = int(band_inner.sum())
        row["band_outer_voxels"] = int(band_outer.sum())
        return row
    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = repr(e)
        return row

def summarize(rows):
    ok = [r for r in rows if r.get("status") == "OK"]
    def collect(key):
        vals = []
        for r in ok:
            v = r.get(key, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)
    def stats(arr):
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
        },
        "outer": {
            "mae_mm": stats(collect("outer_mae_mm")),
            "rmse_mm": stats(collect("outer_rmse_mm")),
            "mae_band_mm": stats(collect("outer_mae_band_mm")),
            "rmse_band_mm": stats(collect("outer_rmse_band_mm")),
            "sign_acc": stats(collect("outer_sign_acc")),
            "sign_acc_band": stats(collect("outer_sign_acc_band")),
        },
    }
    if any("inner_eikonal_band_mean_abs_err" in r for r in ok):
        summary["inner"]["eikonal_band_mean_abs_err"] = stats(collect("inner_eikonal_band_mean_abs_err"))
        summary["outer"]["eikonal_band_mean_abs_err"] = stats(collect("outer_eikonal_band_mean_abs_err"))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=str, default=str(DEFAULT_GT_ROOT))
    ap.add_argument("--pred-root", type=str, default=str(DEFAULT_PRED_ROOT))
    ap.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--band-mm", type=float, default=1.0)
    ap.add_argument("--compute-eikonal", action="store_true",
                    help="Enable auxiliary Eikonal diagnostic. OFF by default for Vanilla INR.")
    args = ap.parse_args()
    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cases = list_cases(gt_root, pred_root)
    if not cases:
        raise RuntimeError(f"No overlapping case folders found between:\n  {gt_root}\n  {pred_root}")
    rows = []
    for case in cases:
        row = eval_one_case(case, gt_root, pred_root, args.band_mm, args.compute_eikonal)
        rows.append(row)
        print(f"[{row['status']}] {case}" + (f" | {row['error']}" if row["status"] != "OK" else ""))
    csv_path = out_root / "metrics_sdf.csv"
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_root / "per_case_sdf.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_root / "summary_sdf.json", "w", encoding="utf-8") as f:
        json.dump(summarize(rows), f, indent=2)
    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Per-case JSON: {out_root / 'per_case_sdf.json'}")
    print(f"Summary JSON: {out_root / 'summary_sdf.json'}")

if __name__ == "__main__":
    main()
