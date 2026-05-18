#!/usr/bin/env python3
"""
Robust QC: check INNER/OUTER have the same "start" and "end" by comparing
extent along the principal vessel axis (PCA-based), in mm.

This avoids false failures from skeleton spurs / thinning artifacts.

For each case under --root:
- Load INNER_trimmed/OUTER_trimmed (fallback to *_clean)
- Resample OUTER to INNER if geometry differs
- Extract point clouds (either skeleton points or mask surface points)
- Compute PCA axis from INNER points (stable reference)
- Project INNER and OUTER points to this axis
- Compare min/max projection (extent) in mm

Outputs:
- Processed/codes/extent_pca_report.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import SimpleITK as sitk


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_binary(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    return sitk.Cast(r.Execute(moving) > 0, sitk.sitkUInt8)


def select_case_files(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    inner_candidates = [case_dir / "INNER_trimmed.nrrd", case_dir / "INNER_clean.nrrd", case_dir / "INNER_binary.nrrd"]
    outer_candidates = [case_dir / "OUTER_trimmed.nrrd", case_dir / "OUTER_clean.nrrd", case_dir / "OUTER_binary.nrrd"]
    inner_path = next((p for p in inner_candidates if p.exists()), None)
    outer_path = next((p for p in outer_candidates if p.exists()), None)
    return inner_path, outer_path


def extract_points_physical(img: sitk.Image, mode: str = "skeleton") -> np.ndarray:
    """
    Returns Nx3 physical points (mm).
    mode:
      - "mask": all foreground voxels (can be large; slower)
      - "surface": surface voxels only (recommended if you want surface extent)
      - "skeleton": skeleton points (fast and stable; recommended for vessel extent)
    """
    img = sitk.Cast(img > 0, sitk.sitkUInt8)

    if mode == "skeleton":
        # Simple and fast: thinning
        sk = sitk.BinaryThinning(img)
        arr = sitk.GetArrayFromImage(sk)  # z,y,x
        idx = np.argwhere(arr > 0)
    elif mode == "surface":
        # Surface voxels = mask - eroded(mask)
        er = sitk.BinaryErode(img, [1, 1, 1], sitk.sitkBall)
        surf = sitk.And(img, sitk.Not(er))
        arr = sitk.GetArrayFromImage(surf)
        idx = np.argwhere(arr > 0)
    elif mode == "mask":
        arr = sitk.GetArrayFromImage(img)
        idx = np.argwhere(arr > 0)
    else:
        raise ValueError("mode must be one of: skeleton, surface, mask")

    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts = []
    for z, y, x in idx:
        p = img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))  # (x,y,z)
        pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA first component axis from Nx3 points.
    Returns (center, axis_unit_vector).
    """
    c = points.mean(axis=0)
    X = points - c
    # Covariance and eig
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)  # ascending
    axis = vecs[:, np.argmax(vals)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return c, axis


def project_extent(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    Project points onto axis: t = (p-center)·axis
    Returns (t_min, t_max) in mm.
    """
    t = (points - center) @ axis
    return float(np.min(t)), float(np.max(t))


def main() -> None:
    ap = argparse.ArgumentParser(description="QC: compare INNER/OUTER start-end extent using PCA projection (mm).")
    ap.add_argument("--root", required=True, type=str, help='Path to "Processed" folder.')
    ap.add_argument("--tol-mm", type=float, default=2.0, help="Tolerance for start/end mismatch in mm.")
    ap.add_argument("--mode", type=str, default="skeleton", choices=["skeleton", "surface", "mask"],
                    help="Point extraction mode. skeleton is recommended.")
    ap.add_argument("--report", type=str, default="extent_pca_report.csv",
                    help="CSV report filename (saved to Processed/codes/).")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    setup_logger(args.log_level)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    case_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.lower() != "codes"]
    if not case_dirs:
        logging.warning(f"No case folders found under: {root}")
        return

    report_path = root / "codes" / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pass_n = 0
    fail_n = 0

    logging.info(f"Found {len(case_dirs)} case folders under: {root}")

    for case_dir in case_dirs:
        inner_path, outer_path = select_case_files(case_dir)
        if inner_path is None or outer_path is None:
            logging.warning(f"Skipping {case_dir.name}: missing inner/outer files.")
            continue

        inner = read_binary(inner_path)
        outer = read_binary(outer_path)

        if not same_geometry(inner, outer):
            logging.warning(f"{case_dir.name}: geometry mismatch; resampling OUTER to INNER.")
            outer = resample_to_reference(outer, inner)

        inner_pts = extract_points_physical(inner, mode=args.mode)
        outer_pts = extract_points_physical(outer, mode=args.mode)

        if inner_pts.shape[0] < 10 or outer_pts.shape[0] < 10:
            # Too few points to define extent reliably
            rows.append({
                "case": case_dir.name,
                "inner_file": inner_path.name,
                "outer_file": outer_path.name,
                "mode": args.mode,
                "status": "INSUFFICIENT_POINTS",
                "start_diff_mm": np.nan,
                "end_diff_mm": np.nan,
                "pass": False,
            })
            fail_n += 1
            logging.info(f"{case_dir.name} | insufficient points | PASS=False")
            continue

        # PCA axis from INNER (reference)
        center, axis = pca_axis(inner_pts)

        i_min, i_max = project_extent(inner_pts, center, axis)
        o_min, o_max = project_extent(outer_pts, center, axis)

        # Start/end differences (absolute)
        start_diff = abs(i_min - o_min)
        end_diff = abs(i_max - o_max)

        passed = (start_diff <= args.tol_mm) and (end_diff <= args.tol_mm)

        rows.append({
            "case": case_dir.name,
            "inner_file": inner_path.name,
            "outer_file": outer_path.name,
            "mode": args.mode,
            "inner_min_t_mm": i_min,
            "inner_max_t_mm": i_max,
            "outer_min_t_mm": o_min,
            "outer_max_t_mm": o_max,
            "start_diff_mm": start_diff,
            "end_diff_mm": end_diff,
            "tol_mm": args.tol_mm,
            "pass": passed,
        })

        if passed:
            pass_n += 1
        else:
            fail_n += 1

        logging.info(
            f"{case_dir.name} | start_diff={start_diff:.3f}mm end_diff={end_diff:.3f}mm | PASS={passed}"
        )

    # Save report
    fieldnames = list(rows[0].keys()) if rows else ["case"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    logging.info(f"Report saved: {report_path}")
    logging.info(f"Summary: PASS={pass_n} | FAIL={fail_n} | TOTAL={pass_n+fail_n}")


if __name__ == "__main__":
    main()
