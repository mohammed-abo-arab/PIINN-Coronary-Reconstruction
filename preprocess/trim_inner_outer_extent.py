#!/usr/bin/env python3
"""
Enforce matched start/end extent between INNER and OUTER masks across all cases.

For each case folder under --root:
- Loads INNER_clean.nrrd and OUTER_clean.nrrd (fallback to *_binary.nrrd if needed)
- Trims INNER to OUTER support (with tolerance in mm)
- Optionally trims OUTER to INNER support (with tolerance in mm)
- Optionally keeps only the largest connected component
- Saves:
    INNER_trimmed.nrrd
    OUTER_trimmed.nrrd
- Generates:
    trim_report.csv

Dependencies:
- SimpleITK
- numpy
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import SimpleITK as sitk


# -----------------------------
# Logging
# -----------------------------
def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# IO + Geometry helpers
# -----------------------------
def read_binary(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    img = sitk.Cast(img > 0, sitk.sitkUInt8)
    return img


def write_nrrd(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


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
    out = r.Execute(moving)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


def mm_to_radius_voxels(img: sitk.Image, radius_mm: float) -> List[int]:
    spacing = img.GetSpacing()  # (sx, sy, sz)
    if radius_mm <= 0:
        return [0, 0, 0]
    return [int(np.ceil(radius_mm / s)) for s in spacing]


def dilate(img: sitk.Image, radius_mm: float) -> sitk.Image:
    rad = mm_to_radius_voxels(img, radius_mm)
    if all(v == 0 for v in rad):
        return img
    return sitk.BinaryDilate(img, rad, sitk.sitkBall)


def connected_components(img: sitk.Image) -> Tuple[sitk.Image, int]:
    cc = sitk.ConnectedComponent(img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled)
    return relabeled, stats.GetNumberOfLabels()


def keep_largest(img: sitk.Image) -> sitk.Image:
    labeled, n = connected_components(img)
    if n == 0:
        z = sitk.Cast(img * 0, sitk.sitkUInt8)
        z.CopyInformation(img)
        return z
    out = sitk.Cast(labeled == 1, sitk.sitkUInt8)
    out.CopyInformation(img)
    return out


def voxel_count(img: sitk.Image) -> int:
    return int(sitk.GetArrayViewFromImage(img).sum())


# -----------------------------
# Trimming logic
# -----------------------------
def trim_to_match_extent(
    inner: sitk.Image,
    outer: sitk.Image,
    tolerance_mm: float = 1.5,
    symmetric: bool = True,
    keep_largest_component: bool = True,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Enforce matched spatial support (start/end extent consistency).

    - INNER is trimmed to dilated OUTER (tolerance_mm)
    - If symmetric=True, OUTER is also trimmed to dilated INNER (tolerance_mm)

    This ensures neither mask extends beyond the other (up to tolerance).
    """

    inner = sitk.Cast(inner > 0, sitk.sitkUInt8)
    outer = sitk.Cast(outer > 0, sitk.sitkUInt8)

    # 1) Trim INNER to OUTER support (with tolerance)
    outer_sup = dilate(outer, tolerance_mm)
    inner_t = sitk.And(inner, outer_sup)

    # 2) Optionally trim OUTER to INNER support (with tolerance)
    if symmetric:
        inner_sup = dilate(inner_t, tolerance_mm)
        outer_t = sitk.And(outer, inner_sup)
    else:
        outer_t = outer

    # 3) Optional: keep largest component for both (stabilizes “same length” notion)
    if keep_largest_component:
        inner_t = keep_largest(inner_t)
        outer_t = keep_largest(outer_t)

    # Ensure metadata retained
    inner_t.CopyInformation(inner)
    outer_t.CopyInformation(outer)

    return inner_t, outer_t


# -----------------------------
# Batch runner
# -----------------------------
def select_case_files(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Prefer clean files; fallback to binary files.
    """
    inner_candidates = [case_dir / "INNER_clean.nrrd", case_dir / "INNER_binary.nrrd"]
    outer_candidates = [case_dir / "OUTER_clean.nrrd", case_dir / "OUTER_binary.nrrd"]

    inner_path = next((p for p in inner_candidates if p.exists()), None)
    outer_path = next((p for p in outer_candidates if p.exists()), None)

    return inner_path, outer_path


def process_root(
    root: Path,
    tolerance_mm: float,
    symmetric: bool,
    keep_largest_component: bool,
    overwrite: bool,
    out_inner_name: str,
    out_outer_name: str,
    report_name: str,
) -> None:
    case_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.lower() != "codes"]

    if not case_dirs:
        logging.warning(f"No case folders found under: {root}")
        return

    report_path = root / "codes" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    logging.info(f"Found {len(case_dirs)} case folders under: {root}")

    for case_dir in case_dirs:
        inner_path, outer_path = select_case_files(case_dir)
        if inner_path is None or outer_path is None:
            logging.warning(f"Skipping {case_dir.name}: missing INNER/OUTER files.")
            continue

        inner = read_binary(inner_path)
        outer = read_binary(outer_path)

        if not same_geometry(inner, outer):
            logging.warning(f"{case_dir.name}: geometry mismatch; resampling OUTER to INNER grid.")
            outer = resample_to_reference(outer, inner)

        inner_vox0, outer_vox0 = voxel_count(inner), voxel_count(outer)
        _, inner_cc0 = connected_components(inner)
        _, outer_cc0 = connected_components(outer)

        inner_t, outer_t = trim_to_match_extent(
            inner=inner,
            outer=outer,
            tolerance_mm=tolerance_mm,
            symmetric=symmetric,
            keep_largest_component=keep_largest_component,
        )

        inner_vox1, outer_vox1 = voxel_count(inner_t), voxel_count(outer_t)
        _, inner_cc1 = connected_components(inner_t)
        _, outer_cc1 = connected_components(outer_t)

        out_inner = case_dir / out_inner_name
        out_outer = case_dir / out_outer_name

        if (out_inner.exists() or out_outer.exists()) and not overwrite:
            logging.info(f"{case_dir.name}: outputs exist; use --overwrite to replace.")
        else:
            write_nrrd(inner_t, out_inner)
            write_nrrd(outer_t, out_outer)
            logging.info(
                f"{case_dir.name} | INNER {inner_vox0:,}->{inner_vox1:,} vox (cc {inner_cc0}->{inner_cc1}) | "
                f"OUTER {outer_vox0:,}->{outer_vox1:,} vox (cc {outer_cc0}->{outer_cc1}) | saved"
            )

        rows.append({
            "case": case_dir.name,
            "inner_in": inner_path.name,
            "outer_in": outer_path.name,
            "inner_vox_before": inner_vox0,
            "outer_vox_before": outer_vox0,
            "inner_cc_before": inner_cc0,
            "outer_cc_before": outer_cc0,
            "inner_vox_after": inner_vox1,
            "outer_vox_after": outer_vox1,
            "inner_cc_after": inner_cc1,
            "outer_cc_after": outer_cc1,
            "tolerance_mm": tolerance_mm,
            "symmetric": symmetric,
            "keep_largest_component": keep_largest_component,
            "inner_out": out_inner_name,
            "outer_out": out_outer_name,
        })

    # Write report
    fieldnames = list(rows[0].keys()) if rows else ["case"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    logging.info(f"Report saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim INNER/OUTER masks to enforce matched start/end extent across dataset."
    )
    parser.add_argument("--root", type=str, required=True, help='Path to "Processed" folder.')
    parser.add_argument("--tolerance-mm", type=float, default=1.5, help="Tolerance (mm) for extent matching.")
    parser.add_argument("--symmetric", action="store_true", help="Also trim OUTER to INNER support (recommended).")
    parser.add_argument("--no-keep-largest", action="store_true", help="Do NOT keep only the largest component.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trimmed outputs.")
    parser.add_argument("--out-inner", type=str, default="INNER_trimmed.nrrd", help="Output inner filename per case.")
    parser.add_argument("--out-outer", type=str, default="OUTER_trimmed.nrrd", help="Output outer filename per case.")
    parser.add_argument("--report", type=str, default="trim_report.csv", help="CSV report filename (written into codes/).")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    args = parser.parse_args()
    setup_logger(args.log_level)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    process_root(
        root=root,
        tolerance_mm=args.tolerance_mm,
        symmetric=args.symmetric,
        keep_largest_component=not args.no_keep_largest,
        overwrite=args.overwrite,
        out_inner_name=args.out_inner,
        out_outer_name=args.out_outer,
        report_name=args.report,
    )


if __name__ == "__main__":
    main()
