#!/usr/bin/env python3
"""
generate_and_qc_sdf.py

Batch-generate Signed Distance Fields (SDFs) for INNER/OUTER vessel masks and run QC.

Key fixes vs v1:
- Containment QC is performed with a tolerance in mm (robust to voxel discretization).
- Optional minimal containment enforcement: OUTER <- OUTER OR INNER.
  This guarantees physical validity (lumen inside wall) while minimally changing OUTER.

Inputs per case (preferred order):
- INNER_trimmed.nrrd, OUTER_trimmed.nrrd
- INNER_clean.nrrd,   OUTER_clean.nrrd
- INNER_binary.nrrd,  OUTER_binary.nrrd

Outputs per case:
- INNER_sdf.nrrd  (float32, mm, inside negative / outside positive)
- OUTER_sdf.nrrd  (float32, mm, inside negative / outside positive)
- Optionally: OUTER_contained.nrrd (if --save-contained-outer)

Global QC output:
- Processed/codes/sdf_qc_report.csv

Dependencies:
- SimpleITK
- numpy
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

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
# IO + geometry
# -----------------------------
def read_binary(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img > 0, sitk.sitkUInt8)


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


def select_case_files(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    inner_candidates = [
        case_dir / "INNER_trimmed.nrrd",
        case_dir / "INNER_clean.nrrd",
        case_dir / "INNER_binary.nrrd",
    ]
    outer_candidates = [
        case_dir / "OUTER_trimmed.nrrd",
        case_dir / "OUTER_clean.nrrd",
        case_dir / "OUTER_binary.nrrd",
    ]
    inner_path = next((p for p in inner_candidates if p.exists()), None)
    outer_path = next((p for p in outer_candidates if p.exists()), None)
    return inner_path, outer_path


# -----------------------------
# Morphology in mm
# -----------------------------
def mm_to_radius_voxels(img: sitk.Image, radius_mm: float) -> List[int]:
    sx, sy, sz = img.GetSpacing()
    if radius_mm <= 0:
        return [0, 0, 0]
    return [int(np.ceil(radius_mm / sx)), int(np.ceil(radius_mm / sy)), int(np.ceil(radius_mm / sz))]


def dilate_mm(mask: sitk.Image, radius_mm: float) -> sitk.Image:
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    rad = mm_to_radius_voxels(mask, radius_mm)
    if all(v == 0 for v in rad):
        return mask
    out = sitk.BinaryDilate(mask, rad, sitk.sitkBall)
    out.CopyInformation(mask)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


# -----------------------------
# Containment
# -----------------------------
def enforce_containment_minimal(inner: sitk.Image, outer: sitk.Image) -> sitk.Image:
    """
    Minimal enforcement: OUTER <- OUTER OR INNER.
    Guarantees inner is inside outer (voxel-wise), without removing anything.
    """
    inner = sitk.Cast(inner > 0, sitk.sitkUInt8)
    outer = sitk.Cast(outer > 0, sitk.sitkUInt8)
    out2 = sitk.Or(outer, inner)
    out2.CopyInformation(outer)
    return sitk.Cast(out2 > 0, sitk.sitkUInt8)


def containment_violation_count_tolerant(inner: sitk.Image, outer: sitk.Image, tol_mm: float) -> int:
    """
    Count voxels where inner==1 but dilated(outer, tol_mm)==0.
    This is the meaningful physical containment check under discretization.
    """
    inner = sitk.Cast(inner > 0, sitk.sitkUInt8)
    outer = sitk.Cast(outer > 0, sitk.sitkUInt8)
    outer_sup = dilate_mm(outer, tol_mm)

    inn = sitk.GetArrayFromImage(inner).astype(np.uint8) > 0
    outsup = sitk.GetArrayFromImage(outer_sup).astype(np.uint8) > 0
    return int(np.sum(inn & (~outsup)))


# -----------------------------
# SDF computation
# -----------------------------
def compute_sdf_mm(mask: sitk.Image) -> sitk.Image:
    """
    Signed distance in mm using Maurer distance.
    Convention: inside negative, outside positive.
    """
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    sdf = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,   # inside negative
        squaredDistance=False,
        useImageSpacing=True,     # mm
    )
    sdf = sitk.Cast(sdf, sitk.sitkFloat32)
    sdf.CopyInformation(mask)
    return sdf


# -----------------------------
# QC metrics
# -----------------------------
def _np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def qc_sdf(mask: sitk.Image, sdf: sitk.Image, band_mm: float = 2.0) -> Dict[str, float]:
    """
    QC checks:
    - sign sanity (inside mean < 0, outside mean > 0)
    - boundary band near zero
    - eikonal approx: |grad(sdf)| ~ 1 in band
    """
    m = _np(mask).astype(np.uint8) > 0
    d = _np(sdf).astype(np.float32)

    inside_vals = d[m]
    outside_vals = d[~m]

    # Robust sampling
    rng = np.random.default_rng(0)

    def sample(arr: np.ndarray, n: int = 200000) -> np.ndarray:
        if arr.size <= n:
            return arr
        idx = rng.choice(arr.size, size=n, replace=False)
        return arr[idx]

    inside_s = sample(inside_vals)
    outside_s = sample(outside_vals)

    # Band around surface
    band = np.abs(d) <= float(band_mm)
    band_vals = d[band]

    # Eikonal in band
    grad_mag = sitk.GradientMagnitude(sdf)
    g = _np(grad_mag).astype(np.float32)
    g_band = g[band]

    out: Dict[str, float] = {}
    out["inside_mean_mm"] = float(np.mean(inside_s)) if inside_s.size else np.nan
    out["inside_p95_mm"] = float(np.percentile(inside_s, 95)) if inside_s.size else np.nan
    out["outside_mean_mm"] = float(np.mean(outside_s)) if outside_s.size else np.nan
    out["outside_p05_mm"] = float(np.percentile(outside_s, 5)) if outside_s.size else np.nan

    out["band_mm"] = float(band_mm)
    out["band_mean_abs_mm"] = float(np.mean(np.abs(band_vals))) if band_vals.size else np.nan
    out["band_p95_abs_mm"] = float(np.percentile(np.abs(band_vals), 95)) if band_vals.size else np.nan

    out["eikonal_band_mean_abs_err"] = float(np.mean(np.abs(g_band - 1.0))) if g_band.size else np.nan
    out["eikonal_band_p95_abs_err"] = float(np.percentile(np.abs(g_band - 1.0), 95)) if g_band.size else np.nan

    return out


def pass_fail_rules(
    qc_inner: Dict[str, float],
    qc_outer: Dict[str, float],
    contain_viol_tolerant: int,
) -> Tuple[bool, List[str]]:
    """
    Conservative QC rules for PINN readiness.
    """
    reasons: List[str] = []

    # Sign sanity
    if not (qc_inner["inside_mean_mm"] < 0 and qc_inner["outside_mean_mm"] > 0):
        reasons.append("INNER sign convention questionable")
    if not (qc_outer["inside_mean_mm"] < 0 and qc_outer["outside_mean_mm"] > 0):
        reasons.append("OUTER sign convention questionable")

    # Boundary band near zero (heuristic)
    if not (qc_inner["band_mean_abs_mm"] <= 0.75 * qc_inner["band_mm"]):
        reasons.append("INNER boundary band not near zero")
    if not (qc_outer["band_mean_abs_mm"] <= 0.75 * qc_outer["band_mm"]):
        reasons.append("OUTER boundary band not near zero")

    # Eikonal (heuristic)
    if not (qc_inner["eikonal_band_mean_abs_err"] <= 0.35):
        reasons.append("INNER eikonal error high")
    if not (qc_outer["eikonal_band_mean_abs_err"] <= 0.35):
        reasons.append("OUTER eikonal error high")

    # Containment tolerant
    if contain_viol_tolerant != 0:
        reasons.append(f"Tolerant containment violated (inner outside dilated outer voxels={contain_viol_tolerant})")

    return (len(reasons) == 0), reasons


# -----------------------------
# Batch runner
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SDFs (mm) for INNER/OUTER and QC them (v2).")
    ap.add_argument("--root", required=True, type=str, help='Path to "Processed" folder.')
    ap.add_argument("--band-mm", type=float, default=2.0, help="QC band around surface (mm).")
    ap.add_argument("--containment-tol-mm", type=float, default=0.5,
                    help="Tolerance (mm) used for containment QC: inner ⊆ dilated(outer, tol).")
    ap.add_argument("--enforce-containment", action="store_true",
                    help="Enforce containment minimally: OUTER <- OUTER OR INNER.")
    ap.add_argument("--save-contained-outer", action="store_true",
                    help="If enforcing containment, also save OUTER_contained.nrrd per case.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing INNER_sdf/OUTER_sdf outputs.")
    ap.add_argument("--report", type=str, default="sdf_qc_report.csv",
                    help="CSV report filename (written into Processed/codes/).")
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

    rows: List[Dict[str, object]] = []
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

        # Optional containment enforcement
        contained_outer_saved = ""
        if args.enforce_containment:
            outer = enforce_containment_minimal(inner, outer)
            if args.save_contained_outer:
                out_contained = case_dir / "OUTER_contained.nrrd"
                write_nrrd(outer, out_contained)
                contained_outer_saved = out_contained.name

        # Tolerant containment QC
        contain_viol_tol = containment_violation_count_tolerant(inner, outer, tol_mm=args.containment_tol_mm)

        # SDF outputs
        out_inner_sdf = case_dir / "INNER_sdf.nrrd"
        out_outer_sdf = case_dir / "OUTER_sdf.nrrd"

        # Compute / save SDFs
        if (out_inner_sdf.exists() or out_outer_sdf.exists()) and not args.overwrite:
            logging.info(f"{case_dir.name}: SDF outputs exist; use --overwrite to replace.")
            inner_sdf = sitk.ReadImage(str(out_inner_sdf)) if out_inner_sdf.exists() else compute_sdf_mm(inner)
            outer_sdf = sitk.ReadImage(str(out_outer_sdf)) if out_outer_sdf.exists() else compute_sdf_mm(outer)
        else:
            inner_sdf = compute_sdf_mm(inner)
            outer_sdf = compute_sdf_mm(outer)
            write_nrrd(inner_sdf, out_inner_sdf)
            write_nrrd(outer_sdf, out_outer_sdf)

        # QC SDF quality
        qc_i = qc_sdf(inner, inner_sdf, band_mm=args.band_mm)
        qc_o = qc_sdf(outer, outer_sdf, band_mm=args.band_mm)

        passed, reasons = pass_fail_rules(qc_i, qc_o, contain_viol_tol)
        if passed:
            pass_n += 1
        else:
            fail_n += 1

        row: Dict[str, object] = {
            "case": case_dir.name,
            "inner_in": inner_path.name,
            "outer_in": outer_path.name,
            "enforce_containment": args.enforce_containment,
            "containment_tol_mm": args.containment_tol_mm,
            "tolerant_containment_violations_vox": contain_viol_tol,
            "contained_outer_saved": contained_outer_saved,
            "inner_sdf": out_inner_sdf.name,
            "outer_sdf": out_outer_sdf.name,
            "band_mm": args.band_mm,
            "pass": passed,
            "fail_reasons": "; ".join(reasons) if reasons else "",
        }
        row.update({f"inner_{k}": v for k, v in qc_i.items()})
        row.update({f"outer_{k}": v for k, v in qc_o.items()})

        rows.append(row)

        logging.info(
            f"{case_dir.name} | PASS={passed} | tol_contain_viol={contain_viol_tol} | "
            f"inner_eik_err={qc_i['eikonal_band_mean_abs_err']:.3f} outer_eik_err={qc_o['eikonal_band_mean_abs_err']:.3f}"
        )
        if not passed:
            logging.warning(f"{case_dir.name} | QC FAIL reasons: {row['fail_reasons']}")

    # Write report
    if rows:
        fieldnames = list(rows[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        logging.info(f"Report saved: {report_path}")
        logging.info(f"Summary: PASS={pass_n} | FAIL={fail_n} | TOTAL={pass_n + fail_n}")
    else:
        logging.warning("No rows written (no valid cases processed).")


if __name__ == "__main__":
    main()
