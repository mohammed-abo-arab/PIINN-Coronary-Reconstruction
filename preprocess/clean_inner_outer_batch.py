#!/usr/bin/env python3
"""
Batch topology-consistent cleanup for coronary INNER/OUTER binary NRRD masks.

Goal:
- Remove outer-only branches (outer leakage) by intersecting OUTER with a dilated INNER.
- Optionally remove inner-only fragments by intersecting INNER with an eroded OUTER.
- Enforce "same topology" pragmatically by keeping only mutually overlapping connected components
  (and optionally keeping the largest consistent component).

Outputs per case folder:
- OUTER_clean.nrrd
- INNER_clean.nrrd

Dependencies:
- SimpleITK
- numpy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import SimpleITK as sitk


# -----------------------------
# Utilities
# -----------------------------

def setup_logger(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_binary_nrrd(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    img = sitk.ReadImage(str(path))
    # Enforce binary (0/1)
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
    """
    Resample 'moving' to match 'reference' grid using nearest neighbor (for labels).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return sitk.Cast(resampler.Execute(moving), sitk.sitkUInt8)


def mm_to_radius_voxels(img: sitk.Image, radius_mm: float) -> List[int]:
    """
    Convert a physical radius in mm to per-axis voxel radii (x,y,z) for SimpleITK morphology.
    """
    spacing = img.GetSpacing()  # (sx, sy, sz)
    if radius_mm <= 0:
        return [0, 0, 0]
    return [int(np.ceil(radius_mm / s)) for s in spacing]


def binary_dilate(img: sitk.Image, radius_mm: float) -> sitk.Image:
    r = mm_to_radius_voxels(img, radius_mm)
    if all(v == 0 for v in r):
        return img
    return sitk.BinaryDilate(img, r, sitk.sitkBall)


def binary_erode(img: sitk.Image, radius_mm: float) -> sitk.Image:
    r = mm_to_radius_voxels(img, radius_mm)
    if all(v == 0 for v in r):
        return img
    return sitk.BinaryErode(img, r, sitk.sitkBall)


def connected_components(img: sitk.Image) -> Tuple[sitk.Image, int]:
    """
    Return (label_image, num_components). labels are 1..N, background 0.
    """
    cc = sitk.ConnectedComponent(img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)  # largest first
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled)
    num = stats.GetNumberOfLabels()
    return relabeled, num


def keep_largest_component(img: sitk.Image) -> sitk.Image:
    labeled, num = connected_components(img)
    if num == 0:
        return sitk.Cast(img * 0, sitk.sitkUInt8)
    largest = sitk.Cast(labeled == 1, sitk.sitkUInt8)
    largest.CopyInformation(img)
    return largest


def overlap_mask(component_labels: sitk.Image, support_mask: sitk.Image) -> sitk.Image:
    """
    Keep only those connected components in component_labels that overlap support_mask.
    component_labels: labeled image (0 background, 1..N components)
    support_mask: binary image (0/1)

    Returns binary mask containing only overlapping components.
    """
    # Convert to arrays
    lab = sitk.GetArrayFromImage(component_labels).astype(np.int32)
    sup = sitk.GetArrayFromImage(support_mask).astype(np.uint8)

    kept = np.zeros_like(lab, dtype=np.uint8)
    labels = np.unique(lab)
    labels = labels[labels != 0]
    if labels.size == 0:
        out = sitk.GetImageFromArray(kept)
        out.CopyInformation(component_labels)
        return sitk.Cast(out, sitk.sitkUInt8)

    for k in labels:
        comp = (lab == k)
        if np.any(sup[comp] > 0):
            kept[comp] = 1

    out = sitk.GetImageFromArray(kept)
    out.CopyInformation(component_labels)
    return sitk.Cast(out, sitk.sitkUInt8)


def summary_stats(img: sitk.Image) -> Tuple[int, int]:
    """
    Returns (voxel_count, num_components).
    """
    vox = int(sitk.GetArrayViewFromImage(img).sum())
    _, num = connected_components(img)
    return vox, num


# -----------------------------
# Core cleanup logic
# -----------------------------

def topology_consistent_cleanup(
    inner: sitk.Image,
    outer: sitk.Image,
    dilation_mm: float = 2.0,
    erosion_mm: float = 0.0,
    keep_largest: bool = True,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Enforce consistent topology between INNER and OUTER.

    Steps:
    1) Ensure OUTER is supported by INNER: OUTER <- OUTER ∩ dilate(INNER)
    2) Optionally ensure INNER is supported by OUTER: INNER <- INNER ∩ erode(OUTER_clean)
    3) Keep only mutually overlapping connected components:
       - Keep INNER components that overlap OUTER_clean
       - Keep OUTER components that overlap dilate(INNER_clean) (robust)
    4) Optionally keep only the largest consistent component (dominant vessel tree)
    5) Final guarantee: INNER_clean ⊂ OUTER_clean (by a last intersection with dilated inner)

    Notes:
    - dilation_mm should be >= expected wall thickness margin (in mm) to avoid over-trimming outer.
    - erosion_mm can be small (e.g., 0.5–1.0 mm) to remove inner spurs not supported by outer.
    """
    inner = sitk.Cast(inner > 0, sitk.sitkUInt8)
    outer = sitk.Cast(outer > 0, sitk.sitkUInt8)

    # (1) Outer supported by inner (dilated)
    inner_d = binary_dilate(inner, dilation_mm)
    outer1 = sitk.And(outer, inner_d)

    # (2) Optional: Inner supported by outer (eroded)
    if erosion_mm > 0:
        outer_e = binary_erode(outer1, erosion_mm)
        inner1 = sitk.And(inner, outer_e)
    else:
        inner1 = inner

    # (3) Mutual overlap on connected components
    inner_lbl, inner_n = connected_components(inner1)
    outer_lbl, outer_n = connected_components(outer1)

    # Keep only components that overlap the other structure
    inner2 = overlap_mask(inner_lbl, outer1)
    # For outer support, use dilated inner2 for robustness
    inner2_d = binary_dilate(inner2, dilation_mm)
    outer2 = overlap_mask(outer_lbl, inner2_d)

    # (4) Optional: keep largest component (dominant tree) for both
    if keep_largest:
        inner3 = keep_largest_component(inner2)
        outer3 = keep_largest_component(outer2)
    else:
        inner3, outer3 = inner2, outer2

    # (5) Final inclusion guarantee: outer3 supported by dilated inner3
    inner3_d = binary_dilate(inner3, dilation_mm)
    outer4 = sitk.And(outer3, inner3_d)

    # Ensure metadata consistency
    inner3.CopyInformation(inner)
    outer4.CopyInformation(outer)

    return inner3, outer4


# -----------------------------
# Batch runner
# -----------------------------

def process_case_folder(
    case_dir: Path,
    inner_name: str,
    outer_name: str,
    dilation_mm: float,
    erosion_mm: float,
    keep_largest: bool,
    overwrite: bool,
) -> None:
    inner_path = case_dir / inner_name
    outer_path = case_dir / outer_name

    if not (inner_path.exists() and outer_path.exists()):
        logging.debug(f"Skipping (missing files): {case_dir.name}")
        return

    logging.info(f"Case: {case_dir.name}")

    inner = read_binary_nrrd(inner_path)
    outer = read_binary_nrrd(outer_path)

    # Geometry alignment
    if not same_geometry(inner, outer):
        logging.warning("  Geometry mismatch. Resampling OUTER to INNER reference grid.")
        outer = resample_to_reference(outer, inner)

    # Before stats
    inner_vox0, inner_cc0 = summary_stats(inner)
    outer_vox0, outer_cc0 = summary_stats(outer)
    logging.info(f"  Before | INNER vox={inner_vox0:,} cc={inner_cc0} | OUTER vox={outer_vox0:,} cc={outer_cc0}")

    inner_clean, outer_clean = topology_consistent_cleanup(
        inner=inner,
        outer=outer,
        dilation_mm=dilation_mm,
        erosion_mm=erosion_mm,
        keep_largest=keep_largest,
    )

    # After stats
    inner_vox1, inner_cc1 = summary_stats(inner_clean)
    outer_vox1, outer_cc1 = summary_stats(outer_clean)
    logging.info(f"  After  | INNER vox={inner_vox1:,} cc={inner_cc1} | OUTER vox={outer_vox1:,} cc={outer_cc1}")

    # Output paths
    out_inner = case_dir / "INNER_clean.nrrd"
    out_outer = case_dir / "OUTER_clean.nrrd"

    if (out_inner.exists() or out_outer.exists()) and not overwrite:
        logging.warning("  Outputs already exist. Use --overwrite to replace them.")
        return

    write_nrrd(inner_clean, out_inner)
    write_nrrd(outer_clean, out_outer)
    logging.info("  Saved: INNER_clean.nrrd, OUTER_clean.nrrd")


def iter_case_folders(root_processed: Path) -> List[Path]:
    """
    Return immediate subfolders that look like case folders.
    """
    return [p for p in root_processed.iterdir() if p.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch cleanup of INNER/OUTER binary NRRDs to enforce topology consistency for PINNs."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=r'Path to "Processed" folder, e.g. T:\MedLAB-research-plan-28102025\one-year-25112025\PINNs\Logkitsi\CoronaryDataset\Processed',
    )
    parser.add_argument("--inner-name", type=str, default="INNER_binary.nrrd", help="Inner binary filename in each case folder.")
    parser.add_argument("--outer-name", type=str, default="OUTER_binary.nrrd", help="Outer binary filename in each case folder.")
    parser.add_argument("--dilation-mm", type=float, default=2.0, help="Dilation radius (mm) applied to INNER to support OUTER cleanup.")
    parser.add_argument("--erosion-mm", type=float, default=0.0, help="Optional erosion radius (mm) applied to OUTER to trim INNER-only fragments.")
    parser.add_argument("--no-keep-largest", action="store_true", help="Do NOT reduce to largest connected component.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing INNER_clean/OUTER_clean outputs.")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    args = parser.parse_args()

    setup_logger(args.log_level)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    keep_largest = not args.no_keep_largest

    case_dirs = iter_case_folders(root)
    if not case_dirs:
        logging.warning(f"No subfolders found under: {root}")
        return

    logging.info(f"Found {len(case_dirs)} case folders under: {root}")
    for case_dir in case_dirs:
        try:
            process_case_folder(
                case_dir=case_dir,
                inner_name=args.inner_name,
                outer_name=args.outer_name,
                dilation_mm=args.dilation_mm,
                erosion_mm=args.erosion_mm,
                keep_largest=keep_largest,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logging.error(f"Failed case {case_dir.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
