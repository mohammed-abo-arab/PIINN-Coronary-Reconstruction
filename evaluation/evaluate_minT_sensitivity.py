#!/usr/bin/env python
"""
evaluate_minT_sensitivity.py

Reviewer Comment 3: sensitivity analysis for w_minT.

Evaluates the same 15 coronary segments under:
    0x, 0.5x, 1x (original PIINN), and 2x w_minT

The script intentionally reuses validate_wall_thickness_revision_v4.py so that
the definitions of wall thickness, ASSD/HD95, nesting, reference loading,
mesh loading, exact point-to-triangle thickness distances, and t_min are
identical to the revision-grade analysis already used in the manuscript.

Primary manuscript table:
    - Outer-surface ASSD (mm)
    - Outer-surface HD95 (mm)
    - Wall-thickness MAE (mm)
    - Wall-thickness RMSE (mm)
    - Mean predicted wall thickness (mm)
    - Wall-thickness bias (mm)
    - Enclosure violation (%)
    - Minimum-separation violation (%)

Secondary QC (saved but not required in the primary table):
    - Inner-surface ASSD / HD95
    - Reference mean wall thickness
    - Non-positive predicted thickness rate
    - Surface-level nesting violations
    - Field-based minimum-separation violation
    - training time / GPU peak memory when available

IMPORTANT
---------
Place this script in the same directory as:
    validate_wall_thickness_revision_v4.py

Default paths match the user's current project:
    D:\Mo_PINNs\Processed

The 1x condition is the already-trained original PIINN:
    outputs_pinn_v3

No retraining is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Reuse the already validated revision analysis.
try:
    import validate_wall_thickness_revision_v4 as v4
except ImportError as exc:
    raise ImportError(
        "Could not import validate_wall_thickness_revision_v4.py. "
        "Put evaluate_minT_sensitivity.py in the same folder as "
        "validate_wall_thickness_revision_v4.py."
    ) from exc


CONDITIONS = [
    ("0x",   "outputs_minT_0x"),
    ("0.5x", "outputs_minT_0_5x"),
    ("1x",   "outputs_pinn_v3"),
    ("2x",   "outputs_minT_2x"),
]

# Primary reviewer-facing metrics.
# column_in_v4, display_name, direction
PRIMARY_METRICS = [
    ("piinn_outer_assd_mm", "Outer-surface ASSD (mm)", "lower"),
    ("piinn_outer_hd95_mm", "Outer-surface HD95 (mm)", "lower"),
    ("mesh_mae_mm", "Wall-thickness MAE (mm)", "lower"),
    ("mesh_rmse_mm", "Wall-thickness RMSE (mm)", "lower"),
    ("mesh_pred_mean", "Mean predicted wall thickness (mm)", "reference"),
    ("mesh_bias_mm", "Wall-thickness bias (mm)", "zero"),
    ("voxel_enclosure_violation_rate", "Enclosure violation (%)", "lower_percent"),
    ("mesh_minT_violation_rate_predicted", "Minimum-separation violation (%)", "lower_percent"),
]

# Useful QC columns retained in the per-case output.
QC_COLUMNS = [
    "case", "condition", "status", "error",
    "piinn_inner_assd_mm", "piinn_inner_hd95_mm",
    "piinn_outer_assd_mm", "piinn_outer_hd95_mm",
    "mesh_ref_mean", "mesh_pred_mean",
    "mesh_mae_mm", "mesh_rmse_mm", "mesh_bias_mm",
    "mesh_pearson_r", "mesh_spearman_rho", "mesh_ccc",
    "voxel_enclosure_violation_rate",
    "surface_inner_outside_outer_tol_rate",
    "surface_outer_inside_lumen_tol_rate",
    "field_minT_violation_rate_reference_wall",
    "mesh_minT_violation_rate_predicted",
    "mesh_minT_violation_rate_reference",
    "mesh_minT_violation_excess_over_reference",
    "mesh_nonpositive_thickness_rate_predicted",
    "t_min_case_mm_from_qc",
    "training_time_sec_from_qc",
    "gpu_peak_mem_mb_from_qc",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate w_minT sensitivity across 0x, 0.5x, 1x, and 2x.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--processed-root",
        type=Path,
        default=Path(r"D:\Mo_PINNs\Processed"),
        help="Root containing GT case folders and all prediction roots.",
    )
    ap.add_argument(
        "--gt-mesh-root",
        type=Path,
        default=None,
        help="GT mesh root. Default: <processed-root>/gt_sdf_meshes",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Default: <processed-root>/revision_minT_sensitivity",
    )
    ap.add_argument(
        "--mesh-points",
        type=int,
        default=100000,
        help="Area-weighted surface samples per mesh, matching v4.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--correspondence-max-mm",
        type=float,
        default=1.0,
        help="Same correspondence QC threshold used by v4.",
    )
    ap.add_argument(
        "--surface-tol-mm",
        type=float,
        default=0.02,
        help="Same surface nesting tolerance used by v4.",
    )
    ap.add_argument(
        "--no-exact-point-to-mesh",
        action="store_true",
        help="Disable exact point-to-triangle distances for wall-thickness analysis. "
             "Not recommended for the final reviewer analysis.",
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional explicit case list. Default: common cases across GT and all four conditions.",
    )
    return ap.parse_args()


def common_cases(
    processed_root: Path,
    roots: Dict[str, Path],
    explicit: List[str] | None,
) -> List[str]:
    if explicit:
        cases = list(explicit)
    else:
        gt_cases = {
            p.name for p in processed_root.iterdir()
            if p.is_dir() and (p / "INNER_SDF.nrrd").exists()
        }
        pred_sets = []
        for label, root in roots.items():
            if not root.exists():
                raise FileNotFoundError(f"Prediction root for {label} does not exist: {root}")
            pred_sets.append({p.name for p in root.iterdir() if p.is_dir() and p.name != "_logs"})
        cases = sorted(set.intersection(gt_cases, *pred_sets))

    if not cases:
        raise RuntimeError("No common cases found across GT and all four conditions.")

    # Strict completeness check for the final outputs needed by v4.
    required_pred = ["pred_inner_sdf.npy", "pred_outer_sdf.npy"]
    problems = []
    for case in cases:
        if not (processed_root / case / "INNER_SDF.nrrd").exists():
            problems.append(f"{case}: missing GT INNER_SDF.nrrd")
        if not (processed_root / case / "OUTER_SDF.nrrd").exists():
            problems.append(f"{case}: missing GT OUTER_SDF.nrrd")
        for label, root in roots.items():
            cdir = root / case
            for f in required_pred:
                if not (cdir / f).exists():
                    problems.append(f"{label}/{case}: missing {f}")

    if problems:
        msg = "\n".join(problems)
        raise RuntimeError(f"Incomplete experiment outputs:\n{msg}")

    return cases


def v4_args(args: argparse.Namespace) -> SimpleNamespace:
    # evaluate_case() accesses these fields.
    return SimpleNamespace(
        mesh_points=int(args.mesh_points),
        seed=int(args.seed),
        correspondence_max_mm=float(args.correspondence_max_mm),
        surface_tol_mm=float(args.surface_tol_mm),
        save_pointwise=False,
        pointwise_csv_max_rows=0,
        exact_point_to_mesh=not bool(args.no_exact_point_to_mesh),
        skip_dilation_baseline=True,   # not needed for Comment 3 sensitivity
        dilation_end_margin_mm=2.0,
        dilation_fixed_offsets_mm=None,
    )


def evaluate_all(
    cases: List[str],
    processed_root: Path,
    roots: Dict[str, Path],
    gt_mesh_root: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    va = v4_args(args)
    total = len(cases) * len(CONDITIONS)
    counter = 0

    for label, _folder in CONDITIONS:
        pred_root = roots[label]
        for case in cases:
            counter += 1
            print(f"[{counter:02d}/{total:02d}] {label:>4s} | {case} ...", flush=True)
            row = v4.evaluate_case(
                case=case,
                gt_root=processed_root,
                pred_root=pred_root,
                gt_mesh_root=gt_mesh_root,
                out_case_dir=out_root / "per_case" / label.replace(".", "_") / case,
                args=va,
            )
            row["condition"] = label
            rows.append(row)

            if row.get("status") == "OK":
                print(
                    "    OK | "
                    f"outer ASSD={v4.safe_float(row.get('piinn_outer_assd_mm')):.4f} mm | "
                    f"thickness MAE={v4.safe_float(row.get('mesh_mae_mm')):.4f} mm | "
                    f"mean thickness={v4.safe_float(row.get('mesh_pred_mean')):.4f} mm | "
                    f"enclosure={100*v4.safe_float(row.get('voxel_enclosure_violation_rate')):.4f}% | "
                    f"minT viol={100*v4.safe_float(row.get('mesh_minT_violation_rate_predicted')):.2f}%",
                    flush=True,
                )
            else:
                print(f"    FAILED | {row.get('error')}", flush=True)

    return pd.DataFrame(rows)


def mean_sd(x: pd.Series) -> Tuple[float, float, int]:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, len(a)


def build_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "OK"].copy()

    numeric_rows = []
    formatted_rows = []

    for col, display, direction in PRIMARY_METRICS:
        nr = {"Metric": display}
        fr = {"Metric": display}
        for label, _ in CONDITIONS:
            sub = ok[ok["condition"] == label]
            mean, sd, n = mean_sd(sub[col]) if col in sub.columns else (np.nan, np.nan, 0)

            # Rates are stored as fractions in v4; manuscript table reports %.
            if direction == "lower_percent":
                mean *= 100.0
                sd *= 100.0

            nr[f"{label}_mean"] = mean
            nr[f"{label}_sd"] = sd
            nr[f"{label}_n"] = n
            fr[label] = f"{mean:.4f} ± {sd:.4f}" if np.isfinite(mean) else "NA"

        numeric_rows.append(nr)
        formatted_rows.append(fr)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)


def holm_adjust(pvalues: List[float]) -> List[float]:
    """Holm step-down adjustment; NaNs remain NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out.tolist()

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out.tolist()


def paired_tests_vs_1x(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary inferential analysis:
    paired Wilcoxon tests compare 0x, 0.5x, and 2x with the original 1x,
    using CASE as the statistical unit. Holm correction is applied across
    the three comparisons separately for each metric.
    """
    ok = df[df["status"] == "OK"].copy()
    results = []

    for col, display, direction in PRIMARY_METRICS:
        metric_rows = []
        for comp in ["0x", "0.5x", "2x"]:
            a = ok[ok["condition"] == comp][["case", col]].rename(columns={col: "comparison"})
            b = ok[ok["condition"] == "1x"][["case", col]].rename(columns={col: "original_1x"})
            m = a.merge(b, on="case", how="inner")
            x = pd.to_numeric(m["comparison"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(m["original_1x"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if direction == "lower_percent":
                x = x * 100.0
                y = y * 100.0

            if len(x) == 0:
                p = np.nan
            elif np.allclose(x, y, rtol=0, atol=1e-15):
                p = 1.0
            else:
                try:
                    p = float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    p = np.nan

            metric_rows.append({
                "Metric": display,
                "Comparison": f"{comp} vs 1x",
                "n_pairs": int(len(x)),
                "comparison_mean": float(np.mean(x)) if len(x) else np.nan,
                "original_1x_mean": float(np.mean(y)) if len(y) else np.nan,
                "mean_difference_comparison_minus_1x": float(np.mean(x - y)) if len(x) else np.nan,
                "wilcoxon_p_raw": p,
            })

        adjusted = holm_adjust([r["wilcoxon_p_raw"] for r in metric_rows])
        for r, padj in zip(metric_rows, adjusted):
            r["wilcoxon_p_holm"] = padj
            results.append(r)

    return pd.DataFrame(results)


def validate_reference_invariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    The reference geometry must be identical across the four conditions.
    This audit catches accidental use of mismatched cases/GT.
    """
    rows = []
    ok = df[df["status"] == "OK"]
    for case, g in ok.groupby("case"):
        vals = pd.to_numeric(g["mesh_ref_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        spread = float(np.max(vals) - np.min(vals)) if len(vals) else np.nan
        rows.append({
            "case": case,
            "n_conditions": int(g["condition"].nunique()),
            "reference_mean_thickness_range_mm": spread,
            "reference_consistent": bool(np.isfinite(spread) and spread < 1e-10),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    gt_mesh_root = (args.gt_mesh_root or (processed_root / "gt_sdf_meshes")).resolve()
    out_root = (args.out_root or (processed_root / "revision_minT_sensitivity")).resolve()

    roots = {
        label: (processed_root / folder).resolve()
        for label, folder in CONDITIONS
    }

    if not processed_root.exists():
        raise FileNotFoundError(processed_root)
    if not gt_mesh_root.exists():
        raise FileNotFoundError(gt_mesh_root)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_case").mkdir(exist_ok=True)

    cases = common_cases(processed_root, roots, args.cases)
    print(f"\nCases found: {len(cases)}")
    print(", ".join(cases))
    print("\nPrediction roots:")
    for label, root in roots.items():
        print(f"  {label:>4s}: {root}")
    print(f"\nExact point-to-mesh thickness distances: {not args.no_exact_point_to_mesh}")
    print(f"Surface samples per mesh: {args.mesh_points:,}\n")

    df = evaluate_all(
        cases=cases,
        processed_root=processed_root,
        roots=roots,
        gt_mesh_root=gt_mesh_root,
        out_root=out_root,
        args=args,
    )

    # Strict final completeness.
    failed = df[df["status"] != "OK"]
    if len(failed):
        df.to_csv(out_root / "per_case_all_conditions.csv", index=False)
        print("\nERROR: one or more evaluations failed:")
        print(failed[["condition", "case", "error"]].to_string(index=False))
        print(f"\nPartial results saved to: {out_root}")
        return 2

    expected = len(cases) * 4
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} successful rows, obtained {len(df)}.")

    # Keep full output and a compact QC output.
    df.to_csv(out_root / "per_case_all_conditions.csv", index=False)

    qc_cols = [c for c in QC_COLUMNS if c in df.columns]
    df[qc_cols].to_csv(out_root / "per_case_primary_and_qc.csv", index=False)

    summary_numeric, summary_formatted = build_summary(df)
    tests = paired_tests_vs_1x(df)
    ref_audit = validate_reference_invariance(df)

    summary_numeric.to_csv(out_root / "minT_sensitivity_summary_numeric.csv", index=False)
    summary_formatted.to_csv(out_root / "minT_sensitivity_table_manuscript.csv", index=False)
    tests.to_csv(out_root / "paired_tests_vs_1x.csv", index=False)
    ref_audit.to_csv(out_root / "reference_consistency_audit.csv", index=False)

    # Workbook for convenient inspection.
    xlsx_path = out_root / "minT_sensitivity_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_formatted.to_excel(writer, sheet_name="Manuscript_Table", index=False)
        summary_numeric.to_excel(writer, sheet_name="Summary_Numeric", index=False)
        df[qc_cols].to_excel(writer, sheet_name="Per_Case_QC", index=False)
        tests.to_excel(writer, sheet_name="Paired_Tests_vs_1x", index=False)
        ref_audit.to_excel(writer, sheet_name="Reference_Audit", index=False)

    manifest = {
        "purpose": "Reviewer Comment 3: w_minT sensitivity",
        "conditions": {k: str(v) for k, v in roots.items()},
        "gt_root": str(processed_root),
        "gt_mesh_root": str(gt_mesh_root),
        "cases": cases,
        "n_cases": len(cases),
        "mesh_points": args.mesh_points,
        "seed": args.seed,
        "exact_point_to_mesh_thickness": not args.no_exact_point_to_mesh,
        "surface_distance_definition": (
            "ASSD/HD95 use the same area-weighted sampled-surface nearest-neighbour "
            "definition as validate_wall_thickness_revision_v4.py and the manuscript."
        ),
        "thickness_definition": (
            "Mesh wall thickness uses shortest Euclidean distance to the opposite "
            "surface; exact point-to-triangle distance is enabled by default."
        ),
        "statistical_unit": "case",
        "primary_metrics": [x[1] for x in PRIMARY_METRICS],
    }
    with open(out_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n============================================================")
    print("PRIMARY MANUSCRIPT TABLE")
    print("============================================================")
    print(summary_formatted.to_string(index=False))

    print("\nReference consistency:")
    print(ref_audit.to_string(index=False))

    print("\nSaved:")
    print(f"  {xlsx_path}")
    print(f"  {out_root / 'minT_sensitivity_table_manuscript.csv'}")
    print(f"  {out_root / 'per_case_primary_and_qc.csv'}")
    print(f"  {out_root / 'paired_tests_vs_1x.csv'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
