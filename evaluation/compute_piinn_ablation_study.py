#!/usr/bin/env python3
"""
compute_piinn_ablation_study.py
================================
Run the SAME revision-grade evaluation on Full PIINN and all four one-component
ablations, verify the same 15 cases, and create publication-ready mean ± sample-SD
summaries plus paired Wilcoxon comparisons against Full PIINN.

IMPORTANT
---------
This script does NOT reuse pre-computed manuscript means. It re-evaluates Full PIINN
and every ablation with the same evaluation engine, mesh sampling count, seed, and
metric definitions. This is necessary for an apples-to-apples ablation table.

Expected ablation layout:
<ablation-root>/
    no_eikonal/<CASE>/...
    no_anatomical/<CASE>/...
    no_region_sampling/<CASE>/...
    no_curriculum/<CASE>/...

Each CASE prediction directory must contain the normal PIINN outputs used by the
validation engine, including pred_inner_sdf.npy and pred_outer_sdf.npy; inner_mesh.ply,
outer_mesh.ply and qc_metrics.json are used when available.
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CASES = [
    "1079_LAD", "1092_LAD", "1113_LAD", "1131_LAD", "1186_LCX",
    "10234_RCA", "10238_RCA", "10250_LAD", "10732_LCX", "10735_RCA",
    "10738_RCA", "11616_LCX", "11621_LAD", "11621_LCX", "11631_LAD",
]

CONFIGS = [
    ("Full PIINN", "full"),
    ("w/o Eikonal", "no_eikonal"),
    ("w/o anatomical constraints", "no_anatomical"),
    ("w/o region-based sampling", "no_region_sampling"),
    ("w/o curriculum", "no_curriculum"),
]

# Primary ablation metrics. These map directly to the manuscript's mesh evaluation.
PRIMARY_METRICS = [
    ("Inner ASSD (mm)", "piinn_inner_assd_mm", "lower"),
    ("Inner HD95 (mm)", "piinn_inner_hd95_mm", "lower"),
    ("Outer ASSD (mm)", "piinn_outer_assd_mm", "lower"),
    ("Outer HD95 (mm)", "piinn_outer_hd95_mm", "lower"),
    ("Wall-thickness MAE (mm)", "mesh_mae_mm", "lower"),
    ("Wall-thickness RMSE (mm)", "mesh_rmse_mm", "lower"),
]

# Useful secondary diagnostics retained in a separate sheet/CSV.
SECONDARY_METRICS = [
    ("Wall-thickness bias (mm)", "mesh_bias_mm"),
    ("Wall-thickness Pearson r", "mesh_pearson_r"),
    ("Wall-thickness Spearman rho", "mesh_spearman_rho"),
    ("Wall-thickness CCC", "mesh_ccc"),
    ("Predicted mean wall thickness (mm)", "mesh_pred_mean"),
    ("Reference mean wall thickness (mm)", "mesh_ref_mean"),
    ("Voxel enclosure violation rate", "voxel_enclosure_violation_rate"),
    ("Minimum-separation violation rate", "field_minT_violation_rate_reference_wall"),
    ("Training time (s)", "training_time_sec_from_qc"),
    ("GPU peak memory (MB)", "gpu_peak_mem_mb_from_qc"),
]


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt-root", type=Path, required=True,
                    help="Root containing the 15 reference case folders")
    ap.add_argument("--full-root", type=Path, required=True,
                    help="Exact prediction root that produced the Full PIINN manuscript results")
    ap.add_argument("--ablation-root", type=Path, required=True,
                    help="Root containing no_eikonal/no_anatomical/no_region_sampling/no_curriculum")
    ap.add_argument("--gt-mesh-root", type=Path, default=None,
                    help="Optional root with per-case inner_gt_from_sdf.ply and outer_gt_from_sdf.ply")
    ap.add_argument("--validator", type=Path,
                    default=Path(__file__).resolve().with_name("validate_wall_thickness_revision_v4.py"),
                    help="Shared validation engine")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output folder for all ablation-study calculations")
    ap.add_argument("--mesh-points", type=int, default=100000,
                    help="Area-weighted sampled points per surface")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exact-point-to-mesh", action="store_true",
                    help="Use exact point-to-triangle distances for thickness; slower. Use only if this matches the chosen manuscript evaluation.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute a configuration even if per_case_summary.csv already exists")
    return ap.parse_args()


def prediction_root(args, key: str) -> Path:
    return args.full_root if key == "full" else args.ablation_root / key


def verify_case_layout(root: Path, label: str):
    if not root.exists():
        raise FileNotFoundError(f"{label} prediction root does not exist: {root}")
    missing_dirs = [c for c in CASES if not (root/c).is_dir()]
    if missing_dirs:
        raise RuntimeError(f"{label}: missing case directories: {missing_dirs}")

    required = ["pred_inner_sdf.npy", "pred_outer_sdf.npy"]
    missing_files = []
    for case in CASES:
        for name in required:
            if not (root/case/name).exists():
                missing_files.append(f"{case}/{name}")
    if missing_files:
        raise RuntimeError(f"{label}: missing required prediction files: {missing_files}")


def run_validator(args, label: str, key: str) -> Path:
    pred_root = prediction_root(args, key)
    verify_case_layout(pred_root, label)
    cfg_out = args.out_root / key
    csv_path = cfg_out / "per_case_summary.csv"
    if csv_path.exists() and not args.force:
        print(f"[reuse] {label}: {csv_path}")
        return csv_path

    cmd = [
        sys.executable, str(args.validator),
        "--gt-root", str(args.gt_root),
        "--pred-root", str(pred_root),
        "--out-root", str(cfg_out),
        "--mesh-points", str(args.mesh_points),
        "--seed", str(args.seed),
        "--skip-dilation-baseline",
        "--cases", *CASES,
    ]
    if args.gt_mesh_root is not None:
        cmd += ["--gt-mesh-root", str(args.gt_mesh_root)]
    if args.exact_point_to_mesh:
        cmd += ["--exact-point-to-mesh"]

    print("\n[run]", label)
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)
    if not csv_path.exists():
        raise RuntimeError(f"Validator completed but did not create {csv_path}")
    return csv_path


def load_and_validate(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "case" not in df.columns or "status" not in df.columns:
        raise RuntimeError(f"{label}: per_case_summary.csv lacks case/status columns")

    got = df["case"].astype(str).tolist()
    duplicates = df.loc[df["case"].astype(str).duplicated(), "case"].astype(str).tolist()
    if duplicates:
        raise RuntimeError(f"{label}: duplicate cases: {duplicates}")
    missing = [c for c in CASES if c not in set(got)]
    extra = [c for c in got if c not in set(CASES)]
    if missing or extra:
        raise RuntimeError(f"{label}: case mismatch. missing={missing}, extra={extra}")

    df = df[df["case"].astype(str).isin(CASES)].copy()
    df["case"] = pd.Categorical(df["case"].astype(str), categories=CASES, ordered=True)
    df = df.sort_values("case").reset_index(drop=True)
    df["case"] = df["case"].astype(str)

    failed = df.loc[df["status"] != "OK", ["case", "status", "error"] if "error" in df.columns else ["case", "status"]]
    if not failed.empty:
        raise RuntimeError(f"{label}: failed evaluations:\n{failed.to_string(index=False)}")
    if len(df) != 15:
        raise RuntimeError(f"{label}: expected 15 evaluated cases, found {len(df)}")

    needed = [k for _, k, _ in PRIMARY_METRICS]
    absent = [k for k in needed if k not in df.columns]
    if absent:
        raise RuntimeError(f"{label}: missing primary metric columns: {absent}")
    for k in needed:
        vals = pd.to_numeric(df[k], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy()).all():
            bad_cases = df.loc[~np.isfinite(vals.to_numpy()), "case"].tolist()
            raise RuntimeError(f"{label}: non-finite values for {k}: {bad_cases}")
    return df


def mean_sd(x):
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, int(len(a))


def fmt(mean, sd, digits=3):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def safe_wilcoxon(full, other):
    x = np.asarray(full, dtype=float)
    y = np.asarray(other, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) == 0:
        return np.nan, 0
    d = y - x
    if np.allclose(d, 0):
        return 1.0, len(x)
    # Two-sided paired test: asks whether the ablated configuration differs from Full PIINN.
    return float(wilcoxon(y, x, alternative="two-sided", zero_method="wilcox").pvalue), len(x)


def holm_adjust(pvals):
    """Holm family-wise adjustment; NaNs remain NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m-rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out


def main():
    args = parse_args()
    args.gt_root = args.gt_root.resolve()
    args.full_root = args.full_root.resolve()
    args.ablation_root = args.ablation_root.resolve()
    args.validator = args.validator.resolve()
    args.out_root = args.out_root.resolve()
    if args.gt_mesh_root is not None:
        args.gt_mesh_root = args.gt_mesh_root.resolve()

    if not args.validator.exists():
        raise FileNotFoundError(f"Validator not found: {args.validator}")
    if not args.gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {args.gt_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    frames = {}
    for label, key in CONFIGS:
        csv_path = run_validator(args, label, key)
        frames[key] = load_and_validate(csv_path, label)

    # Strict paired-case identity check.
    ref_cases = frames["full"]["case"].tolist()
    for label, key in CONFIGS[1:]:
        if frames[key]["case"].tolist() != ref_cases:
            raise RuntimeError(f"{label}: case order/identity differs from Full PIINN")

    # Long per-case primary table.
    long_rows = []
    for label, key in CONFIGS:
        df = frames[key]
        for _, row in df.iterrows():
            rec = {"configuration": label, "key": key, "case": row["case"]}
            for pretty, col, _ in PRIMARY_METRICS:
                rec[pretty] = float(row[col])
            long_rows.append(rec)
    per_case = pd.DataFrame(long_rows)
    per_case.to_csv(args.out_root/"ablation_per_case_primary_metrics.csv", index=False)

    # Publication table: mean ± sample SD over the same 15 cases.
    pub_rows = []
    numeric_rows = []
    for label, key in CONFIGS:
        df = frames[key]
        pub = {"Configuration": label}
        num = {"Configuration": label}
        for pretty, col, _ in PRIMARY_METRICS:
            mean, sd, n = mean_sd(df[col])
            pub[pretty] = fmt(mean, sd, 3)
            num[f"{pretty} mean"] = mean
            num[f"{pretty} SD"] = sd
            num[f"{pretty} n"] = n
        pub_rows.append(pub)
        numeric_rows.append(num)
    publication = pd.DataFrame(pub_rows)
    numeric = pd.DataFrame(numeric_rows)
    publication.to_csv(args.out_root/"ablation_table_mean_sd.csv", index=False)
    numeric.to_csv(args.out_root/"ablation_table_numeric.csv", index=False)

    # Paired Wilcoxon against Full PIINN for each primary metric.
    # Adjustment is applied across the four ablations separately for each metric.
    stats_rows = []
    full = frames["full"]
    for pretty, col, direction in PRIMARY_METRICS:
        block = []
        for label, key in CONFIGS[1:]:
            other = frames[key]
            p, n = safe_wilcoxon(full[col].to_numpy(float), other[col].to_numpy(float))
            delta = other[col].to_numpy(float) - full[col].to_numpy(float)
            block.append({
                "Metric": pretty,
                "Ablation": label,
                "n_pairs": n,
                "Mean paired change (ablation - full)": float(np.mean(delta)),
                "Median paired change (ablation - full)": float(np.median(delta)),
                "Wilcoxon p (two-sided)": p,
                "Better direction": direction,
            })
        adj = holm_adjust([r["Wilcoxon p (two-sided)"] for r in block])
        for r, q in zip(block, adj):
            r["Holm-adjusted p"] = float(q) if np.isfinite(q) else np.nan
            stats_rows.append(r)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(args.out_root/"ablation_paired_statistics.csv", index=False)

    # Secondary diagnostics: preserve them, but do not mix them automatically into the primary manuscript table.
    sec_rows = []
    for label, key in CONFIGS:
        df = frames[key]
        rec = {"Configuration": label}
        for pretty, col in SECONDARY_METRICS:
            if col in df.columns:
                mean, sd, n = mean_sd(df[col])
                rec[pretty] = fmt(mean, sd, 4)
                rec[f"{pretty} [n]"] = n
        sec_rows.append(rec)
    secondary = pd.DataFrame(sec_rows)
    secondary.to_csv(args.out_root/"ablation_secondary_diagnostics.csv", index=False)

    # QC/audit table records sources and any available resource-use metadata per case/configuration.
    audit_cols = [
        "case", "status", "error", "pred_inner_mesh_source", "pred_outer_mesh_source",
        "gt_inner_mesh_source", "gt_outer_mesh_source", "training_time_sec_from_qc",
        "gpu_peak_mem_mb_from_qc", "mesh_distance_backend_exact",
    ]
    audit_parts = []
    for label, key in CONFIGS:
        df = frames[key].copy()
        cols = [x for x in audit_cols if x in df.columns]
        a = df[cols].copy()
        a.insert(0, "Configuration", label)
        audit_parts.append(a)
    audit = pd.concat(audit_parts, ignore_index=True)
    audit.to_csv(args.out_root/"ablation_evaluation_audit.csv", index=False)

    # Excel workbook for inspection/manuscript preparation.
    xlsx = args.out_root/"PIINN_ablation_study_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        publication.to_excel(writer, sheet_name="Publication_Mean_SD", index=False)
        numeric.to_excel(writer, sheet_name="Numeric_Mean_SD", index=False)
        per_case.to_excel(writer, sheet_name="Per_Case_Primary", index=False)
        stats.to_excel(writer, sheet_name="Paired_Statistics", index=False)
        secondary.to_excel(writer, sheet_name="Secondary_Diagnostics", index=False)
        audit.to_excel(writer, sheet_name="Evaluation_Audit", index=False)
        for label, key in CONFIGS:
            frames[key].to_excel(writer, sheet_name=(key[:31]), index=False)

    manifest = {
        "cases": CASES,
        "n_cases": len(CASES),
        "configurations": {label: str(prediction_root(args, key)) for label, key in CONFIGS},
        "validator": str(args.validator),
        "mesh_points_per_surface": args.mesh_points,
        "seed": args.seed,
        "exact_point_to_mesh": bool(args.exact_point_to_mesh),
        "primary_metrics": [x[0] for x in PRIMARY_METRICS],
        "sd_definition": "sample standard deviation across cases (ddof=1)",
        "statistics": "paired two-sided Wilcoxon signed-rank vs Full PIINN; Holm correction across four ablations separately for each metric",
    }
    (args.out_root/"ablation_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== PUBLICATION TABLE (mean ± sample SD; n=15) ===")
    print(publication.to_string(index=False))
    print(f"\nSaved workbook: {xlsx}")
    print("All five configurations were evaluated with the same validator and paired case set.")

if __name__ == "__main__":
    main()
