import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths (as you requested)
# =========================
IN_DIR = Path(r"D:\Mo_PINNs\Processed\outputs_pinn_v6")
OUT_DIR = Path(r"D:\Mo_PINNs\Processed\outputs_pinn_v6\all_qc_metrics")

IN_XLSX = IN_DIR / "PINN_QC_AllCases.xlsx"
OUT_TABLES_XLSX = OUT_DIR / "QC_Summary_Tables.xlsx"

# Output figures
FIG_A = OUT_DIR / "Figure_A_TrainingTime_Distribution.png"
FIG_B = OUT_DIR / "Figure_B_GPUMemory_Usage.png"
FIG_C = OUT_DIR / "Figure_C_Prediction_Stability.png"
FIG_D = OUT_DIR / "Figure_D_Time_vs_Memory_Correlation.png"


# =========================
# Plot style (academic)
# =========================
def apply_academic_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
    })


# =========================
# Helpers
# =========================
def parse_case(case: str):
    """
    Split "1079_LAD" -> (1079, "LAD")
    Works for "10234_RCA" etc.
    """
    m = re.match(r"^\s*(\d+)(?:_(\w+))?\s*$", str(case))
    if not m:
        return (None, None)
    cid = int(m.group(1))
    artery = m.group(2) if m.group(2) else None
    return (cid, artery)


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in QC sheet: {missing}")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# =========================
# Load QC data
# =========================
def load_qc_dataframe(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input file not found: {xlsx_path}")

    # Your workbook has a sheet named "QC_AllCases" in the script we built earlier.
    # If the sheet name differs, we fall back to first sheet.
    xls = pd.ExcelFile(xlsx_path)
    sheet = "QC_AllCases" if "QC_AllCases" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    # Normalize case column
    if "case" not in df.columns:
        raise ValueError("Expected a column named 'case' in the QC sheet.")

    # Derive numeric case_id and artery
    case_ids = []
    arteries = []
    for c in df["case"]:
        cid, art = parse_case(c)
        case_ids.append(cid)
        arteries.append(art)

    df.insert(0, "case_id", case_ids)
    df.insert(1, "artery", arteries)

    # Natural sort by numeric id then artery then case name
    df = df.sort_values(["case_id", "artery", "case"], ascending=[True, True, True]).reset_index(drop=True)

    return df


# =========================
# Build tables
# =========================
def build_table_1_per_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1 — Per-Case Quantitative Summary
    """
    # pick columns if present; keep stable even if some are missing
    preferred = [
        "case", "artery",
        "gpu_name", "total_memory (GB)",
        "torch_cuda_version", "cudnn_version",
        "train_time (min)",
        "gpu_peak_mem (GB)", "gpu_peak_reserved (GB)",
        "pred_wall_count", "pred_t_mean", "pred_t_std",
        "pred_t_min", "pred_t_p50", "pred_t_violation_rate",
    ]
    cols = [c for c in preferred if c in df.columns]
    t1 = df[cols].copy()

    # rounding for presentation
    round_map = {
        "total_memory (GB)": 2,
        "train_time (min)": 2,
        "gpu_peak_mem (GB)": 2,
        "gpu_peak_reserved (GB)": 2,
        "pred_t_mean": 6,
        "pred_t_std": 6,
        "pred_t_min": 6,
        "pred_t_p50": 6,
        "pred_t_violation_rate": 6,
    }
    for c, nd in round_map.items():
        if c in t1.columns:
            t1[c] = safe_numeric(t1[c]).round(nd)

    return t1


def build_table_2_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2 — Global Statistics (mean/std/min/max)
    """
    metrics = [
        "train_time (min)",
        "gpu_peak_mem (GB)",
        "gpu_peak_reserved (GB)",
        "pred_t_mean",
        "pred_t_std",
        "pred_t_min",
        "pred_t_p05",
        "pred_t_p10",
        "pred_t_p50",
        "pred_t_violation_rate",
        "total_memory (GB)",
    ]
    metrics = [m for m in metrics if m in df.columns]

    # Compute stats
    rows = []
    for m in metrics:
        s = safe_numeric(df[m])
        rows.append({
            "metric": m,
            "mean": float(np.nanmean(s.values)),
            "std": float(np.nanstd(s.values, ddof=1)) if np.sum(~np.isnan(s.values)) > 1 else np.nan,
            "min": float(np.nanmin(s.values)),
            "max": float(np.nanmax(s.values)),
            "n": int(np.sum(~np.isnan(s.values))),
        })

    t2 = pd.DataFrame(rows)

    # rounding
    for col in ["mean", "std", "min", "max"]:
        t2[col] = pd.to_numeric(t2[col], errors="coerce")
        # different rounding for time/memory vs thickness
        t2.loc[t2["metric"].str.contains("time", na=False), col] = t2.loc[t2["metric"].str.contains("time", na=False), col].round(2)
        t2.loc[t2["metric"].str.contains("GB", na=False), col] = t2.loc[t2["metric"].str.contains("GB", na=False), col].round(2)
        # thickness stats often need more precision
        mask_thick = t2["metric"].str.contains("pred_t|violation", na=False)
        t2.loc[mask_thick, col] = t2.loc[mask_thick, col].round(6)

    return t2


def write_tables_to_excel(t1: pd.DataFrame, t2: pd.DataFrame, out_xlsx: Path):
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        t1.to_excel(writer, index=False, sheet_name="Table_1_PerCase_Summary")
        t2.to_excel(writer, index=False, sheet_name="Table_2_Global_Statistics")


# =========================
# Figures
# =========================
def fig_a_training_time_distribution(df: pd.DataFrame, out_path: Path):
    ensure_columns(df, ["train_time (min)", "artery"])
    data = df.copy()
    data["train_time (min)"] = safe_numeric(data["train_time (min)"])

    # Group by artery if available
    groups = []
    labels = []
    for art in ["LAD", "LCX", "RCA"]:
        if art in set(data["artery"].dropna().astype(str)):
            vals = data.loc[data["artery"] == art, "train_time (min)"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(art)

    # fallback: overall
    if not groups:
        groups = [data["train_time (min)"].dropna().values]
        labels = ["All"]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_title("Training Time Distribution Across Cases")
    ax.set_xlabel("Artery Group")
    ax.set_ylabel("Training Time (min)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_b_gpu_memory_usage(df: pd.DataFrame, out_path: Path):
    ensure_columns(df, ["gpu_peak_mem (GB)", "artery"])
    data = df.copy()
    data["gpu_peak_mem (GB)"] = safe_numeric(data["gpu_peak_mem (GB)"])

    groups = []
    labels = []
    for art in ["LAD", "LCX", "RCA"]:
        if art in set(data["artery"].dropna().astype(str)):
            vals = data.loc[data["artery"] == art, "gpu_peak_mem (GB)"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(art)

    if not groups:
        groups = [data["gpu_peak_mem (GB)"].dropna().values]
        labels = ["All"]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_title("Peak GPU Memory Usage Across Cases")
    ax.set_xlabel("Artery Group")
    ax.set_ylabel("GPU Peak Memory (GB)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_c_prediction_stability(df: pd.DataFrame, out_path: Path):
    ensure_columns(df, ["case", "pred_t_mean", "pred_t_std"])
    data = df.copy()
    data["pred_t_mean"] = safe_numeric(data["pred_t_mean"])
    data["pred_t_std"] = safe_numeric(data["pred_t_std"])

    # Keep order by case_id then artery (already sorted)
    x = np.arange(len(data))
    y = data["pred_t_mean"].values
    yerr = data["pred_t_std"].values
    labels = data["case"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
    ax.set_title("Prediction Stability Across Cases (pred_t_mean ± pred_t_std)")
    ax.set_xlabel("Case")
    ax.set_ylabel("Predicted Thickness (units as in QC output)")
    ax.grid(True)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_d_time_vs_memory_correlation(df: pd.DataFrame, out_path: Path):
    ensure_columns(df, ["train_time (min)", "gpu_peak_mem (GB)", "case"])
    data = df.copy()
    data["train_time (min)"] = safe_numeric(data["train_time (min)"])
    data["gpu_peak_mem (GB)"] = safe_numeric(data["gpu_peak_mem (GB)"])

    # Drop missing pairs
    data = data.dropna(subset=["train_time (min)", "gpu_peak_mem (GB)"]).copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(data["gpu_peak_mem (GB)"], data["train_time (min)"])
    ax.set_title("Training Time vs Peak GPU Memory")
    ax.set_xlabel("GPU Peak Memory (GB)")
    ax.set_ylabel("Training Time (min)")
    ax.grid(True)

    # Optional: annotate points by case (small font)
    for _, r in data.iterrows():
        ax.text(
            r["gpu_peak_mem (GB)"],
            r["train_time (min)"],
            str(r["case"]),
            fontsize=8,
            ha="left",
            va="bottom"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_academic_style()

    df = load_qc_dataframe(IN_XLSX)

    # Tables
    t1 = build_table_1_per_case(df)
    t2 = build_table_2_global_stats(df)
    write_tables_to_excel(t1, t2, OUT_TABLES_XLSX)

    # Figures
    fig_a_training_time_distribution(df, FIG_A)
    fig_b_gpu_memory_usage(df, FIG_B)
    fig_c_prediction_stability(df, FIG_C)
    fig_d_time_vs_memory_correlation(df, FIG_D)

    print(f"[OK] Tables saved: {OUT_TABLES_XLSX}")
    print(f"[OK] Figure A saved: {FIG_A}")
    print(f"[OK] Figure B saved: {FIG_B}")
    print(f"[OK] Figure C saved: {FIG_C}")
    print(f"[OK] Figure D saved: {FIG_D}")
    print(f"[DONE] Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()