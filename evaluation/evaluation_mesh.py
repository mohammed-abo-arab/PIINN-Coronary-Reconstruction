#!/usr/bin/env python3
"""
evaluation_mesh.py (UPDATED)

Mesh-to-mesh evaluation for PINN SDF reconstruction.

GT meshes:
  <gt_root>/<CASE>/inner_gt_from_sdf.ply
  <gt_root>/<CASE>/outer_gt_from_sdf.ply

Pred meshes:
  <pred_root>/<CASE>/inner_mesh.ply
  <pred_root>/<CASE>/outer_mesh.ply

Outputs (in <out_root>):
  metrics_mesh.csv
  per_case_mesh.json
  summary_mesh.json

Metrics (mm):
  - assd_mm: Average Symmetric Surface Distance (mean NN distance both directions)
  - hd95_mm: Robust Hausdorff 95% (max of 95th percentile both directions)
  - gt_to_pred_mean_mm / pred_to_gt_mean_mm
  - gt_to_pred_hd95_mm / pred_to_gt_hd95_mm

Notes:
  - Distances are computed by sampling points on each mesh surface (area-weighted),
    then using nearest-neighbor distances (KD-tree).
  - Sampling is deterministic given --seed.


Paths:
  D:\Mo_PINNs\Processed\gt_sdf_meshes
  D:\Mo_PINNs\Processed\outputs_Vanilla_INR
  D:\Mo_PINNs\Processed\outputs_Vanilla_INR\evaluation

  run command:
  python evaluation_mesh.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# deps:
#   pip install trimesh scipy
import trimesh
from scipy.spatial import cKDTree


# -----------------------------
# Core geometry utilities
# -----------------------------
def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh; if a scene is provided, merge geometry. Reject empty meshes."""
    m = trimesh.load_mesh(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(m)}")
    if m.vertices is None or len(m.vertices) == 0 or m.faces is None or len(m.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return m


def sample_surface_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Deterministic area-weighted surface sampling.

    We avoid trimesh.sample.sample_surface() directly because it uses numpy's global RNG.
    This implementation samples faces proportional to area and then samples barycentric
    coordinates within each selected triangle.
    """
    # Ensure float64 for stability
    faces = mesh.faces
    verts = mesh.vertices.astype(np.float64)

    # Triangle vertices: (F, 3, 3)
    tri = verts[faces]

    # Triangle areas
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(v0, v1), axis=1)

    area_sum = float(np.sum(areas))
    if not np.isfinite(area_sum) or area_sum <= 0:
        raise ValueError("Mesh has non-positive total surface area.")

    probs = areas / area_sum
    face_idx = rng.choice(len(faces), size=n, replace=True, p=probs)

    tri_sel = tri[face_idx]  # (n, 3, 3)

    # Uniform sampling in triangle using barycentric coords
    u = rng.random(n)
    v = rng.random(n)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v

    pts = (w0[:, None] * tri_sel[:, 0] +
           w1[:, None] * tri_sel[:, 1] +
           w2[:, None] * tri_sel[:, 2])

    return pts.astype(np.float64)


def nn_distances(a_pts: np.ndarray, b_pts: np.ndarray) -> np.ndarray:
    """For each point in a_pts, compute distance to nearest neighbor in b_pts."""
    tree = cKDTree(b_pts)
    dists, _ = tree.query(a_pts, k=1, workers=-1)
    return dists.astype(np.float64)


def symmetric_surface_metrics(gt_pts: np.ndarray, pred_pts: np.ndarray) -> Dict[str, float]:
    """
    Compute symmetric surface metrics using nearest-neighbor distances:
      d_gp: GT -> Pred distances
      d_pg: Pred -> GT distances
    """
    d_gp = nn_distances(gt_pts, pred_pts)
    d_pg = nn_distances(pred_pts, gt_pts)

    # ASSD: average symmetric surface distance
    assd = float(0.5 * (np.mean(d_gp) + np.mean(d_pg)))

    # HD95: robust Hausdorff (95th percentile) symmetrized
    gt_to_pred_hd95 = float(np.percentile(d_gp, 95))
    pred_to_gt_hd95 = float(np.percentile(d_pg, 95))
    hd95 = float(max(gt_to_pred_hd95, pred_to_gt_hd95))

    return {
        "assd_mm": assd,
        "hd95_mm": hd95,
        "gt_to_pred_mean_mm": float(np.mean(d_gp)),
        "pred_to_gt_mean_mm": float(np.mean(d_pg)),
        "gt_to_pred_hd95_mm": gt_to_pred_hd95,
        "pred_to_gt_hd95_mm": pred_to_gt_hd95,
    }


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
    n_points: int,
    seed: int,
) -> Dict[str, object]:
    """
    Evaluate INNER and OUTER surfaces for one case.
    Returns a row-like dict.
    """
    gt_case_dir = gt_root / case
    pred_case_dir = pred_root / case

    gt_inner_path = gt_case_dir / "inner_gt_from_sdf.ply"
    gt_outer_path = gt_case_dir / "outer_gt_from_sdf.ply"

    pred_inner_path = pred_case_dir / "inner_mesh.ply"
    pred_outer_path = pred_case_dir / "outer_mesh.ply"

    row: Dict[str, object] = {
        "case": case,
        "status": "OK",
        "error": "",
        "gt_inner": str(gt_inner_path),
        "gt_outer": str(gt_outer_path),
        "pred_inner": str(pred_inner_path),
        "pred_outer": str(pred_outer_path),
        "n_points": int(n_points),
        "seed": int(seed),
    }

    missing = [p.name for p in [gt_inner_path, gt_outer_path, pred_inner_path, pred_outer_path] if not p.exists()]
    if missing:
        row["status"] = "MISSING"
        row["error"] = "Missing: " + ", ".join(missing)
        return row

    try:
        # Load meshes
        gt_inner = load_mesh(gt_inner_path)
        gt_outer = load_mesh(gt_outer_path)
        pr_inner = load_mesh(pred_inner_path)
        pr_outer = load_mesh(pred_outer_path)

        # Deterministic RNG per surface + per mesh
        rng = np.random.default_rng(seed)

        # Sample points (deterministic but different streams)
        gt_inner_pts = sample_surface_points(gt_inner, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        pr_inner_pts = sample_surface_points(pr_inner, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))

        gt_outer_pts = sample_surface_points(gt_outer, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        pr_outer_pts = sample_surface_points(pr_outer, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))

        # Metrics
        inner_m = symmetric_surface_metrics(gt_inner_pts, pr_inner_pts)
        outer_m = symmetric_surface_metrics(gt_outer_pts, pr_outer_pts)

        # Flatten
        for k, v in inner_m.items():
            row[f"inner_{k}"] = v
        for k, v in outer_m.items():
            row[f"outer_{k}"] = v

        return row

    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = repr(e)
        return row


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok_rows = [r for r in rows if r.get("status") == "OK"]

    def collect(key: str) -> np.ndarray:
        vals = []
        for r in ok_rows:
            v = r.get(key, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    def stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n": 0,
            }
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    # Metrics to summarize (keep consistent structure)
    keys_inner = [
        "inner_assd_mm",
        "inner_hd95_mm",
        "inner_gt_to_pred_mean_mm",
        "inner_pred_to_gt_mean_mm",
        "inner_gt_to_pred_hd95_mm",
        "inner_pred_to_gt_hd95_mm",
    ]
    keys_outer = [
        "outer_assd_mm",
        "outer_hd95_mm",
        "outer_gt_to_pred_mean_mm",
        "outer_pred_to_gt_mean_mm",
        "outer_gt_to_pred_hd95_mm",
        "outer_pred_to_gt_hd95_mm",
    ]

    summary = {
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_failed_or_missing": len(rows) - len(ok_rows),
        "inner": {k.replace("inner_", ""): stats(collect(k)) for k in keys_inner},
        "outer": {k.replace("outer_", ""): stats(collect(k)) for k in keys_outer},
    }
    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=str, default=r"D:\Mo_PINNs\Processed\gt_sdf_meshes",
                    help="Root folder for GT meshes.")
    ap.add_argument("--pred-root", type=str, default=r"D:\Mo_PINNs\Processed\outputs_DeepSDF",
                    help="Root folder for predicted meshes.")
    ap.add_argument("--out-root", type=str, default=r"D:\Mo_PINNs\Processed\outputs_DeepSDF\evaluation",
                    help="Output folder for evaluation results.")
    ap.add_argument("--n-points", type=int, default=100000,
                    help="Surface points to sample per mesh (higher = more stable metrics).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling reproducibility.")
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
        row = eval_one_case(case, gt_root, pred_root, n_points=args.n_points, seed=args.seed)
        rows.append(row)
        print(f"[{row['status']}] {case}" + (f" | {row['error']}" if row["status"] != "OK" else ""))

    # Write CSV (must include _mesh in name)
    csv_path = out_root / "metrics_mesh.csv"
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write per-case JSON (with _mesh)
    per_case_path = out_root / "per_case_mesh.json"
    with open(per_case_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Summary JSON (with _mesh)
    summary_obj = summarize(rows)
    summary_path = out_root / "summary_mesh.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Per-case JSON: {per_case_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()