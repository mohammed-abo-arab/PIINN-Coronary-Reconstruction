#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List
import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_GT_ROOT = Path(r"D:\Mo_PINNs\Processed\gt_sdf_meshes")
DEFAULT_PRED_ROOT = Path(r"D:\Mo_PINNs\Processed\outputs_DeepSDF")
DEFAULT_OUT_ROOT = DEFAULT_PRED_ROOT / "evaluation_mesh"

def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load_mesh(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(m)}")
    if m.vertices is None or len(m.vertices) == 0 or m.faces is None or len(m.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return m

def sample_surface_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    faces = mesh.faces
    verts = mesh.vertices.astype(np.float64)
    tri = verts[faces]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(v0, v1), axis=1)
    area_sum = float(np.sum(areas))
    if not np.isfinite(area_sum) or area_sum <= 0:
        raise ValueError("Mesh has non-positive total surface area.")
    probs = areas / area_sum
    face_idx = rng.choice(len(faces), size=n, replace=True, p=probs)
    tri_sel = tri[face_idx]
    u = rng.random(n)
    v = rng.random(n)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    pts = (w0[:, None] * tri_sel[:, 0] + w1[:, None] * tri_sel[:, 1] + w2[:, None] * tri_sel[:, 2])
    return pts.astype(np.float64)

def nn_distances(a_pts: np.ndarray, b_pts: np.ndarray) -> np.ndarray:
    tree = cKDTree(b_pts)
    dists, _ = tree.query(a_pts, k=1, workers=-1)
    return dists.astype(np.float64)

def symmetric_surface_metrics(gt_pts: np.ndarray, pred_pts: np.ndarray) -> Dict[str, float]:
    d_gp = nn_distances(gt_pts, pred_pts)
    d_pg = nn_distances(pred_pts, gt_pts)
    assd = float(0.5 * (np.mean(d_gp) + np.mean(d_pg)))
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

def list_cases(gt_root: Path, pred_root: Path) -> List[str]:
    gt_cases = {p.name for p in gt_root.iterdir() if p.is_dir()}
    pred_cases = {p.name for p in pred_root.iterdir() if p.is_dir()}
    return sorted(list(gt_cases.intersection(pred_cases)))

def eval_one_case(case: str, gt_root: Path, pred_root: Path, n_points: int, seed: int) -> Dict[str, object]:
    gt_case_dir = gt_root / case
    pred_case_dir = pred_root / case
    gt_inner_path = gt_case_dir / "inner_gt_from_sdf.ply"
    gt_outer_path = gt_case_dir / "outer_gt_from_sdf.ply"
    pred_inner_path = pred_case_dir / "inner_mesh.ply"
    pred_outer_path = pred_case_dir / "outer_mesh.ply"
    row = {
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
        gt_inner = load_mesh(gt_inner_path)
        gt_outer = load_mesh(gt_outer_path)
        pr_inner = load_mesh(pred_inner_path)
        pr_outer = load_mesh(pred_outer_path)
        rng = np.random.default_rng(seed)
        gt_inner_pts = sample_surface_points(gt_inner, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        pr_inner_pts = sample_surface_points(pr_inner, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        gt_outer_pts = sample_surface_points(gt_outer, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        pr_outer_pts = sample_surface_points(pr_outer, n_points, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
        inner_m = symmetric_surface_metrics(gt_inner_pts, pr_inner_pts)
        outer_m = symmetric_surface_metrics(gt_outer_pts, pr_outer_pts)
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
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }
    keys_inner = ["inner_assd_mm","inner_hd95_mm","inner_gt_to_pred_mean_mm","inner_pred_to_gt_mean_mm","inner_gt_to_pred_hd95_mm","inner_pred_to_gt_hd95_mm"]
    keys_outer = ["outer_assd_mm","outer_hd95_mm","outer_gt_to_pred_mean_mm","outer_pred_to_gt_mean_mm","outer_gt_to_pred_hd95_mm","outer_pred_to_gt_hd95_mm"]
    return {
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_failed_or_missing": len(rows) - len(ok_rows),
        "inner": {k.replace("inner_", ""): stats(collect(k)) for k in keys_inner},
        "outer": {k.replace("outer_", ""): stats(collect(k)) for k in keys_outer},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=str, default=str(DEFAULT_GT_ROOT))
    ap.add_argument("--pred-root", type=str, default=str(DEFAULT_PRED_ROOT))
    ap.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--n-points", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
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
        row = eval_one_case(case, gt_root, pred_root, n_points=args.n_points, seed=args.seed)
        rows.append(row)
        print(f"[{row['status']}] {case}" + (f" | {row['error']}" if row["status"] != "OK" else ""))
    csv_path = out_root / "metrics_mesh.csv"
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_root / "per_case_mesh.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_root / "summary_mesh.json", "w", encoding="utf-8") as f:
        json.dump(summarize(rows), f, indent=2)
    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Per-case JSON: {out_root / 'per_case_mesh.json'}")
    print(f"Summary JSON: {out_root / 'summary_mesh.json'}")

if __name__ == "__main__":
    main()
