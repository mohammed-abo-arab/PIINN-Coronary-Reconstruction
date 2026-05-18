# export_gt_mesh_from_sdf_batch.py
# Batch-export GT meshes from INNER_SDF.nrrd / OUTER_SDF.nrrd in each case folder.
#
# Inputs (per case):
#   D:\Mo_PINNs\Processed\<CASE>\INNER_SDF.nrrd
#   D:\Mo_PINNs\Processed\<CASE>\OUTER_SDF.nrrd
#
# Outputs:
#   D:\Mo_PINNs\Processed\gt_sdf_meshes\<CASE>\inner_gt_from_sdf.ply
#   D:\Mo_PINNs\Processed\gt_sdf_meshes\<CASE>\outer_gt_from_sdf.ply
#   D:\Mo_PINNs\Processed\gt_sdf_meshes\gt_mesh_report.csv
#
# Usage (CMD):
#   cd D:\Mo_PINNs\Processed\codes
#   python export_gt_mesh_from_sdf_batch.py --root "D:\Mo_PINNs\Processed" --out "D:\Mo_PINNs\Processed\gt_sdf_meshes"

import csv
import argparse
from pathlib import Path

import numpy as np

# Dependencies:
#   pip install SimpleITK scikit-image trimesh
import SimpleITK as sitk
from skimage.measure import marching_cubes
import trimesh


# -----------------------------
# Helpers
# -----------------------------
def read_nrrd_sitk(nrrd_path: Path):
    """Read NRRD via SimpleITK and return (vol_zyx, origin_xyz, spacing_xyz, direction_3x3)."""
    img = sitk.ReadImage(str(nrrd_path))
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)
    origin_xyz = np.array(img.GetOrigin(), dtype=np.float64)
    spacing_xyz = np.array(img.GetSpacing(), dtype=np.float64)
    direction_3x3 = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    return vol_zyx, origin_xyz, spacing_xyz, direction_3x3


def voxel_vertices_to_world_mm(
    verts_zyx: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    direction_3x3: np.ndarray,
) -> np.ndarray:
    """
    marching_cubes returns verts as (z,y,x) indices.
    Convert to world (x,y,z) in mm:
      world = origin + direction @ (index_xyz * spacing)
    """
    idx_xyz = np.stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]], axis=1).astype(np.float64)
    phys_xyz = idx_xyz * spacing_xyz[None, :]
    world_xyz = (direction_3x3 @ phys_xyz.T).T + origin_xyz[None, :]
    return world_xyz


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    return max(comps, key=lambda m: m.vertices.shape[0])


def export_mesh_from_sdf(
    sdf_path: Path,
    out_ply_path: Path,
    level: float = 0.0,
    step_size: int = 1,
    largest_only: bool = True,
) -> dict:
    """
    Extract isosurface mesh from SDF at 'level' (default 0) and export in world coords (mm).
    Returns mesh stats dict.
    """
    vol_zyx, origin_xyz, spacing_xyz, direction = read_nrrd_sitk(sdf_path)

    if vol_zyx.size == 0 or np.all(~np.isfinite(vol_zyx)):
        raise ValueError(f"Invalid/empty SDF volume: {sdf_path}")

    # marching cubes in index space; we convert vertices using SITK metadata (handles direction correctly)
    verts_zyx, faces, _, _ = marching_cubes(
        vol_zyx,
        level=level,
        spacing=(1.0, 1.0, 1.0),
        step_size=step_size,
        allow_degenerate=False,
    )

    verts_world = voxel_vertices_to_world_mm(verts_zyx, origin_xyz, spacing_xyz, direction)
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)

    if largest_only:
        mesh = keep_largest_component(mesh)

    out_ply_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_ply_path))

    return {
        "verts": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "bounds_min_x": float(mesh.bounds[0, 0]),
        "bounds_min_y": float(mesh.bounds[0, 1]),
        "bounds_min_z": float(mesh.bounds[0, 2]),
        "bounds_max_x": float(mesh.bounds[1, 0]),
        "bounds_max_y": float(mesh.bounds[1, 1]),
        "bounds_max_z": float(mesh.bounds[1, 2]),
    }


def list_case_dirs(root: Path) -> list[Path]:
    """
    Case dirs are direct subfolders of root excluding known non-case folders.
    Adjust exclude list if you add more folders under Processed.
    """
    exclude = {
        "codes",
        "gt_sdf_meshes",
        "Publication",
        "outputs",
        "outputs_pinn_v2",
        "outputs_pinn_v3",
        "_logs",
    }

    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.name in exclude:
            continue
        out.append(p)
    return sorted(out)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"D:\Mo_PINNs\Processed", help="Root folder containing case folders.")
    parser.add_argument("--out", type=str, default=r"D:\Mo_PINNs\Processed\gt_sdf_meshes", help="Output folder for exported meshes.")
    parser.add_argument("--inner-name", type=str, default="INNER_SDF.nrrd", help="Inner SDF filename in each case folder.")
    parser.add_argument("--outer-name", type=str, default="OUTER_SDF.nrrd", help="Outer SDF filename in each case folder.")
    parser.add_argument("--level", type=float, default=0.0, help="Isosurface level (0 for SDF).")
    parser.add_argument("--step-size", type=int, default=1, help="marching_cubes step_size (higher=faster, lower=more detail).")
    parser.add_argument("--keep-all-components", action="store_true", help="Do not filter to largest component.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    report_csv = out_root / "gt_mesh_report.csv"
    case_dirs = list_case_dirs(root)

    rows = []
    ok = 0
    fail = 0

    for case_dir in case_dirs:
        inner_path = case_dir / args.inner_name
        outer_path = case_dir / args.outer_name

        row = {
            "case": case_dir.name,
            "case_dir": str(case_dir),
            "inner_sdf": str(inner_path) if inner_path.exists() else "",
            "outer_sdf": str(outer_path) if outer_path.exists() else "",
            "inner_mesh": "",
            "outer_mesh": "",
            "inner_verts": "",
            "inner_faces": "",
            "outer_verts": "",
            "outer_faces": "",
            "status": "SKIPPED",
            "error": "",
        }

        if not inner_path.exists() or not outer_path.exists():
            row["status"] = "MISSING_INPUTS"
            if not inner_path.exists():
                row["error"] += f"Missing {args.inner_name}. "
            if not outer_path.exists():
                row["error"] += f"Missing {args.outer_name}. "
            rows.append(row)
            print(f"[SKIP] {case_dir.name}: {row['error'].strip()}")
            continue

        try:
            case_out_dir = out_root / case_dir.name
            inner_out = case_out_dir / "inner_gt_from_sdf.ply"
            outer_out = case_out_dir / "outer_gt_from_sdf.ply"

            largest_only = not args.keep_all_components

            inner_stats = export_mesh_from_sdf(
                inner_path, inner_out,
                level=args.level,
                step_size=args.step_size,
                largest_only=largest_only,
            )
            outer_stats = export_mesh_from_sdf(
                outer_path, outer_out,
                level=args.level,
                step_size=args.step_size,
                largest_only=largest_only,
            )

            row["inner_mesh"] = str(inner_out)
            row["outer_mesh"] = str(outer_out)
            row["inner_verts"] = inner_stats["verts"]
            row["inner_faces"] = inner_stats["faces"]
            row["outer_verts"] = outer_stats["verts"]
            row["outer_faces"] = outer_stats["faces"]
            row["status"] = "OK"

            ok += 1
            rows.append(row)
            print(f"[OK] {case_dir.name} -> {case_out_dir}")

        except Exception as e:
            fail += 1
            row["status"] = "FAILED"
            row["error"] = repr(e)
            rows.append(row)
            print(f"[FAIL] {case_dir.name}: {repr(e)}")

    # Write report
    fieldnames = [
        "case", "case_dir",
        "inner_sdf", "outer_sdf",
        "inner_mesh", "outer_mesh",
        "inner_verts", "inner_faces",
        "outer_verts", "outer_faces",
        "status", "error",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. OK={ok}, FAILED={fail}, TOTAL={ok+fail}")
    print(f"Report: {report_csv}")
    print(f"Meshes root: {out_root}")


if __name__ == "__main__":
    main()