from pathlib import Path
from typing import List
import numpy as np
import vtk


# ============================================================
# PATHS (KEEP INPUT PATHS; OUTPUT INTO EVALUATION)
# ============================================================
PRED_ROOT = Path(r"D:\Mo_PINNs\Processed\outputs_pinn_v5")          # per-case folders (INPUT)
OUT_ROOT  = Path(r"D:\Mo_PINNs\Processed\outputs_pinn_v5\reconstruction_3D")  # (OUTPUT) under evaluation
SKIP_DIRS = {"_logs", "3d_reconstruction", "reconstruction_3D"}


# ============================================================
# IO
# ============================================================
def read_ply(path: Path) -> vtk.vtkPolyData:
    r = vtk.vtkPLYReader()
    r.SetFileName(str(path))
    r.Update()
    out = r.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read {path}")
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(out)
    tri.Update()
    return tri.GetOutput()


def write_vtp(path: Path, poly: vtk.vtkPolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(poly)
    if w.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


# ============================================================
# Mesh processing (minimal but robust)
# ============================================================
def clean_poly(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    c = vtk.vtkCleanPolyData()
    c.SetInputData(poly)
    c.Update()
    return c.GetOutput()


def keep_largest(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(conn.GetOutput())
    gf.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(gf.GetOutput())
    tri.Update()
    return tri.GetOutput()


def fill_holes(poly: vtk.vtkPolyData, hole_mm: float) -> vtk.vtkPolyData:
    # Small hole fill is recommended for stable distance computations.
    fh = vtk.vtkFillHolesFilter()
    fh.SetInputData(poly)
    fh.SetHoleSize(float(hole_mm))
    fh.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(fh.GetOutput())
    tri.Update()
    return tri.GetOutput()


def smooth_sinc(poly: vtk.vtkPolyData, iters: int, pass_band: float) -> vtk.vtkPolyData:
    if iters <= 0:
        return poly
    s = vtk.vtkWindowedSincPolyDataFilter()
    s.SetInputData(poly)
    s.SetNumberOfIterations(int(iters))
    s.SetPassBand(float(pass_band))
    s.BoundarySmoothingOff()
    s.FeatureEdgeSmoothingOff()
    s.NonManifoldSmoothingOn()
    s.NormalizeCoordinatesOn()
    s.Update()
    return s.GetOutput()


def decimate(poly: vtk.vtkPolyData, reduction: float) -> vtk.vtkPolyData:
    if reduction <= 0.0:
        return poly
    d = vtk.vtkDecimatePro()
    d.SetInputData(poly)
    d.SetTargetReduction(float(reduction))
    d.PreserveTopologyOn()
    d.BoundaryVertexDeletionOff()
    d.Update()
    return d.GetOutput()


def orient_normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(poly)
    n.AutoOrientNormalsOn()
    n.ConsistencyOn()
    n.SplittingOff()
    n.Update()
    return n.GetOutput()


def append_polys(polys: List[vtk.vtkPolyData]) -> vtk.vtkPolyData:
    app = vtk.vtkAppendPolyData()
    for p in polys:
        app.AddInputData(p)
    app.Update()

    c = vtk.vtkCleanPolyData()
    c.SetInputData(app.GetOutput())
    c.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(c.GetOutput())
    tri.Update()
    return tri.GetOutput()


def estimate_scale_mm(poly: vtk.vtkPolyData) -> float:
    b = poly.GetBounds()
    dx = b[1] - b[0]
    dy = b[3] - b[2]
    dz = b[5] - b[4]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def auto_params_from_scale(scale_mm: float):
    # Same safe defaults as your previous scripts
    inner_hole = max(0.4, 0.008 * scale_mm)
    outer_hole = max(1.0, 0.015 * scale_mm)
    inner_iters = int(np.clip(0.12 * scale_mm, 6, 14))
    outer_iters = int(np.clip(0.18 * scale_mm, 10, 22))
    inner_pb = 0.16
    outer_pb = 0.12
    return inner_hole, outer_hole, inner_iters, outer_iters, inner_pb, outer_pb


# ============================================================
# Thickness utilities
# ============================================================
def to_numpy_points(poly: vtk.vtkPolyData) -> np.ndarray:
    n = poly.GetNumberOfPoints()
    pts = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        pts[i] = poly.GetPoint(i)
    return pts


def attach_scalar(poly: vtk.vtkPolyData, name: str, values: np.ndarray) -> vtk.vtkPolyData:
    if values.shape[0] != poly.GetNumberOfPoints():
        raise ValueError(f"Scalar length mismatch for {name}: {values.shape[0]} vs {poly.GetNumberOfPoints()}")
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(poly.GetNumberOfPoints())
    for i, v in enumerate(values):
        arr.SetValue(i, float(v))
    poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars(name)
    return poly


def add_surface_id(poly: vtk.vtkPolyData, sid: int) -> vtk.vtkPolyData:
    arr = vtk.vtkIntArray()
    arr.SetName("SurfaceId")  # 0=inner, 1=outer
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(poly.GetNumberOfPoints())
    for i in range(poly.GetNumberOfPoints()):
        arr.SetValue(i, int(sid))
    poly.GetPointData().AddArray(arr)
    return poly


def implicit_distance_to_surface(poly: vtk.vtkPolyData) -> vtk.vtkImplicitPolyDataDistance:
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(poly)
    return imp


def point_to_surface_distance_array(src_pts: np.ndarray, target_poly: vtk.vtkPolyData) -> np.ndarray:
    imp = implicit_distance_to_surface(target_poly)
    d = np.empty((src_pts.shape[0],), dtype=np.float32)
    for i, p in enumerate(src_pts):
        d[i] = float(abs(imp.EvaluateFunction((float(p[0]), float(p[1]), float(p[2])))))
    return d


# ============================================================
# Per-case
# ============================================================
def process_case(case_dir: Path) -> None:
    inner_raw = case_dir / "inner_mesh.ply"
    outer_raw = case_dir / "outer_mesh.ply"

    if not inner_raw.exists() or not outer_raw.exists():
        print(f"[SKIP] {case_dir.name} (missing inner_mesh.ply/outer_mesh.ply)")
        return

    out_case = OUT_ROOT / case_dir.name
    out_case.mkdir(parents=True, exist_ok=True)

    print(f"[PROCESS] {case_dir.name}")

    # Load meshes
    inner = read_ply(inner_raw)
    outer = read_ply(outer_raw)

    # Auto params based on outer scale
    scale = estimate_scale_mm(outer)
    inner_hole_mm, outer_hole_mm, inner_iters, outer_iters, inner_pb, outer_pb = auto_params_from_scale(scale)

    # Minimal cleanup
    inner = clean_poly(inner)
    outer = clean_poly(outer)

    # Keep largest outer only (recommended to remove disconnected islands)
    outer = keep_largest(outer)

    # Small hole filling (recommended for stable distance queries / nicer surfaces)
    inner = fill_holes(inner, inner_hole_mm)
    outer = fill_holes(outer, outer_hole_mm)

    # Smoothing (controls realism vs noise)
    inner = smooth_sinc(inner, inner_iters, inner_pb)
    outer = smooth_sinc(outer, outer_iters, outer_pb)

    # Keep full resolution unless you explicitly need lower triangle count
    inner = decimate(inner, 0.0)
    outer = decimate(outer, 0.0)

    inner = orient_normals(inner)
    outer = orient_normals(outer)

    # Thickness
    inner_pts = to_numpy_points(inner)
    outer_pts = to_numpy_points(outer)

    d_in_to_out = point_to_surface_distance_array(inner_pts, outer)  # inner->outer
    d_out_to_in = point_to_surface_distance_array(outer_pts, inner)  # outer->inner (wall thickness visualization)

    inner_col = attach_scalar(inner, "Thickness_mm", d_in_to_out)
    outer_col = attach_scalar(outer, "Thickness_mm", d_out_to_in)

    inner_col = add_surface_id(inner_col, 0)
    outer_col = add_surface_id(outer_col, 1)

    # Save ONLY the requested files
    write_vtp(out_case / "inner_colored_thickness.vtp", inner_col)
    write_vtp(out_case / "outer_colored_thickness.vtp", outer_col)

    artery = append_polys([outer_col, inner_col])
    write_vtp(out_case / "artery_thickness.vtp", artery)

    print(f"[DONE] {case_dir.name} -> {out_case}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for case_dir in PRED_ROOT.iterdir():
        if not case_dir.is_dir():
            continue
        if case_dir.name in SKIP_DIRS:
            continue
        process_case(case_dir)

    print("\nFinished.")
    print(f"Outputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()