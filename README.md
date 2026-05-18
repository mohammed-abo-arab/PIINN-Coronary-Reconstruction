````markdown
# PIINN Coronary Reconstruction

Physics-informed implicit neural network pipeline for coronary artery reconstruction from inner and outer signed distance fields.

## 1. Requirements

- Windows or Linux
- Python 3.10
- Conda/Miniconda
- NVIDIA GPU with CUDA support recommended

Create environment:

```bash
conda create -n venv_pinns python=3.10
conda activate venv_pinns
pip install -r requirements.txt
````

If `requirements.txt` is not available, install the main packages:

```bash
pip install numpy scipy pandas matplotlib openpyxl torch SimpleITK trimesh pyvista vtk scikit-image
```

## 2. Data Structure

Prepare your data as follows:

```text
Processed/
│
├── codes/
│   ├── train_pinn_sdf_reconstruction_v2.py
│   ├── run_queue_2gpus.py
│   ├── generate_and_qc_sdf.py
│   ├── export_gt_mesh_from_sdf_batch.py
│   ├── evaluation_mesh.py
│   ├── evaluation_sdf.py
│   └── qc_analysis_report.py
│
├── Case_001/
│   ├── INNER_SDF.nrrd
│   └── OUTER_SDF.nrrd
│
├── Case_002/
│   ├── INNER_SDF.nrrd
│   └── OUTER_SDF.nrrd
│
└── outputs_pinn/
```

Each case folder must contain:

```text
INNER_SDF.nrrd
OUTER_SDF.nrrd
```

## 3. Run a Single Case

Replace the paths with your own paths:

```bash
python codes/train_pinn_sdf_reconstruction_v2.py ^
  --case-dir "PATH_TO_PROCESSED/Case_001" ^
  --out-root "PATH_TO_PROCESSED/outputs_pinn" ^
  --export-final ^
  --export-each-stage
```

Example:

```bash
python codes/train_pinn_sdf_reconstruction_v2.py ^
  --case-dir "D:/Mo_PINNs/Processed/1079_LAD" ^
  --out-root "D:/Mo_PINNs/Processed/outputs_pinn" ^
  --export-final ^
  --export-each-stage
```

## 4. Run All Cases

On Windows Command Prompt:

```bash
for /D %C in ("PATH_TO_PROCESSED\*") do @if /I not "%~nxC"=="codes" if /I not "%~nxC"=="outputs_pinn" python "PATH_TO_PROCESSED\codes\train_pinn_sdf_reconstruction_v2.py" --case-dir "%C" --out-root "PATH_TO_PROCESSED\outputs_pinn" --export-final --export-each-stage
```

## 5. Run Multi-GPU Queue

From the `codes` folder:

```bash
cd PATH_TO_PROCESSED/codes
conda activate venv_pinns
python run_queue_2gpus.py
```

This automatically distributes cases across two GPUs and saves logs under the output folder.

## 6. Generate and Check SDF Files

```bash
python codes/generate_and_qc_sdf.py ^
  --root "PATH_TO_PROCESSED" ^
  --enforce-containment ^
  --save-contained-outer ^
  --overwrite
```

## 7. Export Ground-Truth Meshes

```bash
python codes/export_gt_mesh_from_sdf_batch.py ^
  --root "PATH_TO_PROCESSED" ^
  --out "PATH_TO_PROCESSED/gt_sdf_meshes"
```

## 8. Mesh-Based Evaluation

```bash
python codes/evaluation_mesh.py ^
  --gt-root "PATH_TO_PROCESSED/gt_sdf_meshes" ^
  --pred-root "PATH_TO_PROCESSED/outputs_pinn" ^
  --out-root "PATH_TO_PROCESSED/evaluation"
```

## 9. SDF-Based Evaluation

```bash
python codes/evaluation_sdf.py ^
  --gt-root "PATH_TO_PROCESSED" ^
  --pred-root "PATH_TO_PROCESSED/outputs_pinn" ^
  --out-root "PATH_TO_PROCESSED/evaluation" ^
  --band-mm 2.0
```

## 10. Generate QC Report

```bash
python codes/qc_analysis_report.py
```

The report generates summary tables and figures for training time, GPU memory usage, prediction stability, and time-memory correlation.

## 11. Output Files

The pipeline generates:

```text
outputs_pinn/
├── Case_001/
│   ├── inner_mesh.ply
│   ├── outer_mesh.ply
│   ├── final_outputs/
│   └── stage_outputs/
│
├── _logs/
│   └── Case_001_gpu0.log
│
└── all_qc_metrics/
    ├── QC_Summary_Tables.xlsx
    ├── Figure_A_TrainingTime_Distribution.png
    ├── Figure_B_GPUMemory_Usage.png
    ├── Figure_C_Prediction_Stability.png
    └── Figure_D_Time_vs_Memory_Correlation.png
```

## 12. Notes

* Replace all `PATH_TO_PROCESSED` values with the location of your own dataset.
* Each case folder must contain both `INNER_SDF.nrrd` and `OUTER_SDF.nrrd`.
* GPU execution is recommended for training.
* If a Python package is missing, install it using:

```bash
pip install package_name
```

## Citation

If you use this code, please cite the associated manuscript:

```text
AboArab M. A. et al. Physics-Informed Implicit Neural Networks for 3D Coronary Artery Reconstruction.
```

