import os
import subprocess
import time
from pathlib import Path

PYTHON_EXE = "python"
HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE / "train_pinn_sdf_reconstruction_ablation.py")

BASE_OUT_ROOT = Path(r"D:\Mo_PINNs\Processed\outputs_pinn_ablation")

ABLATIONS = [
    "no_eikonal",
    "no_anatomical",
    "no_region_sampling",
    "no_curriculum",
]

CASES = [
    r"D:\Mo_PINNs\Processed\1079_LAD",
    r"D:\Mo_PINNs\Processed\1092_LAD",
    r"D:\Mo_PINNs\Processed\1113_LAD",
    r"D:\Mo_PINNs\Processed\1131_LAD",
    r"D:\Mo_PINNs\Processed\1186_LCX",
    r"D:\Mo_PINNs\Processed\10234_RCA",
    r"D:\Mo_PINNs\Processed\10238_RCA",
    r"D:\Mo_PINNs\Processed\10250_LAD",
    r"D:\Mo_PINNs\Processed\10732_LCX",
    r"D:\Mo_PINNs\Processed\10735_RCA",
    r"D:\Mo_PINNs\Processed\10738_RCA",
    r"D:\Mo_PINNs\Processed\11616_LCX",
    r"D:\Mo_PINNs\Processed\11621_LAD",
    r"D:\Mo_PINNs\Processed\11621_LCX",
    r"D:\Mo_PINNs\Processed\11631_LAD",
]

GPUS = [0, 1]
MIN_FREE_MIB = 35000
POLL_SECONDS = 5
GPU_WAIT_SECONDS = 10


def get_free_mib(gpu_id):
    cmd = [
        "nvidia-smi", f"--id={gpu_id}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ]
    return int(subprocess.check_output(cmd, text=True).strip().splitlines()[0])


def wait_for_gpu(gpu_id):
    while True:
        try:
            if get_free_mib(gpu_id) >= MIN_FREE_MIB:
                return
        except Exception:
            pass
        time.sleep(GPU_WAIT_SECONDS)


def completed(out_root, case_name):
    case_out = out_root / case_name
    return (
        (case_out / "qc_metrics.json").exists()
        and (case_out / "pred_inner_sdf.npy").exists()
        and (case_out / "pred_outer_sdf.npy").exists()
        and (case_out / "inner_mesh.ply").exists()
        and (case_out / "outer_mesh.ply").exists()
    )


def launch(case_dir, ablation, gpu_id):
    case_name = Path(case_dir).name
    out_root = BASE_OUT_ROOT / ablation
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_name}_gpu{gpu_id}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        PYTHON_EXE, SCRIPT,
        "--case-dir", case_dir,
        "--out-root", str(out_root),
        "--ablation", ablation,
        "--export-final",
        "--export-each-stage",
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"ABLATION: {ablation}\nCASE: {case_name}\nGPU: {gpu_id}\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    return p, log_path


def main():
    BASE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    queue = []
    for ablation in ABLATIONS:
        out_root = BASE_OUT_ROOT / ablation
        for case_dir in CASES:
            case_name = Path(case_dir).name
            if completed(out_root, case_name):
                print(f"[SKIP COMPLETE] {ablation} | {case_name}")
            else:
                queue.append((ablation, case_dir))

    running = {gpu: None for gpu in GPUS}
    failures = []

    print("=" * 72)
    print("PIINN COMPONENT-WISE ABLATION QUEUE")
    print(f"Pending jobs: {len(queue)} | GPUs: {GPUS}")
    print("=" * 72)

    while queue or any(running.values()):
        for gpu in GPUS:
            job = running[gpu]
            if job is None:
                continue
            p, ablation, case_dir, log_path = job
            ret = p.poll()
            if ret is not None:
                case_name = Path(case_dir).name
                status = "DONE" if ret == 0 else "FAILED"
                print(f"[{status}] GPU {gpu} | {ablation} | {case_name} | exit={ret}")
                if ret != 0:
                    failures.append((ablation, case_name, str(log_path), ret))
                running[gpu] = None

        for gpu in GPUS:
            if running[gpu] is None and queue:
                ablation, case_dir = queue.pop(0)
                print(f"[WAIT] GPU {gpu} | {ablation} | {Path(case_dir).name}")
                wait_for_gpu(gpu)
                p, log_path = launch(case_dir, ablation, gpu)
                running[gpu] = (p, ablation, case_dir, log_path)
                print(f"[START] GPU {gpu} | {ablation} | {Path(case_dir).name}")

        time.sleep(POLL_SECONDS)

    print("=" * 72)
    print("All pending ablation jobs finished.")
    if failures:
        print(f"FAILED JOBS: {len(failures)}")
        for ablation, case_name, log_path, ret in failures:
            print(f"  {ablation} | {case_name} | exit={ret} | {log_path}")
    else:
        print("No subprocess failures detected.")
    print("=" * 72)


if __name__ == "__main__":
    main()
