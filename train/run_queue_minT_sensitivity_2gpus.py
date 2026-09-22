import os
import subprocess
import time
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

PYTHON_EXE = "python"

# Training script that changes ONLY w_minT by the requested scale.
SCRIPT = r"train_pinn_sdf_reconstruction_v3_minT_sensitivity.py"

BASE_OUT_ROOT = Path(r"D:\Mo_PINNs\Processed")

CASES = [
   # r"D:\Mo_PINNs\Processed\1079_LAD",
   # r"D:\Mo_PINNs\Processed\1092_LAD",
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

# Do not rerun 1x here: the original PIINN outputs are already available.
# These are the three additional sensitivity settings requested for Comment 3.
EXPERIMENTS = [
    # {
    #     "label": "0x",
    #     "scale": 0.0,
    #     "out_root": BASE_OUT_ROOT / "outputs_minT_0x",
    # },
    {
        "label": "0_5x",
        "scale": 0.5,
        "out_root": BASE_OUT_ROOT / "outputs_minT_0_5x",
    },
    {
        "label": "2x",
        "scale": 2.0,
        "out_root": BASE_OUT_ROOT / "outputs_minT_2x",
    },
]

# Physical GPU IDs.
GPUS = [0, 1]

# Same free-memory criterion used in the original two-GPU queue.
# NVIDIA L40 is approximately 46 GB; this avoids launching a second job
# on a GPU that is already substantially occupied.
MIN_FREE_MIB = 35000

# Use the same seed for every case/configuration so that w_minT scale is the
# intended experimental variable. The training script itself sets determinism.
SEED = 42

# Keep these identical to the original queue/training workflow.
EXPORT_FINAL = True
EXPORT_EACH_STAGE = True

# If a case already has qc_metrics.json in a sensitivity output folder,
# treat it as completed and skip it. This makes interrupted runs resumable.
SKIP_COMPLETED = True

POLL_SECONDS = 5
GPU_WAIT_SECONDS = 10


# ============================================================
# Helpers
# ============================================================

def get_free_mib(gpu_id: int) -> int:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out.splitlines()[0])


def wait_for_gpu(gpu_id: int) -> None:
    while True:
        try:
            free_mib = get_free_mib(gpu_id)
            if free_mib >= MIN_FREE_MIB:
                return
            print(
                f"[GPU WAIT] GPU {gpu_id}: free={free_mib} MiB "
                f"(need >= {MIN_FREE_MIB} MiB)"
            )
        except Exception as exc:
            print(f"[GPU WAIT] GPU {gpu_id}: nvidia-smi check failed: {exc}")
        time.sleep(GPU_WAIT_SECONDS)


def case_is_complete(out_root: Path, case_dir: str) -> bool:
    case_name = Path(case_dir).name
    qc_path = out_root / case_name / "qc_metrics.json"
    return qc_path.exists()


def make_jobs():
    """
    Build jobs experiment-by-experiment:
      0x   -> all 15 cases
      0.5x -> all 15 cases
      2x   -> all 15 cases

    This keeps each sensitivity level grouped together and easy to audit.
    """
    jobs = []
    for exp in EXPERIMENTS:
        exp["out_root"].mkdir(parents=True, exist_ok=True)
        (exp["out_root"] / "_logs").mkdir(parents=True, exist_ok=True)

        for case_dir in CASES:
            if SKIP_COMPLETED and case_is_complete(exp["out_root"], case_dir):
                print(
                    f"[SKIP] {exp['label']} | {Path(case_dir).name} "
                    f"| existing qc_metrics.json"
                )
                continue

            jobs.append(
                {
                    "label": exp["label"],
                    "scale": exp["scale"],
                    "out_root": exp["out_root"],
                    "case_dir": case_dir,
                }
            )
    return jobs


def launch(job: dict, physical_gpu_id: int):
    case_dir = job["case_dir"]
    case_name = Path(case_dir).name
    label = job["label"]
    scale = job["scale"]
    out_root = Path(job["out_root"])

    log_dir = out_root / "_logs"
    log_path = log_dir / f"{case_name}_minT_{label}_gpu{physical_gpu_id}.log"

    # Mask the process to exactly one physical GPU.
    # Inside the child process, that GPU becomes logical cuda:0.
    env = os.environ.copy()
    # Expose exactly one physical GPU to the child process.
    # Inside that child, the visible GPU is always logical cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    cmd = [
        PYTHON_EXE,
        SCRIPT,
        "--case-dir",
        case_dir,
        "--out-root",
        str(out_root),
        "--minT-scale",
        str(scale),
        "--device",
        "cuda:0",
        "--seed",
        str(SEED),
    ]

    if EXPORT_FINAL:
        cmd.append("--export-final")

    if EXPORT_EACH_STAGE:
        cmd.append("--export-each-stage")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"EXPERIMENT_LABEL: {label}\n")
        f.write(f"MIN_T_SCALE: {scale}\n")
        f.write(f"PHYSICAL_GPU_ID: {physical_gpu_id}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
        f.write(f"SEED: {SEED}\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    return process, log_path


# ============================================================
# Main
# ============================================================

def main():
    # Basic input checks before starting expensive GPU work.
    script_path = Path(SCRIPT)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Training script not found: {script_path.resolve()}\n"
            f"Run this queue from the folder containing {SCRIPT}, "
            "or update SCRIPT with its full path."
        )

    missing_cases = [c for c in CASES if not Path(c).exists()]
    if missing_cases:
        raise FileNotFoundError(
            "The following case folders do not exist:\n  "
            + "\n  ".join(missing_cases)
        )

    jobs = make_jobs()

    print("=" * 78)
    print("PIINN w_minT sensitivity queue")
    print(f"Training script : {SCRIPT}")
    print(f"GPUs            : {GPUS}")
    print(f"Seed            : {SEED}")
    print(f"Jobs to run     : {len(jobs)}")
    print("Experiments     : 0x, 0.5x, 2x")
    print("Original 1x     : NOT rerun (use existing outputs_pinn_v3)")
    print("=" * 78)

    if not jobs:
        print("Nothing to run. All requested sensitivity jobs appear complete.")
        return

    queue = list(jobs)
    running = {gpu: None for gpu in GPUS}

    completed = 0
    failed = []

    while queue or any(running.values()):
        # Check running jobs for completion.
        for gpu in GPUS:
            job_state = running[gpu]
            if job_state is None:
                continue

            process, job, log_path = job_state
            ret = process.poll()

            if ret is not None:
                completed += 1
                case_name = Path(job["case_dir"]).name
                label = job["label"]

                if ret == 0:
                    print(
                        f"[DONE] GPU {gpu} | minT={label} | {case_name} "
                        f"| {completed}/{len(jobs)} | log={log_path}"
                    )
                else:
                    print(
                        f"[FAILED] GPU {gpu} | minT={label} | {case_name} "
                        f"| exit={ret} | log={log_path}"
                    )
                    failed.append((label, case_name, ret, str(log_path)))

                running[gpu] = None

        # Launch new jobs on free GPUs.
        for gpu in GPUS:
            if running[gpu] is not None or not queue:
                continue

            job = queue.pop(0)
            case_name = Path(job["case_dir"]).name

            print(
                f"[WAIT] GPU {gpu} availability | "
                f"minT={job['label']} | {case_name}"
            )
            wait_for_gpu(gpu)

            process, log_path = launch(job, gpu)
            running[gpu] = (process, job, log_path)

            print(
                f"[START] GPU {gpu} | minT={job['label']} | {case_name} "
                f"| remaining={len(queue)} | log={log_path}"
            )

        time.sleep(POLL_SECONDS)

    print("=" * 78)
    print("All requested sensitivity jobs finished.")
    print(f"Successful jobs: {len(jobs) - len(failed)}")
    print(f"Failed jobs    : {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for label, case_name, ret, log_path in failed:
            print(
                f"  minT={label} | {case_name} | exit={ret} | log={log_path}"
            )
        raise SystemExit(1)

    print("\nOutput folders:")
    for exp in EXPERIMENTS:
        print(f"  {exp['label']}: {exp['out_root']}")

    print("\nUse the existing original PIINN folder as the 1x reference:")
    print(r"  D:\Mo_PINNs\Processed\outputs_pinn_v3")


if __name__ == "__main__":
    main()
