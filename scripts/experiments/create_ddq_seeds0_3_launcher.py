"""Create a B200 launcher for DDQ seed 0-3 experiments.

This script is meant to be run inside the B200 Detection-Attack repository.
All generated plans, logs, and result directories stay under the repository.
"""

from __future__ import annotations

import json
import stat
import time

from launcher_common import (
    ANN_FILE,
    IMAGE_DIR,
    PYTHON,
    REPO,
    matrix_model,
    quote,
    validate_common_paths,
)

MODEL = matrix_model("ddq_coco_amnesia", alias="ddq_detr")
CONFIG = str(MODEL["config"])
CHECKPOINT = str(MODEL["checkpoint"])


def _validate_inputs() -> None:
    validate_common_paths([MODEL])


def main() -> None:
    _validate_inputs()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "result" / "score05" / f"ddq_seeds0_3_direct_tensor_{run_id}"
    log_dir = REPO / "logs" / f"score05_ddq_seeds0_3_direct_tensor_{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in range(4):
        out_dir = root / f"seed_{seed}" / "sparse_evo"
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "gpu": seed,
                "seed": seed,
                "num_images": 500,
                "sample_seed": seed,
                "dataset_name": f"COCO_val2017_score05_ddq_seed{seed}",
                "output_dir": str(out_dir),
                "sample_manifest": str(out_dir / "sample_manifest.json"),
                "log": str(log_dir / f"ddq_seed{seed}_sparse_evo.log"),
            }
        )

    launcher = root / "launch_ddq_seeds0_3.sh"
    plan = {
        "run_id": run_id,
        "root": str(root),
        "log_dir": str(log_dir),
        "launcher": str(launcher),
        "image_dir": str(IMAGE_DIR),
        "ann_file": str(ANN_FILE),
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "mmdet_inference_mode": "direct_tensor",
        "jobs": jobs,
    }
    (root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {quote(REPO)}",
        "echo '[launcher] start' $(date)",
        f"echo '[launcher] root {quote(root)}'",
    ]
    pid_vars = []
    for job in jobs:
        pid_var = f"pid_seed{job['seed']}"
        pid_vars.append(pid_var)
        cmd = [
            f"CUDA_VISIBLE_DEVICES={job['gpu']}",
            quote(PYTHON),
            "run_attack.py",
            "--model-type mmdet",
            f"--config {quote(CONFIG)}",
            "--mmdet-inference-mode direct_tensor",
            f"--checkpoint {quote(CHECKPOINT)}",
            f"--image-dir {quote(IMAGE_DIR)}",
            f"--ann-file {quote(ANN_FILE)}",
            "--attack sparse_evo",
            f"--dataset-name {quote(job['dataset_name'])}",
            "--max-query 1000",
            "--score-thr 0.5",
            "--iou-thr 0.5",
            "--success-thr 0.7",
            f"--seed {job['seed']}",
            "--num-images 500",
            "--sample-strategy random",
            f"--sample-seed {job['sample_seed']}",
            f"--sample-manifest {quote(job['sample_manifest'])}",
            "--resume-partial",
            f"--output-dir {quote(job['output_dir'])}",
            "--no-save-snapshots",
            "--pop-size 10",
            "--cr 0.9",
            "--mu 0.01",
            "--log-interval 50",
            f"> {quote(job['log'])} 2>&1 &",
        ]
        lines.append(
            "echo '[launcher] gpu {gpu} ddq seed {seed}'".format(**job)
        )
        lines.append(" ".join(cmd))
        lines.append(f"{pid_var}=$!")

    lines.append("status=0")
    for pid_var in pid_vars:
        lines.append(f"wait ${pid_var} || status=$?")
    lines.append("echo '[launcher] done status='${status} $(date)")
    lines.append("exit ${status}")
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "root": str(root),
                "log_dir": str(log_dir),
                "launcher": str(launcher),
                "jobs": jobs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
