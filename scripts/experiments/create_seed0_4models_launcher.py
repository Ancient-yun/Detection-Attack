"""Create a B200 launcher for seed 0 across four detector models.

This script is meant to be run inside the B200 Detection-Attack repository.
It writes all plans, logs, and result directories under the repository tree.
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

MODELS = [
    matrix_model("ddq_coco_amnesia", alias="ddq_detr", gpu=0),
    matrix_model("atss_coco_amnesia", alias="atss_r50", gpu=1),
    matrix_model("deformable_detr_coco_amnesia", alias="deformable_detr", gpu=2),
    matrix_model("yolov8n_coco", alias="yolov8n", gpu=3),
]


def _validate_inputs() -> None:
    validate_common_paths(MODELS)


def main() -> None:
    _validate_inputs()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "result" / "score05" / f"seed0_4models_direct_tensor_{run_id}"
    log_dir = REPO / "logs" / f"score05_seed0_4models_direct_tensor_{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for model in MODELS:
        out_dir = root / str(model["name"]) / "sparse_evo"
        out_dir.mkdir(parents=True, exist_ok=True)
        job = {
            **model,
            "seed": 0,
            "num_images": 500,
            "sample_seed": 0,
            "dataset_name": f"COCO_val2017_score05_seed0_{model['name']}",
            "output_dir": str(out_dir),
            "sample_manifest": str(out_dir / "sample_manifest.json"),
            "log": str(log_dir / f"{model['name']}_seed0_sparse_evo.log"),
        }
        jobs.append(job)

    plan = {
        "run_id": run_id,
        "root": str(root),
        "log_dir": str(log_dir),
        "launcher": str(root / "launch_seed0_4models.sh"),
        "image_dir": str(IMAGE_DIR),
        "ann_file": str(ANN_FILE),
        "jobs": jobs,
    }
    (root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    launcher = root / "launch_seed0_4models.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {quote(REPO)}",
        "echo '[launcher] start' $(date)",
        f"echo '[launcher] root {quote(root)}'",
    ]
    pid_vars = []
    for job in jobs:
        pid_var = f"pid_{job['name']}"
        pid_vars.append(pid_var)
        cmd = [
            f"CUDA_VISIBLE_DEVICES={job['gpu']}",
            quote(PYTHON),
            "run_attack.py",
            f"--model-type {quote(job['model_type'])}",
        ]
        if job["model_type"] == "mmdet":
            cmd.extend(
                [
                    f"--config {quote(job['config'])}",
                    f"--mmdet-inference-mode {quote(job['mmdet_inference_mode'])}",
                ]
            )
        cmd.extend(
            [
                f"--checkpoint {quote(job['checkpoint'])}",
                f"--image-dir {quote(IMAGE_DIR)}",
                f"--ann-file {quote(ANN_FILE)}",
                "--attack sparse_evo",
                f"--dataset-name {quote(job['dataset_name'])}",
                "--max-query 1000",
                "--score-thr 0.5",
                "--iou-thr 0.5",
                "--success-thr 0.7",
                "--seed 0",
                "--num-images 500",
                "--sample-strategy random",
                "--sample-seed 0",
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
        )
        lines.append(
            "echo '[launcher] gpu {gpu} model {name} seed 0'".format(**job)
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
                "jobs": [
                    {
                        "gpu": job["gpu"],
                        "name": job["name"],
                        "model_type": job["model_type"],
                        "output_dir": job["output_dir"],
                        "log": job["log"],
                    }
                    for job in jobs
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
