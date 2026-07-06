"""Schedule remaining score05 detector experiments on B200.

GPU policy:
- GPU 5 starts with the long model queue first: deformable_detr, atss_r50,
  then yolov8n.
- GPUs 0-3 are currently expected to be occupied by DDQ seed 0-3. When one of
  them becomes free, it takes the shortest remaining job first: yolov8n,
  atss_r50, then deformable_detr.

The scheduler writes all outputs and logs under the repository tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from launcher_common import (
    ANN_FILE,
    IMAGE_DIR,
    PYTHON,
    REPO,
    matrix_model,
    validate_common_paths,
)

DEFAULT_MANAGED_GPUS = [0, 1, 2, 3, 5]
DEFAULT_LONG_GPU = 5
POLL_SECONDS = 60
FREE_MEMORY_MIB = 500

MODELS = {
    "deformable_detr": matrix_model(
        "deformable_detr_coco_amnesia",
        alias="deformable_detr",
        duration_rank=3,
    ),
    "atss_r50": matrix_model(
        "atss_coco_amnesia",
        alias="atss_r50",
        duration_rank=2,
    ),
    "yolov8n": matrix_model(
        "yolov8n_coco",
        alias="yolov8n",
        duration_rank=1,
    ),
}


@dataclass
class ActiveJob:
    job: dict[str, object]
    process: subprocess.Popen[bytes]
    log_handle: TextIO
    started_at: float


def _parse_gpu_list(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_MANAGED_GPUS
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("SCHEDULER_GPUS must contain at least one GPU index")
    return gpus


def _parse_long_gpu(value: str | None) -> int | None:
    if value is None:
        return DEFAULT_LONG_GPU
    value = value.strip()
    if value == "":
        return None
    return int(value)


def _validate_inputs() -> None:
    validate_common_paths(MODELS)


def _gpu_memory_mib() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    memory = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_str, memory_str = [part.strip() for part in line.split(",", 1)]
        memory[int(index_str)] = int(memory_str)
    return memory


def _make_jobs(root: Path, log_dir: Path) -> list[dict[str, object]]:
    jobs = []
    for model_name, model in MODELS.items():
        for seed in range(4):
            out_dir = root / model_name / f"seed_{seed}" / "sparse_evo"
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append(
                {
                    **model,
                    "model": model_name,
                    "seed": seed,
                    "sample_seed": seed,
                    "num_images": 500,
                    "dataset_name": (
                        f"COCO_val2017_score05_{model_name}_seed{seed}"
                    ),
                    "output_dir": str(out_dir),
                    "sample_manifest": str(out_dir / "sample_manifest.json"),
                    "log": str(log_dir / f"{model_name}_seed{seed}.log"),
                }
            )
    return jobs


def _command_for(job: dict[str, object]) -> list[str]:
    command = [
        str(PYTHON),
        "run_attack.py",
        "--model-type",
        str(job["model_type"]),
    ]
    if job["model_type"] == "mmdet":
        command.extend(
            [
                "--config",
                str(job["config"]),
                "--mmdet-inference-mode",
                str(job["mmdet_inference_mode"]),
            ]
        )
    command.extend(
        [
            "--checkpoint",
            str(job["checkpoint"]),
            "--image-dir",
            str(IMAGE_DIR),
            "--ann-file",
            str(ANN_FILE),
            "--attack",
            "sparse_evo",
            "--dataset-name",
            str(job["dataset_name"]),
            "--max-query",
            "1000",
            "--score-thr",
            "0.5",
            "--iou-thr",
            "0.5",
            "--success-thr",
            "0.7",
            "--seed",
            str(job["seed"]),
            "--num-images",
            "500",
            "--sample-strategy",
            "random",
            "--sample-seed",
            str(job["sample_seed"]),
            "--sample-manifest",
            str(job["sample_manifest"]),
            "--resume-partial",
            "--output-dir",
            str(job["output_dir"]),
            "--no-save-snapshots",
            "--pop-size",
            "10",
            "--cr",
            "0.9",
            "--mu",
            "0.01",
            "--log-interval",
            "50",
        ]
    )
    return command


def _pick_job(
    pending: list[dict[str, object]],
    gpu: int,
    long_gpu: int | None,
) -> dict[str, object]:
    reverse = gpu == long_gpu
    ordered = sorted(
        pending,
        key=lambda job: (
            int(job["duration_rank"]),
            -int(job["seed"]) if reverse else int(job["seed"]),
        ),
        reverse=reverse,
    )
    job = ordered[0]
    pending.remove(job)
    return job


def _write_state(
    state_path: Path,
    pending: list[dict[str, object]],
    active: dict[int, ActiveJob],
    finished: list[dict[str, object]],
) -> None:
    state = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pending": [
            {"model": j["model"], "seed": j["seed"]} for j in pending
        ],
        "active": [
            {
                "gpu": gpu,
                "pid": active_job.process.pid,
                "model": active_job.job["model"],
                "seed": active_job.job["seed"],
                "started_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(active_job.started_at),
                ),
            }
            for gpu, active_job in sorted(active.items())
        ],
        "finished": finished,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _launch(job: dict[str, object], gpu: int) -> ActiveJob:
    log_path = Path(str(job["log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TMPDIR", str(REPO / ".pip_tmp"))
    process = subprocess.Popen(
        _command_for(job),
        cwd=REPO,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return ActiveJob(
        job={**job, "gpu": gpu, "pid": process.pid},
        process=process,
        log_handle=log_handle,
        started_at=time.time(),
    )


def main() -> None:
    _validate_inputs()
    managed_gpus = _parse_gpu_list(os.environ.get("SCHEDULER_GPUS"))
    long_gpu = _parse_long_gpu(os.environ.get("SCHEDULER_LONG_GPU"))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "result" / "score05" / f"remaining_auto_{run_id}"
    log_dir = REPO / "logs" / f"score05_remaining_auto_{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = _make_jobs(root, log_dir)
    active: dict[int, ActiveJob] = {}
    finished: list[dict[str, object]] = []
    state_path = root / "scheduler_state.json"
    scheduler_log = log_dir / "scheduler.log"
    plan = {
        "run_id": run_id,
        "root": str(root),
        "log_dir": str(log_dir),
        "managed_gpus": managed_gpus,
        "policy": {
            "long_gpu": long_gpu,
            "long_gpu_policy": "longest remaining first" if long_gpu is not None else None,
            "other_gpus_policy": "shortest remaining first",
        },
        "jobs": pending,
    }
    (root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    with scheduler_log.open("a", encoding="utf-8") as sched:
        def log(message: str) -> None:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            print(line, flush=True)
            sched.write(line + "\n")
            sched.flush()

        log(f"start root={root}")
        while pending or active:
            for gpu, active_job in list(active.items()):
                return_code = active_job.process.poll()
                if return_code is None:
                    continue
                active_job.log_handle.close()
                finished.append(
                    {
                        "gpu": gpu,
                        "pid": active_job.process.pid,
                        "model": active_job.job["model"],
                        "seed": active_job.job["seed"],
                        "return_code": return_code,
                        "elapsed_seconds": round(time.time() - active_job.started_at, 1),
                    }
                )
                log(
                    "finish gpu={gpu} model={model} seed={seed} rc={rc}".format(
                        gpu=gpu,
                        model=active_job.job["model"],
                        seed=active_job.job["seed"],
                        rc=return_code,
                    )
                )
                del active[gpu]

            memory = _gpu_memory_mib()
            for gpu in managed_gpus:
                if gpu in active or not pending:
                    continue
                if memory.get(gpu, 0) >= FREE_MEMORY_MIB:
                    continue
                job = _pick_job(pending, gpu, long_gpu)
                active[gpu] = _launch(job, gpu)
                log(
                    "launch gpu={gpu} pid={pid} model={model} seed={seed}".format(
                        gpu=gpu,
                        pid=active[gpu].process.pid,
                        model=job["model"],
                        seed=job["seed"],
                    )
                )

            _write_state(state_path, pending, active, finished)
            if pending or active:
                time.sleep(POLL_SECONDS)

        _write_state(state_path, pending, active, finished)
        log("done")


if __name__ == "__main__":
    main()
