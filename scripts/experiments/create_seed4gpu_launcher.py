"""Create a B200 launcher that runs one seed across four GPUs.

This script is meant to be run inside the B200 Detection-Attack repository.
It creates image symlink shards and a bash launcher under the repository's
result directory, so no temporary external directory is required.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from launcher_common import (
    ANN_FILE,
    IMAGE_DIR,
    PYTHON,
    REPO,
    matrix_model,
    validate_common_paths,
)
from adversarial_attack.utils.image_selection import select_image_paths


MODEL = matrix_model("ddq_coco_amnesia", alias="ddq_detr")
CONFIG = str(MODEL["config"])
CHECKPOINT = str(MODEL["checkpoint"])


def _make_shard_links(paths: list[str], link_dir: Path) -> None:
    link_dir.mkdir(parents=True, exist_ok=True)
    for src_str in paths:
        src = Path(src_str)
        dst = link_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)


def main() -> None:
    validate_common_paths([MODEL])

    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "result" / "score05" / f"ddq_detr_seed4gpu_{run_id}"
    log_dir = REPO / "logs" / f"score05_ddq_detr_seed4gpu_{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs_by_seed: dict[int, list[dict[str, object]]] = {}
    for seed in range(4):
        selection = select_image_paths(
            IMAGE_DIR,
            num_images=500,
            sample_strategy="random",
            sample_seed=seed,
        )
        selected = selection.selected_image_paths
        if len(selected) != 500:
            raise RuntimeError(
                f"seed {seed}: expected 500 images, got {len(selected)}"
            )

        jobs_by_seed[seed] = []
        for shard_id in range(4):
            start = shard_id * 125
            shard_paths = selected[start:start + 125]
            shard_root = root / f"seed_{seed}" / f"shard_{shard_id}"
            link_dir = shard_root / "image_links"
            out_dir = shard_root / "sparse_evo"
            _make_shard_links(shard_paths, link_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            job = {
                "gpu": shard_id,
                "seed": seed,
                "shard_id": shard_id,
                "start_index": start,
                "attack_seed": seed + start,
                "num_images": len(shard_paths),
                "dataset_name": (
                    f"COCO_val2017_score05_seed{seed}_shard{shard_id}"
                ),
                "link_dir": str(link_dir),
                "output_dir": str(out_dir),
                "log": str(
                    log_dir / f"seed_{seed}_shard_{shard_id}_sparse_evo.log"
                ),
                "first_image": Path(shard_paths[0]).name,
                "last_image": Path(shard_paths[-1]).name,
            }
            jobs_by_seed[seed].append(job)

        (root / f"seed_{seed}" / "shard_plan.json").write_text(
            json.dumps(jobs_by_seed[seed], indent=2) + "\n",
            encoding="utf-8",
        )

    plan = {
        "run_id": run_id,
        "root": str(root),
        "log_dir": str(log_dir),
        "jobs_by_seed": jobs_by_seed,
    }
    (root / "run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    launcher = root / "launch_seed4gpu.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {REPO}",
        "echo '[launcher] start' $(date)",
        f"echo '[launcher] root {root}'",
    ]
    for seed in range(4):
        lines.append(f"echo '[launcher] seed {seed} start' $(date)")
        pid_vars = []
        for job in jobs_by_seed[seed]:
            pid_var = f"pid_s{seed}_g{job['gpu']}"
            pid_vars.append(pid_var)
            cmd = [
                f"CUDA_VISIBLE_DEVICES={job['gpu']}",
                str(PYTHON),
                "run_attack.py",
                f"--config {CONFIG}",
                f"--checkpoint {CHECKPOINT}",
                f"--image-dir {job['link_dir']}",
                f"--ann-file {ANN_FILE}",
                "--attack sparse_evo",
                f"--dataset-name {job['dataset_name']}",
                "--max-query 1000",
                "--score-thr 0.5",
                "--iou-thr 0.5",
                "--success-thr 0.7",
                f"--seed {job['attack_seed']}",
                "--num-images all",
                "--sample-strategy first",
                f"--sample-manifest {job['output_dir']}/sample_manifest.json",
                "--resume-partial",
                f"--output-dir {job['output_dir']}",
                "--no-save-snapshots",
                "--pop-size 10",
                "--cr 0.9",
                "--mu 0.01",
                "--log-interval 50",
                f"> {job['log']} 2>&1 &",
            ]
            lines.append(
                "echo '[launcher] gpu {gpu} seed {seed} shard {shard_id} "
                "attack_seed {attack_seed}'".format(**job)
            )
            lines.append(" ".join(cmd))
            lines.append(f"{pid_var}=$!")
        lines.append("seed_status=0")
        for pid_var in pid_vars:
            lines.append(f"wait ${pid_var} || seed_status=$?")
        lines.append(
            "echo '[launcher] seed "
            + str(seed)
            + " done status='${seed_status}' '$(date)"
        )
        lines.append("if [ ${seed_status} -ne 0 ]; then exit ${seed_status}; fi")
    lines.extend(
        [
            "echo '[launcher] done' $(date)",
            "exit 0",
        ]
    )
    launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "root": str(root),
                "log_dir": str(log_dir),
                "launcher": str(launcher),
                "seeds": 4,
                "shards_per_seed": 4,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
