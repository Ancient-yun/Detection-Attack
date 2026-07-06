from __future__ import annotations

import json
import stat
import time
from pathlib import Path
from typing import Final


REPO: Final = Path("/NHNHOME/WORKSPACE/0226010134_A/daeyun/Detection-Attack")
PYTHON: Final = Path(
    "/NHNHOME/WORKSPACE/0226010134_A/miniconda3/envs/mmdet2_b200/bin/python"
)
IMAGE_DIR: Final = Path("/NHNHOME/WORKSPACE/0226010134_A/data/COCO/val2017")
ANN_FILE: Final = Path(
    "/NHNHOME/WORKSPACE/0226010134_A/data/COCO/annotations/instances_val2017.json"
)
SCRIPT_DIR: Final = REPO / "scripts" / "experiments"


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Pattern not found: {old!r}")
    return text.replace(old, new, 1)


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "result" / "score05" / f"pointwise_multi_sched_score05_{run_id}"
    log_dir = REPO / "logs" / f"score05_pointwise_multi_sched_{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    source = (SCRIPT_DIR / "schedule_remaining_models.py").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "Schedule remaining score05 detector experiments on B200.",
        "Schedule pointwise score05 detector experiments on B200.",
    )
    source = source.replace(
        "DEFAULT_MANAGED_GPUS = [0, 1, 2, 3, 5]",
        "DEFAULT_MANAGED_GPUS = [6, 7]",
    )
    source = source.replace("DEFAULT_LONG_GPU = 5", "DEFAULT_LONG_GPU = 6")
    source = _replace_once(
        source,
        "from launcher_common import (",
        "import sys\n"
        f"sys.path.insert(0, {str(SCRIPT_DIR)!r})\n\n"
        "from launcher_common import (",
    )
    source = _replace_once(
        source,
        'MODELS = {\n    "deformable_detr": matrix_model(',
        'MODELS = {\n'
        '    "ddq_detr": matrix_model(\n'
        '        "ddq_coco_amnesia",\n'
        '        alias="ddq_detr",\n'
        "        duration_rank=4,\n"
        "    ),\n"
        '    "deformable_detr": matrix_model(',
    )
    source = source.replace(
        'root / model_name / f"seed_{seed}" / "sparse_evo"',
        'root / model_name / f"seed_{seed}" / "pointwise_multi_sched"',
    )
    source = source.replace(
        'f"COCO_val2017_score05_{model_name}_seed{seed}"',
        'f"COCO_val2017_score05_pointwise_{model_name}_seed{seed}"',
    )
    source = source.replace(
        'str(log_dir / f"{model_name}_seed{seed}.log")',
        'str(log_dir / f"{model_name}_seed{seed}_pointwise_multi_sched.log")',
    )
    source = source.replace(
        '"--attack",\n            "sparse_evo",',
        '"--attack",\n            "pointwise_multi_sched",',
    )
    sparse_args = "\n".join(
        [
            '            "--no-save-snapshots",',
            '            "--pop-size",',
            '            "10",',
            '            "--cr",',
            '            "0.9",',
            '            "--mu",',
            '            "0.01",',
            '            "--log-interval",',
            '            "50",',
        ]
    )
    pointwise_args = "\n".join(
        [
            '            "--no-save-snapshots",',
            '            "--npix",',
            '            "0.1",',
            '            "--log-interval",',
            '            "50",',
        ]
    )
    source = source.replace(sparse_args, pointwise_args)
    source = source.replace(
        'f"remaining_auto_{run_id}"',
        'f"pointwise_multi_sched_score05_{run_id}"',
    )
    source = source.replace(
        'f"score05_remaining_auto_{run_id}"',
        'f"score05_pointwise_multi_sched_{run_id}"',
    )
    source = source.replace(
        'run_id = time.strftime("%Y%m%d_%H%M%S")',
        f'run_id = "{run_id}"',
    )
    source = source.replace(
        'root = REPO / "result" / "score05" / f"pointwise_multi_sched_score05_{run_id}"',
        f"root = Path({str(root)!r})",
    )
    source = source.replace(
        'log_dir = REPO / "logs" / f"score05_pointwise_multi_sched_{run_id}"',
        f"log_dir = Path({str(log_dir)!r})",
    )
    source = source.replace(
        '"jobs": pending,',
        '"attack": "pointwise_multi_sched",\n'
        '        "npix": 0.1,\n'
        '        "image_dir": str(IMAGE_DIR),\n'
        '        "ann_file": str(ANN_FILE),\n'
        '        "jobs": pending,',
    )

    scheduler = root / "schedule_pointwise_score05.py"
    scheduler.write_text(source, encoding="utf-8")
    scheduler.chmod(scheduler.stat().st_mode | stat.S_IXUSR)

    launcher = root / "run_scheduler.sh"
    launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO}",
                f"export ATTACK_REPO={REPO}",
                f"export ATTACK_PYTHON={PYTHON}",
                f"export ATTACK_IMAGE_DIR={IMAGE_DIR}",
                f"export ATTACK_ANN_FILE={ANN_FILE}",
                "export SCHEDULER_GPUS=6,7",
                "export SCHEDULER_LONG_GPU=6",
                "echo '[launcher] pointwise score05 start' $(date)",
                f"echo '[launcher] root {root}'",
                f"$ATTACK_PYTHON {scheduler}",
                "echo '[launcher] pointwise score05 finished' $(date)",
                "exec bash",
                "",
            ]
        ),
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

    latest = REPO / "result" / "score05" / "pointwise_multi_sched_score05_latest.txt"
    latest.write_text(str(root) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "root": str(root),
                "log_dir": str(log_dir),
                "scheduler": str(scheduler),
                "launcher": str(launcher),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
