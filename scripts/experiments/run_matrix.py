#!/usr/bin/env python
"""Run attack experiments from a JSON matrix."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "attack_reports"
sys.path.insert(0, str(TOOLS))

from matrix_utils import iter_cases, load_matrix, parse_report, project_root, relative, result_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--num-images", default=None)
    parser.add_argument("--attacks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--resume-partial", action="store_true", default=None)
    parser.add_argument("--no-resume-partial", dest="resume_partial", action="store_false")
    return parser.parse_args()


def as_cli_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def output_dir(root: Path, case: dict[str, Any], run_name: str | None) -> Path:
    return root / case["attack"] / case["dataset_name"] / case["model"] / (run_name or case["run_name"])


def is_completed(out_dir: Path, score_thr: float, expected_images: int | None) -> bool:
    report = out_dir / "experiment_report.txt"
    metrics = parse_report(report)
    if not metrics:
        return False
    if metrics.get("score_thr") != as_cli_value(score_thr):
        return False
    if expected_images is not None and metrics.get("total_images") != str(expected_images):
        return False
    return True


def build_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    case: dict[str, Any],
    out_dir: Path,
) -> list[str]:
    defaults = config.get("defaults", {})
    score_thr = args.score_thr
    if score_thr is None:
        score_thr = float(config.get("comparison", {}).get("target_score_thr", 0.5))
    num_images = args.num_images or case.get("num_images", defaults.get("num_images", "all"))

    cmd = [args.python, "run_attack.py"]
    model_type = case.get("model_type", "mmdet")
    if model_type != "mmdet":
        cmd.extend(["--model-type", model_type])
    if case.get("config"):
        cmd.extend(["--config", case["config"]])
    cmd.extend(
        [
            "--checkpoint",
            case["checkpoint"],
            "--image-dir",
            case["image_dir"],
        ]
    )
    if case.get("ann_file"):
        cmd.extend(["--ann-file", case["ann_file"]])
    cmd.extend(
        [
            "--attack",
            case["attack"],
            "--dataset-name",
            case["dataset_name"],
            "--num-images",
            as_cli_value(num_images),
            "--max-query",
            as_cli_value(defaults.get("max_query", 1000)),
            "--score-thr",
            as_cli_value(score_thr),
            "--iou-thr",
            as_cli_value(defaults.get("iou_thr", 0.5)),
            "--success-thr",
            as_cli_value(defaults.get("success_thr", 0.7)),
            "--seed",
            as_cli_value(defaults.get("seed", 42)),
            "--output-dir",
            str(out_dir),
        ]
    )
    resume_partial = args.resume_partial
    if resume_partial is None:
        resume_partial = config.get("run", {}).get("resume_partial", False)
    if resume_partial:
        cmd.append("--resume-partial")

    for key, value in case.get("attack_params", {}).items():
        cmd.extend([f"--{key.replace('_', '-')}", as_cli_value(value)])
    return cmd


def main() -> int:
    args = parse_args()
    config = load_matrix(args.config)
    root = result_root(config, args.result_root)
    score_thr = args.score_thr
    if score_thr is None:
        score_thr = float(config.get("comparison", {}).get("target_score_thr", 0.5))

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

    for case in iter_cases(config, args.attacks, args.targets):
        run_name = args.run_name or case.get("run_name")
        out_dir = output_dir(root, case, run_name)
        if not args.no_skip_completed and is_completed(
            out_dir, score_thr, case.get("expected_images")
        ):
            print(f"[Skip] {relative(out_dir)}")
            continue

        cmd = build_command(args, config, case, out_dir)
        print(shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=project_root(), env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
