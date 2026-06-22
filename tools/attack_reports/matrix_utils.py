"""Shared helpers for experiment-matrix reports and validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_matrix(path: str | Path) -> dict[str, Any]:
    matrix_path = Path(path)
    if not matrix_path.is_absolute():
        matrix_path = project_root() / matrix_path
    with matrix_path.open(encoding="utf-8") as f:
        return json.load(f)


def result_root(config: dict[str, Any], override: str | None = None) -> Path:
    root = Path(override or config.get("result_root", "result"))
    if not root.is_absolute():
        root = project_root() / root
    return root


def selected(value: str, filters: set[str] | None) -> bool:
    if not filters:
        return True
    normalized = value.lower()
    return any(item.lower() in normalized for item in filters)


def iter_cases(
    config: dict[str, Any],
    attacks: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    attack_filter = set(attacks or [])
    target_filter = set(targets or [])
    cases: list[dict[str, Any]] = []
    attack_defaults = config.get("attack_defaults", {})
    default_run_name = config.get("run", {}).get("name")

    for exp in config.get("experiments", []):
        searchable_target = " ".join(
            str(exp.get(key, "")) for key in ("id", "target", "short", "model")
        )
        if not selected(searchable_target, target_filter):
            continue

        for attack in exp.get("attacks", []):
            if attack_filter and attack not in attack_filter:
                continue
            case = dict(exp)
            case["attack"] = attack
            case["attack_params"] = dict(attack_defaults.get(attack, {}))
            case["run_name"] = exp.get("run_name", default_run_name)
            cases.append(case)
    return cases


def scalar(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def mean_value(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*Mean=([0-9.,]+%?)", text, re.MULTILINE)
    return match.group(1).replace(",", "") if match else None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().split()[0].replace("%", "").replace(",", "")
    if cleaned in {"", "N/A", "PENDING"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, Any] = {
        "path": path,
        "score_thr": scalar(text, "Score Threshold"),
        "max_queries": scalar(text, "Max Queries"),
        "iou_thr": scalar(text, "IoU Threshold"),
        "success_thr": scalar(text, "Success Threshold"),
        "total_images": scalar(text, "Total Images"),
        "total_attacked": scalar(text, "Total Images Attacked"),
        "skipped": scalar(text, "Skipped (no detection)"),
        "valid": scalar(text, "Valid Attacks"),
        "success": scalar(text, "Successful Attacks"),
        "image_asr": (scalar(text, "Overall ASR (image)") or "").split()[0] or None,
        "avg_queries": mean_value(text, "Queries Used"),
        "mean_l0": mean_value(text, "L0 Distance (pixels)"),
        "sparsity": mean_value(text, "Sparsity Ratio"),
        "bbox_asr": scalar(text, "BBox-Level ASR"),
        "orig_map": scalar(text, "Original mAP"),
        "adv_map": scalar(text, "Adversarial mAP"),
        "map_drop": scalar(text, "mAP Drop"),
        "elapsed": scalar(text, "Elapsed Time"),
    }
    for key, value in list(metrics.items()):
        if key != "path":
            metrics[f"{key}_num"] = parse_float(value)
    return metrics


def report_score(path: Path) -> float | None:
    return parse_float(parse_report(path).get("score_thr"))


def latest_report(
    root: Path,
    attack: str,
    dataset: str,
    model: str,
    score_thr: float,
    expected_images: int | None = None,
) -> Path | None:
    base = root / attack / dataset / model
    if not base.exists():
        return None

    candidates = [
        path
        for path in base.glob("*/experiment_report.txt")
        if (score := report_score(path)) is not None and abs(score - score_thr) < 1e-9
    ]
    if not candidates:
        return None

    if expected_images is not None:
        full_candidates = [
            path
            for path in candidates
            if parse_report(path).get("total_images") == str(expected_images)
        ]
        if full_candidates:
            candidates = full_candidates

    return max(candidates, key=lambda p: (p.stat().st_mtime, p.parent.name))


def named_report(root: Path, case: dict[str, Any], score_thr: float) -> Path | None:
    run_name = case.get("run_name")
    if run_name:
        report = (
            root
            / case["attack"]
            / case["dataset_name"]
            / case["model"]
            / run_name
            / "experiment_report.txt"
        )
        if report.exists() and report_score(report) == score_thr:
            return report
    return latest_report(
        root,
        case["attack"],
        case["dataset_name"],
        case["model"],
        score_thr,
        case.get("expected_images"),
    )


def relative(path: Path | None) -> str:
    if path is None:
        return "N/A"
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()
