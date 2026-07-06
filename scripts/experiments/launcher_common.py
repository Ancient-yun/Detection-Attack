from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


REPO = env_path("ATTACK_REPO", Path(__file__).resolve().parents[2])
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PYTHON = env_path("ATTACK_PYTHON", sys.executable)
MATRIX_PATH = env_path("ATTACK_MATRIX", REPO / "experiments" / "score05.json")
IMAGE_DIR = env_path("ATTACK_IMAGE_DIR", REPO / "data" / "coco_amnesia" / "val2017")
ANN_FILE = env_path(
    "ATTACK_ANN_FILE",
    REPO / "data" / "coco_amnesia" / "instances_val2017_ori.json",
)


def quote(value: object) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def load_matrix(matrix_path: Path = MATRIX_PATH) -> dict[str, Any]:
    with matrix_path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_key(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized


def experiment_cases(matrix_path: Path = MATRIX_PATH) -> dict[str, dict[str, Any]]:
    cases = {}
    for case in load_matrix(matrix_path).get("experiments", []):
        aliases = {
            case.get("id"),
            case.get("model"),
            case.get("short"),
            normalize_key(case.get("id", "")),
            normalize_key(case.get("model", "")),
            normalize_key(case.get("short", "")),
        }
        for alias in aliases:
            if alias:
                cases[str(alias)] = case
    return cases


def matrix_model(
    key: str,
    *,
    alias: str | None = None,
    gpu: int | None = None,
    duration_rank: int | None = None,
    matrix_path: Path = MATRIX_PATH,
) -> dict[str, object]:
    cases = experiment_cases(matrix_path)
    lookup_key = key if key in cases else normalize_key(key)
    case = cases[lookup_key]
    model = {
        "name": alias or normalize_key(case.get("short") or case["model"]),
        "model": alias or normalize_key(case.get("short") or case["model"]),
        "model_type": case.get("model_type", "mmdet"),
        "checkpoint": case["checkpoint"],
    }
    if case.get("config"):
        model["config"] = case["config"]
    if gpu is not None:
        model["gpu"] = gpu
    if duration_rank is not None:
        model["duration_rank"] = duration_rank
    return model


def validate_common_paths(models: list[dict[str, object]] | dict[str, dict[str, object]]) -> None:
    missing = []
    if not IMAGE_DIR.is_dir():
        missing.append(str(IMAGE_DIR))
    if not ANN_FILE.is_file():
        missing.append(str(ANN_FILE))

    iterable = models.values() if isinstance(models, dict) else models
    for model in iterable:
        config = model.get("config")
        if config and not (REPO / str(config)).is_file():
            missing.append(str(REPO / str(config)))
        if not (REPO / str(model["checkpoint"])).is_file():
            missing.append(str(REPO / str(model["checkpoint"])))

    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
