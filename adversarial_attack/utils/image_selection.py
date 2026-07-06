from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageSelection:
    image_dir: str
    num_images: str
    sample_strategy: str
    sample_seed: int | None
    all_image_paths: list[str]
    selected_image_paths: list[str]


def collect_image_paths(image_dir: str | os.PathLike[str]) -> list[str]:
    root = Path(image_dir)
    paths = [
        os.path.normpath(str(path))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(set(paths))


def _parse_num_images(num_images: str | int | None) -> int | None:
    if num_images is None or str(num_images).lower() == "all":
        return None
    try:
        limit = int(num_images)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid --num-images value {num_images!r}; use "all" or a positive integer.'
        ) from exc
    if limit < 1:
        raise ValueError(f"--num-images must be positive, got {limit}.")
    return limit


def select_image_paths(
    image_dir: str | os.PathLike[str],
    num_images: str | int | None = "all",
    sample_strategy: str = "first",
    sample_seed: int | None = None,
) -> ImageSelection:
    strategy = sample_strategy.lower()
    if strategy not in {"first", "random"}:
        raise ValueError(f"Unsupported sample strategy: {sample_strategy!r}")
    if strategy == "random" and sample_seed is None:
        raise ValueError("--sample-strategy random requires --sample-seed or --seed.")

    all_paths = collect_image_paths(image_dir)
    if not all_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    limit = _parse_num_images(num_images)
    if limit is not None and limit > len(all_paths):
        raise ValueError(
            f"--num-images {limit} exceeds available images ({len(all_paths)}) "
            f"in {image_dir}."
        )

    if limit is None:
        selected = list(all_paths)
    elif strategy == "first":
        selected = all_paths[:limit]
    else:
        selected = random.Random(sample_seed).sample(all_paths, limit)
        selected.sort()

    return ImageSelection(
        image_dir=os.path.normpath(str(image_dir)),
        num_images="all" if limit is None else str(limit),
        sample_strategy=strategy,
        sample_seed=sample_seed,
        all_image_paths=all_paths,
        selected_image_paths=selected,
    )


def build_sample_manifest(selection: ImageSelection) -> dict[str, Any]:
    return {
        "image_dir": selection.image_dir,
        "num_images": selection.num_images,
        "sample_strategy": selection.sample_strategy,
        "sample_seed": selection.sample_seed,
        "selected_image_paths": selection.selected_image_paths,
    }


def write_sample_manifest(
    manifest_path: str | os.PathLike[str],
    selection: ImageSelection,
) -> None:
    path = Path(manifest_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(build_sample_manifest(selection), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def load_sample_manifest(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def validate_sample_manifest(
    manifest_path: str | os.PathLike[str],
    selection: ImageSelection,
) -> None:
    path = Path(manifest_path)
    if not path.exists():
        return

    existing = load_sample_manifest(path)
    expected = build_sample_manifest(selection)
    if existing != expected:
        raise ValueError(
            f"Existing sample manifest differs from current image selection: {path}. "
            "Use a new --output-dir or remove the partial resume state."
        )
