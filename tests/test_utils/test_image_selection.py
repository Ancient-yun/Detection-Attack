from pathlib import Path

import pytest

from adversarial_attack.utils.image_selection import (
    build_sample_manifest,
    select_image_paths,
    validate_sample_manifest,
    write_sample_manifest,
)


def _touch_images(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")


def test_first_strategy_matches_sorted_prefix(tmp_path: Path) -> None:
    _touch_images(
        tmp_path,
        ["b.jpg", "a.png", "nested/c.jpeg", "ignored.txt"],
    )

    selection = select_image_paths(tmp_path, num_images=2)

    assert [Path(p).name for p in selection.selected_image_paths] == [
        "a.png",
        "b.jpg",
    ]


def test_random_strategy_is_reproducible(tmp_path: Path) -> None:
    _touch_images(tmp_path, [f"{index:03d}.jpg" for index in range(20)])

    first = select_image_paths(
        tmp_path,
        num_images=5,
        sample_strategy="random",
        sample_seed=42,
    )
    second = select_image_paths(
        tmp_path,
        num_images=5,
        sample_strategy="random",
        sample_seed=42,
    )

    assert first.selected_image_paths == second.selected_image_paths
    assert first.selected_image_paths == sorted(first.selected_image_paths)


def test_random_strategy_changes_with_seed(tmp_path: Path) -> None:
    _touch_images(tmp_path, [f"{index:03d}.jpg" for index in range(20)])

    first = select_image_paths(
        tmp_path,
        num_images=5,
        sample_strategy="random",
        sample_seed=1,
    )
    second = select_image_paths(
        tmp_path,
        num_images=5,
        sample_strategy="random",
        sample_seed=2,
    )

    assert first.selected_image_paths != second.selected_image_paths


def test_random_strategy_requires_seed(tmp_path: Path) -> None:
    _touch_images(tmp_path, ["a.jpg"])

    with pytest.raises(ValueError, match="requires --sample-seed or --seed"):
        select_image_paths(tmp_path, num_images=1, sample_strategy="random")


def test_num_images_cannot_exceed_candidates(tmp_path: Path) -> None:
    _touch_images(tmp_path, ["a.jpg"])

    with pytest.raises(ValueError, match="exceeds available images"):
        select_image_paths(tmp_path, num_images=2)


def test_manifest_validation_detects_selection_mismatch(tmp_path: Path) -> None:
    _touch_images(tmp_path, [f"{index:03d}.jpg" for index in range(10)])
    manifest_path = tmp_path / "sample_manifest.json"
    original = select_image_paths(
        tmp_path,
        num_images=3,
        sample_strategy="random",
        sample_seed=1,
    )
    changed = select_image_paths(
        tmp_path,
        num_images=3,
        sample_strategy="random",
        sample_seed=2,
    )

    write_sample_manifest(manifest_path, original)

    assert build_sample_manifest(original)["selected_image_paths"]
    with pytest.raises(ValueError, match="Existing sample manifest differs"):
        validate_sample_manifest(manifest_path, changed)
