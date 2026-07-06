from pathlib import Path
from types import SimpleNamespace

import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "experiments"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_matrix import build_command, is_completed  # noqa: E402


def _write_report(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_report.txt").write_text(
        "\n".join(
            [
                "  Score Threshold   : 0.5",
                "  Total Images      : 1",
                "  Overall ASR (image)    : 100.00% (1/1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_completed_requires_final_image_artifacts(tmp_path: Path) -> None:
    _write_report(tmp_path)

    assert not is_completed(tmp_path, 0.5, expected_images=1)

    image_file = tmp_path / "images" / "sample" / "adv.png"
    image_file.parent.mkdir(parents=True)
    for name in ["orig.png", "adv.png", "delta.png", "adv_raw.png", "result.txt"]:
        (image_file.parent / name).write_bytes(b"artifact")

    assert is_completed(tmp_path, 0.5, expected_images=1)


def test_build_command_can_enable_snapshots_without_disabling_final_images(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        python="python",
        score_thr=None,
        num_images=None,
        sample_strategy=None,
        sample_seed=None,
        resume_partial=None,
        save_snapshots=True,
    )
    config = {
        "defaults": {"num_images": 1, "max_query": 10, "seed": 0},
        "comparison": {"target_score_thr": 0.5},
    }
    case = {
        "attack": "sparse_evo",
        "dataset_name": "dataset",
        "model": "yolov8n",
        "run_name": "run",
        "model_type": "yolov8",
        "checkpoint": "ckpt/yolov8n.pt",
        "image_dir": "images",
    }

    cmd = build_command(args, config, case, tmp_path)

    assert "--save-snapshots" in cmd
    assert "--no-save-images" not in cmd
