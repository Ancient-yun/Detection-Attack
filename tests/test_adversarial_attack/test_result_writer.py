from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from adversarial_attack import result_writer


def test_save_attack_results_skips_snapshots_but_keeps_final_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    written_names: list[str] = []

    def fake_imwrite(path, image):
        written_names.append(Path(path).name)
        return True

    def fake_load_visual_images(pipeline, result):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        return base, base.copy(), base.copy(), base.copy(), 1.0, 1.0

    def fail_predict(_tensor):
        raise AssertionError("snapshot prediction should not run when save_snapshots=False")

    monkeypatch.setattr(result_writer.cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(result_writer, "_load_visual_images", fake_load_visual_images)
    monkeypatch.setattr(
        result_writer,
        "compute_benign_map",
        lambda *args, **kwargs: {"orig_mAP": 1.0, "adv_mAP": 0.0, "mAP_drop": 1.0},
    )

    detections = {
        "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
        "labels": np.array([0], dtype=np.int64),
        "scores": np.array([0.9], dtype=np.float32),
    }
    results = [
        {
            "image_path": "sample.jpg",
            "adv_image": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            "n_queries": 1,
            "l0_distance": 2,
            "sparsity_ratio": 0.5,
            "success_rate": 1.0,
            "is_successful": True,
            "match_result": {
                "total": 1,
                "survived": 0,
                "disappeared": 1,
                "misclassified": 0,
            },
            "orig_detections": detections,
            "adv_detections": detections,
            "snapshots": {0: torch.zeros((1, 3, 2, 2), dtype=torch.float32)},
        }
    ]
    pipeline = SimpleNamespace(
        model=SimpleNamespace(classes=["car"], iou_thr=0.5, predict=fail_predict)
    )

    result_writer.save_attack_results(
        pipeline=pipeline,
        results=results,
        output_dir=str(tmp_path),
        save_snapshots=False,
    )

    assert list(tmp_path.glob("attack_results_*.csv"))
    image_dir = tmp_path / "images" / "sample"
    assert image_dir.is_dir()
    assert written_names == ["orig.png", "adv.png", "delta.png", "adv_raw.png"]
    assert not (image_dir / "query_0.png").exists()
    assert "query_0.png" not in (image_dir / "result.txt").read_text(encoding="utf-8")
