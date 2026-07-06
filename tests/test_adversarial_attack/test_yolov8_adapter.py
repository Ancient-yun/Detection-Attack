from types import SimpleNamespace

import numpy as np
import torch

from adversarial_attack.model_adapter import Yolov8ModelAdapter


def test_yolov8_predict_uses_tensor_path_without_numpy_image_conversion() -> None:
    adapter = Yolov8ModelAdapter.__new__(Yolov8ModelAdapter)
    adapter.device = "cpu"
    adapter.score_thr = 0.5
    adapter.iou_thr = 0.5
    adapter.inference_mode = "direct_tensor"

    class FakeDetectionModel:
        end2end = False

        def __call__(self, x: torch.Tensor):
            assert x.shape == (1, 3, 640, 640)
            return torch.zeros((1, 84, 1), dtype=torch.float32), None

    adapter.model = SimpleNamespace(model=FakeDetectionModel())

    def fail_numpy_conversion(_x: torch.Tensor) -> np.ndarray:
        raise AssertionError("predict should not convert tensors to CPU numpy images")

    def fake_nms(*args, **kwargs):
        return [
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 0.9, 2.0]],
                dtype=torch.float32,
            )
        ]

    adapter._tensor_to_numpy_img = fail_numpy_conversion
    adapter._non_max_suppression = fake_nms

    dets = adapter.predict(torch.zeros((1, 3, 640, 640), dtype=torch.float32))

    np.testing.assert_allclose(dets["bboxes"], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(dets["labels"], [2])
    np.testing.assert_allclose(dets["scores"], [0.9], rtol=1e-6)


def test_yolov8_predict_legacy_uses_ultralytics_high_level_path() -> None:
    adapter = Yolov8ModelAdapter.__new__(Yolov8ModelAdapter)
    adapter.score_thr = 0.5
    adapter.inference_mode = "legacy"

    converted = np.zeros((640, 640, 3), dtype=np.uint8)
    adapter._tensor_to_numpy_img = lambda _x: converted

    class FakeBoxes:
        conf = torch.tensor([0.4, 0.9], dtype=torch.float32)
        xyxy = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
            dtype=torch.float32,
        )
        cls = torch.tensor([1.0, 3.0], dtype=torch.float32)

    class FakeHighLevelModel:
        def __call__(self, img_np: np.ndarray, verbose: bool):
            assert img_np is converted
            assert verbose is False
            return [SimpleNamespace(boxes=FakeBoxes())]

    adapter.model = FakeHighLevelModel()

    dets = adapter.predict(torch.zeros((1, 3, 640, 640), dtype=torch.float32))

    np.testing.assert_allclose(dets["bboxes"], [[4.0, 5.0, 6.0, 7.0]])
    np.testing.assert_array_equal(dets["labels"], [3])
    np.testing.assert_allclose(dets["scores"], [0.9], rtol=1e-6)
