from types import SimpleNamespace

import numpy as np
import torch

from adversarial_attack.model_adapter import Yolov8ModelAdapter


def test_yolov8_predict_uses_ultralytics_high_level_path() -> None:
    adapter = Yolov8ModelAdapter.__new__(Yolov8ModelAdapter)
    adapter.score_thr = 0.5

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
