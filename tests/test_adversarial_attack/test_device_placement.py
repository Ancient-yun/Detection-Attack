import numpy as np
import pytest
import torch

from adversarial_attack.attack_pipeline import DetectionAttackPipeline
from adversarial_attack.pointwise import PointWiseAtt
from adversarial_attack.sparse_evo import SpaEvoAtt


class CpuOnlyModel:
    _img_size = (4, 4)
    classes = ["object"]
    device = "cpu"

    def predict_label(self, tensor):
        assert tensor.device.type == "cpu"
        return 1


def test_load_image_respects_cpu_device(tmp_path):
    cv2 = pytest.importorskip("cv2")
    image_path = tmp_path / "image.jpg"
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    pipeline = DetectionAttackPipeline.__new__(DetectionAttackPipeline)
    pipeline.model = CpuOnlyModel()
    pipeline.device = "cpu"

    tensor = pipeline.load_image(str(image_path))

    assert tensor.device.type == "cpu"
    # load_image no longer pre-resizes: the attack works on the original size
    # and each adapter's model function f does its own preprocessing.
    assert tensor.shape == (1, 3, 8, 8)


def test_generate_starting_point_uses_original_image_device():
    pipeline = DetectionAttackPipeline.__new__(DetectionAttackPipeline)
    pipeline.model = CpuOnlyModel()
    pipeline.device = "cpu"
    pipeline.verbose = False

    oimg = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    start_img, n_queries = pipeline.generate_starting_point(oimg, olabel=0, seed=1)

    assert start_img.device.type == "cpu"
    assert n_queries == 1


def test_pointwise_status_check_uses_model_device():
    attack = PointWiseAtt(CpuOnlyModel(), flag=False, verbose=False)
    img = np.zeros((1, 3, 4, 4), dtype=np.float32)

    assert attack._check_adv_status(img, olabel=0, tlabel=-1)


def test_sparse_evo_mask_uses_input_tensor_device():
    attack = SpaEvoAtt(CpuOnlyModel())
    oimg = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    timg = oimg.clone()
    timg[:, :, 1, 2] = 1.0

    indices = attack._compute_mask(oimg, timg)

    assert indices.tolist() == [6]


def test_sparse_evo_apply_mask_uses_tensor_mask_not_cpu_coordinates(monkeypatch):
    attack = SpaEvoAtt(CpuOnlyModel())
    oimg = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    timg = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    mask = np.array([1, 0, 0, 1], dtype=int)

    def fail_coordinate_conversion(*args, **kwargs):
        raise AssertionError("_apply_mask should not convert masks to CPU coordinates")

    monkeypatch.setattr(attack, "_convert_1d_to_2d", fail_coordinate_conversion)

    out = attack._apply_mask(mask, oimg, timg)

    assert torch.equal(out[:, :, 0, 0], timg[:, :, 0, 0])
    assert torch.equal(out[:, :, 0, 1], oimg[:, :, 0, 1])
    assert torch.equal(out[:, :, 1, 0], oimg[:, :, 1, 0])
    assert torch.equal(out[:, :, 1, 1], timg[:, :, 1, 1])
