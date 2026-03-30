"""MMDetection model adapter for adversarial attack compatibility.

Wraps an mmdetection detector model to provide the predict/predict_label
interface that SparseEvo and PointWise attacks expect.
"""

import torch
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

from mmdet.apis import init_detector, inference_detector

from .metrics import match_detections


# Patch torch.load for PyTorch 2.6 compatibility
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args, **{**kwargs, 'weights_only': False}
)


class MMDetModelAdapter:
    """Adapter that wraps mmdetection model for decision-based attacks.

    Converts detection outputs (bboxes, labels, scores) into the
    predict_label() → int interface required by SparseEvo/PointWise.

    Attack success is determined by IoU-based matching:
    - bbox disappeared (no IoU >= threshold match)
    - bbox class changed (IoU match but different label)

    Args:
        config_path: Path to mmdetection config file.
        checkpoint_path: Path to model checkpoint (.pth).
        device: Device string (default: 'cuda:0').
        score_thr: Detection confidence threshold (default: 0.3).
        iou_thr: IoU threshold for bbox matching (default: 0.5).
        success_thr: Minimum attack success rate for predict_label (default: 0.5).
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        score_thr: float = 0.3,
        iou_thr: float = 0.5,
        success_thr: float = 0.5,
    ):
        self.device = device
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.success_thr = success_thr

        # Load mmdetection model
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.model.eval()
        self.classes = self.model.dataset_meta.get('classes', [])

        # Reference detections (set before attack loop)
        self._ref_bboxes = None
        self._ref_labels = None
        self._ref_label_int = 0  # Dummy label for predict_label compatibility

        # Get model input size from config
        self._img_size = self._get_input_size()

    def _get_input_size(self) -> Tuple[int, int]:
        """Extract expected input size from model config."""
        cfg = self.model.cfg
        try:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            for transform in test_pipeline:
                if transform.get('type') in ('Resize',):
                    scale = transform.get('scale', (640, 640))
                    return tuple(scale)
        except Exception:
            pass
        return (640, 640)

    def _tensor_to_numpy_img(self, x: torch.Tensor) -> np.ndarray:
        """Convert [1, C, H, W] tensor in [0,1] range to [H, W, C] uint8 BGR.

        mmdetection's inference_detector expects BGR numpy images.
        """
        if x.dim() == 4:
            x = x[0]
        # [C, H, W] → [H, W, C], RGB → BGR, [0,1] → [0,255]
        img = x.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        return np.ascontiguousarray(img)

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run detection on a tensor image.

        Args:
            x: Image tensor of shape [1, C, H, W], values in [0, 1].

        Returns:
            Dict with keys:
                - 'bboxes': np.ndarray [N, 4] (x1, y1, x2, y2)
                - 'labels': np.ndarray [N] (class indices)
                - 'scores': np.ndarray [N] (confidence scores)
        """
        img_np = self._tensor_to_numpy_img(x)

        with torch.no_grad():
            result = inference_detector(self.model, img_np)

        pred_instances = result.pred_instances

        # Apply score threshold
        mask = pred_instances.scores >= self.score_thr
        bboxes = pred_instances.bboxes[mask].cpu().numpy()
        labels = pred_instances.labels[mask].cpu().numpy()
        scores = pred_instances.scores[mask].cpu().numpy()

        return {
            'bboxes': bboxes,
            'labels': labels,
            'scores': scores,
        }

    def predict_label(self, x: torch.Tensor) -> int:
        """Decision-based prediction for attack compatibility.

        Checks if the attack success rate meets the threshold
        via IoU matching between reference and current detections.

        Args:
            x: Image tensor [1, C, H, W], values in [0, 1].

        Returns:
            -1 if attack succeeded (success_rate >= success_thr),
            self._ref_label_int otherwise (attack failed).
        """
        if self._ref_bboxes is None:
            warnings.warn(
                "Reference detections not set. Call set_reference() first."
            )
            return 0

        dets = self.predict(x)
        result = match_detections(
            self._ref_bboxes, self._ref_labels,
            dets['bboxes'], dets['labels'],
            iou_thr=self.iou_thr,
        )

        # Compute success rate
        if result['total'] == 0:
            return self._ref_label_int

        success_rate = result['attack_success'] / result['total']

        if success_rate >= self.success_thr:
            return -1  # Attack success rate meets threshold
        return self._ref_label_int  # Below threshold → failure

    def set_reference(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Set reference (benign) detections for attack comparison.

        Must be called before running any attack on an image.

        Args:
            x: Benign image tensor [1, C, H, W], values in [0, 1].

        Returns:
            Detection result dict (bboxes, labels, scores).
        """
        dets = self.predict(x)
        self._ref_bboxes = dets['bboxes']
        self._ref_labels = dets['labels']
        self._ref_label_int = 0  # Dummy int for "original label"
        return dets

    def check_attack_success(self, x: torch.Tensor) -> bool:
        """Check if the perturbed image is adversarial.

        Args:
            x: Perturbed image tensor [1, C, H, W].

        Returns:
            True if at least one original detection is disrupted.
        """
        return self.predict_label(x) == -1

    def get_detailed_result(self, x: torch.Tensor) -> Dict:
        """Get detailed attack evaluation results.

        Args:
            x: Perturbed image tensor [1, C, H, W].

        Returns:
            Full match_detections result dict.
        """
        dets = self.predict(x)
        return match_detections(
            self._ref_bboxes, self._ref_labels,
            dets['bboxes'], dets['labels'],
            iou_thr=self.iou_thr,
        )

    def __call__(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        return self.predict(x)
