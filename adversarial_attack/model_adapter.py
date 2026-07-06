"""MMDetection model adapter for adversarial attack compatibility.

Wraps an mmdetection detector model to provide the predict/predict_label
interface that SparseEvo and PointWise attacks expect.
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

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
        inference_mode: 'legacy' uses inference_detector. 'legacy_cached'
            reuses the legacy MMDetection test pipeline. 'direct_tensor'
            feeds GPU tensors directly to model.test_step.
    """

    SUPPORTED_INFERENCE_MODES = {'legacy', 'legacy_cached', 'direct_tensor'}

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        score_thr: float = 0.3,
        iou_thr: float = 0.5,
        success_thr: float = 0.5,
        inference_mode: str = 'legacy',
    ):
        if inference_mode not in self.SUPPORTED_INFERENCE_MODES:
            raise ValueError(
                f"Unsupported mmdet inference mode: {inference_mode!r}. "
                f"Choose from {sorted(self.SUPPORTED_INFERENCE_MODES)}."
            )
        self.device = device
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.success_thr = success_thr
        self.inference_mode = inference_mode

        # Load mmdetection model
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.model.eval()
        self.classes = self.model.dataset_meta.get('classes', [])

        # Reference detections (set before attack loop)
        self._ref_bboxes = None
        self._ref_labels = None
        self._ref_bboxes_tensor = None
        self._ref_labels_tensor = None
        self._ref_label_int = 0  # Dummy label for predict_label compatibility

        # Tensor-only inference settings. Inputs are RGB float tensors in [0, 1].
        self._resize_scale, self._keep_ratio = self._get_resize_info()
        self._img_size = self._resize_scale
        self._mean, self._std = self._get_normalize_tensors()
        self.outputs_original_size = True

    def _get_resize_info(self) -> Tuple[Tuple[int, int], bool]:
        """Extract test-time resize settings from the model config."""
        cfg = self.model.cfg
        try:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            for transform in test_pipeline:
                if transform.get('type') in ('Resize', 'FixScaleResize'):
                    scale = transform.get('scale', (640, 640))
                    keep_ratio = transform.get('keep_ratio', False)
                    return tuple(scale), bool(keep_ratio)
        except Exception:
            pass
        return (640, 640), False

    def _get_normalize_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build RGB [0, 1] normalization tensors from config mean/std."""
        data_preprocessor = self.model.cfg.model.get('data_preprocessor', {})
        mean = data_preprocessor.get('mean', [0.0, 0.0, 0.0])
        std = data_preprocessor.get('std', [255.0, 255.0, 255.0])
        mean = torch.tensor(mean, device=self.device, dtype=torch.float32)
        std = torch.tensor(std, device=self.device, dtype=torch.float32)
        mean = (mean / 255.0).view(1, -1, 1, 1)
        std = (std / 255.0).view(1, -1, 1, 1)
        return mean, std

    def _rescale_size(self, h: int, w: int) -> Tuple[int, int]:
        """Compute keep-ratio resize size using MMDetection scale semantics."""
        target_w, target_h = self._resize_scale
        if self._keep_ratio:
            scale = min(max(target_w, target_h) / max(h, w),
                        min(target_w, target_h) / min(h, w))
            new_w = max(1, int(w * float(scale) + 0.5))
            new_h = max(1, int(h * float(scale) + 0.5))
        else:
            new_w, new_h = target_w, target_h
        return new_h, new_w

    def _preprocess_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, DetDataSample]:
        """Resize and normalize an RGB [0, 1] tensor for direct prediction."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(0) != 1:
            raise ValueError("MMDetModelAdapter only supports batch size 1.")

        x = x.to(device=self.device, dtype=torch.float32).clamp(0, 1)
        _, _, orig_h, orig_w = x.shape
        new_h, new_w = self._rescale_size(orig_h, orig_w)

        if (new_h, new_w) != (orig_h, orig_w):
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear',
                align_corners=False,
            )

        x = (x - self._mean) / self._std

        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_id': 0,
            'img_path': None,
            'ori_shape': (orig_h, orig_w),
            'img_shape': (new_h, new_w),
            'scale_factor': (new_w / orig_w, new_h / orig_h),
            'batch_input_shape': (new_h, new_w),
            'pad_shape': (new_h, new_w),
        })
        return x.contiguous(), data_sample

    def _get_resize_cfg(self) -> tuple[tuple[int, int], bool]:
        """Extract Resize transform settings used by the test pipeline."""
        cfg = self.model.cfg
        try:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            for transform in test_pipeline:
                if transform.get('type') in ('Resize',):
                    scale = transform.get('scale', (640, 640))
                    keep_ratio = transform.get('keep_ratio', True)
                    return tuple(scale), bool(keep_ratio)
        except Exception:
            pass
        return (640, 640), True

    def _build_legacy_test_pipeline(self) -> Compose:
        """Build the same ndarray test pipeline inference_detector creates."""
        cfg = self.model.cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        return Compose(test_pipeline)

    def _tensor_to_numpy_img(self, x: torch.Tensor) -> np.ndarray:
        """Convert [1, C, H, W] tensor in [0,1] range to [H, W, C] uint8 BGR.

        mmdetection's inference_detector expects BGR numpy images.
        """
        if x.dim() == 4:
            x = x[0]

        if x.is_cuda and x.dtype == torch.float32:
            # Preserve legacy input bytes while avoiding a float32 CPU copy.
            img = x.detach().clamp(0, 1).mul(255).to(torch.uint8)
            img = img[[2, 1, 0], :, :].permute(1, 2, 0).contiguous()
            return img.cpu().numpy()

        # [C, H, W] → [H, W, C], RGB → BGR, [0,1] → [0,255]
        img = x.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        return np.ascontiguousarray(img)

    def _resize_bgr_tensor_for_mmdet(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
        """Resize a BGR CHW tensor on GPU like the MMDetection test pipeline.

        Args:
            img: BGR tensor with shape [3, H, W] and range [0, 255].

        Returns:
            Resized tensor, resized (height, width), and scale factor in
            (width_scale, height_scale) order.
        """
        inputs, data_sample = self._preprocess_tensor(x)

        with torch.no_grad():
            result = self.model(
                inputs,
                data_samples=[data_sample],
                mode='predict',
            )[0]

        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_id': 0,
            'ori_shape': (ori_h, ori_w),
            'img_shape': (img_h, img_w),
            'scale_factor': scale_factor,
        })
        return {'inputs': [img], 'data_samples': [data_sample]}

    def _format_prediction(self, result) -> Dict[str, np.ndarray]:
        """Convert a DetDataSample into the attack result format."""
        pred_instances = result.pred_instances
        mask = pred_instances.scores >= self.score_thr
        return {
            'bboxes': pred_instances.bboxes[mask].detach().cpu().numpy(),
            'labels': pred_instances.labels[mask].detach().cpu().numpy(),
            'scores': pred_instances.scores[mask].detach().cpu().numpy(),
        }

    def _format_prediction_tensors(
        self,
        result,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return filtered prediction tensors without a CPU copy."""
        pred_instances = result.pred_instances
        mask = pred_instances.scores >= self.score_thr
        return (
            pred_instances.bboxes[mask].detach(),
            pred_instances.labels[mask].detach(),
            pred_instances.scores[mask].detach(),
        )

    def _predict_legacy(
        self,
        x: torch.Tensor,
        *,
        use_cached_pipeline: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run detection through the original inference_detector path."""
        img_np = self._tensor_to_numpy_img(x)
        test_pipeline = (
            self._legacy_test_pipeline if use_cached_pipeline else None
        )

        with torch.no_grad():
            result = inference_detector(
                self.model,
                img_np,
                test_pipeline=test_pipeline,
            )

        return self._format_prediction(result)

    def _predict_direct_tensor_raw(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run detection with GPU tensor input and keep tensor outputs."""
        data = self._tensor_to_mmdet_data(x)
        with torch.inference_mode():
            result = self.model.test_step(data)[0]
        return self._format_prediction_tensors(result)

    def _predict_direct_tensor(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run detection by feeding GPU tensors directly to model.test_step."""
        bboxes, labels, scores = self._predict_direct_tensor_raw(x)
        return {
            'bboxes': bboxes.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'scores': scores.cpu().numpy(),
        }

    def _match_detections_torch(
        self,
        orig_bboxes: torch.Tensor,
        orig_labels: torch.Tensor,
        adv_bboxes: torch.Tensor,
        adv_labels: torch.Tensor,
    ) -> Dict[str, int]:
        """Match detections on GPU using the same greedy policy as metrics.py."""
        n_orig = int(orig_bboxes.shape[0])
        if n_orig == 0:
            return {
                'total': 0, 'survived': 0,
                'disappeared': 0, 'misclassified': 0,
                'attack_success': 0,
            }

        n_adv = int(adv_bboxes.shape[0])
        if n_adv == 0:
            return {
                'total': n_orig, 'survived': 0,
                'disappeared': n_orig, 'misclassified': 0,
                'attack_success': n_orig,
            }

        orig_bboxes = orig_bboxes.float()
        adv_bboxes = adv_bboxes.float()

        lt = torch.maximum(orig_bboxes[:, None, :2], adv_bboxes[None, :, :2])
        rb = torch.minimum(orig_bboxes[:, None, 2:], adv_bboxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        orig_area = (
            (orig_bboxes[:, 2] - orig_bboxes[:, 0]).clamp(min=0)
            * (orig_bboxes[:, 3] - orig_bboxes[:, 1]).clamp(min=0)
        )
        adv_area = (
            (adv_bboxes[:, 2] - adv_bboxes[:, 0]).clamp(min=0)
            * (adv_bboxes[:, 3] - adv_bboxes[:, 1]).clamp(min=0)
        )
        union = orig_area[:, None] + adv_area[None, :] - inter
        iou_mat = torch.where(
            union > 0,
            inter / union,
            torch.zeros_like(inter),
        )

        survived = 0
        disappeared = 0
        misclassified = 0
        matched_adv: set[int] = set()

        for i in range(n_orig):
            row = iou_mat[i].clone()
            if matched_adv:
                row[list(matched_adv)] = -1.0
            best_j = int(torch.argmax(row).item())
            best_iou = float(row[best_j].item())

            if best_iou < self.iou_thr:
                disappeared += 1
            else:
                matched_adv.add(best_j)
                if int(orig_labels[i].item()) != int(adv_labels[best_j].item()):
                    misclassified += 1
                else:
                    survived += 1

        return {
            'total': n_orig,
            'survived': survived,
            'disappeared': disappeared,
            'misclassified': misclassified,
            'attack_success': disappeared + misclassified,
        }

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
        if self.inference_mode == 'legacy_cached':
            return self._predict_legacy(x, use_cached_pipeline=True)
        if self.inference_mode == 'direct_tensor':
            return self._predict_direct_tensor(x)
        return self._predict_legacy(x)

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

        if (
            self.inference_mode == 'direct_tensor'
            and self._ref_bboxes_tensor is not None
            and self._ref_labels_tensor is not None
        ):
            bboxes, labels, _ = self._predict_direct_tensor_raw(x)
            result = self._match_detections_torch(
                self._ref_bboxes_tensor,
                self._ref_labels_tensor,
                bboxes,
                labels,
            )
        else:
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
        if self.inference_mode == 'direct_tensor':
            bboxes, labels, scores = self._predict_direct_tensor_raw(x)
            self._ref_bboxes_tensor = bboxes
            self._ref_labels_tensor = labels
            dets = {
                'bboxes': bboxes.cpu().numpy(),
                'labels': labels.cpu().numpy(),
                'scores': scores.cpu().numpy(),
            }
        else:
            dets = self.predict(x)
            self._ref_bboxes_tensor = torch.as_tensor(
                dets['bboxes'],
                device=self.device,
                dtype=torch.float32,
            )
            self._ref_labels_tensor = torch.as_tensor(
                dets['labels'],
                device=self.device,
                dtype=torch.long,
            )

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


class Yolov8ModelAdapter:
    """Adapter that wraps ultralytics YOLOv8 model for decision-based attacks.

    Converts YOLO outputs into the predict() dict format required by the pipeline.

    Args:
        checkpoint_path: Path to model checkpoint (e.g., 'yolov8n.pt'). Will download if not exists.
        device: Device string (default: 'cuda:0').
        score_thr: Detection confidence threshold (default: 0.3).
        iou_thr: IoU threshold for bbox matching (default: 0.5).
        success_thr: Minimum attack success rate for predict_label (default: 0.5).
        inference_mode: 'legacy' keeps Ultralytics high-level inference
            semantics. 'direct_tensor' avoids CPU image conversion but can
            produce small numeric differences in detections.
    """

    SUPPORTED_INFERENCE_MODES = {'legacy', 'direct_tensor'}

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:0',
        score_thr: float = 0.3,
        iou_thr: float = 0.5,
        success_thr: float = 0.5,
        inference_mode: str = 'legacy',
    ):
        if inference_mode not in self.SUPPORTED_INFERENCE_MODES:
            raise ValueError(
                f"Unsupported YOLOv8 inference mode: {inference_mode!r}. "
                f"Choose from {self.SUPPORTED_INFERENCE_MODES}"
            )
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Please install ultralytics to use YOLOv8: `pip install ultralytics`"
            )
        try:
            from ultralytics.utils import nms as yolo_nms
            non_max_suppression = yolo_nms.non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression

        self.device = device
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.success_thr = success_thr
        self.inference_mode = inference_mode

        # Load YOLO model
        self.model = YOLO(checkpoint_path)
        self.model.to(device)
        self.model.model.eval()
        self._non_max_suppression = non_max_suppression
        self.classes = list(self.model.names.values())
        
        # Original default input size for standard yolo
        self._img_size = (640, 640)

        # Reference detections
        self._ref_bboxes = None
        self._ref_labels = None
        self._ref_label_int = 0

    def _tensor_to_numpy_img(self, x: torch.Tensor) -> np.ndarray:
        if x.dim() == 4:
            x = x[0]
        img = x.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(img)

    def _predict_legacy(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        img_np = self._tensor_to_numpy_img(x)
        results = self.model(img_np, verbose=False)
        boxes = results[0].boxes
        mask = boxes.conf >= self.score_thr
        return {
            'bboxes': boxes.xyxy[mask].cpu().numpy(),
            'labels': boxes.cls[mask].cpu().numpy().astype(int),
            'scores': boxes.conf[mask].cpu().numpy(),
        }

    def _predict_direct_tensor(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        input_tensor = x.detach().to(self.device)
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()

        with torch.no_grad():
            preds = self.model.model(input_tensor)

        dets = self._non_max_suppression(
            preds,
            conf_thres=self.score_thr,
            iou_thres=self.iou_thr,
            max_det=300,
            nc=0,
            end2end=getattr(self.model.model, "end2end", False),
        )[0]

        if dets.numel() == 0:
            return {
                'bboxes': np.empty((0, 4), dtype=np.float32),
                'labels': np.empty((0,), dtype=int),
                'scores': np.empty((0,), dtype=np.float32),
            }

        bboxes = dets[:, :4].detach().cpu().numpy()
        labels = dets[:, 5].detach().cpu().numpy().astype(int)
        scores = dets[:, 4].detach().cpu().numpy()

        return {
            'bboxes': bboxes,
            'labels': labels,
            'scores': scores,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        if self.inference_mode == 'direct_tensor':
            return self._predict_direct_tensor(x)
        return self._predict_legacy(x)

    def predict_label(self, x: torch.Tensor) -> int:
        if self._ref_bboxes is None:
            warnings.warn("Reference detections not set. Call set_reference() first.")
            return 0

        dets = self.predict(x)
        result = match_detections(
            self._ref_bboxes, self._ref_labels,
            dets['bboxes'], dets['labels'],
            iou_thr=self.iou_thr,
        )

        if result['total'] == 0:
            return self._ref_label_int

        success_rate = result['attack_success'] / result['total']

        if success_rate >= self.success_thr:
            return -1
        return self._ref_label_int

    def set_reference(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        dets = self.predict(x)
        self._ref_bboxes = dets['bboxes']
        self._ref_labels = dets['labels']
        self._ref_label_int = 0
        return dets

    def check_attack_success(self, x: torch.Tensor) -> bool:
        return self.predict_label(x) == -1

    def get_detailed_result(self, x: torch.Tensor) -> Dict:
        dets = self.predict(x)
        return match_detections(
            self._ref_bboxes, self._ref_labels,
            dets['bboxes'], dets['labels'],
            iou_thr=self.iou_thr,
        )

    def __call__(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        return self.predict(x)
