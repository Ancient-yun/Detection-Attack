"""Detection environment: prediction, matching, and reward computation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from .types import Detections, RegionMeta

if TYPE_CHECKING:
    from .tensor_detector import TensorOnlyDetector


def bbox_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


class DetectionEnv:
    """Reward surface for detection attacks.

    References are detector decisions. Scores are not used by the reward. Each
    object state is 0 when it survives with the same class and 1 when it is
    misclassified or disappeared. A reference label below 0 marks an already
    disappeared object for standard reward memory.
    """

    def __init__(
        self,
        detector: TensorOnlyDetector,
        iou_thr: float = 0.5,
        detector_batch_inference: bool = False,
    ) -> None:
        self.detector = detector
        self.device = detector.device
        self.iou_thr = iou_thr
        self.detector_batch_inference = detector_batch_inference

    def predict_images(self, images: list[np.ndarray]) -> list[Detections]:
        if self.detector_batch_inference and hasattr(self.detector, "predict_images"):
            return self.detector.predict_images(images)
        return [self.detector.predict_image(image) for image in images]

    def predict_image_tensors(self, images: list[torch.Tensor]) -> list[Detections]:
        if (
            self.detector_batch_inference
            and hasattr(self.detector, "predict_image_tensors")
        ):
            return self.detector.predict_image_tensors(images)
        return [self.detector.predict_image_tensor(image) for image in images]

    def align_detections_to_refs(self, refs: Detections, current: Detections) -> Detections:
        if len(refs) == 0:
            return Detections(
                bboxes=refs.bboxes.clone(),
                labels=refs.labels.clone(),
                scores=refs.scores.clone(),
            )

        bboxes = refs.bboxes.clone()
        labels = torch.full_like(refs.labels, -1)
        scores = torch.zeros_like(refs.scores)
        if len(current) == 0:
            return Detections(bboxes=bboxes, labels=labels, scores=scores)

        ious = bbox_iou_matrix(refs.bboxes, current.bboxes)
        best_iou, best_idx = ious.max(dim=1)
        same_class = refs.labels[:, None] == current.labels[None, :]
        same_class_ious = ious.masked_fill(~same_class, -1.0)
        best_same_iou, best_same_idx = same_class_ious.max(dim=1)
        same_matched = best_same_iou >= self.iou_thr
        any_matched = best_iou >= self.iou_thr
        matched = same_matched | any_matched
        if matched.any():
            chosen_idx = torch.where(same_matched, best_same_idx, best_idx)
            matched_idx = chosen_idx[matched]
            bboxes[matched] = current.bboxes[matched_idx]
            labels[matched] = current.labels[matched_idx]
            scores[matched] = current.scores[matched_idx]
        return Detections(bboxes=bboxes, labels=labels, scores=scores)

    def object_state_values(self, refs: Detections, adv: Detections) -> torch.Tensor:
        if len(refs) == 0:
            return torch.empty(0, device=self.device)
        missing_refs = refs.labels < 0
        values = torch.zeros(len(refs), device=self.device)
        if len(adv) == 0:
            values[~missing_refs] = 1.0
            return values

        ious = bbox_iou_matrix(refs.bboxes, adv.bboxes)
        best_iou, _ = ious.max(dim=1)
        same_class = refs.labels[:, None] == adv.labels[None, :]
        same_class_ious = ious.masked_fill(~same_class, -1.0)
        best_same_iou, _ = same_class_ious.max(dim=1)
        matched = best_iou >= self.iou_thr
        normal_refs = ~missing_refs
        same_survived = normal_refs & (best_same_iou >= self.iou_thr)

        values[normal_refs & ~same_survived] = 1.0
        values[missing_refs & matched] = 1.0

        return values

    def bbox_guidance_values(self, refs: Detections, adv: Detections) -> torch.Tensor:
        if len(refs) == 0:
            return torch.empty(0, device=self.device)

        values = torch.zeros(len(refs), device=self.device)
        if len(adv) == 0:
            return values

        ious = bbox_iou_matrix(refs.bboxes, adv.bboxes)
        same_class = refs.labels[:, None] == adv.labels[None, :]
        same_class_ious = ious.masked_fill(~same_class, -1.0)
        best_same_iou, _ = same_class_ious.max(dim=1)
        same_survived = (refs.labels >= 0) & (best_same_iou >= self.iou_thr)
        if same_survived.any():
            scale = max(1e-6, 1.0 - self.iou_thr)
            values[same_survived] = (
                (1.0 - best_same_iou[same_survived]) / scale
            ).clamp(min=0.0, max=1.0)
        return values

    def object_objective_values(
        self,
        decision_refs: Detections,
        adv: Detections,
        guidance_refs: Detections | None = None,
        bbox_guidance_weight: float = 0.0,
    ) -> torch.Tensor:
        values = self.object_state_values(decision_refs, adv)
        if bbox_guidance_weight <= 0:
            return values

        guidance_source = guidance_refs if guidance_refs is not None else decision_refs
        guidance = self.bbox_guidance_values(guidance_source, adv)
        if guidance.numel() != values.numel():
            raise ValueError("decision_refs and guidance_refs must be aligned.")
        return values + float(bbox_guidance_weight) * guidance

    def score_regions(
        self,
        changed_by_image: dict[int, np.ndarray],
        refs_by_image: list[Detections],
        regions: list[RegionMeta],
        previous_values_by_image: list[torch.Tensor] | None = None,
        reward_mode: str = "discrepancy",
        guidance_refs_by_image: list[Detections] | None = None,
        bbox_guidance_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[int, Detections]]:
        if reward_mode not in {"discrepancy", "standard"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        image_indices = list(changed_by_image)
        predictions = self.predict_images([changed_by_image[idx] for idx in image_indices])
        adv_by_image = dict(zip(image_indices, predictions))
        return self._score_region_detections(
            adv_by_image,
            refs_by_image,
            regions,
            previous_values_by_image=previous_values_by_image,
            guidance_refs_by_image=guidance_refs_by_image,
            bbox_guidance_weight=bbox_guidance_weight,
        )

    def score_region_tensors(
        self,
        changed_by_image: dict[int, torch.Tensor],
        refs_by_image: list[Detections],
        regions: list[RegionMeta],
        previous_values_by_image: list[torch.Tensor] | None = None,
        reward_mode: str = "discrepancy",
        guidance_refs_by_image: list[Detections] | None = None,
        bbox_guidance_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[int, Detections]]:
        if reward_mode not in {"discrepancy", "standard"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        image_indices = list(changed_by_image)
        predictions = self.predict_image_tensors([
            changed_by_image[idx] for idx in image_indices
        ])
        adv_by_image = dict(zip(image_indices, predictions))
        return self._score_region_detections(
            adv_by_image,
            refs_by_image,
            regions,
            previous_values_by_image=previous_values_by_image,
            guidance_refs_by_image=guidance_refs_by_image,
            bbox_guidance_weight=bbox_guidance_weight,
        )

    def _score_region_detections(
        self,
        adv_by_image: dict[int, Detections],
        refs_by_image: list[Detections],
        regions: list[RegionMeta],
        previous_values_by_image: list[torch.Tensor] | None = None,
        guidance_refs_by_image: list[Detections] | None = None,
        bbox_guidance_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[int, Detections]]:
        value_cache: dict[int, torch.Tensor] = {}
        for image_idx, adv in adv_by_image.items():
            guidance_refs = (
                guidance_refs_by_image[image_idx]
                if guidance_refs_by_image is not None
                else None
            )
            value_cache[image_idx] = self.object_objective_values(
                refs_by_image[image_idx],
                adv,
                guidance_refs=guidance_refs,
                bbox_guidance_weight=bbox_guidance_weight,
            )

        flat_rewards = []
        flat_state_values = []
        by_image_values: dict[int, list[torch.Tensor]] = defaultdict(list)
        for region in regions:
            values = value_cache.get(region.image_index)
            if values is None or region.object_index >= values.numel():
                current_value = torch.tensor(0.0, device=self.device)
            else:
                current_value = values[region.object_index]

            previous_value = torch.tensor(0.0, device=self.device)
            if previous_values_by_image is not None:
                previous_values = previous_values_by_image[region.image_index].to(self.device)
                if region.object_index < previous_values.numel():
                    previous_value = previous_values[region.object_index]

            reward = current_value - previous_value
            flat_rewards.append(reward)
            flat_state_values.append(current_value)
            by_image_values[region.image_index].append(current_value)

        image_state_values = {
            image_idx: torch.stack(values).mean()
            for image_idx, values in by_image_values.items()
        }
        return (
            torch.stack(flat_rewards),
            torch.stack(flat_state_values),
            image_state_values,
            adv_by_image,
        )
