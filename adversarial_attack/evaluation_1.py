"""Final attack and COCO metric evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from .dataset import CocoDetectionDataset, VocDetectionDataset
from .env import bbox_iou_matrix
from .types import Detections, ImageSample


COCO_BBOX_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def l0_pixels(original: np.ndarray, attacked: np.ndarray) -> int:
    return int(np.any(original != attacked, axis=2).sum())


def match_summary(refs: Detections, adv: Detections, iou_thr: float) -> dict[str, int]:
    total = len(refs)
    if total == 0:
        return {"total": 0, "survived": 0, "misclassified": 0, "disappeared": 0}
    if len(adv) == 0:
        return {"total": total, "survived": 0, "misclassified": 0, "disappeared": total}

    ious = bbox_iou_matrix(refs.bboxes, adv.bboxes)
    survived = 0
    misclassified = 0
    disappeared = 0

    for idx in range(total):
        best_iou, best_j = ious[idx].max(dim=0)
        if float(best_iou.item()) < iou_thr:
            disappeared += 1
        elif int(refs.labels[idx].item()) == int(adv.labels[best_j].item()):
            survived += 1
        else:
            misclassified += 1

    return {
        "total": total,
        "survived": survived,
        "misclassified": misclassified,
        "disappeared": disappeared,
    }


def detections_to_coco_predictions(
    samples: list[ImageSample],
    detections: list[Detections],
    label_to_cat_id: dict[int, int],
) -> list[dict]:
    predictions = []
    for sample, dets in zip(samples, detections):
        cpu = dets.to_cpu_dict()
        for bbox, label, score in zip(cpu["bboxes"], cpu["labels"], cpu["scores"]):
            category_id = label_to_cat_id.get(int(label))
            if category_id is None:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            predictions.append(
                {
                    "image_id": int(sample.image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return predictions


def evaluate_coco_bbox(
    dataset: CocoDetectionDataset | VocDetectionDataset,
    predictions: list[dict],
    img_ids: list[int],
) -> dict[str, float]:
    if not predictions:
        return {name: -1.0 for name in COCO_BBOX_METRIC_NAMES}

    coco_dt = dataset.coco.loadRes(predictions)
    coco_eval = COCOeval(dataset.coco, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        name: float(value)
        for name, value in zip(COCO_BBOX_METRIC_NAMES, coco_eval.stats)
    }


def coco_metric_delta(
    clean_metrics: dict[str, float],
    attacked_metrics: dict[str, float],
) -> dict[str, float | None]:
    """Return attacked-clean deltas for every COCO bbox metric.

    COCO uses negative values for unavailable buckets, such as no medium
    objects in a selected subset. Those deltas are reported as null.
    """
    deltas: dict[str, float | None] = {}
    for name in COCO_BBOX_METRIC_NAMES:
        clean = clean_metrics.get(name, -1.0)
        attacked = attacked_metrics.get(name, -1.0)
        if clean < 0 or attacked < 0:
            deltas[name] = None
        else:
            deltas[name] = attacked - clean
    return deltas


def build_final_summary(
    samples: list[ImageSample],
    original_images: list[np.ndarray],
    attacked_images: list[np.ndarray],
    original_refs: list[Detections],
    final_dets: list[Detections],
    iou_thr: float,
) -> dict:
    total_l0 = 0
    total_pixels = 0
    totals = {"total": 0, "survived": 0, "misclassified": 0, "disappeared": 0}

    per_image = []
    for sample, original, attacked, refs, adv in zip(
        samples, original_images, attacked_images, original_refs, final_dets
    ):
        l0 = l0_pixels(original, attacked)
        total_l0 += l0
        total_pixels += int(original.shape[0] * original.shape[1])

        matched = match_summary(refs, adv, iou_thr)
        for key in totals:
            totals[key] += matched[key]

        per_image.append(
            {
                "image_id": int(sample.image_id),
                "file_name": sample.file_name,
                "l0": l0,
                "pixel_ratio": l0 / max(1, original.shape[0] * original.shape[1]),
                "reference_detections": len(refs),
                **matched,
            }
        )

    attack_success = totals["misclassified"] + totals["disappeared"]
    return {
        "num_images": len(samples),
        "total_l0": total_l0,
        "pixel_ratio": total_l0 / max(1, total_pixels),
        "attack_success_rate": attack_success / max(1, totals["total"]),
        "matching": totals,
        "per_image": per_image,
    }


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
