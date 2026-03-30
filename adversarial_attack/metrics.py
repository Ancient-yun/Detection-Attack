"""Distance metrics and detection matching for adversarial attacks.

Uses mmdet's built-in bbox_overlaps for IoU computation.
Provides L0 distance and detection-level attack success evaluation.
"""

import torch
import numpy as np
from typing import Dict

from mmdet.evaluation.functional import bbox_overlaps


# ======================== L0 Distance ========================

def compute_l0(img1: torch.Tensor, img2: torch.Tensor) -> int:
    """Compute pixel-wise L0 distance between two images.

    Counts the number of pixels that differ in at least one channel.

    Args:
        img1: Image tensor of shape [N, C, H, W].
        img2: Image tensor of shape [N, C, H, W].

    Returns:
        Number of differing pixels.
    """
    diff = torch.abs(img1 - img2)
    pixel_diff = (diff > 0.0).any(dim=1)  # [N, H, W]
    return pixel_diff.sum().item()


def compute_l0_approx(img1: torch.Tensor, img2: torch.Tensor) -> int:
    """Approximate L0 distance using channel-sum comparison.

    Args:
        img1: Image tensor of shape [N, C, H, W].
        img2: Image tensor of shape [N, C, H, W].

    Returns:
        Approximate number of differing pixels.
    """
    diff = torch.abs(torch.sum(img1, 1) - torch.sum(img2, 1))
    return (diff > 0.0).sum().item()


# ======================== IoU ========================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes using mmdet.

    Args:
        box1: Array of [x1, y1, x2, y2].
        box2: Array of [x1, y1, x2, y2].

    Returns:
        IoU value in [0, 1].
    """
    bboxes1 = np.array(box1, dtype=np.float32).reshape(1, 4)
    bboxes2 = np.array(box2, dtype=np.float32).reshape(1, 4)
    return float(bbox_overlaps(bboxes1, bboxes2, mode='iou')[0, 0])


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix using mmdet.

    Args:
        boxes1: Array of shape [N, 4], each row [x1, y1, x2, y2].
        boxes2: Array of shape [M, 4], each row [x1, y1, x2, y2].

    Returns:
        IoU matrix of shape [N, M].
    """
    bboxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    bboxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    return bbox_overlaps(bboxes1, bboxes2, mode='iou')


# ======================== Detection Matching ========================

def match_detections(
    orig_bboxes: np.ndarray,
    orig_labels: np.ndarray,
    adv_bboxes: np.ndarray,
    adv_labels: np.ndarray,
    iou_thr: float = 0.5,
) -> Dict[str, int]:
    """Match original detections against adversarial detections using IoU.

    For each original bbox, find the best IoU match in the adversarial
    detections and determine if it survived, disappeared, or was misclassified.

    Args:
        orig_bboxes: Original bboxes [N, 4].
        orig_labels: Original class labels [N].
        adv_bboxes: Adversarial bboxes [M, 4].
        adv_labels: Adversarial class labels [M].
        iou_thr: IoU threshold for matching (default: 0.5).

    Returns:
        Dict with keys:
            - 'total': total original detections
            - 'survived': still detected with same class
            - 'disappeared': no matching bbox found
            - 'misclassified': bbox matched but class changed
            - 'attack_success': disappeared + misclassified
    """
    n_orig = len(orig_bboxes)

    if n_orig == 0:
        return {
            'total': 0, 'survived': 0,
            'disappeared': 0, 'misclassified': 0,
            'attack_success': 0,
        }

    n_adv = len(adv_bboxes)

    if n_adv == 0:
        return {
            'total': n_orig, 'survived': 0,
            'disappeared': n_orig, 'misclassified': 0,
            'attack_success': n_orig,
        }

    # Compute IoU matrix using mmdet
    iou_mat = bbox_overlaps(
        np.asarray(orig_bboxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(adv_bboxes, dtype=np.float32).reshape(-1, 4),
        mode='iou',
    )

    # Greedy matching: for each orig bbox, find best adv match
    survived = 0
    disappeared = 0
    misclassified = 0
    matched_adv = set()

    for i in range(n_orig):
        row = iou_mat[i].copy()
        for j in matched_adv:
            row[j] = -1.0
        best_j = int(row.argmax())
        best_iou = row[best_j]

        if best_iou < iou_thr:
            disappeared += 1
        else:
            matched_adv.add(best_j)
            if orig_labels[i] != adv_labels[best_j]:
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


def compute_attack_success_rate(
    orig_bboxes: np.ndarray,
    orig_labels: np.ndarray,
    adv_bboxes: np.ndarray,
    adv_labels: np.ndarray,
    iou_thr: float = 0.5,
) -> float:
    """Compute attack success rate.

    success_rate = (disappeared + misclassified) / total_original_bboxes

    Args:
        orig_bboxes: Original bboxes [N, 4].
        orig_labels: Original class labels [N].
        adv_bboxes: Adversarial bboxes [M, 4].
        adv_labels: Adversarial class labels [M].
        iou_thr: IoU threshold for matching.

    Returns:
        Attack success rate in [0.0, 1.0].
    """
    result = match_detections(orig_bboxes, orig_labels,
                              adv_bboxes, adv_labels, iou_thr)
    if result['total'] == 0:
        return 0.0
    return result['attack_success'] / result['total']
