"""Lightweight structures shared by the detection attack modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ImageSample:
    image_id: int
    file_name: str
    path: Path
    image: np.ndarray  # RGB uint8, HWC


@dataclass
class Detections:
    bboxes: torch.Tensor  # [N, 4], xyxy in original image coordinates
    labels: torch.Tensor  # [N]
    scores: torch.Tensor  # [N]

    def __len__(self) -> int:
        return int(self.bboxes.shape[0])

    @property
    def device(self) -> torch.device:
        return self.bboxes.device

    def select(self, mask: torch.Tensor) -> "Detections":
        return Detections(
            bboxes=self.bboxes[mask],
            labels=self.labels[mask],
            scores=self.scores[mask],
        )

    def to_cpu_dict(self) -> dict[str, np.ndarray]:
        return {
            "bboxes": self.bboxes.detach().cpu().numpy(),
            "labels": self.labels.detach().cpu().numpy(),
            "scores": self.scores.detach().cpu().numpy(),
        }


@dataclass
class RegionMeta:
    image_index: int
    object_index: int
    label: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2; x2/y2 are exclusive
    attack_pixels: int

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])
