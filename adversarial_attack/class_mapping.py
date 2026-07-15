"""Class-space mappings for cross-dataset detector inference."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

VOC_CLASSES: Tuple[str, ...] = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)

# Class-name correspondence from:
# https://gist.github.com/kaixin96/58f5ddcdd1fc74c0a0f0427fba99cd09
COCO_TO_VOC_CLASS_NAMES = {
    'airplane': 'aeroplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'dining table': 'diningtable',
    'dog': 'dog',
    'horse': 'horse',
    'motorcycle': 'motorbike',
    'person': 'person',
    'potted plant': 'pottedplant',
    'sheep': 'sheep',
    'couch': 'sofa',
    'train': 'train',
    'tv': 'tvmonitor',
}

SUPPORTED_CLASS_MAPPINGS = ('none', 'coco-to-voc')


def _normalize_class_name(class_name: str) -> str:
    """Normalize class metadata across MMDetection checkpoint versions."""
    return ' '.join(class_name.strip().lower().replace('_', ' ').split())


@dataclass(frozen=True)
class ResolvedClassMapping:
    """A resolved source-index to target-index mapping.

    Attributes:
        target_classes: Class names exposed by the adapter after mapping.
        source_to_target: Target label for each source label. ``-1`` marks a
            source class that must be removed. ``None`` disables remapping.
    """

    target_classes: Tuple[str, ...]
    source_to_target: Optional[Tuple[int, ...]]

    def to_tensor(self, device: str) -> Optional[torch.Tensor]:
        """Create the label lookup tensor on the inference device.

        Args:
            device: Torch device used for detector inference.

        Returns:
            A source-to-target lookup tensor, or ``None`` when disabled.
        """
        if self.source_to_target is None:
            return None
        return torch.tensor(
            self.source_to_target,
            dtype=torch.long,
            device=device,
        )


def resolve_class_mapping(
    source_classes: Sequence[str],
    mapping_name: str = 'none',
) -> ResolvedClassMapping:
    """Resolve a named class mapping against a model's class metadata.

    The Gist uses sparse COCO category IDs, while MMDetection predictions use
    contiguous model labels. Resolving by class name avoids mixing those two
    index spaces.

    Args:
        source_classes: Class names in the detector output order.
        mapping_name: Mapping to apply. Supported values are ``none`` and
            ``coco-to-voc``.

    Returns:
        Target class metadata and an optional source-to-target lookup table.

    Raises:
        ValueError: If the mapping name is unsupported or required COCO class
            names are absent from the model metadata.
    """
    normalized_source_classes = tuple(
        _normalize_class_name(class_name) for class_name in source_classes)

    if mapping_name == 'none':
        return ResolvedClassMapping(
            target_classes=tuple(source_classes),
            source_to_target=None,
        )
    if mapping_name != 'coco-to-voc':
        raise ValueError(f'Unsupported class mapping: {mapping_name!r}. '
                         f'Choose from {SUPPORTED_CLASS_MAPPINGS}.')

    missing_classes = sorted(
        set(COCO_TO_VOC_CLASS_NAMES) - set(normalized_source_classes))
    if missing_classes:
        raise ValueError(
            'coco-to-voc requires COCO class metadata. Missing source '
            f'classes: {", ".join(missing_classes)}')

    voc_indices = {
        class_name: index
        for index, class_name in enumerate(VOC_CLASSES)
    }
    source_to_target = tuple(
        voc_indices[COCO_TO_VOC_CLASS_NAMES[class_name]] if class_name in
        COCO_TO_VOC_CLASS_NAMES else -1
        for class_name in normalized_source_classes)
    return ResolvedClassMapping(
        target_classes=VOC_CLASSES,
        source_to_target=source_to_target,
    )


def filter_and_remap_detections(
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float,
    source_to_target: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply score filtering and an optional class-space mapping.

    Args:
        bboxes: Predicted boxes with shape ``[N, 4]``.
        labels: Source-space labels with shape ``[N]``.
        scores: Confidence scores with shape ``[N]``.
        score_thr: Minimum confidence score to retain.
        source_to_target: Lookup from source labels to target labels. Entries
            equal to ``-1`` are outside the target dataset and are removed.

    Returns:
        Filtered boxes, target-space labels, and scores.
    """
    labels = labels.to(dtype=torch.long)
    keep = scores >= score_thr

    if source_to_target is not None:
        if source_to_target.device != labels.device:
            source_to_target = source_to_target.to(labels.device)
        mapped_labels = source_to_target[labels]
        keep = keep & (mapped_labels >= 0)
        labels = mapped_labels

    return bboxes[keep], labels[keep], scores[keep]
