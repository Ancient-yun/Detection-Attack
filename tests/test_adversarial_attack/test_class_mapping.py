import pytest
import torch

from adversarial_attack.class_mapping import (
    VOC_CLASSES,
    filter_and_remap_detections,
    resolve_class_mapping,
)
from mmdet.datasets import CocoDataset, VOCDataset

COCO_CLASSES = CocoDataset.METAINFO['classes']


def test_coco_to_voc_mapping_matches_dataset_metadata():
    resolved = resolve_class_mapping(COCO_CLASSES, 'coco-to-voc')

    assert resolved.target_classes == tuple(VOCDataset.METAINFO['classes'])
    assert resolved.target_classes == VOC_CLASSES
    assert resolved.source_to_target is not None
    assert sum(label >= 0 for label in resolved.source_to_target) == 20

    source_to_target = resolved.source_to_target
    assert source_to_target[COCO_CLASSES.index('airplane')] == 0
    assert source_to_target[COCO_CLASSES.index('motorcycle')] == 13
    assert source_to_target[COCO_CLASSES.index('person')] == 14
    assert source_to_target[COCO_CLASSES.index('couch')] == 17
    assert source_to_target[COCO_CLASSES.index('truck')] == -1


def test_coco_to_voc_mapping_accepts_legacy_underscore_metadata():
    legacy_classes = tuple(
        class_name.replace(' ', '_') for class_name in COCO_CLASSES)

    resolved = resolve_class_mapping(legacy_classes, 'coco-to-voc')

    assert resolved.source_to_target is not None
    assert resolved.source_to_target[legacy_classes.index(
        'dining_table')] == 10
    assert resolved.source_to_target[legacy_classes.index(
        'potted_plant')] == 15


def test_filter_remaps_voc_classes_and_removes_non_voc_classes():
    resolved = resolve_class_mapping(COCO_CLASSES, 'coco-to-voc')
    source_to_target = resolved.to_tensor('cpu')
    bboxes = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    labels = torch.tensor([
        COCO_CLASSES.index('person'),
        COCO_CLASSES.index('truck'),
        COCO_CLASSES.index('airplane'),
        COCO_CLASSES.index('couch'),
        COCO_CLASSES.index('dining table'),
    ])
    scores = torch.tensor([0.9, 0.95, 0.8, 0.7, 0.1])

    filtered_bboxes, filtered_labels, filtered_scores = (
        filter_and_remap_detections(
            bboxes,
            labels,
            scores,
            score_thr=0.3,
            source_to_target=source_to_target,
        ))

    assert torch.equal(filtered_bboxes, bboxes[[0, 2, 3]])
    assert filtered_labels.tolist() == [14, 0, 17]
    assert torch.equal(filtered_scores, scores[[0, 2, 3]])


def test_none_mapping_preserves_source_label_space():
    resolved = resolve_class_mapping(COCO_CLASSES, 'none')
    assert resolved.target_classes == tuple(COCO_CLASSES)
    assert resolved.source_to_target is None

    bboxes = torch.zeros((2, 4))
    labels = torch.tensor([7.0, 4.0])
    scores = torch.tensor([0.9, 0.2])
    filtered_bboxes, filtered_labels, filtered_scores = (
        filter_and_remap_detections(
            bboxes,
            labels,
            scores,
            score_thr=0.3,
        ))

    assert filtered_bboxes.shape == (1, 4)
    assert filtered_labels.tolist() == [7]
    assert filtered_scores.tolist() == pytest.approx([0.9])


def test_coco_to_voc_mapping_rejects_non_coco_metadata():
    with pytest.raises(ValueError, match='requires COCO class metadata'):
        resolve_class_mapping(VOC_CLASSES, 'coco-to-voc')
