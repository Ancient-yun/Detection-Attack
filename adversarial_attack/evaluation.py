from __future__ import annotations

import json
import os
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools.coco import COCO

from .evaluation_1 import detections_to_coco_predictions, evaluate_coco_bbox
from .types import Detections, ImageSample


def dets_to_eval_format(
    dets: dict[str, np.ndarray],
    n_classes: int,
) -> list[np.ndarray]:
    per_class = []
    labels = np.asarray(dets["labels"])
    for class_index in range(n_classes):
        mask = labels == class_index
        if mask.any():
            cls_bboxes = np.asarray(dets["bboxes"])[mask]
            cls_scores = np.asarray(dets["scores"])[mask].reshape(-1, 1)
            per_class.append(
                np.hstack([cls_bboxes, cls_scores]).astype(np.float32)
            )
        else:
            per_class.append(np.zeros((0, 5), dtype=np.float32))
    return per_class


def compute_benign_map(
    model: Any,
    results: list[dict[str, Any]],
    iou_thr: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    dataset, samples, label_to_cat_id = _build_benign_coco_dataset(model, results)
    clean = evaluate_coco_bbox(
        dataset, detections_to_coco_predictions(samples, [_as_detections(r['orig_detections']) for r in results], label_to_cat_id), dataset.coco.getImgIds()
    )
    adversarial = evaluate_coco_bbox(
        dataset, detections_to_coco_predictions(samples, [_as_detections(r['adv_detections']) for r in results], label_to_cat_id), dataset.coco.getImgIds()
    )
    map_result = _coco_map_result(clean, adversarial)

    if verbose:
        print(f"\n[Pipeline] === Benign mAP (IoU={iou_thr}) ===")
        print(f"  Benign clean AP50 : {clean['AP50']:.4f}")
        print(f"  Benign adv AP50   : {adversarial['AP50']:.4f}")

    return map_result


def compute_gt_map(
    model: Any,
    results: list[dict[str, Any]],
    ann_file: str,
    iou_thr: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    if os.path.isdir(ann_file):
        raise ValueError('evaluation_1 COCO evaluation requires a COCO JSON annotation file.')
    dataset, samples, label_to_cat_id = _build_gt_coco_dataset(model, results, ann_file)
    clean = evaluate_coco_bbox(dataset, detections_to_coco_predictions(samples, [_as_detections(r['orig_detections']) for r in results], label_to_cat_id), dataset.coco.getImgIds())
    adversarial = evaluate_coco_bbox(dataset, detections_to_coco_predictions(samples, [_as_detections(r['adv_detections']) for r in results], label_to_cat_id), dataset.coco.getImgIds())
    map_result = _coco_map_result(clean, adversarial)

    if verbose:
        print(f"\n[Pipeline] === GT mAP (IoU={iou_thr}) ===")
        print(f"  GT clean AP50 : {clean['AP50']:.4f}")
        print(f"  GT adv AP50   : {adversarial['AP50']:.4f}")

    return map_result


def _as_detections(dets: dict[str, Any]) -> Detections:
    return Detections(torch.as_tensor(dets['bboxes'], dtype=torch.float32), torch.as_tensor(dets['labels'], dtype=torch.int64), torch.as_tensor(dets['scores'], dtype=torch.float32))


def _coco_map_result(clean: dict[str, float], adversarial: dict[str, float]) -> dict[str, Any]:
    drop = float('nan') if clean['AP50'] < 0 or adversarial['AP50'] < 0 else clean['AP50'] - adversarial['AP50']
    return {'orig_mAP': clean['AP50'], 'adv_mAP': adversarial['AP50'], 'mAP_drop': drop, 'clean_coco_metrics': clean, 'adv_coco_metrics': adversarial, 'per_class_ap': []}


def _build_coco(dataset: dict[str, Any]) -> SimpleNamespace:
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return SimpleNamespace(coco=coco)


def _build_benign_coco_dataset(model: Any, results: list[dict[str, Any]]):
    categories = [{'id': index + 1, 'name': name} for index, name in enumerate(model.classes)]
    images, annotations, samples = [], [], []
    ann_id = 1
    for image_id, result in enumerate(results, 1):
        image_path = result['image_path']
        images.append({'id': image_id, 'file_name': os.path.basename(image_path), 'width': 1, 'height': 1})
        samples.append(ImageSample(image_id, os.path.basename(image_path), image_path, np.zeros((1, 1, 3), dtype=np.uint8)))
        for box, label in zip(result['orig_detections']['bboxes'], result['orig_detections']['labels']):
            x1, y1, x2, y2 = map(float, box)
            annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': int(label) + 1, 'bbox': [x1, y1, x2-x1, y2-y1], 'area': max(0, x2-x1)*max(0, y2-y1), 'iscrowd': 0})
            ann_id += 1
    return _build_coco({'images': images, 'annotations': annotations, 'categories': categories}), samples, {i: i + 1 for i in range(len(model.classes))}


def _build_gt_coco_dataset(model: Any, results: list[dict[str, Any]], ann_file: str):
    with open(ann_file, 'r', encoding='utf-8') as file:
        source = json.load(file)
    wanted = {os.path.basename(r['image_path']) for r in results}
    images = [image for image in source['images'] if os.path.basename(image['file_name']) in wanted]
    image_ids = {image['id'] for image in images}
    annotations = [ann for ann in source['annotations'] if ann['image_id'] in image_ids]
    names = {category['name']: category['id'] for category in source['categories']}
    label_to_cat_id = {label: names[name] for label, name in enumerate(model.classes) if name in names}
    if not label_to_cat_id:
        raise ValueError('Model classes do not match annotation category names.')
    by_name = {os.path.basename(image['file_name']): image for image in images}
    samples = [ImageSample(by_name[os.path.basename(r['image_path'])]['id'], os.path.basename(r['image_path']), r['image_path'], np.zeros((1, 1, 3), dtype=np.uint8)) for r in results]
    return _build_coco({'images': images, 'annotations': annotations, 'categories': source['categories']}), samples, label_to_cat_id


def _load_annotations(
    ann_file: str,
    model_h: int,
    model_w: int,
) -> dict[str, dict[str, list[Any]]]:
    if os.path.isdir(ann_file):
        return _load_yolo_annotations(ann_file, model_h, model_w)
    return _load_coco_annotations(ann_file, model_h, model_w)


def _load_yolo_annotations(
    ann_dir: str,
    model_h: int,
    model_w: int,
) -> dict[str, dict[str, list[Any]]]:
    file_to_anns: dict[str, dict[str, list[Any]]] = {}
    for txt_name in os.listdir(ann_dir):
        if not txt_name.endswith(".txt"):
            continue
        base_name = os.path.splitext(txt_name)[0]
        file_to_anns[base_name] = {"bboxes": [], "labels": []}
        txt_path = os.path.join(ann_dir, txt_name)
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, width, height = map(float, parts[1:5])
                x1 = (xc - width / 2) * model_w
                y1 = (yc - height / 2) * model_h
                x2 = (xc + width / 2) * model_w
                y2 = (yc + height / 2) * model_h
                file_to_anns[base_name]["bboxes"].append([x1, y1, x2, y2])
                file_to_anns[base_name]["labels"].append(cls_id)
    return file_to_anns


def _load_coco_annotations(
    ann_file: str,
    model_h: int,
    model_w: int,
) -> dict[str, dict[str, list[Any]]]:
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    cat_ids = sorted([category["id"] for category in coco_data["categories"]])
    cat_id_to_idx = {category_id: index for index, category_id in enumerate(cat_ids)}
    img_id_to_file = {
        image["id"]: image["file_name"] for image in coco_data["images"]
    }
    img_id_to_size = {
        image["id"]: (image["height"], image["width"])
        for image in coco_data["images"]
    }
    file_to_anns: dict[str, dict[str, list[Any]]] = {}

    for ann in coco_data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        fname = img_id_to_file[ann["image_id"]]
        base_name = os.path.splitext(os.path.basename(fname))[0]
        if base_name not in file_to_anns:
            file_to_anns[base_name] = {"bboxes": [], "labels": []}

        x, y, width, height = ann["bbox"]
        # Tensor-only pipeline: detections are returned in ORIGINAL image
        # coordinates (the model resizes internally and rescales boxes back), so
        # keep GT in original coords too — no model-input scaling.
        file_to_anns[base_name]["bboxes"].append(
            [x, y, x + width, y + height]
        )
        file_to_anns[base_name]["labels"].append(cat_id_to_idx[ann["category_id"]])

    return file_to_anns
