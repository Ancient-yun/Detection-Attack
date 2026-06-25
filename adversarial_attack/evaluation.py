from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from mmdet.evaluation.functional import eval_map


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
    n_classes = len(model.classes)
    annotations = []
    orig_det_results = []
    adv_det_results = []

    for result in results:
        orig_dets = result["orig_detections"]
        adv_dets = result["adv_detections"]
        annotations.append(
            {
                "bboxes": np.asarray(orig_dets["bboxes"], dtype=np.float32).reshape(-1, 4),
                "labels": np.asarray(orig_dets["labels"], dtype=np.int64).reshape(-1),
            }
        )
        orig_det_results.append(dets_to_eval_format(orig_dets, n_classes))
        adv_det_results.append(dets_to_eval_format(adv_dets, n_classes))

    orig_mAP, _ = eval_map(
        orig_det_results,
        annotations,
        iou_thr=iou_thr,
        logger="silent",
    )
    adv_mAP, adv_details = eval_map(
        adv_det_results,
        annotations,
        iou_thr=iou_thr,
        logger="silent",
    )
    per_class_ap = [
        detail["ap"].item() if detail["ap"].size > 0 else 0.0
        for detail in adv_details
    ]
    map_result = {
        "orig_mAP": float(orig_mAP),
        "adv_mAP": float(adv_mAP),
        "per_class_ap": per_class_ap,
        "mAP_drop": float(orig_mAP - adv_mAP),
    }

    if verbose:
        print(f"\n[Pipeline] === Benign mAP (IoU={iou_thr}) ===")
        print(f"  Benign orig mAP : {orig_mAP:.4f}")
        print(f"  Benign adv mAP  : {adv_mAP:.4f}")
        print(f"  mAP Drop        : {map_result['mAP_drop']:.4f}")

    return map_result


def compute_gt_map(
    model: Any,
    results: list[dict[str, Any]],
    ann_file: str,
    iou_thr: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    n_classes = len(model.classes)
    model_h, model_w = model._img_size
    file_to_anns = _load_annotations(ann_file, model_h, model_w)
    annotations = []
    orig_det_results = []
    adv_det_results = []

    for result in results:
        image_name = os.path.basename(result["image_path"])
        base_name = os.path.splitext(image_name)[0]
        orig_dets = result["orig_detections"]
        adv_dets = result["adv_detections"]

        if base_name in file_to_anns:
            gt_bboxes = np.array(file_to_anns[base_name]["bboxes"], dtype=np.float32)
            gt_labels = np.array(file_to_anns[base_name]["labels"], dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

        annotations.append({"bboxes": gt_bboxes, "labels": gt_labels})
        orig_det_results.append(dets_to_eval_format(orig_dets, n_classes))
        adv_det_results.append(dets_to_eval_format(adv_dets, n_classes))

    orig_mAP, _ = eval_map(
        orig_det_results,
        annotations,
        iou_thr=iou_thr,
        logger="silent",
    )
    adv_mAP, adv_details = eval_map(
        adv_det_results,
        annotations,
        iou_thr=iou_thr,
        logger="silent",
    )
    per_class_ap = [
        detail["ap"].item() if detail["ap"].size > 0 else 0.0
        for detail in adv_details
    ]
    map_result = {
        "orig_mAP": float(orig_mAP),
        "adv_mAP": float(adv_mAP),
        "per_class_ap": per_class_ap,
        "mAP_drop": float(orig_mAP - adv_mAP),
    }

    if verbose:
        print(f"\n[Pipeline] === GT mAP (IoU={iou_thr}) ===")
        print(f"  GT orig mAP : {orig_mAP:.4f}")
        print(f"  GT adv mAP  : {adv_mAP:.4f}")
        print(f"  mAP Drop    : {map_result['mAP_drop']:.4f}")

    return map_result


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
        orig_h, orig_w = img_id_to_size[ann["image_id"]]
        scale_x = model_w / orig_w
        scale_y = model_h / orig_h
        file_to_anns[base_name]["bboxes"].append(
            [
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ]
        )
        file_to_anns[base_name]["labels"].append(cat_id_to_idx[ann["category_id"]])

    return file_to_anns
