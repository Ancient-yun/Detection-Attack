from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import torch

from .evaluation import compute_benign_map, compute_gt_map


def draw_detections(
    img_bgr: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    classes: list[str],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    vis = img_bgr.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
    ]

    for bbox, label, score in zip(bboxes, labels, scores):
        color = colors[int(label) % len(colors)]
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        cls_name = classes[label] if label < len(classes) else f"cls_{label}"
        text = f"{cls_name} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        cv2.rectangle(
            vis,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            vis,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    return vis


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image[:, :, ::-1].copy()


def save_attack_results(
    pipeline: Any,
    results: list[dict[str, Any]],
    output_dir: str,
    ann_file: str | None = None,
    save_snapshots: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"attack_results_{timestamp}.csv")
    _write_csv(csv_path, results)

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for result in results:
        _save_image_result(
            pipeline,
            result,
            image_dir,
            ann_file,
            save_snapshots=save_snapshots,
        )

    print(f"[Pipeline] Results saved to {output_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {image_dir}")
    print("  (orig / adv / delta / adv_raw / result.txt per image subdirectory)")
    if save_snapshots:
        print("  Snapshots: saved")
    else:
        print("  Snapshots: skipped")


def _write_csv(csv_path: str, results: list[dict[str, Any]]) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "n_queries",
                "l0_distance",
                "sparsity_ratio",
                "total_bboxes",
                "survived",
                "disappeared",
                "misclassified",
                "success_rate",
            ]
        )
        for result in results:
            match = result["match_result"]
            writer.writerow(
                [
                    result["image_path"],
                    result["n_queries"],
                    result["l0_distance"],
                    f"{result['sparsity_ratio']:.6f}",
                    match["total"],
                    match["survived"],
                    match["disappeared"],
                    match["misclassified"],
                    f"{result['success_rate']:.4f}",
                ]
            )


def _save_image_result(
    pipeline: Any,
    result: dict[str, Any],
    image_dir: str,
    ann_file: str | None,
    save_snapshots: bool,
) -> None:
    basename = os.path.splitext(os.path.basename(result["image_path"]))[0]
    per_image_dir = os.path.join(image_dir, basename)
    os.makedirs(per_image_dir, exist_ok=True)

    orig_dets = result["orig_detections"]
    adv_dets = result["adv_detections"]
    orig_bgr, adv_bgr, orig_bgr_up, adv_bgr_up, scale_x, scale_y = _load_visual_images(
        pipeline, result
    )
    orig_vis = draw_detections(
        orig_bgr_up,
        orig_dets["bboxes"],
        orig_dets["labels"],
        orig_dets["scores"],
        pipeline.model.classes,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    adv_vis = draw_detections(
        adv_bgr_up,
        adv_dets["bboxes"],
        adv_dets["labels"],
        adv_dets["scores"],
        pipeline.model.classes,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    cv2.imwrite(os.path.join(per_image_dir, "orig.png"), orig_vis)
    cv2.imwrite(os.path.join(per_image_dir, "adv.png"), adv_vis)
    cv2.imwrite(os.path.join(per_image_dir, "delta.png"), _delta_image(orig_bgr, adv_bgr))
    cv2.imwrite(os.path.join(per_image_dir, "adv_raw.png"), adv_bgr)

    snapshots = result.get("snapshots", {})
    if save_snapshots:
        _save_snapshots(pipeline, snapshots, per_image_dir)
    _write_text_result(
        pipeline,
        result,
        per_image_dir,
        ann_file,
        save_snapshots=save_snapshots,
    )


def _load_visual_images(
    pipeline: Any,
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    orig_img = cv2.imread(result["image_path"])
    scale_x, scale_y = 1.0, 1.0

    if orig_img is not None:
        orig_h, orig_w = orig_img.shape[:2]
        model_h, model_w = pipeline.model._img_size
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        orig_bgr = cv2.resize(orig_img, (model_w, model_h))
    else:
        orig_bgr = tensor_to_bgr(pipeline.load_image(result["image_path"]))

    adv_bgr = tensor_to_bgr(result["adv_image"])

    if orig_img is not None:
        orig_bgr_up = cv2.resize(
            orig_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        adv_bgr_up = cv2.resize(
            adv_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        orig_bgr_up = orig_bgr.copy()
        adv_bgr_up = adv_bgr.copy()

    return orig_bgr, adv_bgr, orig_bgr_up, adv_bgr_up, scale_x, scale_y


def _delta_image(orig_bgr: np.ndarray, adv_bgr: np.ndarray) -> np.ndarray:
    delta = cv2.absdiff(orig_bgr, adv_bgr)
    delta_max = delta.max()
    if delta_max > 0:
        return (delta.astype(np.float32) / delta_max * 255).astype(np.uint8)
    return delta


def _save_snapshots(
    pipeline: Any,
    snapshots: dict[int, torch.Tensor],
    per_image_dir: str,
) -> None:
    for query_num, snap_tensor in sorted(snapshots.items()):
        snap_bgr = tensor_to_bgr(snap_tensor)
        snap_dets = pipeline.model.predict(snap_tensor)
        snap_vis = draw_detections(
            snap_bgr,
            snap_dets["bboxes"],
            snap_dets["labels"],
            snap_dets["scores"],
            pipeline.model.classes,
        )
        cv2.imwrite(os.path.join(per_image_dir, f"query_{query_num}.png"), snap_vis)


def _write_text_result(
    pipeline: Any,
    result: dict[str, Any],
    per_image_dir: str,
    ann_file: str | None,
    save_snapshots: bool,
) -> None:
    txt_path = os.path.join(per_image_dir, "result.txt")
    orig_dets = result["orig_detections"]
    adv_dets = result["adv_detections"]
    match = result["match_result"]
    snapshots = result.get("snapshots", {})

    with open(txt_path, "w") as f:
        f.write(f"Image: {result['image_path']}\n")
        f.write(f"Queries used: {result['n_queries']}\n")
        f.write(f"L0 distance: {result['l0_distance']}\n")
        f.write(
            f"Sparsity ratio: {result['sparsity_ratio']:.6f} "
            f"({result['sparsity_ratio']:.2%})\n"
        )
        f.write("\n--- Detection Results ---\n")
        f.write(f"Original bboxes: {match['total']}\n")
        f.write(f"Survived: {match['survived']}\n")
        f.write(f"Disappeared: {match['disappeared']}\n")
        f.write(f"Misclassified: {match['misclassified']}\n")
        f.write(
            f"Attack success rate: {result['success_rate']:.4f} "
            f"({result['success_rate']:.2%})\n"
        )
        attack_result = "SUCCEEDED" if result["is_successful"] else "FAILED"
        f.write(f"Attack result: {attack_result}\n")

        benign_map = compute_benign_map(
            pipeline.model, [result], iou_thr=pipeline.model.iou_thr, verbose=False
        )
        f.write(f"\n--- Benign mAP (IoU={pipeline.model.iou_thr}) ---\n")
        f.write(f"  Orig: {benign_map['orig_mAP']:.4f}\n")
        f.write(f"  Adv : {benign_map['adv_mAP']:.4f}\n")
        f.write(f"  Drop: {benign_map['mAP_drop']:.4f}\n")

        if ann_file:
            gt_map = compute_gt_map(
                pipeline.model,
                [result],
                ann_file,
                iou_thr=pipeline.model.iou_thr,
                verbose=False,
            )
            f.write(f"\n--- GT mAP (IoU={pipeline.model.iou_thr}) ---\n")
            f.write(f"  Orig: {gt_map['orig_mAP']:.4f}\n")
            f.write(f"  Adv : {gt_map['adv_mAP']:.4f}\n")
            f.write(f"  Drop: {gt_map['mAP_drop']:.4f}\n")

        _write_detection_section(f, "Original Detections", orig_dets, pipeline.model.classes)
        _write_detection_section(f, "Adversarial Detections", adv_dets, pipeline.model.classes)
        if save_snapshots and snapshots:
            f.write("\n--- Snapshots ---\n")
            for query_num in sorted(snapshots.keys()):
                f.write(f"  query_{query_num}.png\n")


def _write_detection_section(
    f: Any,
    title: str,
    dets: dict[str, np.ndarray],
    classes: list[str],
) -> None:
    f.write(f"\n--- {title} ---\n")
    for index, (bbox, label, score) in enumerate(
        zip(dets["bboxes"], dets["labels"], dets["scores"])
    ):
        cls_name = classes[label] if label < len(classes) else f"cls_{label}"
        f.write(
            f"  [{index}] {cls_name}: "
            f"bbox={bbox.astype(int).tolist()}, score={score:.4f}\n"
        )
