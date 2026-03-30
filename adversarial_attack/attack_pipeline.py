"""End-to-end adversarial attack pipeline for mmdetection models.

Orchestrates the full attack flow: load model, load image,
generate starting point, run attack, evaluate results.
"""

import os
import json
import torch
import numpy as np
import cv2
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from mmdet.evaluation.functional import eval_map

from .model_adapter import MMDetModelAdapter, Yolov8ModelAdapter
from .sparse_evo import SpaEvoAtt
from .pointwise import PointWiseAtt
from .metrics import compute_l0, match_detections


class DetectionAttackPipeline:
    """Full pipeline for adversarial attacks on object detection models.

    Handles: image loading → benign inference → starting point generation
    → attack execution → result evaluation and saving.

    Args:
        model_type: 'mmdet' or 'yolov8'.
        config_path: Path to mmdetection config file.
        checkpoint_path: Path to model checkpoint (.pth or .pt).
        attack_method: 'sparse_evo' or 'pointwise'.
        device: CUDA device string.
        score_thr: Detection confidence threshold.
        iou_thr: IoU threshold for attack success matching.
        success_thr: Minimum success rate to declare attack successful.
        log_interval: Print progress every N queries.
        attack_kwargs: Additional kwargs for the attack class.
    """

    SUPPORTED_ATTACKS = {
        'sparse_evo', 'pointwise', 'pointwise_multi', 'pointwise_multi_sched',
    }

    def __init__(
        self,
        model_type: str = 'mmdet',
        config_path: str = None,
        checkpoint_path: str = None,
        attack_method: str = 'sparse_evo',
        device: str = 'cuda:0',
        score_thr: float = 0.3,
        iou_thr: float = 0.5,
        success_thr: float = 0.5,
        log_interval: int = 50,
        **attack_kwargs,
    ):
        if attack_method not in self.SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unsupported attack: {attack_method}. "
                f"Choose from {self.SUPPORTED_ATTACKS}"
            )

        self.attack_method = attack_method
        self.verbose = True
        self.device = device
        self.success_thr = success_thr
        self.log_interval = log_interval

        # Initialize model adapter
        print(f"[Pipeline] Loading {model_type} model from {checkpoint_path}...")
        if model_type == 'mmdet':
            self.model = MMDetModelAdapter(
                config_path, checkpoint_path,
                device=device, score_thr=score_thr, iou_thr=iou_thr,
                success_thr=success_thr,
            )
        elif model_type == 'yolov8':
            self.model = Yolov8ModelAdapter(
                checkpoint_path,
                device=device, score_thr=score_thr, iou_thr=iou_thr,
                success_thr=success_thr,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        print(
            f"[Pipeline] Model loaded. "
            f"Classes: {len(self.model.classes)}, "
            f"Input size: {self.model._img_size}"
        )

        # Initialize attack
        if attack_method == 'sparse_evo':
            self.attack = SpaEvoAtt(
                model=self.model,
                flag=False,  # Untargeted for detection
                log_interval=log_interval,
                **attack_kwargs,
            )
        elif attack_method in ('pointwise', 'pointwise_multi',
                                'pointwise_multi_sched'):
            self.attack = PointWiseAtt(
                model=self.model,
                flag=False,  # Untargeted for detection
                log_interval=log_interval,
            )

        self.attack_kwargs = attack_kwargs

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load image and convert to [1, C, H, W] tensor in [0, 1].

        Args:
            image_path: Path to the image file.

        Returns:
            Image tensor [1, 3, H, W], float, in [0, 1], on CUDA.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Resize to model input size
        h, w = self.model._img_size
        img = cv2.resize(img, (w, h))

        # BGR → RGB, [0,255] → [0,1], HWC → CHW
        img = img[:, :, ::-1].copy()
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).float().cuda()
        return tensor

    def generate_starting_point(
        self,
        oimg: torch.Tensor,
        olabel: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Generate an adversarial starting point using random noise.

        Tries various scales of salt-and-pepper noise until the model's
        detection output changes (at least one bbox disrupted).

        Args:
            oimg: Original image tensor [1, C, H, W].
            olabel: Original label (0 for detection adapter).
            seed: Random seed.

        Returns:
            Tuple of (starting_point_tensor, n_queries_used).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        c = oimg.shape[1]
        h = oimg.shape[2]
        w = oimg.shape[3]
        scales = [1, 2, 4, 8, 16, 32]
        n_queries = 0

        for scale in scales:
            # Generate block-wise random noise
            sh, sw = h // scale, w // scale
            noise = torch.rand(1, c, sh, sw).cuda()

            # Upscale to original size
            if scale > 1:
                noise = torch.nn.functional.interpolate(
                    noise, size=(h, w), mode='nearest'
                )
            else:
                noise = noise

            noise = noise.clamp(0, 1)
            n_queries += 1
            pred = self.model.predict_label(noise)

            if pred != olabel:
                if self.verbose:
                    print(
                        f"[Pipeline] Starting point found at scale={scale}, "
                        f"queries={n_queries}"
                    )
                return noise, n_queries

        # Fallback: pure random image
        for _ in range(100):
            noise = torch.rand_like(oimg).cuda()
            n_queries += 1
            pred = self.model.predict_label(noise)
            if pred != olabel:
                if self.verbose:
                    print(
                        f"[Pipeline] Starting point found (random), "
                        f"queries={n_queries}"
                    )
                return noise, n_queries

        if self.verbose:
            print("[Pipeline] WARNING: Could not find adversarial starting point")
        return torch.rand_like(oimg).cuda(), n_queries

    def run_attack(
        self,
        image_path: str,
        max_query: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict:
        """Run adversarial attack on a single image.

        Full pipeline: load → infer → start point → attack → evaluate.

        Args:
            image_path: Path to the input image.
            max_query: Maximum model queries for the attack.
            seed: Random seed.

        Returns:
            Dict with:
                - 'image_path': input path
                - 'adv_image': adversarial image tensor
                - 'n_queries': total queries used
                - 'l0_distance': final L0 distance
                - 'l0_trace': L0 distance over queries
                - 'success_rate': (disappeared+misclassified) / total
                - 'match_result': detailed detection matching result
                - 'orig_detections': original detection results
                - 'adv_detections': adversarial detection results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Pipeline] Attacking: {image_path}")
            print(f"{'='*60}")

        # 1. Load image
        oimg = self.load_image(image_path)
        if self.verbose:
            print(f"[Pipeline] Image loaded: {oimg.shape}")

        # 2. Benign inference (set reference)
        ref_dets = self.model.set_reference(oimg)
        n_orig_bboxes = len(ref_dets['bboxes'])
        if self.verbose:
            print(
                f"[Pipeline] Benign detection: {n_orig_bboxes} objects found"
            )

        if n_orig_bboxes == 0:
            if self.verbose:
                print("[Pipeline] No detections in original image, skipping.")
            return {
                'image_path': image_path,
                'adv_image': oimg,
                'n_queries': 0,
                'l0_distance': 0,
                'l0_trace': np.array([]),
                'sparsity_ratio': 0.0,
                'success_rate': 0.0,
                'is_successful': False,
                'match_result': match_detections(
                    np.array([]), np.array([]),
                    np.array([]), np.array([]),
                ),
                'orig_detections': ref_dets,
                'adv_detections': ref_dets,
            }

        if self.verbose:
            for i, (bbox, label) in enumerate(
                zip(ref_dets['bboxes'], ref_dets['labels'])
            ):
                cls_name = self.model.classes[label] if label < len(self.model.classes) else f"cls_{label}"
                print(f"  [{i}] {cls_name}: {bbox.astype(int)}")

        # 3. Generate starting point
        olabel = 0  # adapter convention: 0 = "original"
        tlabel = -1  # adapter convention: -1 = "attack success"
        start_img, start_queries = self.generate_starting_point(
            oimg, olabel, seed
        )

        # 4. Run attack
        if self.verbose:
            print(f"[Pipeline] Running {self.attack_method} attack...")
        total_queries = start_queries
        remaining_budget = max_query - start_queries
        snapshot_interval = max(1, max_query // 5)
        snapshots = {0: start_img.clone()}

        if self.attack_method == 'sparse_evo':
            adv_img, attack_queries, l0_trace, evo_snapshots = self.attack.evo_perturb(
                oimg, start_img, olabel, tlabel,
                max_query=remaining_budget,
                snapshot_interval=snapshot_interval,
            )
            snapshots.update(evo_snapshots)
        elif self.attack_method == 'pointwise':
            oimg_np = oimg.cpu().numpy()
            timg_np = start_img.cpu().numpy()
            adv_flat, attack_queries, l0_trace, pw_snaps = self.attack.pw_perturb(
                oimg_np, timg_np, olabel, tlabel, max_query=remaining_budget,
                snapshot_interval=snapshot_interval,
            )
            snapshots.update(pw_snaps)
            adv_img = torch.from_numpy(
                adv_flat.reshape(oimg.shape)
            ).float().cuda()
        elif self.attack_method == 'pointwise_multi':
            oimg_np = oimg.cpu().numpy()
            timg_np = start_img.cpu().numpy()
            npix = self.attack_kwargs.get('npix', 196)
            adv_flat, attack_queries, l0_trace, pw_snaps = \
                self.attack.pw_perturb_multiple(
                    oimg_np, timg_np, olabel, tlabel,
                    npix=npix, max_query=remaining_budget,
                    snapshot_interval=snapshot_interval,
                )
            snapshots.update(pw_snaps)
            adv_img = torch.from_numpy(
                adv_flat.reshape(oimg.shape)
            ).float().cuda()
        elif self.attack_method == 'pointwise_multi_sched':
            oimg_np = oimg.cpu().numpy()
            timg_np = start_img.cpu().numpy()
            npix = self.attack_kwargs.get('npix', 196)
            adv_flat, attack_queries, l0_trace, pw_snaps = \
                self.attack.pw_perturb_multiple_scheduling(
                    oimg_np, timg_np, olabel, tlabel,
                    npix=npix, max_query=remaining_budget,
                    snapshot_interval=snapshot_interval,
                )
            snapshots.update(pw_snaps)
            adv_img = torch.from_numpy(
                adv_flat.reshape(oimg.shape)
            ).float().cuda()

        total_queries += attack_queries

        # 5. Evaluate results
        adv_dets = self.model.predict(adv_img)
        final_l0 = compute_l0(oimg, adv_img)
        match_result = match_detections(
            ref_dets['bboxes'], ref_dets['labels'],
            adv_dets['bboxes'], adv_dets['labels'],
            iou_thr=self.model.iou_thr,
        )
        success_rate = (
            match_result['attack_success'] / match_result['total']
            if match_result['total'] > 0 else 0.0
        )

        # Compute sparsity ratio
        total_pixels = oimg.shape[2] * oimg.shape[3]  # H * W
        sparsity_ratio = final_l0 / total_pixels

        is_successful = success_rate >= self.success_thr

        if self.verbose:
            print(f"\n[Pipeline] === Results ===")
            print(f"  Queries used: {total_queries}")
            print(f"  L0 distance: {final_l0}")
            print(f"  Sparsity ratio: {sparsity_ratio:.4f} ({sparsity_ratio:.2%})")
            print(f"  Original bboxes: {match_result['total']}")
            print(f"  Survived: {match_result['survived']}")
            print(f"  Disappeared: {match_result['disappeared']}")
            print(f"  Misclassified: {match_result['misclassified']}")
            print(f"  Attack success rate: {success_rate:.2%}")
            print(f"  Success threshold: {self.success_thr:.2%}")
            print(f"  Attack {'SUCCEEDED' if is_successful else 'FAILED'}")

        return {
            'image_path': image_path,
            'adv_image': adv_img,
            'n_queries': total_queries,
            'l0_distance': final_l0,
            'l0_trace': l0_trace if isinstance(l0_trace, np.ndarray)
                        else l0_trace.cpu().numpy(),
            'success_rate': success_rate,
            'sparsity_ratio': sparsity_ratio,
            'is_successful': is_successful,
            'match_result': match_result,
            'orig_detections': ref_dets,
            'adv_detections': adv_dets,
            'snapshots': snapshots,
        }

    def run_batch_attack(
        self,
        image_paths: List[str],
        max_query: int = 10000,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Run attack on multiple images sequentially.

        Args:
            image_paths: List of image file paths.
            max_query: Maximum queries per image.
            seed: Random seed (incremented per image).

        Returns:
            List of result dicts (one per image).
        """
        self.verbose = False
        self.attack.verbose = False
        results = []
        n_success = 0
        pbar = tqdm(image_paths, desc="Attacking", unit="img")

        for i, path in enumerate(pbar):
            img_seed = seed + i if seed is not None else None
            result = self.run_attack(path, max_query, img_seed)
            results.append(result)

            if result['is_successful']:
                n_success += 1

            # Update tqdm postfix with running stats
            avg_asr = np.mean([r['success_rate'] for r in results])
            avg_l0 = np.mean([r['l0_distance'] for r in results])
            pbar.set_postfix(
                ASR=f"{avg_asr:.0%}",
                L0=f"{avg_l0:.0f}",
                ok=f"{n_success}/{i+1}",
            )

        pbar.close()
        self.verbose = True
        self.attack.verbose = True

        # Print final summary
        if results:
            avg_rate = np.mean([r['success_rate'] for r in results])
            avg_queries = np.mean([r['n_queries'] for r in results])
            avg_l0 = np.mean([r['l0_distance'] for r in results])
            print(f"\n{'='*60}")
            print(f"[Pipeline] Batch Summary ({len(results)} images)")
            print(f"  Avg success rate: {avg_rate:.2%}")
            print(f"  Avg queries: {avg_queries:.0f}")
            print(f"  Avg L0: {avg_l0:.0f}")
            print(f"{'='*60}")

        return results

    @staticmethod
    def _draw_detections(
        img_bgr: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        classes: list,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on an image.

        Args:
            img_bgr: BGR image [H, W, 3], uint8.
            bboxes: Bboxes [N, 4].
            labels: Class labels [N].
            scores: Confidence scores [N].
            classes: List of class names.

        Returns:
            Image with bboxes drawn (copy).
        """
        vis = img_bgr.copy()
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
        ]

        for i, (bbox, label, score) in enumerate(
            zip(bboxes, labels, scores)
        ):
            color = colors[int(label) % len(colors)]
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            cls_name = classes[label] if label < len(classes) else f'cls_{label}'
            text = f'{cls_name} {score:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(vis, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert [1, C, H, W] tensor in [0,1] to BGR uint8 [H, W, 3]."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return img[:, :, ::-1].copy()

    def save_results(
        self,
        results: List[Dict],
        output_dir: str,
    ) -> None:
        """Save attack results to disk.

        Saves:
        - Original image with bboxes
        - Adversarial image with bboxes
        - Delta image (pixel difference heatmap)
        - Raw adversarial image
        - Summary CSV with per-image metrics

        Args:
            results: List of result dicts from run_attack.
            output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV summary
        csv_path = os.path.join(output_dir, f"attack_results_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_path', 'n_queries', 'l0_distance', 'sparsity_ratio',
                'total_bboxes', 'survived', 'disappeared',
                'misclassified', 'success_rate',
            ])
            for r in results:
                writer.writerow([
                    r['image_path'],
                    r['n_queries'],
                    r['l0_distance'],
                    f"{r['sparsity_ratio']:.6f}",
                    r['match_result']['total'],
                    r['match_result']['survived'],
                    r['match_result']['disappeared'],
                    r['match_result']['misclassified'],
                    f"{r['success_rate']:.4f}",
                ])

        # Save images with bbox visualization
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        for r in results:
            basename = os.path.splitext(
                os.path.basename(r['image_path'])
            )[0]

            # Create per-image subdirectory: images/이미지이름/
            per_img_dir = os.path.join(img_dir, basename)
            os.makedirs(per_img_dir, exist_ok=True)

            orig_dets = r['orig_detections']
            adv_dets = r['adv_detections']
            match = r['match_result']

            # Convert tensors to BGR images
            # Re-load original image for visualization
            orig_img = cv2.imread(r['image_path'])
            if orig_img is not None:
                h, w = self.model._img_size
                orig_bgr = cv2.resize(orig_img, (w, h))
            else:
                orig_bgr = self._tensor_to_bgr(
                    self.load_image(r['image_path'])
                )
            adv_bgr = self._tensor_to_bgr(r['adv_image'])

            # Draw bboxes (no title text on images)
            orig_vis = self._draw_detections(
                orig_bgr, orig_dets['bboxes'], orig_dets['labels'],
                orig_dets['scores'], self.model.classes,
            )
            adv_vis = self._draw_detections(
                adv_bgr, adv_dets['bboxes'], adv_dets['labels'],
                adv_dets['scores'], self.model.classes,
            )

            # Save individual images into per-image subdirectory
            cv2.imwrite(
                os.path.join(per_img_dir, "orig.png"),
                orig_vis,
            )
            cv2.imwrite(
                os.path.join(per_img_dir, "adv.png"),
                adv_vis,
            )

            # Save delta image (pixel difference heatmap)
            delta = cv2.absdiff(orig_bgr, adv_bgr)
            # Amplify for visibility: scale to full [0, 255] range
            delta_max = delta.max()
            if delta_max > 0:
                delta_vis = (delta.astype(np.float32) / delta_max * 255).astype(np.uint8)
            else:
                delta_vis = delta
            cv2.imwrite(
                os.path.join(per_img_dir, "delta.png"),
                delta_vis,
            )

            # Save raw adversarial image (no bboxes)
            cv2.imwrite(
                os.path.join(per_img_dir, "adv_raw.png"),
                adv_bgr,
            )

            # Save snapshot images at query intervals
            snapshots = r.get('snapshots', {})
            if snapshots:
                for query_num, snap_tensor in sorted(snapshots.items()):
                    snap_bgr = self._tensor_to_bgr(snap_tensor)
                    snap_dets = self.model.predict(snap_tensor)
                    snap_vis = self._draw_detections(
                        snap_bgr, snap_dets['bboxes'], snap_dets['labels'],
                        snap_dets['scores'], self.model.classes,
                    )
                    cv2.imwrite(
                        os.path.join(per_img_dir, f"query_{query_num}.png"),
                        snap_vis,
                    )

            # Save per-image result text file
            txt_path = os.path.join(per_img_dir, "result.txt")
            with open(txt_path, 'w') as f:
                f.write(f"Image: {r['image_path']}\n")
                f.write(f"Queries used: {r['n_queries']}\n")
                f.write(f"L0 distance: {r['l0_distance']}\n")
                f.write(f"Sparsity ratio: {r['sparsity_ratio']:.6f} ({r['sparsity_ratio']:.2%})\n")
                f.write(f"\n--- Detection Results ---\n")
                f.write(f"Original bboxes: {match['total']}\n")
                f.write(f"Survived: {match['survived']}\n")
                f.write(f"Disappeared: {match['disappeared']}\n")
                f.write(f"Misclassified: {match['misclassified']}\n")
                f.write(f"Attack success rate: {r['success_rate']:.4f} ({r['success_rate']:.2%})\n")
                f.write(f"Attack result: {'SUCCEEDED' if r['is_successful'] else 'FAILED'}\n")
                f.write(f"\n--- Original Detections ---\n")
                for i, (bbox, label, score) in enumerate(
                    zip(orig_dets['bboxes'], orig_dets['labels'], orig_dets['scores'])
                ):
                    cls_name = self.model.classes[label] if label < len(self.model.classes) else f'cls_{label}'
                    f.write(f"  [{i}] {cls_name}: bbox={bbox.astype(int).tolist()}, score={score:.4f}\n")
                f.write(f"\n--- Adversarial Detections ---\n")
                for i, (bbox, label, score) in enumerate(
                    zip(adv_dets['bboxes'], adv_dets['labels'], adv_dets['scores'])
                ):
                    cls_name = self.model.classes[label] if label < len(self.model.classes) else f'cls_{label}'
                    f.write(f"  [{i}] {cls_name}: bbox={bbox.astype(int).tolist()}, score={score:.4f}\n")
                if snapshots:
                    f.write(f"\n--- Snapshots ---\n")
                    for query_num in sorted(snapshots.keys()):
                        f.write(f"  query_{query_num}.png\n")

        print(f"[Pipeline] Results saved to {output_dir}")
        print(f"  CSV: {csv_path}")
        print(f"  Images: {img_dir}")
        print(f"  (orig / adv / delta / adv_raw / snapshots / result.txt per image subdirectory)")

    def _dets_to_eval_format(
        self,
        dets: Dict[str, np.ndarray],
        n_classes: int,
    ) -> List[np.ndarray]:
        """Convert detection dict to eval_map per-class format.

        Args:
            dets: Dict with 'bboxes', 'labels', 'scores'.
            n_classes: Number of classes.

        Returns:
            List of ndarray (n, 5) per class.
        """
        per_class = []
        for c in range(n_classes):
            mask = dets['labels'] == c
            if mask.any():
                cls_bboxes = dets['bboxes'][mask]
                cls_scores = dets['scores'][mask].reshape(-1, 1)
                per_class.append(
                    np.hstack([cls_bboxes, cls_scores]).astype(np.float32)
                )
            else:
                per_class.append(np.zeros((0, 5), dtype=np.float32))
        return per_class

    def compute_benign_map(
        self,
        results: List[Dict],
        iou_thr: float = 0.5,
    ) -> Dict:
        """Compute mAP using benign model predictions as GT.

        Measures how much the adversarial attack degrades detections
        relative to the model's own benign predictions.

        Args:
            results: List of result dicts from run_attack.
            iou_thr: IoU threshold for mAP evaluation.

        Returns:
            Dict with orig_mAP, adv_mAP, per_class_ap, mAP_drop.
        """
        n_classes = len(self.model.classes)

        annotations = []
        orig_det_results = []
        adv_det_results = []

        for r in results:
            orig_dets = r['orig_detections']
            adv_dets = r['adv_detections']

            annotations.append({
                'bboxes': np.asarray(orig_dets['bboxes'], dtype=np.float32).reshape(-1, 4),
                'labels': np.asarray(orig_dets['labels'], dtype=np.int64).reshape(-1),
            })
            orig_det_results.append(self._dets_to_eval_format(orig_dets, n_classes))
            adv_det_results.append(self._dets_to_eval_format(adv_dets, n_classes))

        orig_mAP, _ = eval_map(
            orig_det_results, annotations,
            iou_thr=iou_thr, logger='silent',
        )
        adv_mAP, adv_details = eval_map(
            adv_det_results, annotations,
            iou_thr=iou_thr, logger='silent',
        )

        per_class_ap = [
            d['ap'].item() if d['ap'].size > 0 else 0.0
            for d in adv_details
        ]

        result = {
            'orig_mAP': float(orig_mAP),
            'adv_mAP': float(adv_mAP),
            'per_class_ap': per_class_ap,
            'mAP_drop': float(orig_mAP - adv_mAP),
        }

        print(f"\n[Pipeline] === Benign mAP (IoU={iou_thr}) ===")
        print(f"  Benign orig mAP : {orig_mAP:.4f}")
        print(f"  Benign adv mAP  : {adv_mAP:.4f}")
        print(f"  mAP Drop        : {result['mAP_drop']:.4f}")

        return result

    def compute_gt_map(
        self,
        results: List[Dict],
        ann_file: str,
        iou_thr: float = 0.5,
    ) -> Dict:
        """Compute mAP using real GT annotations.

        Measures actual detection performance before/after attack
        against ground truth labels (COCO JSON or YOLO format directory).

        Args:
            results: List of result dicts from run_attack.
            ann_file: Path to COCO annotation JSON or YOLO txt dir.
            iou_thr: IoU threshold for mAP evaluation.

        Returns:
            Dict with orig_mAP, adv_mAP, per_class_ap, mAP_drop.
        """
        n_classes = len(self.model.classes)
        model_h, model_w = self.model._img_size
        file_to_anns = {}

        if os.path.isdir(ann_file):
            # Assume YOLO format directory (e.g., labels/val)
            for txt_name in os.listdir(ann_file):
                if not txt_name.endswith('.txt'):
                    continue
                base_name = os.path.splitext(txt_name)[0]
                file_to_anns[base_name] = {'bboxes': [], 'labels': []}

                txt_path = os.path.join(ann_file, txt_name)
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])

                            # YOLO matches: box bounds in [0, 1] normalized scale
                            # Convert directly to model_w, model_h coordinates since gt_map
                            # uses model_w/model_h coordinate space inside eval_map
                            x1 = (xc - w / 2) * model_w
                            y1 = (yc - h / 2) * model_h
                            x2 = (xc + w / 2) * model_w
                            y2 = (yc + h / 2) * model_h

                            file_to_anns[base_name]['bboxes'].append([x1, y1, x2, y2])
                            file_to_anns[base_name]['labels'].append(cls_id)
        else:
            # Assume COCO JSON
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)

            # Build category_id -> contiguous index mapping
            cat_ids = sorted([c['id'] for c in coco_data['categories']])
            cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

            img_id_to_file = {
                img['id']: img['file_name'] for img in coco_data['images']
            }
            img_id_to_size = {
                img['id']: (img['height'], img['width']) for img in coco_data['images']
            }
            for ann in coco_data['annotations']:
                if ann.get('iscrowd', 0):
                    continue
                fname = img_id_to_file[ann['image_id']]
                base_name = os.path.splitext(os.path.basename(fname))[0]
                if base_name not in file_to_anns:
                    file_to_anns[base_name] = {'bboxes': [], 'labels': []}

                # COCO bbox: [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']

                # Scale from orig size directly to model input size here
                orig_h, orig_w = img_id_to_size[ann['image_id']]
                scale_x = model_w / orig_w
                scale_y = model_h / orig_h

                file_to_anns[base_name]['bboxes'].append([
                    x * scale_x,
                    y * scale_y,
                    (x + w) * scale_x,
                    (y + h) * scale_y
                ])
                file_to_anns[base_name]['labels'].append(cat_id_to_idx[ann['category_id']])

        annotations = []
        orig_det_results = []
        adv_det_results = []

        for r in results:
            fname = os.path.basename(r['image_path'])
            base_name = os.path.splitext(fname)[0]
            orig_dets = r['orig_detections']
            adv_dets = r['adv_detections']

            if base_name in file_to_anns:
                gt_bboxes = np.array(file_to_anns[base_name]['bboxes'], dtype=np.float32)
                gt_labels = np.array(file_to_anns[base_name]['labels'], dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.zeros((0,), dtype=np.int64)

            annotations.append({
                'bboxes': gt_bboxes,
                'labels': gt_labels,
            })
            orig_det_results.append(self._dets_to_eval_format(orig_dets, n_classes))
            adv_det_results.append(self._dets_to_eval_format(adv_dets, n_classes))

        orig_mAP, _ = eval_map(
            orig_det_results, annotations,
            iou_thr=iou_thr, logger='silent',
        )
        adv_mAP, adv_details = eval_map(
            adv_det_results, annotations,
            iou_thr=iou_thr, logger='silent',
        )

        per_class_ap = [
            d['ap'].item() if d['ap'].size > 0 else 0.0
            for d in adv_details
        ]

        result = {
            'orig_mAP': float(orig_mAP),
            'adv_mAP': float(adv_mAP),
            'per_class_ap': per_class_ap,
            'mAP_drop': float(orig_mAP - adv_mAP),
        }

        print(f"\n[Pipeline] === GT mAP (IoU={iou_thr}) ===")
        print(f"  GT orig mAP : {orig_mAP:.4f}")
        print(f"  GT adv mAP  : {adv_mAP:.4f}")
        print(f"  mAP Drop    : {result['mAP_drop']:.4f}")

        return result
