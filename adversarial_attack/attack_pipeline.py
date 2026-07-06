"""End-to-end adversarial attack pipeline for mmdetection models.

Orchestrates the full attack flow: load model, load image,
generate starting point, run attack, evaluate results.
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .model_adapter import MMDetModelAdapter, Yolov8ModelAdapter
from .sparse_evo import SpaEvoAtt
from .pointwise import PointWiseAtt
from .metrics import compute_l0, match_detections
from .evaluation import (
    compute_benign_map as evaluate_benign_map,
    compute_gt_map as evaluate_gt_map,
    dets_to_eval_format,
)
from .result_writer import draw_detections, save_attack_results, tensor_to_bgr


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
        mmdet_inference_mode: MMDetection inference path. 'legacy' uses
            inference_detector; 'direct_tensor' avoids CPU image conversion.
        yolo_inference_mode: YOLOv8 inference path. 'legacy' preserves
            Ultralytics high-level prediction semantics; 'direct_tensor'
            avoids CPU image conversion but is not bitwise-equivalent.
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
        mmdet_inference_mode: str = 'legacy',
        yolo_inference_mode: str = 'legacy',
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
                inference_mode=mmdet_inference_mode,
            )
        elif model_type == 'yolov8':
            self.model = Yolov8ModelAdapter(
                checkpoint_path,
                device=device, score_thr=score_thr, iou_thr=iou_thr,
                success_thr=success_thr,
                inference_mode=yolo_inference_mode,
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

    def _target_device(self) -> torch.device:
        return torch.device(getattr(self.model, "device", self.device))

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load image and convert to [1, C, H, W] tensor in [0, 1].

        Args:
            image_path: Path to the image file.

        Returns:
            Image tensor [1, 3, H, W], float, in [0, 1].
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
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self._target_device())
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
            noise = torch.rand(1, c, sh, sw, dtype=oimg.dtype, device=oimg.device)

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
            noise = torch.rand_like(oimg)
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
        return torch.rand_like(oimg), n_queries

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
            ).float().to(oimg.device)
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
            ).float().to(oimg.device)
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
            ).float().to(oimg.device)

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
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> np.ndarray:
        return draw_detections(
            img_bgr,
            bboxes,
            labels,
            scores,
            classes,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_bgr(tensor)

    def save_results(
        self,
        results: List[Dict],
        output_dir: str,
        ann_file: str = None,
        save_snapshots: bool = False,
    ) -> None:
        save_attack_results(
            self,
            results,
            output_dir,
            ann_file,
            save_snapshots=save_snapshots,
        )

    def _dets_to_eval_format(
        self,
        dets: Dict[str, np.ndarray],
        n_classes: int,
    ) -> List[np.ndarray]:
        return dets_to_eval_format(dets, n_classes)

    def compute_benign_map(
        self,
        results: List[Dict],
        iou_thr: float = 0.5,
        verbose: bool = True,
    ) -> Dict:
        return evaluate_benign_map(self.model, results, iou_thr, verbose)

    def compute_gt_map(
        self,
        results: List[Dict],
        ann_file: str,
        iou_thr: float = 0.5,
        verbose: bool = True,
    ) -> Dict:
        return evaluate_gt_map(self.model, results, ann_file, iou_thr, verbose)
