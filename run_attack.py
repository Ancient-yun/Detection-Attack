"""Entry point for running adversarial attacks on mmdetection models.

Usage:
    # Single image attack with SparseEvo
    python run_attack.py \
        --config rtmdet_tiny_8xb32-300e_coco.py \
        --checkpoint rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
        --image demo/demo.jpg \
        --attack sparse_evo \
        --max-query 10000

    # Multiple images with PointWise
    python run_attack.py \
        --config rtmdet_tiny_8xb32-300e_coco.py \
        --checkpoint rtmdet_tiny_*.pth \
        --image-dir data/test_images/ \
        --attack pointwise \
        --max-query 5000

    # PointWise multi-pixel variant
    python run_attack.py \
        --config rtmdet_tiny_8xb32-300e_coco.py \
        --checkpoint rtmdet_tiny_*.pth \
        --image demo/demo.jpg \
        --attack pointwise_multi \
        --npix 196 \
        --max-query 10000
"""

import os
import glob
import time
import torch
from argparse import ArgumentParser

# PyTorch 2.6 compatibility patch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args, **{**kwargs, 'weights_only': False}
)

from adversarial_attack import DetectionAttackPipeline
from utils import build_output_dir, save_experiment_report


def parse_args():
    parser = ArgumentParser(
        description='Adversarial attack on mmdetection models'
    )
    parser.add_argument(
        '--model-type', default='mmdet',
        choices=['mmdet', 'yolov8'],
        help='Target model architecture framework (default: mmdet)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to mmdetection config file (Required for mmdet)',
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to model checkpoint (.pth or .pt)',
    )
    parser.add_argument(
        '--image', default=None,
        help='Path to a single input image',
    )
    parser.add_argument(
        '--image-dir', default=None,
        help='Path to directory of images for batch attack',
    )
    parser.add_argument(
        '--num-images', type=int, default=None,
        help='Limit number of images to attack (default: all)',
    )
    parser.add_argument(
        '--attack', default='sparse_evo',
        choices=['sparse_evo', 'pointwise', 'pointwise_multi',
                 'pointwise_multi_sched'],
        help='Attack method (default: sparse_evo)',
    )
    parser.add_argument(
        '--max-query', type=int, default=10000,
        help='Maximum number of model queries (default: 10000)',
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='Detection score threshold (default: 0.3)',
    )
    parser.add_argument(
        '--iou-thr', type=float, default=0.5,
        help='IoU threshold for attack success matching (default: 0.5)',
    )
    parser.add_argument(
        '--success-thr', type=float, default=0.5,
        help='Minimum attack success rate to declare success (default: 0.5)',
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device (default: cuda:0)',
    )
    parser.add_argument(
        '--output-dir', default='outputs/attack_results',
        help='Output directory (default: outputs/attack_results)',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility',
    )
    # SparseEvo-specific
    parser.add_argument(
        '--pop-size', type=int, default=10,
        help='[SparseEvo] Population size (default: 10)',
    )
    parser.add_argument(
        '--cr', type=float, default=0.9,
        help='[SparseEvo] Crossover rate (default: 0.9)',
    )
    parser.add_argument(
        '--mu', type=float, default=0.01,
        help='[SparseEvo] Mutation rate (default: 0.01)',
    )
    # PointWise-specific
    parser.add_argument(
        '--npix', type=float, default=0.1,
        help='[PointWise Multi] Pixels per group. If < 1.0, treated as ratio '
             'of total pixels (e.g., 0.1 = 10%%). Default: 0.1',
    )
    parser.add_argument(
        '--log-interval', type=int, default=50,
        help='Print progress every N queries (default: 50)',
    )
    parser.add_argument(
        '--ann-file', default=None,
        help='Path to COCO annotation JSON file or YOLO txt dir for GT eval',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if args.model_type == 'mmdet' and args.config is None:
        raise ValueError("--config is required when --model-type is mmdet")

    if args.image is None and args.image_dir is None:
        raise ValueError("Must specify --image or --image-dir")

    # Build output directory: result/[attack]/[model]/[date]/
    if args.output_dir == 'outputs/attack_results':
        # Default: use structured path
        if args.model_type == 'mmdet':
            output_dir = build_output_dir(args.attack, args.config)
        else:
            # Fallback for yolov8 checkpoint structure
            output_dir = build_output_dir(args.attack, args.checkpoint)
    else:
        output_dir = args.output_dir

    # Build attack kwargs
    attack_kwargs = {}
    if args.attack == 'sparse_evo':
        attack_kwargs = {
            'pop_size': args.pop_size,
            'cr': args.cr,
            'mu': args.mu,
            'seed': args.seed,
        }
    elif args.attack in ('pointwise_multi', 'pointwise_multi_sched'):
        attack_kwargs = {'npix': args.npix}

    # Create pipeline
    pipeline = DetectionAttackPipeline(
        model_type=args.model_type,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        attack_method=args.attack,
        device=args.device,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        success_thr=args.success_thr,
        log_interval=args.log_interval,
        **attack_kwargs,
    )

    # Collect image paths
    if args.image is not None:
        image_paths = [args.image]
    else:
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in extensions:
            # Search flat directory
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            # Search nested directories recursively (e.g., images/val/)
            image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
        
        # Remove duplicates
        image_paths = list(set(image_paths))
        image_paths.sort()
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {args.image_dir}"
            )

    # Limit number of images
    if args.num_images is not None:
        image_paths = image_paths[:args.num_images]

    print(f"[Main] {len(image_paths)} image(s) to attack")
    print(f"[Main] Attack method: {args.attack}")
    print(f"[Main] Max queries: {args.max_query}")
    print(f"[Main] Output dir: {output_dir}")

    # Run attacks
    start_time = time.time()

    if len(image_paths) == 1:
        results = [
            pipeline.run_attack(
                image_paths[0],
                max_query=args.max_query,
                seed=args.seed,
            )
        ]
    else:
        results = pipeline.run_batch_attack(
            image_paths,
            max_query=args.max_query,
            seed=args.seed,
        )

    elapsed_time = time.time() - start_time

    # Save results (images + CSV)
    pipeline.save_results(results, output_dir)

    # Compute mAP
    # Benign: model predictions on original images as GT
    benign_map = pipeline.compute_benign_map(results, iou_thr=args.iou_thr)

    # GT: real COCO annotations as GT (if annotation file provided)
    gt_map = None
    if args.ann_file:
        gt_map = pipeline.compute_gt_map(
            results, args.ann_file, iou_thr=args.iou_thr,
        )

    # Save comprehensive experiment report (txt)
    save_experiment_report(
        results, args, output_dir, elapsed_time,
        benign_map=benign_map, gt_map=gt_map,
    )

    print(f"[Main] Total time: {elapsed_time:.1f}s")
    print("[Main] Done!")


if __name__ == '__main__':
    main()
