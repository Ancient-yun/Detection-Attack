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
import pickle
import time
import torch
from argparse import ArgumentParser
from tqdm import tqdm

# PyTorch 2.6 compatibility patch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args, **{**kwargs, 'weights_only': False}
)

from adversarial_attack import DetectionAttackPipeline
from adversarial_attack.class_mapping import SUPPORTED_CLASS_MAPPINGS
from adversarial_attack.utils import build_output_dir, save_experiment_report
from adversarial_attack.utils.image_selection import (
    select_image_paths,
    validate_sample_manifest,
    write_sample_manifest,
)


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
        '--num-images', default='all',
        help='Limit number of images to attack ("all" or an integer, default: "all")',
    )
    parser.add_argument(
        '--sample-strategy',
        default='first',
        choices=['first', 'random'],
        help='Image selection strategy for --image-dir (default: first)',
    )
    parser.add_argument(
        '--sample-seed',
        type=int,
        default=None,
        help='Random seed for --sample-strategy random. Defaults to --seed.',
    )
    parser.add_argument(
        '--sample-manifest',
        default=None,
        help=(
            'Path to write selected image manifest. Defaults to '
            'output-dir/sample_manifest.json.'
        ),
    )
    parser.add_argument(
        '--dataset-name', default='dataset',
        help='Name of the dataset for organizing results (default: dataset)',
    )
    parser.add_argument(
        '--class-map',
        default='none',
        choices=SUPPORTED_CLASS_MAPPINGS,
        help=(
            'Map detector outputs into another dataset label space. Use '
            'coco-to-voc for COCO checkpoints on Pascal VOC images.'
        ),
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
    # Deprecated and ignored: the attack is now tensor-only for every model
    # (original-size image in, preprocessing inside the model). Kept so existing
    # launch scripts that still pass these flags do not error.
    parser.add_argument(
        '--mmdet-inference-mode', default=None,
        help='Deprecated and ignored (attack is tensor-only).',
    )
    parser.add_argument(
        '--yolo-inference-mode', default=None,
        help='Deprecated and ignored (attack is tensor-only).',
    )
    parser.add_argument(
        '--output-dir', default='outputs/attack_results',
        help='Output directory (default: outputs/attack_results)',
    )
    parser.add_argument(
        '--save-snapshots',
        dest='save_snapshots',
        action='store_true',
        default=False,
        help='Save intermediate query_*.png snapshot visualizations.',
    )
    parser.add_argument(
        '--no-save-snapshots',
        dest='save_snapshots',
        action='store_false',
        help='Skip intermediate query_*.png snapshots while keeping final images.',
    )
    parser.add_argument(
        '--resume-partial', action='store_true',
        help=(
            'Resume a long run from output-dir/partial_results.pkl and save '
            'one lightweight checkpoint after each completed image.'
        ),
    )
    parser.add_argument(
        '--partial-file', default=None,
        help='Optional partial checkpoint path for --resume-partial.',
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


def _norm_image_path(path):
    return os.path.abspath(os.path.normpath(path))


def _strip_result_for_partial(result):
    """Keep enough data to finish reports while avoiding huge checkpoints."""
    stripped = {}
    for key, value in result.items():
        if key in ('snapshots', 'l0_trace'):
            continue
        if key == 'adv_image' and isinstance(value, torch.Tensor):
            stripped['adv_image_uint8'] = (
                value.detach().cpu().mul(255).round().clamp(0, 255)
                .to(torch.uint8)
            )
            continue
        if isinstance(value, torch.Tensor):
            stripped[key] = value.detach().cpu()
        else:
            stripped[key] = value
    return stripped


def _restore_partial_result(result):
    if 'adv_image' not in result and 'adv_image_uint8' in result:
        restored = dict(result)
        restored['adv_image'] = restored.pop('adv_image_uint8').float() / 255.0
        return restored
    return result


def _load_partial_results(partial_path):
    if not os.path.exists(partial_path):
        return []
    with open(partial_path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        results = payload.get('results', [])
    else:
        results = payload
    return [_restore_partial_result(r) for r in results]


def _save_partial_results(partial_path, results, args):
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    tmp_path = partial_path + '.tmp'
    payload = {
        'args': vars(args),
        'results': [_strip_result_for_partial(r) for r in results],
        'updated_at': time.time(),
    }
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, partial_path)


def _run_attacks_with_partial_resume(pipeline, image_paths, args, partial_path):
    loaded_results = _load_partial_results(partial_path)
    results_by_path = {
        _norm_image_path(r['image_path']): r
        for r in loaded_results
        if isinstance(r, dict) and 'image_path' in r
    }
    if results_by_path:
        print(
            f"[Partial] Loaded {len(results_by_path)} completed image(s): "
            f"{partial_path}"
        )

    pbar = tqdm(image_paths, desc="Attacking", unit="img")
    for i, path in enumerate(pbar):
        key = _norm_image_path(path)
        if key in results_by_path:
            pbar.set_postfix(done=f"{len(results_by_path)}/{len(image_paths)}")
            continue

        img_seed = args.seed + i if args.seed is not None else None
        result = pipeline.run_attack(path, max_query=args.max_query, seed=img_seed)
        results_by_path[key] = _strip_result_for_partial(result)

        ordered_partial = [
            results_by_path[_norm_image_path(p)]
            for p in image_paths
            if _norm_image_path(p) in results_by_path
        ]
        _save_partial_results(partial_path, ordered_partial, args)
        pbar.set_postfix(done=f"{len(results_by_path)}/{len(image_paths)}")

    return [
        _restore_partial_result(results_by_path[_norm_image_path(p)])
        for p in image_paths
        if _norm_image_path(p) in results_by_path
    ]


def main():
    args = parse_args()

    # Validate input
    if args.model_type == 'mmdet' and args.config is None:
        raise ValueError("--config is required when --model-type is mmdet")

    if args.image is None and args.image_dir is None:
        raise ValueError("Must specify --image or --image-dir")

    # Build output directory: result/[attack]/[dataset_name]/[model]/[date]/
    if args.output_dir == 'outputs/attack_results':
        # Default: use structured path
        if args.model_type == 'mmdet':
            output_dir = build_output_dir(args.attack, args.dataset_name, args.config)
        else:
            # Fallback for yolov8 checkpoint structure
            output_dir = build_output_dir(args.attack, args.dataset_name, args.checkpoint)
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

    sample_manifest_path = args.sample_manifest or os.path.join(
        output_dir, 'sample_manifest.json'
    )

    # Collect image paths
    if args.image is not None:
        image_paths = [args.image]
        sample_selection = None
    else:
        sample_seed = args.sample_seed
        if sample_seed is None:
            sample_seed = args.seed
        sample_selection = select_image_paths(
            args.image_dir,
            num_images=args.num_images,
            sample_strategy=args.sample_strategy,
            sample_seed=sample_seed,
        )
        image_paths = sample_selection.selected_image_paths
        if args.resume_partial:
            validate_sample_manifest(sample_manifest_path, sample_selection)
        write_sample_manifest(sample_manifest_path, sample_selection)

    print(f"[Main] {len(image_paths)} image(s) to attack")
    print(f"[Main] Attack method: {args.attack}")
    print(f"[Main] Class mapping: {args.class_map}")
    print(f"[Main] Max queries: {args.max_query}")
    print(f"[Main] Output dir: {output_dir}")
    if sample_selection is not None:
        print(f"[Main] Sample strategy: {sample_selection.sample_strategy}")
        print(f"[Main] Sample seed: {sample_selection.sample_seed}")
        print(f"[Main] Sample manifest: {sample_manifest_path}")
    if args.resume_partial:
        partial_path = args.partial_file or os.path.join(
            output_dir, 'partial_results.pkl'
        )
        print(f"[Main] Partial resume: {partial_path}")

    # Create pipeline after sample selection so invalid sampling fails fast.
    pipeline = DetectionAttackPipeline(
        model_type=args.model_type,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        attack_method=args.attack,
        device=args.device,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        success_thr=args.success_thr,
        class_mapping=args.class_map,
        log_interval=args.log_interval,
        **attack_kwargs,
    )

    # Run attacks
    start_time = time.time()

    if args.resume_partial:
        results = _run_attacks_with_partial_resume(
            pipeline, image_paths, args, partial_path,
        )
    elif len(image_paths) == 1:
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
    pipeline.save_results(
        results,
        output_dir,
        ann_file=args.ann_file,
        save_snapshots=args.save_snapshots,
    )

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
