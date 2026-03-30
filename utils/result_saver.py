"""Experiment result saving utilities.

Handles structured output directory creation and
comprehensive experiment report generation.
"""

import os
import numpy as np
from datetime import datetime


def build_output_dir(attack_method, config_path):
    """Build output directory: result/[attack_method]/[model_name]/[date]/"""
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("result", attack_method, model_name, date_str)


def save_experiment_report(results, args, output_dir, elapsed_time,
                           benign_map=None, gt_map=None):
    """Save comprehensive experiment results to a txt report file.

    Saves:
        [1] Experiment configuration
        [2] Overall summary (success/fail counts, image-level ASR)
        [3] Aggregate metrics with statistics (mean/std/min/max/median)
        [4] Benign mAP (model predictions as GT)
        [5] GT mAP (real COCO annotations as GT)
        [6] Efficiency metrics
        [7] Per-image detailed results table

    Args:
        results: List of result dicts from attack pipeline.
        args: Parsed CLI arguments.
        output_dir: Output directory path.
        elapsed_time: Total elapsed time in seconds.
        benign_map: Benign mAP results (model predictions as GT).
        gt_map: GT mAP results (real COCO annotations as GT).

    Returns:
        Path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "experiment_report.txt")

    n_images = len(results)
    n_success = sum(1 for r in results if r['is_successful'])
    n_skip = sum(1 for r in results if r['n_queries'] == 0)
    valid_results = [r for r in results if r['n_queries'] > 0]

    # Aggregate metrics
    if valid_results:
        queries_list = [r['n_queries'] for r in valid_results]
        l0_list = [r['l0_distance'] for r in valid_results]
        sparsity_list = [r['sparsity_ratio'] for r in valid_results]
        sr_list = [r['success_rate'] for r in valid_results]
        survived_list = [r['match_result']['survived'] for r in valid_results]
        disappeared_list = [r['match_result']['disappeared'] for r in valid_results]
        misclassified_list = [r['match_result']['misclassified'] for r in valid_results]
        total_bbox_list = [r['match_result']['total'] for r in valid_results]
    else:
        queries_list = l0_list = sparsity_list = sr_list = [0]
        survived_list = disappeared_list = misclassified_list = total_bbox_list = [0]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  ADVERSARIAL ATTACK EXPERIMENT REPORT\n")
        f.write("=" * 70 + "\n\n")

        # [1] Experiment Configuration
        _write_config_section(f, args, n_images, elapsed_time)

        # [2] Overall Summary
        _write_summary_section(f, n_images, n_skip, n_success, valid_results)

        # [3] Aggregate Metrics
        _write_aggregate_section(
            f, queries_list, l0_list, sparsity_list, sr_list,
            total_bbox_list, survived_list, disappeared_list, misclassified_list,
        )

        # [4] Benign mAP (model predictions as GT)
        section_num = 4
        if benign_map is not None:
            _write_map_section(f, benign_map, args, section_num,
                               "Benign mAP (Model Predictions as GT)")
            section_num += 1

        # [5] GT mAP (real COCO annotations as GT)
        if gt_map is not None:
            _write_map_section(f, gt_map, args, section_num,
                               "GT mAP (COCO Annotations as GT)")
            section_num += 1

        # Efficiency Metrics
        _write_efficiency_section(f, valid_results, queries_list, elapsed_time,
                                  section_num)
        section_num += 1

        # Per-Image Results
        _write_per_image_section(f, results, section_num)

        f.write("\n" + "=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[Main] Experiment report saved: {report_path}")
    return report_path


# ====================== Internal Helpers ======================

def _write_stat(f, name, values, fmt=".2f", is_pct=False):
    """Write a single statistic line (mean/std/min/max/median)."""
    arr = np.array(values, dtype=float)
    suffix = "%" if is_pct else ""
    mult = 100 if is_pct else 1
    f.write(
        f"  {name:<25s}: "
        f"Mean={arr.mean()*mult:{fmt}}{suffix}  "
        f"Std={arr.std()*mult:{fmt}}{suffix}  "
        f"Min={arr.min()*mult:{fmt}}{suffix}  "
        f"Max={arr.max()*mult:{fmt}}{suffix}  "
        f"Median={np.median(arr)*mult:{fmt}}{suffix}\n"
    )


def _write_config_section(f, args, n_images, elapsed_time):
    """[1] Experiment Configuration section."""
    f.write("[1] Experiment Configuration\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Date/Time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Config            : {args.config}\n")
    f.write(f"  Checkpoint        : {args.checkpoint}\n")
    f.write(f"  Model Name        : {os.path.splitext(os.path.basename(args.config))[0]}\n")
    f.write(f"  Attack Method     : {args.attack}\n")
    f.write(f"  Device            : {args.device}\n")
    f.write(f"  Max Queries       : {args.max_query}\n")
    f.write(f"  Score Threshold   : {args.score_thr}\n")
    f.write(f"  IoU Threshold     : {args.iou_thr}\n")
    f.write(f"  Success Threshold : {args.success_thr}\n")
    f.write(f"  Random Seed       : {args.seed}\n")
    if args.attack == 'sparse_evo':
        f.write(f"  Population Size   : {args.pop_size}\n")
        f.write(f"  Crossover Rate    : {args.cr}\n")
        f.write(f"  Mutation Rate     : {args.mu}\n")
    elif args.attack == 'pointwise_multi':
        f.write(f"  Pixels per Group  : {args.npix}\n")
    dataset_name = args.image_dir if args.image_dir else args.image
    f.write(f"  Dataset/Input     : {dataset_name}\n")
    f.write(f"  Total Images      : {n_images}\n")
    f.write(f"  Elapsed Time      : {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)\n")
    f.write("\n")


def _write_summary_section(f, n_images, n_skip, n_success, valid_results):
    """[2] Overall Summary section."""
    f.write("[2] Overall Summary\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Total Images Attacked  : {n_images}\n")
    f.write(f"  Skipped (no detection) : {n_skip}\n")
    f.write(f"  Valid Attacks           : {len(valid_results)}\n")
    f.write(f"  Successful Attacks     : {n_success}\n")
    f.write(f"  Failed Attacks         : {len(valid_results) - n_success}\n")
    overall_asr = n_success / len(valid_results) if valid_results else 0.0
    f.write(f"  Overall ASR (image)    : {overall_asr:.2%} ({n_success}/{len(valid_results)})\n")
    f.write("\n")


def _write_aggregate_section(
    f, queries_list, l0_list, sparsity_list, sr_list,
    total_bbox_list, survived_list, disappeared_list, misclassified_list,
):
    """[3] Aggregate Metrics section."""
    f.write("[3] Aggregate Metrics (over valid attacks)\n")
    f.write("-" * 50 + "\n")

    _write_stat(f, "Queries Used", queries_list, fmt=".0f")
    _write_stat(f, "L0 Distance (pixels)", l0_list, fmt=".0f")
    _write_stat(f, "Sparsity Ratio", sparsity_list, fmt=".4f", is_pct=True)
    _write_stat(f, "Attack Success Rate", sr_list, fmt=".2f", is_pct=True)
    _write_stat(f, "Total BBoxes (orig)", total_bbox_list, fmt=".1f")
    _write_stat(f, "Survived BBoxes", survived_list, fmt=".1f")
    _write_stat(f, "Disappeared BBoxes", disappeared_list, fmt=".1f")
    _write_stat(f, "Misclassified BBoxes", misclassified_list, fmt=".1f")

    # Bbox-level ASR
    total_bboxes_all = sum(total_bbox_list)
    total_disappeared_all = sum(disappeared_list)
    total_misclassified_all = sum(misclassified_list)
    total_survived_all = sum(survived_list)
    bbox_asr = ((total_disappeared_all + total_misclassified_all) /
                 total_bboxes_all if total_bboxes_all > 0 else 0.0)
    f.write(f"\n  --- BBox-Level Summary ---\n")
    f.write(f"  Total BBoxes           : {total_bboxes_all}\n")
    f.write(f"  Survived               : {total_survived_all}\n")
    f.write(f"  Disappeared            : {total_disappeared_all}\n")
    f.write(f"  Misclassified          : {total_misclassified_all}\n")
    f.write(f"  BBox-Level ASR         : {bbox_asr:.2%}\n")
    f.write("\n")


def _write_map_section(f, map_result, args, section_num, title):
    """mAP Evaluation section."""
    f.write(f"[{section_num}] {title}\n")
    f.write("-" * 50 + "\n")
    f.write(f"  IoU Threshold          : {args.iou_thr}\n")
    f.write(f"  Original mAP           : {map_result['orig_mAP']:.4f}\n")
    f.write(f"  Adversarial mAP        : {map_result['adv_mAP']:.4f}\n")
    f.write(f"  mAP Drop               : {map_result['mAP_drop']:.4f}\n")
    drop_pct = (map_result['mAP_drop'] / map_result['orig_mAP'] * 100
                if map_result['orig_mAP'] > 0 else 0.0)
    f.write(f"  mAP Drop Rate          : {drop_pct:.2f}%\n")

    per_class_ap = map_result.get('per_class_ap', [])
    if per_class_ap:
        f.write(f"\n  --- Per-Class AP (Adversarial) ---\n")
        for i, ap in enumerate(per_class_ap):
            f.write(f"  Class {i:<4d} : AP = {ap:.4f}\n")
    f.write("\n")


def _write_efficiency_section(f, valid_results, queries_list, elapsed_time,
                              section_num):
    """Efficiency Metrics section."""
    f.write(f"[{section_num}] Efficiency Metrics\n")
    f.write("-" * 50 + "\n")
    if valid_results:
        avg_time_per_image = elapsed_time / len(valid_results)
        avg_queries = np.mean(queries_list)
        f.write(f"  Avg Time per Image     : {avg_time_per_image:.1f}s\n")
        f.write(f"  Avg Queries per Image  : {avg_queries:.0f}\n")
        f.write(f"  Total Queries          : {sum(queries_list)}\n")
        success_queries = [r['n_queries'] for r in valid_results if r['is_successful']]
        if success_queries:
            f.write(f"  Avg Queries (success)  : {np.mean(success_queries):.0f}\n")
        fail_queries = [r['n_queries'] for r in valid_results if not r['is_successful']]
        if fail_queries:
            f.write(f"  Avg Queries (fail)     : {np.mean(fail_queries):.0f}\n")
    f.write("\n")


def _write_per_image_section(f, results, section_num):
    """Per-Image Results table."""
    f.write(f"[{section_num}] Per-Image Results\n")
    f.write("-" * 70 + "\n")
    header = (
        f"  {'No.':<5s} {'Image':<35s} {'Queries':<8s} {'L0':<8s} "
        f"{'Sparsity':<10s} {'ASR':<8s} {'Surv':<5s} {'Dis':<5s} "
        f"{'Mis':<5s} {'Result':<8s}\n"
    )
    f.write(header)
    f.write("  " + "-" * 98 + "\n")

    for i, r in enumerate(results):
        img_name = os.path.basename(r['image_path'])
        if len(img_name) > 33:
            img_name = img_name[:30] + "..."
        status = "SUCCESS" if r['is_successful'] else ("SKIP" if r['n_queries'] == 0 else "FAIL")
        f.write(
            f"  {i+1:<5d} {img_name:<35s} {r['n_queries']:<8d} "
            f"{r['l0_distance']:<8d} {r.get('sparsity_ratio', 0):<10.4f} "
            f"{r['success_rate']:<8.2%} "
            f"{r['match_result']['survived']:<5d} "
            f"{r['match_result']['disappeared']:<5d} "
            f"{r['match_result']['misclassified']:<5d} "
            f"{status:<8s}\n"
        )
