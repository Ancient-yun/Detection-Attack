#!/usr/bin/env python
"""Validate experiment reports described by a matrix config."""

from __future__ import annotations

import argparse

from matrix_utils import (
    iter_cases,
    load_matrix,
    named_report,
    parse_report,
    relative,
    result_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument("--success-thr", type=float, default=None)
    parser.add_argument("--attacks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_matrix(args.config)
    root = result_root(config, args.result_root)
    comparison = config.get("comparison", {})
    defaults = config.get("defaults", {})
    score_thr = args.score_thr
    if score_thr is None:
        score_thr = float(comparison.get("target_score_thr", 0.5))
    max_query = args.max_query
    if max_query is None:
        max_query = int(defaults.get("max_query", 1000))
    success_thr = args.success_thr
    if success_thr is None:
        success_thr = float(defaults.get("success_thr", 0.7))

    all_ok = True
    for case in iter_cases(config, args.attacks, args.targets):
        report = named_report(root, case, score_thr)
        metrics = parse_report(report)
        issues: list[str] = []
        if not metrics:
            issues.append("missing report")
        else:
            checks = {
                "Score Threshold": (metrics.get("score_thr"), str(score_thr)),
                "Max Queries": (metrics.get("max_queries"), str(max_query)),
                "Success Threshold": (metrics.get("success_thr"), str(success_thr)),
                "Total Images": (
                    metrics.get("total_images"),
                    str(case.get("expected_images")),
                ),
            }
            for label, (actual, expected) in checks.items():
                if actual != expected:
                    issues.append(f"{label} expected {expected}, got {actual}")

        prefix = f"{case['target']} | {case['attack']}"
        if issues:
            all_ok = False
            print(f"FAIL | {prefix} | {relative(report)} | {'; '.join(issues)}")
        else:
            print(f"OK   | {prefix} | {relative(report)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
