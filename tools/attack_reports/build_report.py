#!/usr/bin/env python
"""Build a comparison report from an experiment matrix."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from typing import Any

from matrix_utils import (
    iter_cases,
    latest_report,
    load_matrix,
    named_report,
    parse_report,
    project_root,
    relative,
    result_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--baseline-score-thr", type=float, default=None)
    parser.add_argument("--target-score-thr", type=float, default=None)
    parser.add_argument("--attacks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--format", choices=["md", "html"], default="md")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def metric(report: dict[str, Any], key: str) -> str:
    value = report.get(key)
    return "N/A" if value in {None, ""} else str(value)


def collect_rows(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_matrix(args.config)
    root = result_root(config, args.result_root)
    comparison = config.get("comparison", {})
    baseline_score = args.baseline_score_thr
    if baseline_score is None:
        baseline_score = float(comparison.get("baseline_score_thr", 0.3))
    target_score = args.target_score_thr
    if target_score is None:
        target_score = float(comparison.get("target_score_thr", 0.5))

    rows = []
    for case in iter_cases(config, args.attacks, args.targets):
        baseline_path = latest_report(
            root,
            case["attack"],
            case["baseline_dataset_name"],
            case["model"],
            baseline_score,
            case.get("expected_images"),
        )
        target_path = named_report(root, case, target_score)
        rows.append(
            {
                "case": case,
                "baseline_path": baseline_path,
                "target_path": target_path,
                "baseline": parse_report(baseline_path),
                "target": parse_report(target_path),
            }
        )
    return config, rows


def render_md(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    comparison = config.get("comparison", {})
    baseline_score = comparison.get("baseline_score_thr", "baseline")
    target_score = comparison.get("target_score_thr", "target")
    lines = [
        f"# {config.get('name', 'experiment')} comparison",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Baseline score threshold: {baseline_score}",
        f"- Target score threshold: {target_score}",
        "",
        "| Target | Attack | Baseline Report | Target Report | Images | ASR baseline | ASR target | AvgQ baseline | AvgQ target | Mean L0 baseline | Mean L0 target | mAP Drop baseline | mAP Drop target |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        case = row["case"]
        baseline = row["baseline"]
        target = row["target"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case["target"],
                    case["attack"],
                    relative(row["baseline_path"]),
                    relative(row["target_path"]),
                    metric(target, "total_images"),
                    metric(baseline, "image_asr"),
                    metric(target, "image_asr"),
                    metric(baseline, "avg_queries"),
                    metric(target, "avg_queries"),
                    metric(baseline, "mean_l0"),
                    metric(target, "mean_l0"),
                    metric(baseline, "map_drop"),
                    metric(target, "map_drop"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_html(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    body_rows = []
    for row in rows:
        case = row["case"]
        baseline = row["baseline"]
        target = row["target"]
        body_rows.append(
            "<tr>"
            f"<td>{esc(case['target'])}</td>"
            f"<td><code>{esc(case['attack'])}</code></td>"
            f"<td><code>{esc(relative(row['baseline_path']))}</code></td>"
            f"<td><code>{esc(relative(row['target_path']))}</code></td>"
            f"<td class='num'>{esc(metric(target, 'total_images'))}</td>"
            f"<td class='num'>{esc(metric(baseline, 'image_asr'))}</td>"
            f"<td class='num'>{esc(metric(target, 'image_asr'))}</td>"
            f"<td class='num'>{esc(metric(baseline, 'avg_queries'))}</td>"
            f"<td class='num'>{esc(metric(target, 'avg_queries'))}</td>"
            f"<td class='num'>{esc(metric(baseline, 'mean_l0'))}</td>"
            f"<td class='num'>{esc(metric(target, 'mean_l0'))}</td>"
            f"<td class='num'>{esc(metric(baseline, 'map_drop'))}</td>"
            f"<td class='num'>{esc(metric(target, 'map_drop'))}</td>"
            "</tr>"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{esc(config.get('name', 'experiment'))} comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; background: #f9fafb; position: sticky; top: 0; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
    .num {{ text-align: right; white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>{esc(config.get('name', 'experiment'))} comparison</h1>
  <p>Generated: {esc(generated)}</p>
  <table>
    <thead>
      <tr>
        <th>Target</th><th>Attack</th><th>Baseline Report</th><th>Target Report</th>
        <th>Images</th><th>ASR baseline</th><th>ASR target</th>
        <th>AvgQ baseline</th><th>AvgQ target</th>
        <th>Mean L0 baseline</th><th>Mean L0 target</th>
        <th>mAP Drop baseline</th><th>mAP Drop target</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    config, rows = collect_rows(args)
    text = render_html(config, rows) if args.format == "html" else render_md(config, rows)

    output = args.output
    if output is None:
        ext = "html" if args.format == "html" else "md"
        output = f"outputs/reports/{config.get('name', 'experiment')}/comparison.{ext}"

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_root() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(relative(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
