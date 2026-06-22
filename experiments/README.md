# Experiment Matrices

Experiment-specific values live in JSON matrix files instead of code. Runners
and reports read these files, so a new threshold or dataset should be added as a
new matrix or CLI override instead of copying scripts.

Common commands:

```bash
python scripts/experiments/run_matrix.py --config experiments/score05.json --dry-run
python tools/attack_reports/validate_reports.py --config experiments/score05.json
python tools/attack_reports/build_report.py --config experiments/score05.json --format html
python scripts/experiments/run_matrix.py --config experiments/baseline.json --targets yolov8n --dry-run
```
