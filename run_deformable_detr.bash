#!/usr/bin/env bash
set -euo pipefail

python scripts/experiments/run_matrix.py \
  --config experiments/baseline.json \
  --targets deformable \
  "$@"
