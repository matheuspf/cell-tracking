#!/usr/bin/env bash
# Additive, resumable local execution. Does not install into existing environments.
set -euo pipefail
study_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
study_python=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$study_repo/tools"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$study_repo"
for study_stage in "$@"; do
  "$study_python" -m annotation_selection "$study_stage"
done
