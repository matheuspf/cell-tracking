#!/usr/bin/env bash
set -euo pipefail
v4_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
v4_python=${MULTIDATA_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v4_python=$2; shift 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$v4_repo/tools:$v4_repo/work/annotation-selection-v1/official/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$v4_python" -m multidata_training_v4 "$@"
