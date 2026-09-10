#!/usr/bin/env bash
set -euo pipefail
v5_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
v5_python=${V5_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v5_python=$2; shift 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$v5_repo/tools:$v5_repo/work/annotation-selection-v1/official/src${PYTHONPATH:+:$PYTHONPATH}"
v5_module=${1:-preflight}; shift || true
exec "$v5_python" -m "image_native_tracking_v5.$v5_module" "$@"
