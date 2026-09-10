#!/usr/bin/env bash
set -euo pipefail
v6_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
v6_python=${V6_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v6_python=$2; shift 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$v6_repo/tools:$v6_repo/work/annotation-selection-v1/official/src"
exec "$v6_python" -m segmentation_tracking_v6 "${@:-run}"
