#!/usr/bin/env bash
set -euo pipefail
cd /root/code/kaggle/cell-tracking
source scripts/root_remote_env.sh
export PYTHONPATH="$PWD/tools:$PWD/work/annotation-selection-v1/official/src"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
exec /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m strong_tracker_v2 "$@"
