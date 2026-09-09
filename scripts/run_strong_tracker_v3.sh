#!/usr/bin/env bash
set -euo pipefail
v3_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
v3_python=${STRONG_TRACKER_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v3_python=$2; shift 2; fi
if [[ ! -x "$v3_python" ]]; then echo "Study interpreter unavailable: $v3_python" >&2; exit 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$v3_repo/tools:$v3_repo/work/annotation-selection-v1/official/src${PYTHONPATH:+:$PYTHONPATH}"
# Preserve user library paths; this host injects one incompatible toolkit path.
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  v3_lib=":$LD_LIBRARY_PATH:"
  v3_lib=${v3_lib//:\/usr\/local\/cuda\/lib64:/:}
  v3_lib=${v3_lib#:}; v3_lib=${v3_lib%:}
  if [[ -n "$v3_lib" ]]; then export LD_LIBRARY_PATH="$v3_lib"; else unset LD_LIBRARY_PATH; fi
fi
# Bound aggregate CPU execution even when CUDA/Zarr create idle support threads.
v3_cpu_set=$("$v3_python" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:32])))')
exec taskset --cpu-list "$v3_cpu_set" "$v3_python" -m strong_tracker_v3 "$@"
