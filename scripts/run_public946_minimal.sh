#!/usr/bin/env bash
set -euo pipefail
study_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$study_repo/tools${PYTHONPATH:+:$PYTHONPATH}"
exec "${PUBLIC946_PYTHON:-/kaggle/envs/cell-tracking-notebooks/bin/python}" -m public946_minimal "$@"
