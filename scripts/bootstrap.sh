#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

if ! conda run -n cell-tracking python --version >/dev/null 2>&1; then
  conda create -y -n cell-tracking python=3.12 pip
fi
conda run --no-capture-output -n cell-tracking python -m pip install -r requirements-local.txt
conda run --no-capture-output -n cell-tracking python -m playwright install chromium
conda run --no-capture-output -n cell-tracking python -m ipykernel install \
  --user --name cell-tracking --display-name "Python (cell-tracking)"
conda run --no-capture-output -n cell-tracking python -c 'import json,sys; from pathlib import Path; from jupyter_client.kernelspec import KernelSpecManager; p=Path(KernelSpecManager().get_kernel_spec("cell-tracking").resource_dir)/"kernel.json"; s=json.loads(p.read_text()); s["env"]={"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PATH":str(Path(sys.executable).parent)+":${PATH}"}; p.write_text(json.dumps(s,indent=2)+"\n")'
conda run --no-capture-output -n cell-tracking python tools/download_data.py
conda run --no-capture-output -n cell-tracking python tools/inspect_data.py
PYTHONPATH=tools conda run --no-capture-output -n cell-tracking python -m kaggle_extract sync
conda run --no-capture-output -n cell-tracking python -m pip check

printf 'Ready. Activate with: conda activate cell-tracking\n'
