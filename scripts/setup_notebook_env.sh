#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
notebook_env=/kaggle/envs/cell-tracking-notebooks

if [ ! -x "$notebook_env/bin/python" ]; then
  conda create -y --prefix "$notebook_env" python=3.12 pip
fi
"$notebook_env/bin/python" - <<'PY'
import hashlib
from pathlib import Path

wheel = Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/wheels/tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl")
expected = "5111a9058edb31272fd5c6d01ccfe6146cf190ed96b100cb3d27411a6a91ff4d"
if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
    raise RuntimeError("The tracksdata wheel differs from the verified upstream build")
PY
"$notebook_env/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$notebook_env/bin/python" -m pip install -r requirements-notebooks.txt
cp requirements-notebooks.txt "$notebook_env/requirements-local.txt"
"$notebook_env/bin/python" -m pip check
"$notebook_env/bin/python" -m ipykernel install --user \
  --name cell-tracking-notebooks --display-name "Python (cell-tracking notebooks)"
"$notebook_env/bin/python" - <<'PY'
import json
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

path = Path(KernelSpecManager().get_kernel_spec("cell-tracking-notebooks").resource_dir) / "kernel.json"
spec = json.loads(path.read_text())
spec["env"] = {
    "PATH": "/kaggle/envs/cell-tracking-notebooks/bin:${PATH}",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/repo/src",
    "POLARS_PREFER_PKG": "32",
    "BIOHUB_ALLOW_PIP_INSTALL": "0",
}
path.write_text(json.dumps(spec, indent=2) + "\n")
PY
printf 'Activate with: conda activate /kaggle/envs/cell-tracking-notebooks\n'
