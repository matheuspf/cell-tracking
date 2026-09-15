"""Record the actual isolated runtimes without importing models or using GPUs."""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "work/detector-screen-20260914"
ENVIRONMENTS = [
    "cell-tracking-notebooks", "detector-screen-torch", "detector-screen-tf2162",
    "detector-screen-organoid", "detector-screen-pacmap", "detector-screen-cellpose",
]
PACKAGES = [
    "numpy", "scipy", "scikit-image", "torch", "torchvision", "triton",
    "tensorflow", "keras", "tf-keras", "stardist", "csbdeep", "spotiflow",
    "cellpose", "dinov3", "h5py", "tifffile", "tracksdata", "zarr", "polars",
    "model-factory", "pacmap",
]


def inspect_runtime(name):
    executable = Path("/kaggle/envs") / name / "bin/python"
    script = """
import importlib.metadata as md, json, platform, sys
packages = {}
for name in json.loads(sys.argv[1]):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        pass
print(json.dumps(dict(executable=sys.executable, python=platform.python_version(), packages=packages)))
"""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    result = subprocess.check_output([str(executable), "-c", script, json.dumps(PACKAGES)], env=env, text=True)
    return name, json.loads(result)


def main():
    with ThreadPoolExecutor(max_workers=len(ENVIRONMENTS)) as executor:
        runtimes = dict(executor.map(inspect_runtime, ENVIRONMENTS))
    result = dict(runtimes=runtimes,
                  scope="Installed distribution metadata in the inference and evaluation runtimes; no model imports or GPU operations.",
                  tensorflow_note="StarDist and AnyStar select TF_USE_LEGACY_KERAS=1 per process; NucVerse uses Keras 3. OrganoidTracker selects the PyTorch Keras backend.")
    path = ROOT / "evaluation/runtimes.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
