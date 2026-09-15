"""Track and score each whole clip as its independently running Cellpose job finishes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.cellpose_ultrack.prepare import ROOT, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    panel = json.loads((args.root / "panel.json").read_text())
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", POLARS_MAX_THREADS="2", CUDA_VISIBLE_DEVICES="")
    for clip in panel["clips"]:
        name = clip["dataset"]
        expected = [args.root / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.json"
                    for t in range(clip["shape"][0])]
        print(f"Waiting for the complete clip: {name}", flush=True)
        while not all(p.exists() for p in expected):
            failures = list((args.root / "cellpose/errors").glob("*.json"))
            if failures:
                raise RuntimeError(f"Cellpose reported an error: {failures}")
            time.sleep(5)
        for step, python, module, extra_env in [
            ("track", "/kaggle/envs/cell-tracking-ultrack-v6/bin/python", "tools.cellpose_ultrack.track",
             {"PYTHONPATH": "/home/mpf/code/kaggle/ultrack"}),
            ("score", "/kaggle/envs/cell-tracking-notebooks/bin/python", "tools.cellpose_ultrack.evaluate", {}),
        ]:
            path = args.root / f"{name}-{step}.log"
            print(f"Starting {step}: {name}", flush=True)
            with path.open("a") as stream:
                subprocess.run([python, "-m", module, "--root", str(args.root), "--datasets", name],
                               cwd=REPO, env={**env, **extra_env}, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f"Complete {step}: {name}", flush=True)
    subprocess.run(["/kaggle/envs/cell-tracking-notebooks/bin/python", "-m", "tools.cellpose_ultrack.evaluate",
                    "--root", str(args.root)], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
