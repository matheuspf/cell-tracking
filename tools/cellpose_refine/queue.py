"""Resume feature-dependent fits, then freeze predictions before target evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from .common import EMBRYOS, REPO, WORK, config, now, write_json


def ready(source):
    manifest = json.loads((WORK / "inputs" / ("train-" + source + ".json")).read_text())
    paths = [Path(r["cache_path"]) for r in manifest["natural_rows"] + manifest["jitter_rows"]]
    return sum(p.exists() and p.with_suffix(".json").exists() for p in paths), len(paths)


def execute(name, module, arguments):
    log = WORK / (name + ".log")
    print(json.dumps({"created_utc": now(), "stage": name, "status": "starting", "log": str(log)}), flush=True)
    with log.open("a") as f:
        result = subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(WORK / "queue-status.json", {"updated_utc": now(), "status": "failed", "stage": name, "log": str(log), "exit_code": result.returncode})
        raise RuntimeError(name + " failed; see " + str(log))


def main():
    cfg = config()
    for source in EMBRYOS:
        last = None
        while True:
            have, need = ready(source)
            state = {"updated_utc": now(), "status": "waiting_for_features", "source": source, "available": have, "required": need}
            write_json(WORK / "queue-status.json", state)
            if have == need:
                break
            if last != have:
                print(json.dumps(state), flush=True); last = have
            time.sleep(15)
        for seed in cfg["seeds"]:
            execute(f"train-{source}-{seed}", "tools.cellpose_refine.train", ["--source", source, "--seed", str(seed)])
    execute("inference", "tools.cellpose_refine.infer", [])
    # Use the verified matching environment, independently of the Cellpose environment.
    log = WORK / "evaluation.log"
    with log.open("a") as f:
        result = subprocess.run(["/kaggle/envs/cell-tracking-notebooks/bin/python", "-u", "-m", "tools.cellpose_refine.evaluate"],
                                cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Evaluation failed; see " + str(log))
    write_json(WORK / "queue-status.json", {"updated_utc": now(), "status": "complete", "stage": "four fits and locked target evaluation"})
    print("ALL FOUR FITS AND EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
