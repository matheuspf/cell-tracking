"""Replay a locked trained detector on one renamed raw volume per held-out embryo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from tools.cellpose_refine.common import (EMBRYOS, REPO, RESULTS, WORK, config,
                                         now, sha, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    rows = json.loads((WORK / "inputs/inference.json").read_text())["rows"]
    selected = [next(r for r in rows if r["embryo"] == e and r["role"] == "pilot") for e in EMBRYOS]
    seed = config()["seeds"][0]
    plan = {"created_utc": now(), "seed": seed,
            "selection": "First pilot frame in frozen inference manifest per embryo; no scores used",
            "keys": [r["key"] for r in selected], "script_sha256": sha(__file__)}
    plan_path = RESULTS / "trained-raw-replay-plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert all(old[k] == v for k, v in plan.items() if k != "created_utc")
    else:
        write_json(plan_path, plan)
    lock_path = WORK / "predictions-lock.json"
    while args.wait and not lock_path.exists():
        time.sleep(15)
    lock = json.loads(lock_path.read_text())
    checks = []
    for index, row in enumerate(selected):
        model = next(m for m in lock["models"] if m["seed"] == seed and m["target"] == row["embryo"])
        assert model["source"] != row["embryo"] and sha(model["path"]) == model["sha256"]
        volume = WORK / "replay" / f"heldout-{index}.npy"
        output = WORK / "replay" / f"heldout-{index}-output.npz"
        volume.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image_path"], volume)
        subprocess.run([sys.executable, "-u", "-m", "tools.cellpose_refine.infer",
                        "--volume", str(volume), "--head", model["path"], "--output", str(output)],
                       cwd=REPO, check=True)
        raw_receipt = json.loads(output.with_suffix(".json").read_text())
        cached_path = WORK / "predictions" / f"seed-{seed}" / (row["key"] + ".npz")
        expected = next(p for p in lock["predictions"] if p["key"] == row["key"] and p["seed"] == seed)
        assert sha(cached_path) == expected["prediction_sha256"]
        with np.load(output, allow_pickle=False) as raw, np.load(cached_path, allow_pickle=False) as cached, np.load(row["prediction_path"], allow_pickle=False) as original:
            equal = {k: np.array_equal(raw[k], cached[k]) for k in cached.files}
            equal["baseline_centers_zyx"] = np.array_equal(raw["baseline_centers_zyx"], original["centers_zyx"].astype(np.float64))
            equal["original_scores"] = np.array_equal(raw["scores"], original["scores"])
        with np.load(row["cache_path"], allow_pickle=False) as features:
            equal["feature_arrays"] = raw_receipt["feature_array_sha256"] == {
                k: hashlib.sha256(features[k].tobytes()).hexdigest() for k in ("feature", "patch", "geometry")}
        equal["source_identity"] = raw_receipt["source"] == model["source"]
        equal["checkpoint_identity"] = raw_receipt["checkpoint_sha256"] == model["sha256"]
        checks.append({"key": row["key"], "target_embryo": row["embryo"], "training_source": model["source"],
                       "seed": seed, "exact_equality": equal, "raw_receipt": raw_receipt})
        print(json.dumps({"raw_replay": row["key"], "checks": equal}), flush=True)
    passed = all(all(r["exact_equality"].values()) for r in checks)
    write_json(RESULTS / "trained-raw-replay.json", {
        "created_utc": now(), "status": "passed" if passed else "failed", "checks": checks,
        "plan_sha256": sha(plan_path), "predictions_lock_sha256": sha(lock_path),
        "scope": "Two renamed raw volumes, one per held-out embryo, primary seed. Full Cellpose plus refiner with annotation/cache access blocked in inference; not complete clips or a Kaggle runtime check."})
    assert passed


if __name__ == "__main__":
    main()
