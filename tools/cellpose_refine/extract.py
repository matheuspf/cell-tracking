"""Build immutable query feature caches, keeping labeled and inference queries separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import zarr

from .common import (EMBRYOS, REPO, RESULTS, WORK, config, configure, json_hash,
                     locked_gpu, now, save_npz, sha, write_json)
from .features import FrozenFeatures


def load_volume(row):
    if row.get("image_path"):
        return np.load(row["image_path"], allow_pickle=False)
    return np.asarray(zarr.open_group(row["raw_path"], mode="r")["0"][row["time"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["all", "inference", "jitter"], default="all")
    parser.add_argument("--embryo", choices=EMBRYOS)
    args = parser.parse_args()
    configure()
    cfg = config()
    lock = json.loads((WORK / "feature-lock.json").read_text())
    assert json_hash(cfg) == lock["config_sha256"]
    for name, expected in lock["feature_code_sha256"].items():
        assert sha(REPO / "tools/cellpose_refine" / name) == expected, name
    receipt = json.loads((RESULTS / "preflight.json").read_text())
    assert receipt["status"] == "passed" and receipt["config_sha256"] == json_hash(cfg)
    jobs = []
    if args.kind in ("all", "inference"):
        assert sha(WORK / "inputs/inference.json") == lock["inference_manifest_sha256"]
        jobs += [("inference", r) for r in json.loads((WORK / "inputs/inference.json").read_text())["rows"]]
    if args.kind in ("all", "jitter"):
        for embryo in EMBRYOS:
            path = WORK / "inputs" / ("train-" + embryo + ".json")
            assert sha(path) == lock["source_summaries"][embryo]["manifest_sha256"]
            jobs += [("jitter", r) for r in json.loads(path.read_text())["jitter_rows"]]
    if args.embryo:
        jobs = [(kind, row) for kind, row in jobs if row["embryo"] == args.embryo]
    # Complete one source cohort first so its registered fits can start sooner.
    # Plane batches and the locked numerical producer remain identical per frame.
    jobs.sort(key=lambda item: (EMBRYOS.index(item[1]["embryo"]), item[0] != "inference"))
    n_done = 0
    with locked_gpu() as queue:
        extractor = FrozenFeatures()
    started = time.perf_counter()
    for kind, row in jobs:
        dest = Path(row["cache_path"])
        sidecar = dest.with_suffix(".json")
        input_path = Path(row["prediction_path"] if kind == "inference" else row["query_path"])
        input_sha = row["prediction_sha256"] if kind == "inference" else row["query_sha256"]
        if dest.exists() and sidecar.exists():
            old = json.loads(sidecar.read_text())
            assert old["config_sha256"] == json_hash(cfg) and old["query_input_sha256"] == input_sha
            assert old["cache_sha256"] == sha(dest)
            n_done += 1
            continue
        assert sha(input_path) == input_sha
        with np.load(input_path, allow_pickle=False) as p:
            queries = p["centers_zyx" if kind == "inference" else "queries"].astype(np.float64)
        volume = load_volume(row)
        assert list(volume.shape) == row["shape"] and np.isfinite(volume).all()
        volume_sha = __import__("hashlib").sha256(volume.tobytes()).hexdigest()
        frame_start = time.perf_counter()
        if len(queries):
            with locked_gpu() as frame_queue:
                queue += frame_queue
                torch.cuda.reset_peak_memory_stats()
                f, p, g = extractor.extract(volume, queries)
        else:
            # An empty query set executes normalization only; no CUDA work occurs.
            frame_queue = 0.
            f, p, g = extractor.extract(volume, queries)
        save_npz(dest, feature=f, patch=p, geometry=g, queries=queries)
        elapsed = time.perf_counter() - frame_start
        peak = torch.cuda.max_memory_reserved()
        assert peak < cfg["training"]["max_gpu_reserved_gib"] * 1024**3
        info = {"created_utc": now(), "kind": kind, "key": row["key"], "embryo": row["embryo"],
                "config_sha256": json_hash(cfg), "query_input_sha256": input_sha, "volume_sha256": volume_sha,
                "cache_sha256": sha(dest), "queries": len(queries), "seconds": elapsed, "gpu_queue_seconds": frame_queue,
                "feature_seconds": extractor.last_seconds if len(queries) else {"queried_planes": 0}, "gpu_reserved_bytes": peak,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "fitted_preprocessing": False, "annotation_read": "none" if kind == "inference" else "prebuilt source query coordinates; no target scoring"}
        write_json(sidecar, info)
        n_done += 1
        print(json.dumps({"done": n_done, "total": len(jobs), "kind": kind, "key": row["key"],
                          "queries": len(queries), "seconds": round(elapsed, 3), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        if n_done % 20 == 0:
            write_json(WORK / "extraction-progress.json", {"updated_utc": now(), "done": n_done, "total": len(jobs),
                       "last_key": row["key"], "kind": kind, "queue_seconds": queue, "elapsed_seconds": time.perf_counter() - started})
    write_json(WORK / "extraction-complete.json", {"created_utc": now(), "completed_jobs": n_done, "expected_jobs": len(jobs),
                                                   "config_sha256": json_hash(cfg), "kind": args.kind, "embryo": args.embryo})


if __name__ == "__main__":
    main()
