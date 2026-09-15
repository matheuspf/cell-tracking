"""Fit only a source-embryo residual head; target scores are never read here."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .common import (EMBRYOS, REPO, RESULTS, WORK, assert_source, config, configure,
                     json_hash, locked_gpu, now, sha, validate_ancestry, write_json)
from .model import Refiner, sparse_offset_loss


def install_source_guard(source):
    """Reject reads of the other embryo's raw data, labels and feature caches."""
    other = next(e for e in EMBRYOS if e != source)
    checked_prefixes = ("/cellpose-refine-v1/", "/detector-screen-20260914/", "/biohub-cell-tracking-during-development/train/")
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if any(p in path for p in checked_prefixes):
                if f"/{other}/" in path or f"/{other}_" in path or f"train-{other}.json" in path:
                    raise PermissionError("Source-only training guard rejected other embryo: " + path)
    sys.addaudithook(audit)


def load_source(source, cfg):
    manifest_path = WORK / "inputs" / ("train-" + source + ".json")
    lock = json.loads((WORK / "feature-lock.json").read_text())
    assert sha(manifest_path) == lock["source_summaries"][source]["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_embryo"] == source
    validate_ancestry(source, manifest)
    rows = manifest["natural_rows"] + manifest["jitter_rows"]
    assert_source(source, rows)
    groups = {name: [] for name in ("feature", "patch", "geometry", "target_um", "weight", "natural")}
    reads = []
    for row in rows:
        feature_path = Path(row["cache_path"])
        sidecar_path = feature_path.with_suffix(".json")
        record = json.loads(sidecar_path.read_text())
        assert record["key"] == row["key"] and record["embryo"] == source
        assert record["config_sha256"] == json_hash(cfg)
        assert sha(feature_path) == record["cache_sha256"]
        natural = "target_path" in row
        assert record["query_input_sha256"] == row["prediction_sha256" if natural else "query_sha256"]
        target_path = Path(row["target_path"] if natural else row["query_path"])
        target_hash = row["target_sha256"] if natural else row["query_sha256"]
        assert sha(target_path) == target_hash
        with np.load(target_path, allow_pickle=False) as target, np.load(feature_path, allow_pickle=False) as feature:
            valid = target["weight"] > 0
            assert len(valid) == len(feature["feature"])
            if not natural:
                np.testing.assert_array_equal(feature["queries"], target["queries"])
            for name in ("feature", "patch", "geometry"):
                groups[name].append(feature[name][valid])
            for name in ("target_um", "weight"):
                groups[name].append(target[name][valid])
            groups["natural"].append(np.full(int(valid.sum()), natural, dtype=bool))
        reads.append({"key": row["key"], "kind": "natural" if natural else "jitter",
                      "cache_sha256": record["cache_sha256"], "target_sha256": target_hash})
    arrays = {k: np.concatenate(v) for k, v in groups.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    assert len(arrays["weight"]) and (arrays["weight"] > 0).all()
    return arrays, manifest, reads


def lock_training(cfg):
    path = WORK / "training-lock.json"
    source_paths = [*(REPO / "tools/cellpose_refine").glob("*.py"),
                    *(REPO / "tools/detector_screen" / name for name in ("cellpose_adapter.py", "analyze_headroom.py", "evaluate.py")),
                    REPO / "tools/annotation_selection/metric_adapter.py"]
    sources = {str(p.relative_to(REPO)): sha(p) for p in source_paths}
    expected = {"config_sha256": json_hash(cfg), "source_sha256": sources,
                "feature_lock_sha256": sha(WORK / "feature-lock.json")}
    if path.exists():
        old = json.loads(path.read_text())
        for key, value in expected.items():
            if old[key] != value:
                raise ValueError("Training lock mismatch: " + key)
        return old
    payload = {"created_utc": now(), **expected, "target_scores_opened": False,
               "retained_fits": [{"source": s, "target": next(e for e in EMBRYOS if e != s), "seed": seed}
                                 for s in EMBRYOS for seed in cfg["seeds"]],
               "checkpoint_selection": "fixed final update; all four fits must finish before scoring"}
    write_json(path, payload)
    write_json(RESULTS / "training-lock.json", payload)
    return payload


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def run(source, seed):
    configure()
    cfg = config()
    if source not in EMBRYOS or seed not in cfg["seeds"]:
        raise ValueError("Unregistered fit")
    lock = lock_training(cfg)
    install_source_guard(source)
    fit_id = f"source-{source}-seed-{seed}"
    dest = WORK / "models" / fit_id
    final = dest / "final.pt"
    if final.exists():
        receipt = json.loads((dest / "receipt.json").read_text())
        assert receipt["checkpoint_sha256"] == sha(final) and receipt["training_lock_sha256"] == sha(WORK / "training-lock.json")
        print("Already complete " + fit_id, flush=True)
        return
    arrays, manifest, reads = load_source(source, cfg)
    target_embryo = next(e for e in EMBRYOS if e != source)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    tc = cfg["training"]
    started = time.perf_counter()
    with locked_gpu() as queued:
        torch.cuda.reset_peak_memory_stats()
        data = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
        del arrays
        model = Refiner(cfg).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        natural = torch.nonzero(data["natural"], as_tuple=False).flatten()
        jitter = torch.nonzero(~data["natural"], as_tuple=False).flatten()
        history = []
        resume = dest / "resume.pt"
        first = 1
        if resume.exists():
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            assert saved["config_sha256"] == json_hash(cfg) and saved["source"] == source and saved["seed"] == seed
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["cpu_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"])
            first = saved["update"] + 1
            history = saved["history"]
        for step in range(first, tc["updates"] + 1):
            if step <= tc["warmup_updates"]:
                factor = step / tc["warmup_updates"]
            else:
                progress = (step - tc["warmup_updates"]) / (tc["updates"] - tc["warmup_updates"])
                factor = tc["final_learning_rate_fraction"] + (1 - tc["final_learning_rate_fraction"]) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = tc["learning_rate"] * factor
            half = tc["batch_size"] // 2
            assert len(natural) and len(jitter)
            ix = torch.cat([natural[torch.randint(len(natural), (half,), device="cuda")],
                            jitter[torch.randint(len(jitter), (tc["batch_size"] - half,), device="cuda")]])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data["feature"][ix], data["patch"][ix], data["geometry"][ix])
            loss = sparse_offset_loss(prediction, data["target_um"][ix], data["weight"][ix], tc["loss_beta_um"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite source loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                item = {"update": step, "source_loss": float(loss.detach()), "gradient_norm": float(grad), "learning_rate": optimizer.param_groups[0]["lr"]}
                history.append(item)
                print(json.dumps({"fit": fit_id, **item}), flush=True)
            if step % tc["checkpoint_interval"] == 0:
                payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "source": source, "seed": seed,
                           "config_sha256": json_hash(cfg), "update": step, "history": history,
                           "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
                save_checkpoint(resume, payload)
                assert torch.cuda.max_memory_reserved() < tc["max_gpu_reserved_gib"] * 1024**3
        torch.cuda.synchronize()
        model.eval()
        save_checkpoint(final, {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg,
                                "source": source, "target": target_embryo, "seed": seed, "update": tc["updates"],
                                "training_lock_sha256": sha(WORK / "training-lock.json")})
        peak = torch.cuda.max_memory_reserved()
    receipt = {"created_utc": now(), "fit": fit_id, "source": source, "target": target_embryo, "seed": seed,
               "config_sha256": json_hash(cfg), "training_lock_sha256": sha(WORK / "training-lock.json"),
               "checkpoint_sha256": sha(final), "updates": tc["updates"], "seconds_including_queue": time.perf_counter() - started,
               "gpu_queue_seconds": queued, "peak_gpu_reserved_bytes": peak,
               "supervised_queries": {"natural": len(natural), "jitter_and_center": len(jitter)},
               "adaptation_exposure": manifest["adaptation_exposure"], "inherited_exposure": manifest["inherited_exposure"],
               "training_data_reads": reads, "source_loss_history": history, "target_scores_read": False,
               "read_guard": "Python audit hook rejects other-embryo raw/cached training data paths; this is not a kernel sandbox"}
    write_json(dest / "receipt.json", receipt)
    write_json(RESULTS / (fit_id + ".json"), {k: v for k, v in receipt.items() if k != "training_data_reads"})
    print(json.dumps({"fit": fit_id, "status": "complete", "seconds": receipt["seconds_including_queue"], "checkpoint_sha256": receipt["checkpoint_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=EMBRYOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.source, args.seed)


if __name__ == "__main__":
    main()
