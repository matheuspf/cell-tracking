"""Annotation-free residual inference and raw-volume replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .common import (CHECKPOINT, EMBRYOS, REPO, RESULTS, SPACING, WORK, bounded_integer,
                     config, configure, json_hash, locked_gpu, now, save_npz, sha, write_json)
from .model import Refiner


def install_inference_guard(raw=False):
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(args[0])
        prohibited = (".geff/", "ground_truth.json", "/queries/", "/inputs/train-", "/evaluation/gt/")
        if any(s in path for s in prohibited):
            raise PermissionError("Inference attempted annotation access: " + path)
        if raw and (("/features/" in path and "cellpose-refine-v1" in path) or "/predictions/" in path or "/pre_ilp_" in path):
            raise PermissionError("Raw replay attempted cached prediction/feature access: " + path)
    sys.addaudithook(audit)


def load_head(path, cfg):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert json_hash(state["config"]) == json_hash(cfg)
    head = Refiner(cfg)
    head.load_state_dict(state["model"], strict=True)
    return head.eval().requires_grad_(False).cuda(), state


@torch.inference_mode()
def refine_cached(head, queries, feature, patch, geometry, shape, cfg):
    chunks = []
    for begin in range(0, len(queries), 256):
        x = [torch.as_tensor(a[begin:begin + 256], device="cuda") for a in (feature, patch, geometry)]
        chunks.append(head(*x).cpu().numpy())
    delta = np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32)
    assert (np.linalg.norm(delta, axis=1) <= cfg["movement_limit_um"] + 1e-5).all()
    continuous = np.clip(queries + delta / SPACING, 0, np.asarray(shape) - 1)
    integer, projected = bounded_integer(queries, continuous, shape, cfg["movement_limit_um"])
    return {"centers_zyx": integer.astype(np.float64), "float_centers_zyx": continuous,
            "offset_um": delta, "integer_bound_projected": projected}


def all_heads_ready(cfg):
    lock_path = WORK / "training-lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["config_sha256"] == json_hash(cfg)
    for filename, expected in lock["source_sha256"].items():
        assert sha(REPO / filename) == expected, filename
    models = []
    for source in EMBRYOS:
        for seed in cfg["seeds"]:
            folder = WORK / "models" / f"source-{source}-seed-{seed}"
            receipt = json.loads((folder / "receipt.json").read_text())
            checkpoint = folder / "final.pt"
            assert receipt["checkpoint_sha256"] == sha(checkpoint)
            assert receipt["training_lock_sha256"] == sha(lock_path) and receipt["target_scores_read"] is False
            models.append({"source": source, "seed": seed, "target": receipt["target"], "path": str(checkpoint), "sha256": receipt["checkpoint_sha256"]})
    return models


def cached_inference():
    cfg = config()
    models = all_heads_ready(cfg)
    install_inference_guard()
    torch.use_deterministic_algorithms(True)
    rows = json.loads((WORK / "inputs/inference.json").read_text())["rows"]
    receipts = []
    for seed in cfg["seeds"]:
        with locked_gpu() as queued:
            heads = {}
            for m in models:
                if m["seed"] == seed:
                    head, state = load_head(m["path"], cfg)
                    heads[m["target"]] = (head, state, m)
        for row in rows:
            head, state, model_info = heads[row["embryo"]]
            assert state["source"] != row["embryo"] and state["target"] == row["embryo"]
            out = WORK / "predictions" / f"seed-{seed}" / (row["key"] + ".npz")
            if out.exists() and out.with_suffix(".json").exists():
                old = json.loads(out.with_suffix(".json").read_text())
                assert old["checkpoint_sha256"] == model_info["sha256"] and old["prediction_sha256"] == sha(out)
                receipts.append(old)
                continue
            start = time.perf_counter()
            cache = Path(row["cache_path"])
            info = json.loads(cache.with_suffix(".json").read_text())
            assert info["kind"] == "inference" and info["annotation_read"] == "none"
            assert info["cache_sha256"] == sha(cache) and info["config_sha256"] == json_hash(cfg)
            assert sha(row["prediction_path"]) == row["prediction_sha256"]
            with np.load(cache, allow_pickle=False) as f, np.load(row["prediction_path"], allow_pickle=False) as b:
                queries, scores = b["centers_zyx"].astype(np.float64), b["scores"].copy()
                np.testing.assert_array_equal(queries, f["queries"])
                with locked_gpu() as wait:
                    arrays = refine_cached(head, queries, f["feature"], f["patch"], f["geometry"], row["shape"], cfg)
                save_npz(out, **arrays, scores=scores)
            receipt = {"key": row["key"], "embryo": row["embryo"], "role": row["role"], "seed": seed,
                       "training_source": state["source"], "checkpoint_sha256": model_info["sha256"],
                       "prediction_sha256": sha(out), "original_prediction_sha256": row["prediction_sha256"],
                       "feature_cache_sha256": info["cache_sha256"], "count": len(queries),
                       "integer_projection_count": int(arrays["integer_bound_projected"].sum()),
                       "seconds": time.perf_counter() - start, "gpu_queue_seconds": wait,
                       "annotation_access": False, "candidate_count_and_scores_unchanged": True}
            write_json(out.with_suffix(".json"), receipt)
            receipts.append(receipt)
        print(json.dumps({"seed": seed, "predicted_frames": len(rows)}), flush=True)
    lock = {"created_utc": now(), "models": models, "config_sha256": json_hash(cfg),
            "training_lock_sha256": sha(WORK / "training-lock.json"), "predictions": receipts,
            "frame_count": len(rows) * len(cfg["seeds"]), "target_scores_opened": False}
    write_json(WORK / "predictions-lock.json", lock)
    write_json(RESULTS / "predictions-lock.json", lock)


def raw_inference(volume_path, head_path, output_path):
    """The complete current raw-volume detector, independent of proposal/feature caches."""
    install_inference_guard(raw=True)
    from scipy.ndimage import mean
    from scipy.special import expit
    from skimage.measure import regionprops_table
    from cellpose import models
    from .features import FrozenFeatures
    cfg = config()
    volume = np.load(volume_path, allow_pickle=False)
    started = time.perf_counter()
    with locked_gpu() as queued:
        extractor = FrozenFeatures()
        cp = models.CellposeModel(device=torch.device("cpu"), pretrained_model=str(CHECKPOINT), use_bfloat16=True)
        cp.net = extractor.net
        cp.device = torch.device("cuda:0"); cp.gpu = True
        masks, flows, _ = cp.eval(volume, z_axis=0, channel_axis=None, do_3D=True, anisotropy=4., batch_size=8,
                                 normalize=True, diameter=None, resample=True, flow_threshold=.4, cellprob_threshold=0.,
                                 flow3D_smooth=0, min_size=15, max_size_fraction=.4, niter=None, augment=False,
                                 tile_overlap=.1, bsize=None)
        props = regionprops_table(masks.astype(np.int32), properties=("label", "centroid"))
        ids = np.asarray(props["label"], dtype=np.int32)
        # Preserve the benchmark adapter's serialization precision before sampling features.
        queries = np.column_stack([props[f"centroid-{i}"] for i in range(3)]).astype(np.float32).astype(np.float64)
        scores = np.asarray(mean(expit(np.asarray(flows[2], dtype=np.float32)), labels=masks, index=ids), dtype=np.float32)
        feature, patch, geometry = extractor.extract(volume, queries)
        # Cellpose's import enables TF32 globally; the fitted FP32 head used it disabled.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        head, state = load_head(head_path, cfg)
        refined = refine_cached(head, queries, feature, patch, geometry, volume.shape, cfg)
    save_npz(output_path, **refined, scores=scores, baseline_centers_zyx=queries)
    write_json(Path(output_path).with_suffix(".json"), {"created_utc": now(), "source": state["source"],
               "input_sha256": sha(volume_path), "checkpoint_sha256": sha(head_path), "prediction_sha256": sha(output_path),
               "feature_array_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                        for name, value in (("feature", feature), ("patch", patch), ("geometry", geometry))},
               "seconds": time.perf_counter() - started, "gpu_queue_seconds": queued,
               "annotation_access": False, "cached_prediction_or_feature_access": False,
               "read_guard": "Python audit hook; renamed raw input only, no kernel sandbox"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure()
    if args.volume:
        if not args.head or not args.output:
            parser.error("Raw replay requires --head and --output")
        raw_inference(args.volume, args.head, args.output)
    else:
        cached_inference()


if __name__ == "__main__":
    main()
