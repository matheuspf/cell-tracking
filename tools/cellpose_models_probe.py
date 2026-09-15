"""Pinned additional Cellpose models on the existing six-clip, 12-frame pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work/cellpose-embedding-probe-20260914"
OUT = REPO / "results/cellpose-embedding-probe-20260914"
SCREEN = REPO / "work/detector-screen-20260914"
GPU_LOCK = Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); tmp.replace(p)


def prepare():
    panel = read(SCREEN / "panel.json")
    rows = [r for r in panel["frames"] if r["role"] == "pilot"]
    assert len(rows) == 12 and len({r["dataset"] for r in rows}) == 6
    models = {name: read(WORK / "weights" / f"{name}.json") for name in ("cpdino", "cpsam_v2")}
    for model in models.values():
        assert sha(model["path"]) == model["sha256"]
    plan = {"study": "cellpose-embedding-probe-20260914", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frames": rows, "image_sha256": {r["key"]: sha(r["image_path"]) for r in rows}, "models": models,
            "cellpose_revision": subprocess.check_output(["git", "-C", "/home/mpf/code/kaggle/cellpose", "rev-parse", "HEAD"], text=True).strip(),
            "cellpose_source": {name: sha(Path('/home/mpf/code/kaggle/cellpose/cellpose') / name) for name in ('models.py', 'vit.py', 'core.py', 'transforms.py', 'dynamics.py')},
            "detector_recipe": {"native_ZYX": True, "anisotropy": 4.0, "do_3D": True, "normalize": True,
                                "batch_size": 4, "diameter": None, "flow_threshold": .4, "cellprob_threshold": 0.,
                                "min_size": 15, "max_size_fraction": .4, "niter": None, "augment": False,
                                "tile_overlap": .1, "bsize": None, "float_and_integer_centroids": True},
            "union_rule": "Keep all current DINO-B proposals, then append alternate-model centers only when their physical distance from every retained center exceeds 2um; within each added bank use descending original confidence. No cross-model confidence calibration.",
            "native_embedding": "Frozen primary and secondary TemporalUNet encoders and existing node transformers, freshly sampled at each proposed point; no transfer of cached incumbent node identities.",
            "evaluation": "All new pilot predictions and native pair scores frozen before scoring. Detector recall at1-7um and sparse supported-edge ranking. This 12-frame pilot does not measure a complete-clip competition score.",
            "exposure": "All 6 pilot clips are included in the incumbent secondary training manifest. No new fit; exploratory integration/proposal test, not held-out generalization.",
            "script_sha256": sha(__file__)}
    dest = OUT / "pilot-plan.json"
    if dest.exists():
        old = read(dest)
        assert {k:v for k,v in old.items() if k != "created_utc"} == {k:v for k,v in plan.items() if k != "created_utc"}
    else:
        write(dest, plan)
    print(json.dumps({"frames": len(rows), "models": list(models), "plan": str(dest)}), flush=True)


def deny_labels():
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = str(Path(os.fsdecode(args[0])).resolve())
            if ".geff" in name or "ground_truth.json" in name or "/evaluation/gt/" in name:
                raise PermissionError("Annotation reads disabled during detector/embedding inference")
    sys.addaudithook(audit)


def detect(name):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["NUMBA_NUM_THREADS"] = "4"
    deny_labels()
    import numpy as np
    import torch
    import cv2
    from cellpose import models
    from scipy.ndimage import mean
    from scipy.special import expit
    from skimage.measure import regionprops_table
    import tifffile
    from tools.detector_screen.cellpose_adapter import gpu_lock

    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(4)
    plan = read(OUT / "pilot-plan.json")
    assert plan["script_sha256"] == sha(__file__)
    checkpoint = plan["models"][name]
    assert sha(checkpoint["path"]) == checkpoint["sha256"]
    for source, expected in plan['cellpose_source'].items():
        assert sha(Path('/home/mpf/code/kaggle/cellpose/cellpose') / source) == expected
    folder = WORK / "predictions" / name; folder.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(device=torch.device("cpu"), pretrained_model=checkpoint["path"], use_bfloat16=True)
    expected_backbone = "dino_vitl" if name == "cpdino" else "sam_vitl"
    assert model.backbone == expected_backbone
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
    state = {k.removeprefix("module."): v for k,v in state.items()}
    missing = model.net.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys and not (set(missing.missing_keys) - {"diam_mean", "diam_labels"})
    del state
    for row in plan["frames"]:
        dest = folder / (row["key"] + ".npz"); receipt = dest.with_suffix(".json")
        if receipt.exists():
            old = read(receipt)
            assert old["plan_sha256"] == sha(OUT / "pilot-plan.json") and sha(dest) == old["prediction_sha256"]
            continue
        assert sha(row["image_path"]) == plan["image_sha256"][row["key"]]
        volume = np.load(row["image_path"], allow_pickle=False)
        assert list(volume.shape) == row["shape"] and np.isfinite(volume).all()
        began = time.perf_counter()
        with gpu_lock(GPU_LOCK) as queue:
            torch.cuda.reset_peak_memory_stats()
            model.net.to("cuda:0"); model.device = torch.device("cuda:0"); model.gpu = True
            try:
                masks, flows, _ = model.eval(volume, z_axis=0, channel_axis=None, do_3D=True,
                    anisotropy=row["spacing_um"][0]/row["spacing_um"][1], batch_size=4,
                    normalize=True, diameter=None, resample=True, flow_threshold=.4,
                    cellprob_threshold=0., flow3D_smooth=0, min_size=15,
                    max_size_fraction=.4, niter=None, augment=False, tile_overlap=.1, bsize=None)
                torch.cuda.synchronize(); peak = torch.cuda.max_memory_reserved()
            finally:
                model.net.to("cpu"); model.device = torch.device("cpu"); model.gpu = False; torch.cuda.empty_cache()
        cell_score = np.asarray(flows[2], dtype=np.float32)
        assert masks.shape == volume.shape == cell_score.shape
        props = regionprops_table(masks.astype(np.int32), properties=("label", "centroid", "area"))
        ids = np.asarray(props["label"], np.int32)
        centers = np.column_stack([props[f"centroid-{a}"] for a in range(3)]).astype(np.float32)
        scores = np.asarray(mean(expit(cell_score), masks, ids), np.float32) if len(ids) else np.empty(0, np.float32)
        assert np.isfinite(centers).all() and np.isfinite(scores).all()
        assert len(centers) == len(scores) and ((centers >= 0) & (centers <= np.asarray(volume.shape)-1)).all()
        np.savez_compressed(dest, centers_zyx=centers, scores=scores, instance_ids=ids, volumes_voxels=np.asarray(props["area"], np.int32))
        tifffile.imwrite(folder / f'{row["key"]}-masks.tif', masks.astype(np.uint32), compression="zlib", maxworkers=4, metadata={"axes":"ZYX"})
        item = {"key": row["key"], "model": name, "candidates": len(centers), "backbone": model.backbone,
                "processing_seconds_excluding_queue": time.perf_counter()-began-queue, "gpu_queue_seconds": queue,
                "gpu_peak_reserved_bytes": peak, "plan_sha256": sha(OUT / "pilot-plan.json"),
                "prediction_sha256": sha(dest), "mask_sha256": sha(folder / f'{row["key"]}-masks.tif'),
                "all_learned_weights_loaded": True, "missing_metadata": list(missing.missing_keys)}
        write(receipt, item); print(json.dumps(item), flush=True)
        del masks, flows, cell_score, volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare", "detect")); parser.add_argument("--model", choices=("cpdino", "cpsam_v2"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    else:
        if args.model is None: parser.error("--model is required for detect")
        detect(args.model)
