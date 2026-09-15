"""Frozen pretrained Cellpose cpdino-vitb 3D center-screen adapter.

Uses the upstream CellposeModel.eval implementation and default normalization,
flows, and mask reconstruction. Physical anisotropy comes only from image
metadata; this adapter never reads annotations or trains a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        started = time.perf_counter()
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield time.perf_counter() - started
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def configure_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "4"


def run(args):
    configure_environment()
    import numpy as np
    import torch
    import torchvision
    import cv2
    from scipy.special import expit
    from scipy.ndimage import mean
    from skimage.measure import regionprops_table
    import tifffile
    from cellpose import models

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(4)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)  # Prevent upstream fallback to another model.
    frames = json.loads(args.panel.read_text())["frames"]
    if args.role:
        frames = [frame for frame in frames if frame["role"] == args.role]
    if args.keys:
        frames = [frame for frame in frames if frame["key"] in args.keys]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise ValueError("No selected frames")
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    preset = {
        "model": "cpdino-vitb",
        "checkpoint_sha256": sha256(checkpoint),
        "cellpose_revision": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "dinov3_revision": subprocess.check_output(["git", "-C", str(args.dinov3_repo), "rev-parse", "HEAD"], text=True).strip(),
        "input": "native ZYX single fluorescence channel, z_axis=0",
        "do_3D": True,
        "anisotropy": "spacing_um[0] / spacing_um[1]; equal XY spacing required",
        "diameter": None,
        "resample": True,
        "normalize": True,
        "normalization_description": "upstream global 3D 1st/99th percentiles",
        "use_bfloat16": True,
        "batch_size": args.batch_size,
        "bsize": None,
        "effective_bsize": 384,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "flow3D_smooth": 0,
        "min_size": 15,
        "max_size_fraction": 0.4,
        "niter": None,
        "effective_niter": 200,
        "augment": False,
        "tile_overlap": 0.1,
        "center_definition": "geometric centroid of each reconstructed 3D instance in native ZYX coordinates",
        "score_definition": "mean sigmoid of upstream summed orthogonal cell-score logits over instance; uncalibrated confidence proxy",
        "annotation_access": "none",
    }
    preset_hash = hashlib.sha256(json.dumps(preset, sort_keys=True).encode()).hexdigest()
    pending = []
    for frame in frames:
        sidecar = args.output / f"{frame['key']}.json"
        prediction = args.output / f"{frame['key']}.npz"
        if sidecar.exists() and prediction.exists() and not args.force:
            if json.loads(sidecar.read_text())["preset_sha256"] != preset_hash:
                raise ValueError(f"Preset mismatch: {sidecar}")
            print(f"Already complete: {frame['key']}", flush=True)
        else:
            pending.append(frame)
    assets = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": "https://github.com/MouseLand/cellpose",
        "repo_path": str(args.repo),
        "code_license": "BSD-3-Clause",
        "dinov3_repo_url": "https://github.com/facebookresearch/dinov3",
        "dinov3_code_license": "DINOv3 License, 2025-08-19",
        "checkpoint_path": str(checkpoint),
        "checkpoint_source": "https://huggingface.co/mouseland/cellpose-sam",
        "checkpoint_license_metadata": "BSD-3-Clause in Hugging Face model card",
        "training_data_license_statement": "Upstream Cellpose README states all Cellpose models trained on CC-BY-NC data; this is separate from code/card licensing",
        "environment": sys.prefix,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "panel_sha256": sha256(args.panel),
        "preset": preset,
        "preset_sha256": preset_hash,
        "requested_frames": [frame["key"] for frame in frames],
        "gpu_lock": str(args.gpu_lock),
        "gpu_lock_scope": "entire upstream eval per frame; model moved to CPU and CUDA cache released before unlocking",
    }
    write_json(args.artifacts / "assets.json", assets)
    if not pending:
        return
    # Construct and audit weights on CPU; DINOv3 pretrained=False does not fetch Meta weights.
    started = time.perf_counter()
    model = models.CellposeModel(device=torch.device("cpu"), pretrained_model=str(checkpoint), use_bfloat16=True)
    if model.backbone != "dino_vitb":
        raise ValueError(f"Wrong detected backbone: {model.backbone}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.net.load_state_dict(state, strict=False)
    # The published checkpoint predates two nontrainable diameter metadata
    # parameters added to BaseModel. Upstream initializes both to 30; eval with
    # diameter=None never uses them to rescale the image.
    allowed_metadata = {"diam_labels", "diam_mean"}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed_metadata:
        raise ValueError(f"Checkpoint architecture mismatch: {incompatible}")
    default_metadata = {}
    for name in incompatible.missing_keys:
        parameter = getattr(model.net, name)
        if parameter.requires_grad or not torch.all(parameter == 30):
            raise ValueError(f"Unexpected default metadata parameter: {name}")
        default_metadata[name] = parameter.tolist()
    del state
    model_load_seconds = time.perf_counter() - started
    write_json(args.artifacts / "model_load.json", {
        "seconds": model_load_seconds,
        "parameters": sum(parameter.numel() for parameter in model.net.parameters()),
        "backbone": model.backbone,
        "all_learned_parameters_matched": True,
        "missing_nontrainable_metadata_defaults": default_metadata,
        "unexpected_state_dict_keys": incompatible.unexpected_keys,
        "pretrained_model": model.pretrained_model,
    })
    phase_times = {}
    for method_name, phase_name in (("_run_net", "network_and_resize"), ("_compute_masks", "mask_reconstruction")):
        original = getattr(model, method_name)

        def measured(*a, _original=original, _phase=phase_name, **kw):
            began = time.perf_counter()
            result = _original(*a, **kw)
            phase_times[_phase] = time.perf_counter() - began
            return result

        setattr(model, method_name, measured)
    records = []
    for frame in pending:
        key = frame["key"]
        print(f"FRAME START {key}", flush=True)
        started = time.perf_counter()
        try:
            image_path = Path(frame["image_path"])
            volume = np.load(image_path, allow_pickle=False)
            spacing = np.asarray(frame["spacing_um"], dtype=np.float64)
            if volume.ndim != 3 or not np.isfinite(volume).all() or not np.isclose(spacing[1], spacing[2]):
                raise ValueError("Expected finite native ZYX with equal XY pixel spacing")
            loaded = time.perf_counter()
            anisotropy = float(spacing[0] / spacing[1])
            phase_times.clear()
            with gpu_lock(args.gpu_lock) as queue_seconds:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                torch.cuda.reset_peak_memory_stats()
                gpu_started = time.perf_counter()
                model.device = torch.device("cuda:0")
                model.gpu = True
                try:
                    model.net.to(model.device)
                    masks, flows, styles = model.eval(
                        volume, z_axis=0, channel_axis=None, do_3D=True,
                        anisotropy=anisotropy, batch_size=args.batch_size,
                        normalize=True, diameter=None, resample=True,
                        flow_threshold=0.4, cellprob_threshold=0.0,
                        flow3D_smooth=0, min_size=15, max_size_fraction=0.4,
                        niter=None, augment=False, tile_overlap=0.1, bsize=None,
                    )
                    torch.cuda.synchronize()
                    gpu_peak = torch.cuda.max_memory_allocated()
                    gpu_reserved_peak = torch.cuda.max_memory_reserved()
                finally:
                    model.net.to("cpu")
                    model.device = torch.device("cpu")
                    model.gpu = False
                    torch.cuda.empty_cache()
                gpu_seconds = time.perf_counter() - gpu_started
            cell_score = np.asarray(flows[2], dtype=np.float32)
            if masks.shape != volume.shape or cell_score.shape != volume.shape:
                raise ValueError(f"Native shape mismatch: {masks.shape}, {cell_score.shape}, {volume.shape}")
            if not np.isfinite(cell_score).all():
                raise ValueError("Nonfinite cell scores")
            props = regionprops_table(masks.astype(np.int32), intensity_image=cell_score, properties=("label", "centroid", "area", "intensity_mean"))
            ids = np.asarray(props["label"], dtype=np.int32)
            centers = np.column_stack([props[f"centroid-{axis}"] for axis in range(3)]).astype(np.float32)
            scores = np.asarray(mean(expit(cell_score), masks, ids), dtype=np.float32)
            if centers.size and (np.any(centers < 0) or np.any(centers > np.asarray(volume.shape) - 1)):
                raise ValueError("Out-of-bounds native center")
            artifact_dir = args.artifacts / "volumes"
            artifact_dir.mkdir(exist_ok=True)
            tifffile.imwrite(artifact_dir / f"{key}-masks.tif", masks.astype(np.uint32), compression="zlib", maxworkers=4, metadata={"axes": "ZYX"})
            np.savez_compressed(artifact_dir / f"{key}-cell-scores.npz", cell_score_zyx=cell_score)
            np.savez_compressed(
                args.output / f"{key}.npz", centers_zyx=centers, scores=scores,
                instance_ids=ids, volumes_voxels=np.asarray(props["area"], dtype=np.int32),
                mean_cell_scores=np.asarray(props["intensity_mean"], dtype=np.float32),
            )
            finished = time.perf_counter()
            record = {
                "key": key, "dataset": frame["dataset"], "time": frame["time"], "role": frame["role"],
                "preset_sha256": preset_hash, "input_sha256": sha256(image_path),
                "input_shape_zyx": list(volume.shape), "spacing_um": frame["spacing_um"],
                "anisotropy": anisotropy, "internal_shape_zyx": [int(volume.shape[0] * anisotropy), *volume.shape[1:]],
                "n_centers": len(centers), "foreground_voxels": int(np.count_nonzero(masks)),
                "timing_seconds": {"load": loaded - started, **phase_times, "upstream_eval_and_model_transfer": gpu_seconds, "total": finished - started - queue_seconds, "gpu_lock_wait": queue_seconds, "wall_including_gpu_queue": finished - started},
                "gpu_peak_allocated_bytes": gpu_peak, "gpu_peak_reserved_bytes": gpu_reserved_peak,
                "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
            write_json(args.output / f"{key}.json", record)
            records.append(record)
            print(f"FRAME DONE {key}: {len(centers)} centers, {record['timing_seconds']['total']:.2f} compute seconds, {queue_seconds:.2f} seconds GPU queue", flush=True)
            del masks, flows, styles, cell_score, volume
        except Exception as error:
            write_json(args.artifacts / "errors" / f"{key}.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise
    write_json(args.artifacts / f"run-{int(time.time())}.json", {"model_load_seconds": model_load_seconds, "frames": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/cellpose"))
    parser.add_argument("--dinov3-repo", type=Path, default=Path("/home/mpf/code/kaggle/dinov3"))
    parser.add_argument("--panel", type=Path, default=Path("work/detector-screen-20260914/panel.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work/detector-screen-20260914/cellpose/weights/cpdino-vitb"))
    parser.add_argument("--output", type=Path, default=Path("work/detector-screen-20260914/predictions/cellpose_cpdino_vitb"))
    parser.add_argument("--artifacts", type=Path, default=Path("work/detector-screen-20260914/cellpose"))
    parser.add_argument("--gpu-lock", type=Path, default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    parser.add_argument("--role", default="pilot")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
