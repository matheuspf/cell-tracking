"""Run the official OrganoidTracker 2 position checkpoint on prepared raw frames.

Inference uses upstream normalization, temporal ordering, patch extraction,
nearest-neighbor resizing, Z interpolation, and peak calling. The adapter retains
fractional coordinates and inverts the actual pixel-center resize transform.
Only the position checkpoint is used; no links or ground-truth labels are read.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def gpu_lock_when_ready(path: Path, torch, minimum_free_bytes: int):
    """Yield only when this frame's measured GPU footprint can safely fit."""
    last_notice = 0.0
    while True:
        with gpu_lock(path):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= minimum_free_bytes:
                yield free_bytes
                return
            now = time.monotonic()
            if now - last_notice >= 30:
                print(json.dumps({"status": "waiting_for_gpu_memory", "free_gpu_gib": free_bytes / 2**30,
                                  "required_gpu_gib": minimum_free_bytes / 2**30}), flush=True)
                last_notice = now
        # The lock is released while other detectors finish or release VRAM.
        time.sleep(2)


def inverse_resize(model_coords, native_patch, model_patch, corner):
    return (model_coords + 0.5) * native_patch / model_patch - 0.5 + corner


def in_native_core(native_coords, native_patch, native_buffers, corner):
    core_min = corner + native_buffers - 0.5
    core_max = corner + native_patch - native_buffers - 0.5
    return np.all((native_coords >= core_min) & (native_coords < core_max), axis=1)


def verify_geometry(resize) -> dict:
    native = np.array([39, 327, 327])
    target = np.array([32, 416, 416])
    probes = np.array([[1, 32, 32], [9, 160, 204], [16, 240, 320], [30, 383, 383]])
    mapped = inverse_resize(probes, native, target, np.zeros(3))
    # Independently inspect nearest-neighbor samples from coordinate-ramp images.
    samples = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = native[axis]
        ramp = np.broadcast_to(np.arange(native[axis], dtype=np.float32).reshape(shape), native)
        resized = resize(ramp, target, order=0, clip=False, preserve_range=True, anti_aliasing=False)
        samples.append(resized[tuple(probes.T)])
    assert np.array_equal(np.stack(samples, axis=1), np.floor(mapped + 0.5)), "Pixel-center inverse disagrees with actual resampling"
    buffers = np.array([1, 25, 25])
    corners = [np.array([-1, -25, -25]), np.array([36, -25, -25])]
    # Include a point exactly on a tile-core boundary and the volume's last Z plane.
    native_probes = np.array([[0, 0, 0], [36.499, 127.2, 63.1], [36.5, 127.2, 63.1], [63, 255, 255]])
    ownership = np.stack([in_native_core(native_probes, native, buffers, corner) for corner in corners])
    assert np.array_equal(ownership.sum(axis=0), np.ones(len(native_probes))), "Native tile cores overlap or leave gaps"
    assert np.array_equal(ownership[:, 2], [False, True]), "Boundary has wrong tile owner"
    # Conversion of a known physical position round-trips through model coordinates.
    corner = corners[1]
    model_probe = (native_probes - corner + 0.5) * target / native - 0.5
    error = float(np.max(np.abs(inverse_resize(model_probe, native, target, corner) - native_probes)))
    assert error < 1e-10
    return {"nearest_resize_coordinate_ramps_passed": True, "tile_boundary_unique_ownership_passed": True, "roundtrip_max_error_native_px": error}


def predict_frame(model, frame: dict, args, imports) -> tuple[np.ndarray, np.ndarray, dict]:
    TimePoint, Image, split_patches, reconstruct, peak_local_max, resize = imports
    source_spacing = np.asarray(frame["spacing_um"], dtype=np.float64)
    model_spacing = np.asarray([2.0, 0.32, 0.32], dtype=np.float64)
    nominal_scale = source_spacing / model_spacing
    # Official checkpoint settings.json specifies time_window=[0,1].
    image_paths = [frame["image_path"], frame["next_path"]]
    arrays = [np.load(path, mmap_mode="r") for path in image_paths]
    if arrays[0].ndim != 3 or arrays[0].shape != arrays[1].shape:
        raise ValueError(f"Expected aligned current/next ZYX images: {[a.shape for a in arrays]}")
    full_images = {TimePoint(i): Image(array) for i, array in enumerate(arrays)}
    buffers = np.asarray([1, 32, 32], dtype=np.int64)
    model_patch = np.asarray([32, args.patch_xy + 64, args.patch_xy + 64], dtype=np.int64)
    native_patch = np.asarray([int(size / scale) for size, scale in zip(model_patch, nominal_scale)])
    native_buffers = np.asarray([int(size / scale) for size, scale in zip(buffers, nominal_scale)])
    if np.any(native_patch - 2 * native_buffers <= 0):
        raise ValueError("Nonpositive tile stride")
    if float(np.max(arrays[0])) == float(np.min(arrays[0])) and float(np.max(arrays[1])) == float(np.min(arrays[1])):
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), {"flat_input": True, "patches": 0}
    points: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    patches_done = 0
    heatmap_min, heatmap_max = float("inf"), float("-inf")
    for patch in split_patches(
        TimePoint(0), full_images,
        patch_shape_zyx_px=tuple(model_patch),
        buffer_size_zyx_px=tuple(buffers),
        scale_factors_zyx=tuple(nominal_scale),
        intensity_quantiles=(0.01, 0.99),
        tophat_mask=0,
    ):
        input_array = np.empty((*model_patch, 2), dtype=np.float32)
        for channel in range(2):
            input_array[..., channel] = resize(
                patch.array[..., channel], output_shape=tuple(model_patch), order=0,
                clip=False, preserve_range=True, anti_aliasing=False,
            )
        prediction = model.predict(input_array[None], verbose=0)[0, ..., 0]
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite network prediction")
        heatmap_min = min(heatmap_min, float(prediction.min()))
        heatmap_max = max(heatmap_max, float(prediction.max()))
        interpolated, z_divisor = reconstruct(prediction, 5)
        coords = peak_local_max(
            interpolated, min_distance=6, threshold_abs=args.threshold, exclude_border=False,
        )
        if len(coords):
            scores = interpolated[tuple(coords.T)]
            model_coords = coords.astype(np.float64)
            model_coords[:, 0] = model_coords[:, 0] / z_divisor - 1
            # skimage.resize's grid uses pixel centers: inverse is (p+.5)/scale-.5.
            # Invert the actual integer crop size, not the requested nominal scale.
            corner = np.asarray(patch.corner_zyx, dtype=np.float64)
            native_coords = inverse_resize(model_coords, native_patch, model_patch, corner)
            # Give each native tile core exclusive ownership. This also avoids
            # upstream's upper-Z buffer check against the interpolated Z length.
            inside = in_native_core(native_coords, native_patch, native_buffers, corner)
            inside &= np.all((native_coords >= 0) & (native_coords < np.asarray(arrays[0].shape)), axis=1)
            points.append(native_coords[inside])
            confidences.append(scores[inside])
        patches_done += 1
    centers = np.concatenate(points, axis=0).astype(np.float32) if points else np.empty((0, 3), np.float32)
    scores = np.concatenate(confidences, axis=0).astype(np.float32) if confidences else np.empty(0, np.float32)
    if len(scores):
        order = np.argsort(-scores, kind="stable")
        centers, scores = centers[order], scores[order]
    metadata = {
        "image_paths": image_paths,
        "native_shape_zyx": list(arrays[0].shape),
        "source_spacing_um_zyx": source_spacing.tolist(),
        "model_spacing_um_zyx": model_spacing.tolist(),
        "nominal_scale_factors_zyx": nominal_scale.tolist(),
        "actual_scale_factors_zyx": (model_patch / native_patch).tolist(),
        "model_patch_shape_zyx": model_patch.tolist(),
        "native_patch_shape_zyx": native_patch.tolist(),
        "native_buffer_zyx": native_buffers.tolist(),
        "patches": patches_done,
        "heatmap_min": heatmap_min,
        "heatmap_max": heatmap_max,
        "centers": len(centers),
    }
    return centers, scores, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/OrganoidTracker"))
    parser.add_argument("--panel", type=Path, default=Path("work/detector-screen-20260914/panel.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("work/detector-screen-20260914/organoid"))
    parser.add_argument("--model", type=Path, default=Path("work/detector-screen-20260914/organoid/models/model_positions"))
    parser.add_argument("--output", type=Path, default=Path("work/detector-screen-20260914/predictions/organoid"))
    parser.add_argument("--role", choices=["pilot", "assessment", "all"], default="pilot")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patch-xy", type=int, default=352)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--minimum-free-gpu-gib", type=float, default=13.0)
    parser.add_argument("--lock", type=Path, default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    args = parser.parse_args()
    if args.patch_xy % 32:
        parser.error("--patch-xy must be a multiple of 32")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    import _keras_environment
    _keras_environment.activate()
    import keras
    import torch
    from keras.src.backend.torch.core import device_scope
    from skimage.feature import peak_local_max
    from skimage.transform import resize
    from organoid_tracker.core import TimePoint
    from organoid_tracker.core.images import Image
    from organoid_tracker.neural_network.position_detection_cnn.position_predictor import _split_into_patches
    from organoid_tracker.neural_network.position_detection_cnn.peak_calling import reconstruct_volume
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    settings = json.loads((args.model / "settings.json").read_text())
    if settings.get("type") != "positions" or settings.get("time_window") != [0, 1]:
        raise ValueError(f"Unexpected checkpoint settings: {settings}")
    load_start = time.perf_counter()
    with device_scope("cpu"):
        model = keras.saving.load_model(args.model / "model.keras", compile=False, safe_mode=True)
    load_seconds = time.perf_counter() - load_start
    if model.input_shape != (1, 32, None, None, 2):
        raise ValueError(f"Unexpected input shape: {model.input_shape}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this screen")
    frames = [f for f in json.loads(args.panel.read_text())["frames"] if args.role == "all" or f["role"] == args.role]
    if args.limit:
        frames = frames[:args.limit]
    config = {
        "model_name": "OrganoidTracker 2 pretrained intestinal organoid position network",
        "code_url": "https://github.com/jvzonlab/OrganoidTracker",
        "code_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "model_url": "https://zenodo.org/records/18479952",
        "model_sha256": sha256(args.model / "model.keras"),
        "model_settings": settings,
        "model_parameter_count": model.count_params(),
        "training_domain": "mouse intestinal organoids; confocal H2B-mCherry; XYZ 0.32,0.32,2 micrometers",
        "model_license": "CC-BY-4.0 (Zenodo record)",
        "code_license": "MIT neural network files; GPL-2.0 other upstream files",
        "runtime": sys.executable,
        "runtime_packages": {n: importlib.metadata.version(n) for n in ["torch", "keras", "numpy", "scipy", "scikit-image", "h5py"]},
        "model_load_seconds": load_seconds,
        "model_input_shape": list(model.input_shape),
        "threshold": args.threshold,
        "peak_min_distance_px": 6,
        "mid_layers": 5,
        "patch_xy_unbuffered": args.patch_xy,
        "intensity_quantiles": [0.01, 0.99],
        "background_subtraction": False,
        "prediction_is_probability": False,
        "coordinate_transform": "native=(model+0.5)*native_patch/model_patch-0.5+crop_corner; Z first inverted from upstream reconstruction by z/6-1",
        "adapter_changes": ["preserve fractional coordinates", "invert actual pixel-center resize transform", "exclusive native-core tile ownership instead of upper-Z buffer check against interpolated length"],
        "panel_sha256": sha256(args.panel),
        "ground_truth_access": False,
        "minimum_free_gpu_gib_before_frame": args.minimum_free_gpu_gib,
    }
    config_path = args.artifacts / f"config-{args.role}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    imports = (TimePoint, Image, _split_into_patches, reconstruct_volume, peak_local_max, resize)
    geometry = verify_geometry(resize)
    (args.artifacts / "geometry-verification.json").write_text(json.dumps(geometry, indent=2) + "\n")
    for index, frame in enumerate(frames):
        output = args.output / f"{frame['key']}.npz"
        sidecar = output.with_suffix(".json")
        if output.exists() and sidecar.exists():
            old = json.loads(sidecar.read_text())
            if old.get("model_sha256") != config["model_sha256"] or old.get("threshold") != args.threshold or old.get("patch_xy_unbuffered") != args.patch_xy:
                raise ValueError(f"Existing output uses different configuration: {output}")
            print(json.dumps({"key": frame["key"], "status": "already_complete"}), flush=True)
            continue
        wait_start = time.perf_counter()
        with gpu_lock_when_ready(args.lock, torch, int(args.minimum_free_gpu_gib * 2**30)) as free_gpu_bytes:
            wait_seconds = time.perf_counter() - wait_start
            start = time.perf_counter()
            with device_scope("cuda:0"), torch.inference_mode():
                model.to("cuda:0")
                torch.cuda.reset_peak_memory_stats()
                try:
                    centers, scores, detail = predict_frame(model, frame, args, imports)
                    torch.cuda.synchronize()
                    max_memory = torch.cuda.max_memory_allocated()
                finally:
                    model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
            idle_gpu_memory = torch.cuda.memory_allocated()
            seconds = time.perf_counter() - start
        np.savez_compressed(output, centers_zyx=centers, scores=scores)
        detail.update({
            "key": frame["key"], "dataset": frame["dataset"], "time": frame["time"], "role": frame["role"],
            "inference_seconds": seconds, "gpu_lock_wait_seconds": wait_seconds,
            "peak_gpu_memory_bytes": max_memory, "model_sha256": config["model_sha256"],
            "free_gpu_memory_bytes_before_frame": free_gpu_bytes,
            "idle_gpu_memory_bytes_after_frame": idle_gpu_memory,
            "threshold": args.threshold, "patch_xy_unbuffered": args.patch_xy,
            "config_path": str(config_path.resolve()),
        })
        sidecar.write_text(json.dumps(detail, indent=2) + "\n")
        print(json.dumps({"index": index + 1, "total": len(frames), "key": frame["key"], "centers": len(centers), "seconds": seconds, "peak_gpu_gb": max_memory / 1e9}), flush=True)


if __name__ == "__main__":
    main()
