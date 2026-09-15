#!/usr/bin/env python3
"""Screen the published SH-SY5Y PAC-MAP checkpoint on native ZYX volumes.

Uses the pinned upstream model, resampling, normalization, patch padding, map
reconstruction, distance-aware peak extraction and close-point merge functions.
Full-frame map assembly is necessary because the published benchmark evaluates
precut patches. This adapter does not apply its evaluation-only border mask.
Upstream PAC-MAP code is CC-BY-NC-SA-4.0; local research screen only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


REPO_COMMIT = "693c51385ab53b9fbad504793a41dff5b1ed6b77"
UNET_COMMIT = "4eb255149219dc1734142a598686dee668d75b42"
SPACING = np.array([1.9999, 0.3594, 0.3594])
PATCH = np.array([46, 256, 256])
STRIDE = np.array([40, 224, 224])
THRESHOLD = 0.1
MIN_DISTANCE = 5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def gpu_lock(path, required_gib=6.0):
    """Release the lock when other frameworks retain too much device memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("a") as handle:
        while True:
            fcntl.flock(handle, fcntl.LOCK_EX)
            free, _ = torch.cuda.mem_get_info()
            if free >= required_gib * 1024**3:
                break
            fcntl.flock(handle, fcntl.LOCK_UN)
            print(json.dumps({"status": "waiting_for_gpu_memory", "free_bytes": free}), flush=True)
            time.sleep(5)
        try:
            yield time.perf_counter() - started
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_upstream(repo, checkpoint):
    sys.path.insert(0, str(repo))
    architecture = importlib.import_module("pytorch-3dunet.pytorch3dunet.unet3d.model")
    utils = importlib.import_module("pacmap.utils")
    points = importlib.import_module("pacmap.prob2points")
    reconstruction = importlib.import_module("pacmap.depatchify")
    # Upstream passes a one-element argwhere row into math.ceil; NumPy 2 removed
    # that implicit scalar conversion. Preserve its exact scalar index value.
    condensed_to_square = utils.condensed_to_square
    utils.condensed_to_square = lambda index, count: condensed_to_square(
        np.asarray(index).item(), count)
    # Same construction as pacmap.train.create_model, without training imports.
    model = architecture.UNet3D(in_channels=1, out_channels=1,
                                is_segmentation=False, final_sigmoid=False,
                                f_maps=32, num_levels=4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval(), utils, points, reconstruction


def prepare(raw, spacing, utils):
    image = utils.rescale_voxels(raw, np.asarray(spacing), SPACING)
    resampled_shape = np.array(image.shape)
    image = utils.normalize_per_channel(image[:, None], [0.1], [99.9])[:, 0].astype(np.float32)
    padded_shape = PATCH * np.ceil(resampled_shape / PATCH).astype(int)
    padding = utils.get_padding(image, shape=padded_shape, multichannel=False)
    image = np.pad(image, padding, mode="constant")
    structure = (np.array(image.shape) - PATCH) // STRIDE + 1
    starts = list(itertools.product(*(range(0, int(length - patch + 1), int(stride))
                                     for length, patch, stride in zip(image.shape, PATCH, STRIDE))))
    covered_end = (structure - 1) * STRIDE + PATCH
    low = np.array([pair[0] for pair in padding])
    if np.any(covered_end < low + resampled_shape):
        raise ValueError("Source patch layout leaves part of the unpadded frame uncovered")
    return image, resampled_shape, padding, structure, starts


def predict_patches(image, starts, model):
    predictions = []
    for start in starts:
        slices = tuple(slice(int(s), int(s + p)) for s, p in zip(start, PATCH))
        patch = torch.from_numpy(np.ascontiguousarray(image[slices]))
        # Dataset normalization, then the second normalization in pred_model.py.
        for _ in range(2):
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-20)
        patch = patch[None, None].cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(patch)
        predictions.append(prediction[0, 0].float().cpu().numpy())
        del patch, prediction
    torch.cuda.synchronize()
    return predictions


def infer_frame(frame, model, utils, point_module, reconstruction, lock_path):
    started = time.perf_counter()
    raw = np.load(frame["image_path"], allow_pickle=False)
    spacing = np.asarray(frame["spacing_um"], dtype=float)
    if raw.ndim != 3 or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Expected a native ZYX image and three positive spacings")
    image, shape, padding, structure, starts = prepare(raw, spacing, utils)
    effective_spacing = spacing * np.array(raw.shape) / shape
    prepared = time.perf_counter()
    with gpu_lock(lock_path) as waited:
        model.cuda()
        torch.cuda.reset_peak_memory_stats()
        predicted_at = time.perf_counter()
        try:
            predictions = predict_patches(image, starts, model)
            peak_memory = torch.cuda.max_memory_allocated()
            network_seconds = time.perf_counter() - predicted_at
        finally:
            model.cpu()
            torch.cuda.empty_cache()
    post_started = time.perf_counter()
    patches = np.stack(predictions).reshape(tuple(structure) + tuple(PATCH))
    assembled = reconstruction.depatchify_merge(patches, image.shape, STRIDE)
    crop = tuple(slice(int(pair[0]), int(pair[0] + length))
                 for pair, length in zip(padding, shape))
    distance_map = assembled[crop]
    if distance_map.shape != tuple(shape) or not np.isfinite(distance_map).all():
        raise ValueError("Invalid reconstructed distance map")
    points = point_module.get_points(distance_map, min_distance=MIN_DISTANCE,
                                    threshold_abs=THRESHOLD, exclude_border=False,
                                    intensity_as_spacing=True, top_down=False,
                                    voxelsize=effective_spacing)
    raw_count = len(points)
    if len(points):
        points = utils.merge_close_points(points, voxelsize=effective_spacing,
                                          threshold=MIN_DISTANCE).astype(int)
        scores = distance_map[tuple(points.T)].astype(np.float32)
    else:
        points = np.empty((0, 3), dtype=np.int64)
        scores = np.empty(0, dtype=np.float32)
    # skimage.transform.rescale/resize uses pixel-center grid geometry.
    centers = (points.astype(float) + 0.5) * (np.array(raw.shape) / shape) - 0.5
    clipped = int(np.any((centers < 0) | (centers > np.array(raw.shape) - 1), axis=1).sum())
    centers = np.clip(centers, 0, np.array(raw.shape) - 1).astype(np.float32)
    if centers.shape != (len(scores), 3) or not np.isfinite(centers).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid native centers or distance scores")
    finished = time.perf_counter()
    metadata = {
        "key": frame["key"], "dataset": frame["dataset"], "time": frame["time"],
        "role": frame["role"], "method": "pacmap", "input_path": frame["image_path"],
        "native_shape_zyx": list(raw.shape), "spacing_um": spacing.tolist(),
        "resampled_shape_zyx": shape.tolist(), "effective_spacing_um": effective_spacing.tolist(),
        "padded_shape_zyx": list(image.shape), "padding_zyx": padding,
        "patch_structure": structure.tolist(), "patch_count": len(starts),
        "raw_candidates": raw_count, "count": len(centers),
        "boundary_coordinates_clipped": clipped,
        "score_type": "raw predicted nearest-neighbor distance amplitude in micrometers",
        "map_min": float(distance_map.min()), "map_max": float(distance_map.max()),
        "preprocess_seconds": prepared - started, "gpu_lock_wait_seconds": waited,
        "network_seconds": network_seconds, "postprocess_seconds": finished - post_started,
        "inference_seconds": finished - started - waited,
        "wall_seconds": finished - started, "peak_cuda_memory_bytes": peak_memory,
    }
    return centers, scores, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/PAC-MAP"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--role", choices=["pilot", "assessment", "all"], default="all")
    parser.add_argument("--key", action="append")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-lock", type=Path,
                        default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    root = args.panel.resolve().parent
    artifacts = args.artifact_root or root / "pacmap"
    output = args.output_root or root / "predictions" / "pacmap"
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    unet_commit = subprocess.check_output(["git", "-C", str(args.repo / "pytorch-3dunet"),
                                          "rev-parse", "HEAD"], text=True).strip()
    if commit != REPO_COMMIT or unet_commit != UNET_COMMIT:
        raise ValueError("Unexpected upstream revision")
    provenance = {
        "method": "PAC-MAP", "repo": "https://github.com/DeVosLab/PAC-MAP",
        "repo_commit": commit, "unet_fork_commit": unet_commit,
        "code_license": "CC-BY-NC-SA-4.0", "weight_record": "https://zenodo.org/records/14138806",
        "weight_record_license": "CC-BY-4.0 (verified Zenodo API metadata)",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_selection": "SH-SY5Y pacmap-finetuned-0; nuclear-specific LSFM, first published seed",
        "architecture": "UNet3D, f_maps=32, num_levels=4, one input/output, regression, no sigmoid",
        "target_spacing_um_zyx": SPACING.tolist(), "patch_shape_zyx": PATCH.tolist(),
        "patch_stride_zyx": STRIDE.tolist(), "normalization_percentiles": [0.1, 99.9],
        "patch_normalization": "two source float32 min-max normalizations, eps=1e-20",
        "precision": "CUDA autocast float16, as source pred_model.py",
        "padding_and_reconstruction": "upstream get_padding and depatchify_merge; mean overlaps",
        "threshold_abs": THRESHOLD, "threshold_source": "performance.py command-line default 0.1",
        "min_distance_um": MIN_DISTANCE, "intensity_as_spacing": True, "top_down": False,
        "source_merge_close_points_um": MIN_DISTANCE,
        "native_coordinate_conversion": "(resampled_point + 0.5) * native_shape / resampled_shape - 0.5; boundary clip",
        "adaptations": [
            "Full-volume inference joins the source patch predictions before upstream peak extraction.",
            "The spheroid bounding-box crop is omitted because competition frames are already crops.",
            "The source benchmark's evaluation-only 5 um border mask is not applied to full-frame predictions.",
            "The regression amplitude is retained as score; it is not a calibrated detection probability.",
            "NumPy 2 compatibility shim converts upstream singleton condensed-distance indices to scalars.",
            "No evaluation labels, threshold/scale tuning, temporal linking or training are used."
        ],
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "adapter_sha256": sha256(Path(__file__)),
    }
    save_json(artifacts / "provenance.json", provenance)
    panel = json.loads(args.panel.read_text())
    frames = [frame for frame in panel["frames"]
              if (args.role == "all" or frame["role"] == args.role)
              and (not args.key or frame["key"] in args.key)]
    if args.max_frames:
        frames = frames[:args.max_frames]
    model, utils, point_module, reconstruction = load_upstream(args.repo, args.checkpoint)
    records = []
    for frame in frames:
        destination = output / f"{frame['key']}.npz"
        sidecar = artifacts / "frames" / f"{frame['key']}.json"
        if destination.exists() and sidecar.exists() and not args.overwrite:
            cached = json.loads(sidecar.read_text())
            if not destination.with_suffix(".json").exists():
                save_json(destination.with_suffix(".json"), cached)
            records.append(cached)
            continue
        print(json.dumps({"key": frame["key"], "status": "starting"}), flush=True)
        centers, scores, metadata = infer_frame(frame, model, utils, point_module,
                                                reconstruction, args.gpu_lock)
        metadata["provenance"] = str(artifacts / "provenance.json")
        with destination.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, centers_zyx=centers, scores=scores)
        destination.with_suffix(".npz.tmp").replace(destination)
        save_json(sidecar, metadata)
        save_json(destination.with_suffix(".json"), metadata)
        records.append(metadata)
        print(json.dumps({"key": frame["key"], "status": "complete", "count": len(centers),
                          "inference_seconds": metadata["inference_seconds"],
                          "peak_cuda_memory_bytes": metadata["peak_cuda_memory_bytes"]}), flush=True)
    save_json(artifacts / f"run_{args.role}.json", {"frames": records, "count": len(records)})


if __name__ == "__main__":
    main()
