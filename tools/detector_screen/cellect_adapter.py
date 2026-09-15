#!/usr/bin/env python3
"""Run CELLECT's pretrained detector on the common native-ZYX screening panel.

The patch detector is called directly from the pinned external repository. The
intraframe grouping below follows its ``inference.py`` first-frame branch and
uses that branch's group-mean centroid. No temporal assignments or short-track
filter are applied. Adapted grouping code is GPL-2.0-only, as is upstream.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


REPO_COMMIT = "3586070926f7f1fd5d8df37456861d22bdc63236"
DETECTOR = "U-ext+-x3rd-149.0-4.6540.pth"
INTRAFRAME = "EN+-x3rd-149.0-4.6540.pth"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(path)


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        started = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield time.perf_counter() - started
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_models(repo: Path):
    sys.path.insert(0, str(repo))
    architecture = importlib.import_module("unetext3Dn_con7")
    # recoloss allocates its convolution kernel on CUDA during import. Its import
    # and all callers therefore run while holding the shared GPU lock.
    helpers = importlib.import_module("recoloss")
    detector = architecture.UNet3D(2, 6).eval()
    intraframe = architecture.EXNet(64, 6).eval()
    detector.load_state_dict(torch.load(repo / "model" / DETECTOR,
                                      map_location="cpu", weights_only=True))
    intraframe.load_state_dict(torch.load(repo / "model" / INTRAFRAME,
                                        map_location="cpu", weights_only=True))
    return detector, intraframe, helpers


def detect_with_scores(detector, helpers, volume):
    """Preserve upstream candidate selection and carry an extra score column.

    Upstream returns division probabilities, not detection scores. We append the
    foreground softmax to the feature array solely so that its existing patch
    overlap filter carries scores with their candidates; the 64 learned features
    used by the intraframe MLP are unchanged.
    """
    original = helpers.vsup256infer

    def scored_patch(model, patch, labels):
        result = original(model, patch, labels)
        positions, features, sizes, division, label, exclusion, logits = result
        if len(positions):
            b, y, x, z = positions.unbind(1)
            scores = logits[b, :, y, x, z].softmax(1)[:, 1]
        else:
            scores = features.new_empty((0,))
        features = torch.cat([features, scores[:, None]], dim=1)
        return positions, features, sizes, division, label, exclusion, logits

    helpers.vsup256infer = scored_patch
    try:
        positions, features, division, sizes, _, _ = helpers.run_unet_on_patches_infer(
            detector, volume, torch.zeros_like(volume[:, 0]))
    finally:
        helpers.vsup256infer = original
    return positions[:, 1:].float(), features[:, :64], division, sizes, features[:, 64]


def deduplicate(positions_yxz, features, division, sizes, scores, matcher, helpers, zratio):
    """Translate upstream first-frame grouping and its group-mean centroids."""
    count = len(positions_yxz)
    if count <= 2:
        return positions_yxz, scores, np.ones(count, dtype=np.int64)
    positions = positions_yxz.clone()
    positions[:, 2] *= zratio
    query_projection = positions.clone()
    query_projection[:, 2] = query_projection[:, 2] // zratio
    squares = query_projection.square().sum(1, keepdim=True)
    distances = squares.expand(count, count) + squares.expand(count, count).T
    distances.addmm_(query_projection, query_projection.T, beta=1, alpha=-2)
    indices = distances.topk(min(6, count), largest=False).indices
    while indices.shape[1] < 6:
        indices = torch.cat([indices, indices[:, -1:]], dim=1)
    indices = indices[:, 1:]
    near_features, near_positions, near_sizes, near_division = helpers.sort_feature(
        features, positions, positions, indices, sizes, sizes, division, division, n=5)
    distances = near_positions.square().sum(-1).sqrt()
    similarity = matcher(features[:, None], near_features, near_positions,
                         near_sizes, near_division).sigmoid()
    strong = (similarity[:, :-1] > 0.5).sum(1)
    high = similarity > 0.9999
    moderate = similarity > 0.6
    indices, distances, sizes = indices.cpu(), distances.cpu(), sizes.cpu()
    strong, high, moderate = strong.cpu(), high.cpu(), moderate.cpu()
    matched = set()
    groups = {}
    for i in range(count):
        if i not in matched:
            group = [i]
            if strong[i] > 0:
                # Upstream deliberately examines four of its five neighbors.
                for j in range(4):
                    if high[i, j] or (moderate[i, j] and
                                     distances[i, j] < max(5, 0.6 * sizes[i])):
                        neighbor = int(indices[i, j])
                        matched.add(neighbor)
                        group.append(neighbor)
            groups[i] = group
    covered = set()
    retained = []
    for anchor, members in groups.items():
        if anchor not in covered:
            members = np.unique(members).tolist()
            retained.append(members)
            covered.update(members)
    centers = torch.stack([positions_yxz[group].mean(0) for group in retained])
    group_scores = torch.stack([scores[group].max() for group in retained])
    return centers, group_scores, np.array([len(g) for g in retained], dtype=np.int64)


def infer_frame(frame, detector, matcher, helpers):
    started = time.perf_counter()
    raw = np.load(frame["image_path"], allow_pickle=False)
    context_path = frame.get("next_path") or frame["image_path"]
    following = np.load(context_path, allow_pickle=False)
    if raw.ndim != 3 or following.shape != raw.shape:
        raise ValueError(f"Expected matching ZYX frames, got {raw.shape} and {following.shape}")
    spacing = np.asarray(frame["spacing_um"], dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"Invalid native-ZYX spacing: {spacing}")
    if not np.isclose(spacing[1], spacing[2], rtol=1e-3):
        raise ValueError("CELLECT requires matching X/Y spacing for this unresampled screen")
    zratio = float(spacing[0] / spacing[1])
    # Upstream tim() transposes TIFF ZYX to YXZ, despite naming its axes XYZ.
    raw_yxz = np.stack([raw.transpose(1, 2, 0), following.transpose(1, 2, 0)])
    tensor = torch.from_numpy(np.clip(raw_yxz, 0, 65535).astype(np.float32))[None].cuda()
    positive = tensor[tensor > 0]
    if not positive.numel():
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), {
            "raw_candidates": 0, "count": 0, "native_shape_zyx": list(raw.shape),
            "zratio": zratio, "inference_seconds": time.perf_counter() - started,
            "all_zero_input": True}, np.empty(0, np.int64)
    minimum = positive.min()
    del positive
    tensor = torch.log1p(torch.clamp_min(tensor, minimum) + 1900)
    # The source handles shallow Z stacks but has negative slice starts for XY
    # smaller than one patch. Pad small XY at the high edge, then reject padding.
    original_yxz = tensor.shape[2:]
    pad_y = max(0, 256 - tensor.shape[2])
    pad_x = max(0, 256 - tensor.shape[3])
    if pad_y or pad_x:
        tensor = F.pad(tensor, (0, 0, 0, pad_x, 0, pad_y), value=float(tensor.min()))
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    positions, features, division, sizes, scores = detect_with_scores(detector, helpers, tensor)
    valid = (positions >= 0).all(1)
    for axis, length in enumerate(original_yxz):
        valid &= positions[:, axis] < length
    positions, features, division, sizes, scores = (
        value[valid] for value in (positions, features, division, sizes, scores))
    torch.cuda.synchronize()
    detected = time.perf_counter()
    raw_count = len(positions)
    centers, scores, group_sizes = deduplicate(
        positions, features, division, sizes, scores, matcher, helpers, zratio)
    centers = centers[:, [2, 0, 1]].cpu().numpy().astype(np.float32)
    scores = scores.cpu().numpy().astype(np.float32)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    metadata = {
        "native_shape_zyx": list(raw.shape), "input_dtype": str(raw.dtype),
        "spacing_um": spacing.tolist(), "zratio": zratio,
        "input_pair": [frame["image_path"], str(context_path)],
        "context_fallback_to_target": context_path == frame["image_path"],
        "input_shape_bcyxz": list(tensor.shape), "extra_high_xy_padding": [pad_y, pad_x],
        "raw_candidates": raw_count, "count": len(centers),
        "preprocess_seconds": prepared - started, "detector_seconds": detected - prepared,
        "intraframe_seconds": finished - detected, "inference_seconds": finished - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    return centers, scores, metadata, group_sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/CELLECT"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--gpu-lock", type=Path,
                        default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    parser.add_argument("--role", choices=["pilot", "assessment", "all"], default="all")
    parser.add_argument("--key", action="append")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    artifact_root = args.artifact_root or args.panel.parent / "cellect"
    output_root = args.output_root or args.panel.parent / "predictions" / "cellect"
    output_root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REPO_COMMIT:
        raise ValueError(f"Expected reviewed CELLECT commit {REPO_COMMIT}; found {commit}")
    provenance = {
        "method": "cellect", "source_url": "https://github.com/zzz333za/CELLECT",
        "repo_path": str(args.repo), "repo_commit": commit, "license": "GPL-2.0 (repository LICENSE)",
        "checkpoint_license": "No separate checkpoint license found; checkpoints distributed in source repository",
        "architecture": "unetext3Dn_con7.UNet3D(2,6) + EXNet(64,6)",
        "checkpoints": [{"filename": name, "size_bytes": (args.repo / "model" / name).stat().st_size,
                         "sha256": sha256(args.repo / "model" / name),
                         "source_url": f"https://github.com/zzz333za/CELLECT/blob/{commit}/model/{name}"}
                        for name in [DETECTOR, INTRAFRAME]],
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "fixed_settings": {"xy_pool": 1, "high": 65535, "low": 0, "thresh0": 0,
                           "enhance": 1, "log_offset": 1900, "zratio": "native spacing_z / spacing_y",
                           "resampling": "none", "tile_yxz": [256, 256, 32],
                           "tile_step_yxz": [248, 248, 28], "detection_threshold": "upstream hard candidate selection"},
        "score_definition": "Max source foreground softmax (Uout channel 1) across group members; unused for selection",
        "centroid_definition": "Mean of source first-frame intraframe-group candidate coordinates",
        "tracking_applied": False, "cpu_threads": 4,
        "notes": ["Source masks 2 voxels at XY tile boundaries and 1 voxel at Z boundaries",
                  "High-side XY padding added only when source would otherwise slice from a negative start",
                  "Uses target and next frame; target is duplicated only if no next_path is supplied",
                  "No training or assessment-label tuning"],
    }
    save_json(artifact_root / "provenance.json", provenance)
    panel = json.loads(args.panel.read_text())
    frames = panel if isinstance(panel, list) else panel["frames"]
    frames = [frame for frame in frames if (args.role == "all" or frame["role"] == args.role)
              and (not args.key or frame["key"] in args.key)]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("No panel frames match selection")
    detector = matcher = helpers = None
    records = []
    for frame in frames:
        destination = output_root / (frame["key"] + ".npz")
        sidecar = destination.with_suffix(".json")
        if destination.exists() and sidecar.exists() and not args.overwrite:
            print(json.dumps({"key": frame["key"], "status": "cached"}), flush=True)
            records.append(json.loads(sidecar.read_text()))
            continue
        print(json.dumps({"key": frame["key"], "status": "waiting_for_gpu"}), flush=True)
        with gpu_lock(args.gpu_lock) as waited:
            if detector is None:
                detector, matcher, helpers = load_models(args.repo)
            detector.cuda()
            matcher.cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                centers, scores, metadata, group_sizes = infer_frame(frame, detector, matcher, helpers)
            detector.cpu()
            matcher.cpu()
            torch.cuda.empty_cache()
        if centers.shape != (len(scores), 3) or not np.isfinite(centers).all() or not np.isfinite(scores).all():
            raise ValueError("Detector produced invalid coordinates or scores")
        metadata.update({"key": frame["key"], "dataset": frame["dataset"], "time": frame["time"],
                         "role": frame["role"], "method": "cellect", "gpu_lock_wait_seconds": waited,
                         "provenance": str(artifact_root / "provenance.json")})
        with destination.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, centers_zyx=centers, scores=scores, group_sizes=group_sizes)
        destination.with_suffix(".npz.tmp").replace(destination)
        save_json(sidecar, metadata)
        records.append(metadata)
        print(json.dumps({"key": frame["key"], "status": "complete", "count": len(centers),
                          "raw_candidates": metadata["raw_candidates"],
                          "inference_seconds": metadata["inference_seconds"]}), flush=True)
    save_json(artifact_root / f"run_{args.role}.json", {"frames": records, "count": len(records)})


if __name__ == "__main__":
    main()
