"""Run official AnyStar-mix weights with a frozen, physically isotropic preset.

The checkpoint is synthetic-only StarDist3D. This adapter reads current images
only, retains predicted polyhedron origins as centers, and performs no linking.
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
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "work/detector-screen-20260914"
WEIGHTS_SHA256 = "12d013468d02a3c77f772bbbb8929275fe75a994ad797c42e3d1169cb0b2b073"
CONFIG_SHA256 = "1783c3dcfd38c0f5254664aabcac2b407f4e5cf36ab491b6334784317cc832fe"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def normalize_like_upstream(raw: np.ndarray) -> np.ndarray:
    """Exact float32 clipping/min-max operations from AnyStar infer.FileData."""
    image = raw.astype(np.float32)
    upper = np.percentile(image, 99.9)
    image = np.clip(image, 0, upper)
    spread = image.max() - image.min()
    if spread == 0:
        return np.zeros_like(image)
    return (image - image.min()) / spread


def invert_resize(points, actual_zoom):
    return (np.asarray(points, np.float64).reshape(-1, 3) + 0.5) / actual_zoom - 0.5


def finish_neural_stage(generator):
    stages = []
    for stage in generator:
        if not isinstance(stage, str):
            raise RuntimeError("StarDist returned results before its NMS boundary")
        stages.append(stage)
        if stage == "nms":
            return stages
    raise RuntimeError("StarDist did not yield its NMS boundary")


def finish_cpu_stage(generator):
    result = None
    for result in generator:
        pass
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Unexpected StarDist instance result")
    return result


def mask_centroids(details, shape, actual_zoom):
    from stardist.geometry import polyhedron_to_label
    from skimage.measure import regionprops_table
    labels = polyhedron_to_label(
        details["dist"], details["points"], details["rays"], shape=shape,
        prob=details["prob"], verbose=False,
    )
    props = regionprops_table(labels, properties=("label", "centroid"))
    source_indices = np.asarray(props["label"], dtype=np.int64) - 1
    geometric = np.column_stack([props[f"centroid-{axis}"] for axis in range(3)])
    geometric = invert_resize(geometric, actual_zoom)
    scores = np.asarray(details["prob"], dtype=np.float32)[source_indices]
    return geometric, scores, source_indices


def verify_geometry(model, zoom, tile_iterator) -> dict:
    # Pixel-center inverse verified against independent coordinate-ramp images.
    native = np.asarray([64, 256, 256])
    desired = np.asarray([4.0, 1.0, 1.0])
    probes = np.asarray([[2, 3, 4], [64, 65, 66], [128, 129, 130], [250, 251, 252]])
    expected = invert_resize(probes, desired)
    observed = []
    for axis in range(3):
        axis_shape = [1, 1, 1]
        axis_shape[axis] = native[axis]
        ramp = np.broadcast_to(np.arange(native[axis], dtype=np.float64).reshape(axis_shape), native)
        result = zoom(ramp, desired, order=1, grid_mode=True, mode="nearest", prefilter=False)
        observed.append(result[tuple(probes.T)])
    max_error = float(np.max(np.abs(np.stack(observed, axis=1) - expected)))
    assert max_error < 1e-10
    actual = np.asarray([256, 256, 256]) / native
    physical_probe = np.asarray([[0, 0, 0], [30.3, 127.2, 90.8], [63, 255, 255]])
    model_probe = (physical_probe + 0.5) * actual - 0.5
    assert np.allclose(invert_resize(model_probe, actual), physical_probe, atol=1e-12)

    # Inspect the exact upstream tile iterator and verify exclusive write coverage.
    overlap = model._axes_tile_overlap("ZYXC")
    blocks = model._axes_div_by("ZYXC")
    block_overlap = tuple(int(np.ceil(o / b)) for o, b in zip(overlap, blocks))
    marker = np.zeros((256, 256, 256, 1), dtype=np.uint8)
    writes = np.zeros_like(marker)
    shapes = []
    for tile, source_slice, output_slice in tile_iterator(marker, (4, 4, 4, 1), blocks, block_overlap):
        assert tile[source_slice].shape == writes[output_slice].shape
        writes[output_slice] += 1
        shapes.append(list(tile.shape))
    assert np.all(writes == 1), "Upstream tiling loses or duplicates spatial ownership"

    # Call both actual public and split-stage StarDist methods on identical
    # deterministic neural outputs. This checks that releasing the lock at 'nms'
    # changes neither the proposed centers nor CPU polyhedron suppression.
    marker = np.zeros((64, 64, 64), dtype=np.float32)
    known = np.asarray([[5, 6, 7], [20, 30, 40], [50, 51, 52]])
    marker[tuple(known.T)] = [0.6, 0.8, 0.9]
    original_predict = model.keras_model.predict

    def known_neural_output(batch, **kwargs):
        return [batch.copy(), np.full((*batch.shape[:-1], model.config.n_rays), 3.0, dtype=np.float32)]

    model.keras_model.predict = known_neural_output
    try:
        kwargs = dict(axes="ZYX", n_tiles=(1, 1, 1), prob_thresh=0.5, nms_thresh=0.3,
                      return_labels=False, show_tile_progress=False)
        _, reference = model.predict_instances(marker, **kwargs)
        generator = model._predict_instances_generator(marker, **kwargs)
        stages = finish_neural_stage(generator)
        _, split = finish_cpu_stage(generator)
        assert np.array_equal(reference["points"], split["points"])
        assert np.array_equal(reference["prob"], split["prob"])
        assert {tuple(p) for p in split["points"]} == {tuple(p) for p in known}
        # A fully occluded source ID leaves a label gap. Confidence must still
        # follow the original ID, without relabel_sequential changing its index.
        duplicate = dict(points=np.asarray([[32, 32, 32], [32, 32, 32]]),
                         dist=np.full((2, model.config.n_rays), 3.0, np.float32),
                         prob=np.asarray([0.6, 0.9], np.float32), rays=split["rays"])
        geometric, geometric_scores, source_indices = mask_centroids(duplicate, marker.shape, np.ones(3))
        assert np.array_equal(source_indices, [1])
        assert np.array_equal(geometric_scores, np.asarray([0.9], np.float32))
        assert np.max(np.abs(geometric - 32)) < 0.25
    finally:
        model.keras_model.predict = original_predict
    return {
        "coordinate_ramp_max_error_native_px": max_error,
        "pixel_center_roundtrip_passed": True,
        "upstream_tile_write_coverage_unique": True,
        "tile_overlap_zyxc": [int(x) for x in overlap],
        "tile_divisibility_zyxc": [int(x) for x in blocks],
        "n_tiles_requested": [4, 4, 4],
        "n_tiles_actual": len(shapes),
        "tile_shapes_zyxc": [list(s) for s in sorted(set(tuple(s) for s in shapes))],
        "actual_public_method_vs_split_generator_fixture_identical": True,
        "fixture_known_centers_recovered": len(known),
        "mask_confidence_mapping_with_occluded_id_passed": True,
        "fixture_stages": stages,
        "ground_truth_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/AnyStar"))
    parser.add_argument("--panel", type=Path, default=ROOT / "panel.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "anystar")
    parser.add_argument("--model", type=Path, default=ROOT / "anystar/models/anystar-mix")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions/anystar")
    parser.add_argument("--centroid-output", type=Path, default=ROOT / "predictions/anystar_centroid")
    parser.add_argument("--role", choices=["pilot", "assessment", "all"], default="pilot")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-mb", type=int, default=3500)
    parser.add_argument("--lock", type=Path, default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.centroid_output.mkdir(parents=True, exist_ok=True)
    if sha256(args.model / "weights_best.h5") != WEIGHTS_SHA256 or sha256(args.model / "config.json") != CONFIG_SHA256:
        raise RuntimeError("AnyStar checkpoint hashes differ from the verified official download")
    import tensorflow as tf
    from scipy.ndimage import zoom
    from stardist.models import StarDist3D
    from csbdeep.internals.predict import tile_iterator
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_mb)])
    if not args.check_only and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CUDA is required for panel inference")
    started = time.monotonic()
    with gpu_lock(args.lock) if not args.check_only else contextlib.nullcontext():
        model = StarDist3D(None, name=args.model.name, basedir=str(args.model.parent))
        model.keras_model.trainable = False
        geometry = verify_geometry(model, zoom, tile_iterator)
    model_setup_seconds = time.monotonic() - started
    (args.artifacts / "geometry-verification.json").write_text(json.dumps(geometry, indent=2) + "\n")
    cfg = {
        "model": "AnyStar-mix official paper checkpoint",
        "code_url": "https://github.com/neel-dey/AnyStar",
        "code_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "model_url": "https://drive.google.com/drive/folders/1yiY_vBR2GQW9zJzgUPRWeIecN4ZnCi3c",
        "checkpoint_sha256": WEIGHTS_SHA256, "checkpoint_config_sha256": CONFIG_SHA256,
        "parameter_count": model.keras_model.count_params(),
        "input_shape": model.keras_model.input_shape,
        "output_shape": model.keras_model.output_shape,
        "training_data": "purely synthetic domain-randomized star-convex shapes; no real training images; isotropic 64^3 patches",
        "code_license": "MIT", "weights_license": "No separate license in the linked Drive folder; upstream repository MIT",
        "normalization": "upstream float32; clip between 0 and raw99.9th percentile; min-max scale to[0,1]; then interpolate",
        "resampling": "scipy.ndimage.zoom order1, grid_mode=True, mode=nearest, prefilter=False",
        "scale_policy": "source spacing divided by minimum source spacing; isotropic grid preserving native finest-axis spacing",
        "scale_caveat": "Checkpoint has no absolute micrometer calibration; isotropy is documented. XY scale is retained without label tuning.",
        "probability_threshold": 0.5, "nms_threshold": 0.3,
        "threshold_source": "Official infer.py CLI defaults; nms0.3 also used in official fluorescence Colab example",
        "n_tiles_policy": "ceil(resampled dimension/64); panel uses(4,4,4). Actual overlap/tiles recorded in geometry file.",
        "center_definition": "StarDist predicted polyhedron origins / centerness points; no instance rasterization",
        "additional_center_definition": "anystar_centroid: geometric centroid of original-ID, probability-ordered polyhedron rasterization; same NMS results",
        "coordinate_inverse": "native=(model_point+0.5)/actual_zoom-0.5",
        "temporal_inputs": "current image only", "ground_truth_access": False,
        "runtime": sys.executable,
        "runtime_packages": {n: importlib.metadata.version(n) for n in ["tensorflow", "tf_keras", "stardist", "csbdeep", "numpy", "scipy", "h5py"]},
        "tf_logical_device_memory_limit_mb": args.gpu_memory_mb,
        "model_setup_seconds_including_geometry": model_setup_seconds,
        "panel_sha256": sha256(args.panel),
    }
    (args.artifacts / ("config-check.json" if args.check_only else f"config-{args.role}.json")).write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({"status": "setup_complete", "parameters": cfg["parameter_count"], "geometry": geometry}), flush=True)
    if args.check_only:
        return
    frames = [f for f in json.loads(args.panel.read_text())["frames"] if args.role == "all" or f["role"] == args.role]
    if args.limit:
        frames = frames[:args.limit]
    for index, row in enumerate(frames):
        dest = args.output / f"{row['key']}.npz"
        sidecar = dest.with_suffix(".json")
        centroid_dest = args.centroid_output / dest.name
        centroid_sidecar = centroid_dest.with_suffix(".json")
        if dest.exists() and sidecar.exists() and centroid_dest.exists() and centroid_sidecar.exists():
            old = json.loads(sidecar.read_text())
            if old.get("checkpoint_sha256") != WEIGHTS_SHA256 or old.get("probability_threshold") != 0.5 or old.get("nms_threshold") != 0.3:
                raise RuntimeError(f"Existing prediction has a different configuration: {dest}")
            print(json.dumps({"key": row["key"], "status": "already_complete"}), flush=True)
            continue
        started = time.monotonic()
        raw = np.load(row["image_path"], mmap_mode="r")
        if raw.ndim != 3:
            raise ValueError("Panel image is not ZYX")
        spacing = np.asarray(row["spacing_um"], dtype=np.float64)
        desired_zoom = spacing / spacing.min()
        image = zoom(normalize_like_upstream(raw), desired_zoom, order=1, grid_mode=True, mode="nearest", prefilter=False)
        actual_zoom = np.asarray(image.shape) / np.asarray(raw.shape)
        n_tiles = tuple(int(np.ceil(s / 64)) for s in image.shape)
        prep_seconds = time.monotonic() - started
        waited = time.monotonic()
        with gpu_lock(args.lock):
            wait_seconds = time.monotonic() - waited
            started = time.monotonic()
            tf.config.experimental.reset_memory_stats("GPU:0")
            generator = model._predict_instances_generator(
                image, axes="ZYX", n_tiles=n_tiles, prob_thresh=0.5, nms_thresh=0.3,
                return_labels=False, show_tile_progress=False,
            )
            stages = finish_neural_stage(generator)
            neural_seconds = time.monotonic() - started
            gpu_memory = tf.config.experimental.get_memory_info("GPU:0")
        started = time.monotonic()
        _, details = finish_cpu_stage(generator)
        nms_seconds = time.monotonic() - started
        centers = invert_resize(details["points"], actual_zoom)
        scores = np.asarray(details["prob"], dtype=np.float32)
        started = time.monotonic()
        geometric, geometric_scores, geometric_source_indices = mask_centroids(details, image.shape, actual_zoom)
        geometric_inside = np.all((geometric >= 0) & (geometric < np.asarray(raw.shape)), axis=1)
        geometric, geometric_scores = geometric[geometric_inside].astype(np.float32), geometric_scores[geometric_inside]
        geometric_source_indices = geometric_source_indices[geometric_inside]
        centroid_seconds = time.monotonic() - started
        inside = np.all((centers >= 0) & (centers < np.asarray(raw.shape)), axis=1)
        removed_outside = int((~inside).sum())
        centers, scores = centers[inside].astype(np.float32), scores[inside]
        if centers.shape != (len(scores), 3) or not np.isfinite(centers).all() or not np.isfinite(scores).all():
            raise RuntimeError("Invalid predicted centers/scores")
        np.savez_compressed(dest, centers_zyx=centers, scores=scores)
        seconds = prep_seconds + neural_seconds + nms_seconds
        meta = dict(
            key=row["key"], dataset=row["dataset"], time=row["time"], role=row["role"],
            image_path=row["image_path"], checkpoint_sha256=WEIGHTS_SHA256,
            centers=len(centers), probability_threshold=0.5, nms_threshold=0.3,
            source_spacing_um_zyx=spacing.tolist(), native_shape_zyx=list(raw.shape),
            desired_zoom_zyx=desired_zoom.tolist(), actual_zoom_zyx=actual_zoom.tolist(),
            input_shape_zyx=list(image.shape), n_tiles_requested=list(n_tiles),
            tiles_completed=stages.count("tile"), out_of_bounds_centers_removed=removed_outside,
            preprocess_seconds=prep_seconds, neural_seconds=neural_seconds, nms_seconds=nms_seconds,
            inference_seconds=neural_seconds + nms_seconds, seconds=seconds,
            gpu_lock_wait_seconds=wait_seconds, gpu_memory_bytes=gpu_memory,
            ground_truth_access=False,
        )
        sidecar.write_text(json.dumps(meta, indent=2) + "\n")
        np.savez_compressed(centroid_dest, centers_zyx=geometric, scores=geometric_scores,
                            source_polyhedron_indices=geometric_source_indices)
        centroid_meta = dict(meta, centers=len(geometric), mask_centroid_seconds=centroid_seconds,
                             seconds=seconds + centroid_seconds,
                             center_definition="Geometric mask centroid, source polyhedron IDs preserved for confidence assignment")
        centroid_sidecar.write_text(json.dumps(centroid_meta, indent=2) + "\n")
        print(json.dumps({"index": index + 1, "total": len(frames), "key": row["key"], "centers": len(centers), "seconds": seconds,
                          "neural_seconds": neural_seconds, "nms_seconds": nms_seconds}), flush=True)
        del generator, details, image, raw
        gc.collect()


if __name__ == "__main__":
    main()
