"""Run the released NucVerse3D generalized model on a frozen frame panel.

This uses the upstream model, preprocessing, overlap blending and instance
reconstruction. The adapter captures foreground probabilities before upstream
argmax, bounds CPU parallelism, and omits gradient trajectories that upstream
does not subsequently use. It never reads annotations or trains a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        print(f"Waiting for GPU lock: {path}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("Acquired GPU lock", flush=True)
        yield handle


def configure_environment() -> None:
    # Bound total numerical parallelism even when upstream requests n_jobs=-1.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("TF_USE_LEGACY_KERAS", None)


@contextmanager
def persistent_prediction_session(enabled: bool):
    """Avoid upstream per-patch global cleanup while retaining its computation."""
    if not enabled:
        yield
        return
    import gc
    import importlib

    backend = importlib.import_module("tensorflow.keras.backend")
    original_clear, original_collect = backend.clear_session, gc.collect
    backend.clear_session = lambda *args, **kwargs: None
    gc.collect = lambda *args, **kwargs: 0
    try:
        yield
    finally:
        backend.clear_session, gc.collect = original_clear, original_collect
        gc.collect()


def locked_prediction(handle, upstream, image, model, batch_size, persistent):
    """Serialize GPU work while allowing other models during CPU stages."""
    waiting = time.perf_counter()
    fcntl.flock(handle, fcntl.LOCK_EX)
    waited = time.perf_counter() - waiting
    try:
        with persistent_prediction_session(persistent):
            result = upstream.prediction(image, model, batch_size=batch_size)
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
    return result, waited


def install_upstream_hooks(upstream, image_utils):
    """Capture probabilities and avoid retaining 200 unused trajectory copies."""
    import numpy as np

    original_stitch = image_utils.stitch_joblib_3d
    original_interp = image_utils.linear_interp_vol
    original_gradient = upstream.gradient_descent_momentum
    captured = {}

    def bounded_stitch(*args, **kwargs):
        kwargs["n_jobs"] = 4
        return original_stitch(*args, **kwargs)

    def capture_interp(img, output_shape, patch_size, step, k):
        result = original_interp(img, output_shape, patch_size, step, k)
        if k == 2:
            captured["foreground_probability"] = result[..., 1].copy()
        return result

    def gradient_last_only(gradient, start, learn_rate=0.2, n_iter=200, tolerance=1e-6, momentum=0.8):
        vector = np.array(start, dtype=gradient.dtype)
        diff = 0
        for _ in range(n_iter):
            gradients = upstream.grad_point(gradient, vector)
            diff = learn_rate * np.array(gradients) + momentum * diff
            vector = vector + diff
        final = vector.astype("int16")
        return final, [final]

    # A numerical equivalence check exercises the upstream update and integer
    # quantization before enabling the memory-saving execution path.
    rng = np.random.default_rng(1427)
    gradient = rng.normal(0, 0.02, (8, 9, 10, 3)).astype(np.float64)
    starts = np.asarray([[3, 4, 5], [3, 5, 6], [3, 5, 7]], dtype=np.int16)
    expected, history = original_gradient(gradient, starts, n_iter=9)
    actual, short_history = gradient_last_only(gradient, starts, n_iter=9)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(short_history[-1], history[-1])
    image_utils.stitch_joblib_3d = bounded_stitch
    image_utils.linear_interp_vol = capture_interp
    upstream.gradient_descent_momentum = gradient_last_only
    return captured


def load_panel(path: Path, role: str | None, limit: int | None, keys: list[str] | None):
    panel = json.loads(path.read_text())
    frames = panel if isinstance(panel, list) else panel["frames"]
    if role:
        frames = [x for x in frames if x["role"] == role]
    if keys:
        frames = [x for x in frames if x["key"] in keys]
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise ValueError("No frames matched the requested panel selection")
    return frames


def run(args):
    configure_environment()
    sys.path.insert(0, str(args.repo / "src"))
    frames = load_panel(args.panel, args.role, args.limit, args.keys)
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(args.checkpoint)
    revision = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    preset = {
        "checkpoint_name": "resunet_combined_1000/best_model.weights.h5",
        "checkpoint_sha256": checkpoint_hash,
        "repo_revision": revision,
        "input_axes": "ZYX",
        "geometric_scale_zyx": [1.0, 1.0, 1.0],
        "preprocessing": "upstream min-max, uneven intensity correction, unsupervised Wiener per slice, percentiles 2/99.8",
        "preprocessing_stochasticity": "upstream unsupervised_wiener uses its default RNG; realized outputs are retained",
        "center_definition": "geometric centroid of reconstructed upstream instance mask",
        "score_definition": "mean pre-argmax foreground probability over each reconstructed instance; uncalibrated",
        "gradient_descent": {"iterations": 200, "learning_rate": 0.2, "momentum": 0.8},
        "centroid_smoothing_sigma": 1,
        "centroid_threshold_std_multiplier": 5,
        "instance_assignment_radius_voxels": 5,
        "minimum_instance_voxels": 20,
        "cpu_process_limit": 4,
        "cpu_numeric_threads_per_job": 1,
        "adapter_changes": ["capture probabilities before argmax", "bound joblib workers", "discard unused intermediate trajectory history after exact synthetic equivalence check"],
    }
    preset_hash = hashlib.sha256(json.dumps(preset, sort_keys=True).encode()).hexdigest()
    pending = []
    for frame in frames:
        key = frame["key"]
        if Path(key).name != key:
            raise ValueError(f"Unsafe panel key {key!r}")
        output_path, sidecar = args.output / f"{key}.npz", args.output / f"{key}.json"
        if output_path.exists() and sidecar.exists() and not args.force:
            existing = json.loads(sidecar.read_text())
            if existing.get("preset_sha256") != preset_hash:
                raise ValueError(f"Existing output has a different preset: {sidecar}")
            print(f"Already complete: {key}", flush=True)
        else:
            pending.append(frame)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": "https://github.com/Segovia-lab/NucVerse3D",
        "repo_path": str(args.repo),
        "repo_revision": revision,
        "code_license": "MIT",
        "code_license_path": str(args.repo / "LICENSE"),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_license": "CC-BY-4.0 (Zenodo deposit metadata)",
        "checkpoint_source": "https://zenodo.org/records/18517324",
        "environment": sys.prefix,
        "panel_sha256": file_sha256(args.panel),
        "preset": preset,
        "preset_sha256": preset_hash,
        "requested_frames": [x["key"] for x in frames],
        "annotation_access": "none",
        "network_execution": "persistent_session" if args.persistent_session else "upstream_cleanup",
        "gpu_lock_scope": "model loading and each frame's network prediction; CPU preprocessing/reconstruction outside lock",
        "gpu_memory_limit_mb": args.gpu_memory_mb,
    }
    write_json(args.artifacts / "assets.json", manifest)
    if not pending:
        return
    with gpu_lock(args.gpu_lock) as shared_lock:
        import numpy as np
        import tensorflow as tf
        from skimage.measure import regionprops_table

        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        import config
        config.DEFAULT_N_JOBS = 4
        config.MEMORY_GROWTH_ENABLED = False
        import nuclei_prediction as upstream
        from preprocessing.image_preprocessing import preprocessing
        from utils import image as image_utils
        from keras_models.attention_unet_3d import Attention_ResUNet_3D

        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            raise RuntimeError("TensorFlow cannot see the required CUDA GPU")
        tf.config.set_logical_device_configuration(
            devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_mb)]
        )
        captured = install_upstream_hooks(upstream, image_utils)
        model_start = time.perf_counter()
        model = Attention_ResUNet_3D(config.INPUT_SHAPE, config.DROPOUT_RATE, config.BATCH_NORM, config.FILTER_NUM)
        model.load_weights(args.checkpoint)
        model_load_seconds = time.perf_counter() - model_start
        write_json(args.artifacts / "model_load.json", {"seconds": model_load_seconds, "tensorflow": tf.__version__, "parameters": int(model.count_params()), "strict_weights_load": True, "gpu_memory_limit_mb": args.gpu_memory_mb})
        fcntl.flock(shared_lock, fcntl.LOCK_UN)
        results = []
        for frame in pending:
            key = frame["key"]
            started = time.perf_counter()
            print(f"FRAME START {key}", flush=True)
            try:
                image_path = Path(frame["image_path"])
                image = np.load(image_path, allow_pickle=False)
                if image.ndim != 3 or not np.isfinite(image).all():
                    raise ValueError(f"Expected finite 3D image, got {image.shape}")
                if image.max() == image.min():
                    raise ValueError("Constant image is undefined for upstream min-max normalization")
                shape = image.shape
                image = image_utils.normalize(image).astype(np.float16)
                loaded = time.perf_counter()
                prepared = preprocessing(image).astype(np.float16)
                preprocessed = time.perf_counter()
                captured.clear()
                reference = None
                gpu_wait_seconds = 0.0
                if args.verify_session:
                    reference_start = time.perf_counter()
                    (reference_binary, reference_gradients), ref_wait = locked_prediction(shared_lock, upstream, prepared, model, config.PREDICTION_BATCH_SIZE, False)
                    gpu_wait_seconds += ref_wait
                    reference_probability = captured.pop("foreground_probability")
                    reference = (reference_binary, reference_gradients, reference_probability, time.perf_counter() - reference_start - ref_wait)
                network_start = time.perf_counter()
                (binary, gradients), network_wait = locked_prediction(shared_lock, upstream, prepared, model, config.PREDICTION_BATCH_SIZE, args.persistent_session)
                gpu_wait_seconds += network_wait
                if reference is not None:
                    ref_binary, ref_gradients, ref_probability, reference_seconds = reference
                    probability_error = float(np.max(np.abs(captured["foreground_probability"].astype(np.float32) - ref_probability.astype(np.float32))))
                    gradient_error = float(np.max(np.abs(gradients.astype(np.float32) - ref_gradients.astype(np.float32))))
                    identical_binary = bool(np.array_equal(binary, ref_binary))
                    verification = {"key": key, "identical_binary": identical_binary, "max_abs_probability_difference": probability_error, "max_abs_gradient_difference": gradient_error, "upstream_seconds": reference_seconds, "persistent_seconds": time.perf_counter() - network_start - network_wait}
                    write_json(args.artifacts / f"session-equivalence-{key}.json", verification)
                    if not identical_binary or probability_error > 2 ** -10 or gradient_error > 2 ** -10:
                        raise ValueError(f"Session optimization exceeds float16 numerical tolerance: {verification}")
                probability = image_utils.unpad_img(captured.pop("foreground_probability"), shape)
                if not np.isfinite(probability).all() or not np.isfinite(gradients).all():
                    raise ValueError("Nonfinite network output")
                predicted = time.perf_counter()
                frame_dir = args.artifacts / "instances" / key
                frame_dir.mkdir(parents=True, exist_ok=True)
                if np.any(binary):
                    seeds, starts, ends = upstream.centroid_estimation(binary, gradients, str(frame_dir))
                    assignments = upstream.calculate_labels(seeds, ends)
                    labels = upstream.nuclei_segmentation(binary, starts, assignments, str(frame_dir), key, shape)
                else:
                    labels = np.zeros(shape, dtype=np.uint16)
                if not np.isfinite(labels).all():
                    raise ValueError("Nonfinite reconstructed mask")
                props = regionprops_table(labels.astype(np.int32), intensity_image=probability.astype(np.float32), properties=("label", "centroid", "intensity_mean", "area"))
                centers = np.column_stack([props[f"centroid-{i}"] for i in range(3)]).astype(np.float32)
                scores = np.asarray(props["intensity_mean"], dtype=np.float32)
                if reference is not None:
                    from scipy.optimize import linear_sum_assignment
                    from scipy.spatial.distance import cdist

                    reference_dir = args.artifacts / "reference-instances" / key
                    reference_dir.mkdir(parents=True, exist_ok=True)
                    ref_seeds, ref_starts, ref_ends = upstream.centroid_estimation(ref_binary, ref_gradients, str(reference_dir))
                    ref_assignments = upstream.calculate_labels(ref_seeds, ref_ends)
                    ref_labels = upstream.nuclei_segmentation(ref_binary, ref_starts, ref_assignments, str(reference_dir), key, shape)
                    ref_probability = image_utils.unpad_img(ref_probability, shape).astype(np.float32)
                    ref_props = regionprops_table(ref_labels.astype(np.int32), intensity_image=ref_probability, properties=("label", "centroid", "intensity_mean"))
                    ref_centers = np.column_stack([ref_props[f"centroid-{i}"] for i in range(3)])
                    ref_scores = np.asarray(ref_props["intensity_mean"])
                    if len(ref_centers) != len(centers):
                        raise ValueError(f"Session optimization changed instance count: {len(ref_centers)} vs {len(centers)}")
                    distances = cdist(centers * np.asarray(frame["spacing_um"]), ref_centers * np.asarray(frame["spacing_um"]))
                    row, col = linear_sum_assignment(distances)
                    center_error = float(distances[row, col].max(initial=0))
                    score_error = float(np.abs(scores[row] - ref_scores[col]).max(initial=0))
                    verification.update({"n_reference_centers": len(ref_centers), "n_persistent_centers": len(centers), "max_center_displacement_um": center_error, "max_instance_score_difference": score_error, "center_tolerance_um": 0.1, "score_tolerance": 0.001})
                    verification["accepted"] = center_error <= 0.1 and score_error <= 0.001
                    write_json(args.artifacts / f"session-equivalence-{key}.json", verification)
                    if not verification["accepted"]:
                        raise ValueError(f"Session optimization changed reconstructed centers: {verification}")
                    del reference, ref_binary, ref_gradients, ref_probability, ref_labels
                if centers.size and (np.any(centers < 0) or np.any(centers > np.asarray(shape) - 1)):
                    raise ValueError("Out-of-bounds center after native-grid reconstruction")
                np.savez_compressed(args.output / f"{key}.npz", centers_zyx=centers, scores=scores, instance_ids=np.asarray(props["label"], dtype=np.int32), volumes_voxels=np.asarray(props["area"], dtype=np.int32))
                finished = time.perf_counter()
                record = {
                    "key": key,
                    "dataset": frame["dataset"],
                    "time": frame["time"],
                    "role": frame["role"],
                    "input_shape_zyx": list(shape),
                    "spacing_um": frame["spacing_um"],
                    "input_sha256": file_sha256(image_path),
                    "preset_sha256": preset_hash,
                    "network_execution": "persistent_session" if args.persistent_session else "upstream_cleanup",
                    "n_centers": len(centers),
                    "foreground_voxels": int(np.count_nonzero(binary)),
                    "timing_seconds": {"load_normalize": loaded - started, "preprocessing": preprocessed - loaded, "network_and_stitch": predicted - preprocessed - gpu_wait_seconds, "reconstruction_and_write": finished - predicted, "total": finished - started - gpu_wait_seconds, "gpu_lock_wait": gpu_wait_seconds, "wall_including_gpu_queue": finished - started},
                    "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "gpu_memory": tf.config.experimental.get_memory_info("GPU:0"),
                }
                write_json(args.output / f"{key}.json", record)
                results.append(record)
                print(f"FRAME DONE {key}: {len(centers)} centers, {finished - started - gpu_wait_seconds:.2f} compute seconds, {gpu_wait_seconds:.2f} seconds GPU queue", flush=True)
                del image, prepared, binary, gradients, probability, labels
            except Exception as error:
                record = {"key": key, "error": repr(error), "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started}
                write_json(args.artifacts / "errors" / f"{key}.json", record)
                print(record["traceback"], flush=True)
                if not args.continue_on_error:
                    raise
        write_json(args.artifacts / f"run-{int(time.time())}.json", {"model_load_seconds": model_load_seconds, "preset_sha256": preset_hash, "frames": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/mpf/code/kaggle/NucVerse3D"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/mpf/code/kaggle/NucVerse3D/models/resunet_combined_1000/best_model.weights.h5"))
    parser.add_argument("--panel", type=Path, default=Path("work/detector-screen-20260914/panel.json"))
    parser.add_argument("--output", type=Path, default=Path("work/detector-screen-20260914/predictions/nucverse"))
    parser.add_argument("--artifacts", type=Path, default=Path("work/detector-screen-20260914/nucverse"))
    parser.add_argument("--gpu-lock", type=Path, default=Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock"))
    parser.add_argument("--gpu-memory-mb", type=int, default=3500)
    parser.add_argument("--role")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--persistent-session", action="store_true")
    parser.add_argument("--verify-session", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
