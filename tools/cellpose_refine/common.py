from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/cellpose-refine-v1"
RESULTS = REPO / "results/cellpose-refine-v1"
SCREEN = REPO / "work/detector-screen-20260914"
DATA = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
SPACING = np.array([1.625, .40625, .40625], dtype=np.float64)
EMBRYOS = ("44b6", "6bba")
CHECKPOINT = SCREEN / "cellpose/weights/cpdino-vitb"
CHECKPOINT_SHA = "3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f"
GPU_LOCK = Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def configure():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import cv2
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def config():
    return json.loads((REPO / "configs/cellpose-refine-v1.json").read_text())


def assert_source(embryo, rows):
    if embryo not in EMBRYOS:
        raise ValueError("Unknown source embryo")
    for r in rows:
        if r["embryo"] != embryo or r["dataset"].split("_")[0] != embryo:
            raise ValueError("Non-source row in training input")


def validate_ancestry(source, artifact):
    """Fail closed for declared adaptation exposure; inherited uncertainty is separate."""
    exposure = artifact["adaptation_exposure"]
    expected = {"labels", "unlabeled_images", "calibration", "teacher_construction", "normalization_fit"}
    if set(exposure) != expected:
        raise ValueError("Incomplete adaptation exposure schema")
    for field, seen in exposure.items():
        if not isinstance(seen, list) or set(seen) - {source}:
            raise ValueError(f"Non-source exposure: {field}")
    for parent in artifact.get("parents", []):
        validate_ancestry(source, parent)


def native_to_isotropic(points):
    """OpenCV half-pixel resize coordinates, not a zero-origin scalar multiply."""
    return (np.asarray(points, dtype=np.float64) + .5) * np.array([4., 1., 1.]) - .5


def bounded_integer(points, refined, shape, limit_um=3.):
    """Closest integer output within the physical bound from the rounded baseline."""
    base = np.clip(np.rint(points), 0, np.asarray(shape) - 1).astype(np.int64)
    continuous = np.clip(refined, 0, np.asarray(shape) - 1)
    proposed = np.rint(continuous).astype(np.int64)
    bad = np.linalg.norm((proposed - base) * SPACING, axis=1) > limit_um + 1e-9
    maxima = np.floor(limit_um / SPACING).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in maxima], indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.linalg.norm(grid * SPACING, axis=1) <= limit_um + 1e-9]
    for i in np.flatnonzero(bad):
        options = base[i] + grid
        options = options[((options >= 0) & (options < np.asarray(shape))).all(axis=1)]
        error = np.square((options - continuous[i]) * SPACING).sum(axis=1)
        proposed[i] = options[np.argmin(error)]
    assert (np.linalg.norm((proposed - base) * SPACING, axis=1) <= limit_um + 1e-9).all()
    return proposed, bad


@contextmanager
def locked_gpu():
    from tools.detector_screen.cellpose_adapter import gpu_lock
    with gpu_lock(GPU_LOCK) as wait:
        yield wait
