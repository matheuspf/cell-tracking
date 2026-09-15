"""Paths and receipts, separate from earlier studies and their mutable status."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HOCT_REASSESSMENT_OUTPUT", str(REPO / "work/hoct-reassessment-20260914")))
RESULTS = Path(os.environ.get("HOCT_REASSESSMENT_RESULTS", str(REPO / "results/hoct-reassessment-20260914")))
CONFIG = REPO / "configs/hoct-reassessment-20260914.json"
OLD = Path("/kaggle/working/cell-tracking/image-native-tracking-v5")
HOCT_SOURCE = OLD / "inference_package_validation/python"
CP = REPO / "work/cellpose-ultrack-20260914"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def setup():
    dependencies = ROOT / "python"
    if dependencies.exists() and str(dependencies) not in sys.path:
        sys.path.insert(0, str(dependencies))
    if str(HOCT_SOURCE) not in sys.path:
        sys.path.insert(0, str(HOCT_SOURCE))
    import torch
    import hoct
    torch.set_num_threads(2)
    torch.jit.set_fusion_strategy([])
    if Path(hoct.__file__).resolve().parent != HOCT_SOURCE / "hoct":
        raise RuntimeError("Unexpected HOCT source")
    return hoct


def load_model(name="general_v1"):
    hoct = setup()
    path = OLD / "models/hoct" / f"{name}.pt"
    if sha(path) != read(CONFIG)["models"][name]:
        raise ValueError("Checkpoint hash differs from the locked protocol")
    return hoct.load_model(path, device="cpu")


def stamp():
    return dict(config_sha256=sha(CONFIG), code={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def pilot():
    names = read(CONFIG)["pilot_clips"]
    clips = {c["dataset"]: c for c in read(CP / "panel.json")["clips"]}
    return [clips[name] for name in names]
