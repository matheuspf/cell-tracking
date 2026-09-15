"""Independent outputs, shared frozen observations and bounded GPU use."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tools.hoct_reassessment.common import REPO, CP, arrays, read, save, sha, write

ROOT = Path(os.environ.get("OTHER_TRACKERS_OUTPUT", str(REPO / "work/other-trackers-20260914")))
RESULTS = REPO / "results/other-trackers-20260914"
CONFIG = REPO / "configs/other-trackers-20260914.json"
BANK = REPO / "work/hoct-reassessment-20260914/cellpose/features"
LOCK = Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock")
ORG = REPO.parent / "OrganoidTracker"
CEL = REPO.parent / "CELLECT"


def clips():
    panel = {c["dataset"]: c for c in read(CP / "panel.json")["clips"]}
    return [panel[n] for n in read(CONFIG)["pilot_clips"]]


def bank(name):
    path = BANK / f"{name}.npz"
    receipt = read(path.with_suffix(".json"))
    if sha(path) != receipt["sha256"]:
        raise ValueError("Frozen Cellpose input changed")
    data = arrays(path)
    return {k: data[k] for k in ("nodes", "positions")}


def guard():
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0]).resolve()
        if any(p.endswith(".geff") or p in {"evaluation", "matches", "strong-tracker-v3", "segmentation-tracking-v6"}
               for p in path.parts):
            raise PermissionError("Tracker inference cannot read labels or incumbent evidence")
        old = Path("/kaggle/working/cell-tracking/image-native-tracking-v5")
        if old in path.parents:
            raise PermissionError("Tracker inference cannot read native baseline artifacts")
    sys.addaudithook(audit)


def verify_repo(path, family):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    assert actual == read(CONFIG)[family]["revision"]
    assert not subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip()


class Lease:
    def __init__(self, models, seconds=10., fraction=.38):
        import torch
        self.torch = torch
        self.models = models
        self.seconds = seconds
        self.handle = None
        self.queued = self.elapsed = 0.
        self.leases = self.batches = self.peak = 0
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(fraction)

    def acquire(self):
        if self.handle is not None:
            return
        self.handle = LOCK.open("a+")
        started = time.monotonic()
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.queued += time.monotonic() - started
        self.started = time.monotonic()
        self.leases += 1
        self.torch.cuda.reset_peak_memory_stats()
        for model in self.models:
            model.to("cuda:0")

    def finish_batch(self):
        self.batches += 1
        if time.monotonic() - self.started >= self.seconds:
            self.release()

    def release(self):
        if self.handle is None:
            return
        try:
            self.torch.cuda.synchronize()
            self.peak = max(self.peak, self.torch.cuda.max_memory_reserved())
            for model in self.models:
                model.to("cpu")
            self.torch.cuda.empty_cache()
            self.elapsed += time.monotonic() - self.started
        finally:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def receipt(self):
        return dict(queue_seconds=self.queued, leased_seconds=self.elapsed, leases=self.leases,
                    batches=self.batches, peak_reserved_gib=self.peak/2**30, lease_limit_seconds=self.seconds,
                    precision="float32; TF32 and autocast disabled", shared_lock=str(LOCK))


class Images:
    def __init__(self, name, length=100):
        self.name, self.length = name, length
        self.cached = {}

    def raw(self, t):
        t = max(0, min(self.length-1, int(t)))
        if t not in self.cached:
            path = CP / "images" / self.name / f"t{t:03}.npy"
            self.cached[t] = np.load(path, allow_pickle=False, mmap_mode="r")
        for old in list(self.cached):
            if abs(old-t) > 3:
                del self.cached[old]
        return self.cached[t]

    def get_image(self, time_point):
        from organoid_tracker.core.images import Image
        return Image(self.raw(time_point.time_point_number()))
