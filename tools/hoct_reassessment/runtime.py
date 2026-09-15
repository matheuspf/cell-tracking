"""Bounded HOCT GPU leases shared with the existing local inference/training queue."""
from __future__ import annotations

import fcntl
from pathlib import Path
import time

import torch

GPU_LOCK = Path("/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock")


class CooperativeModel(torch.nn.Module):
    """Expose a CPU interface to upstream inference; execute short FP32 GPU leases."""
    def __init__(self, model, batches_per_lease=2048, seconds_per_lease=10.):
        super().__init__()
        self.base = model.cpu().eval()
        self.limit = batches_per_lease
        self.seconds_limit = seconds_per_lease
        self.in_lease = 0
        self.handle = None
        self.queued_seconds = 0.
        self.gpu_seconds = 0.
        self.leases = 0
        self.peak_reserved = 0
        self.batches = 0
        self.lease_started = None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.30)

    def acquire(self):
        self.handle = GPU_LOCK.open("a+")
        queued = time.monotonic()
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.queued_seconds += time.monotonic() - queued
        self.lease_started = time.monotonic()
        self.leases += 1
        torch.cuda.reset_peak_memory_stats()
        self.base.cuda()

    def release(self):
        if self.handle is None:
            return
        try:
            torch.cuda.synchronize()
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
            self.base.cpu()
            torch.cuda.empty_cache()
            self.gpu_seconds += time.monotonic() - self.lease_started
        finally:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
            self.in_lease = 0

    def forward(self, *inputs):
        if self.handle is None:
            self.acquire()
        try:
            device_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
            output = self.base(*device_inputs)
            result = tuple(x.cpu() for x in output)
            del output, device_inputs
            self.in_lease += 1
            self.batches += 1
            if self.in_lease >= self.limit or time.monotonic() - self.lease_started >= self.seconds_limit:
                self.release()
            return result
        except BaseException:
            self.release()
            raise

    def receipt(self):
        return dict(device="CUDA FP32; upstream CPU interface; TF32 and autocast disabled",
                    shared_lock=str(GPU_LOCK), batches_per_lease=self.limit, leases=self.leases,
                    seconds_per_lease=self.seconds_limit,
                    batches=self.batches, queue_seconds=self.queued_seconds, leased_seconds=self.gpu_seconds,
                    peak_reserved_gib=self.peak_reserved / 2**30, process_memory_cap_fraction=.30)
