"""Monitor the complete worker tree and cooperate with the existing GPU lease."""
import contextlib
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import psutil

from .common import Blocked, GPU_LOCK, WORK, now, write_json


def gpu_snapshot():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        # Numerical libraries can update the process environment while this
        # background sampler spawns a child. Give exec a stable explicit copy.
        text=True, capture_output=True, check=True, env=os.environ.copy(),
    )
    memory, utilization = result.stdout.strip().splitlines()[0].split(',')
    return dict(total_gib=float(memory) / 1024, utilization_percent=float(utilization))


def process_snapshot():
    rows = []
    for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'memory_info']):
        try:
            args = p.info['cmdline'] or []
            if any('python' in a or 'train_native' in a for a in args[:4]):
                rows.append(dict(pid=p.pid, ppid=p.ppid(), command=args,
                                 rss_gib=p.memory_info().rss / 2**30))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rows


class Monitor:
    def __init__(self, path, gpu_limit=20., rss_limit=50.):
        self.path, self.gpu_limit, self.rss_limit = path, gpu_limit, rss_limit
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.peak_gpu = self.peak_rss = 0.
        self.peak_study_rss = 0.
        self.cache_bytes = 0
        self.last_cache_check = 0.
        self.peak_threads = 0
        self.samples = 0
        self.error = None

    def sample(self):
        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        rss = threads = 0
        for p in processes:
            with contextlib.suppress(psutil.NoSuchProcess):
                rss += p.memory_info().rss
                threads += p.num_threads()
        gpu = gpu_snapshot()['total_gib']
        # Include independent study workers, not just this subprocess's tree.
        study = {p.pid: p for p in processes}
        for process in psutil.process_iter(['cmdline']):
            try:
                if any('pipeline_error_training' in arg or 'pipeline-error-training-20260915' in arg
                       for arg in (process.info['cmdline'] or [])):
                    study[process.pid] = process
                    study.update({p.pid: p for p in process.children(recursive=True)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        study_rss = 0
        for process in study.values():
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                study_rss += process.memory_info().rss
        self.peak_gpu = max(self.peak_gpu, gpu)
        self.peak_rss = max(self.peak_rss, rss / 2**30)
        self.peak_study_rss = max(self.peak_study_rss, study_rss / 2**30)
        self.peak_threads = max(self.peak_threads, threads)
        self.samples += 1
        if gpu >= self.gpu_limit or max(rss, study_rss) / 2**30 >= self.rss_limit:
            self.error = f'Resource envelope reached: total GPU {gpu:.3f} GiB, tree RSS {rss/2**30:.3f} GiB, study RSS {study_rss/2**30:.3f} GiB'
        if time.monotonic()-self.last_cache_check >= 30.:
            size = 0
            for folder, _, files in os.walk(WORK):
                for name in files:
                    with contextlib.suppress(FileNotFoundError):
                        size += (Path(folder)/name).stat().st_size
            self.cache_bytes = size
            self.last_cache_check = time.monotonic()
            if size >= 80*2**30:
                self.error = 'New study artifacts reached the 80 GiB cache ceiling'

    def check(self):
        from .budget import check as check_budget
        check_budget()
        if self.error:
            raise Blocked(self.error)
        if shutil.disk_usage(WORK).free < 20 * 2**30:
            raise Blocked('Study filesystem below the 20 GiB free-space floor')

    def _loop(self):
        while not self.stop.wait(1.):
            try:
                self.sample()
            except (OSError, subprocess.SubprocessError) as exc:
                self.error = f'Resource monitor failed: {exc}'

    def __enter__(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.sample()
        self.check()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()
        write_json(self.path, dict(
            created=now(), wall_seconds=time.monotonic()-self.started,
            peak_total_gpu_gib=self.peak_gpu, peak_tree_rss_gib=self.peak_rss,
            peak_all_study_process_rss_gib=self.peak_study_rss, study_disk_gib=self.cache_bytes/2**30,
            peak_os_threads=self.peak_threads, sampling_seconds=1., samples=self.samples,
            resource_error=self.error, logical_cpu_affinity=psutil.Process().cpu_affinity(),
            gpu_time_note='Total-device samples include other programs; wall time is not kernel time.',
        ))
        if exc[0] is None:
            self.check()


def cpu_budget(cores=16):
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:cores]))
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS', 'ZARR_THREADING_MAX_WORKERS']:
        os.environ[key] = '1'
    # blosc2 otherwise overrides NUMEXPR_NUM_THREADS while importing tracksdata.
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    os.environ['ZARR_ASYNC_CONCURRENCY'] = '2'
    # Donfig uses a double underscore for nested configuration keys.
    os.environ['ZARR_ASYNC__CONCURRENCY'] = '2'
    os.environ['ZARR_THREADING__MAX_WORKERS'] = '2'
    import sys
    if 'zarr' in sys.modules:
        sys.modules['zarr'].config.set({'async.concurrency': 2, 'threading.max_workers': 2})


class Lease:
    """Nonblocking admission; caller checkpoints and releases between short batches."""
    def __init__(self, required_gib=4., wait=False, timeout=3600.):
        self.required_gib = required_gib
        self.wait, self.timeout = wait, timeout
        self.handle = None
        self.budget_token = None

    def __enter__(self):
        self.handle = GPU_LOCK.open('a+')
        try:
            started = time.monotonic()
            while True:
                try:
                    fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not self.wait or time.monotonic()-started >= self.timeout:
                        raise
                    time.sleep(.2)
            if gpu_snapshot()['total_gib'] + self.required_gib >= 20:
                raise Blocked('Other GPU use leaves insufficient room under the 20 GiB total cap')
            from .budget import begin
            self.budget_token = begin()
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *exc):
        try:
            from .budget import finish
            finish(self.budget_token)
        finally:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
