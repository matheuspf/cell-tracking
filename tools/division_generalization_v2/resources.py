"""Short cooperative GPU leases, bounded study RSS, and complete wall accounting."""
import contextlib
import fcntl
import os
import shutil
import threading
import time
from pathlib import Path

import psutil

from pipeline_error_training.resources import gpu_snapshot, process_snapshot
from .common import WORK, GPU_LOCK, append_json, now, read_json, write_json

ACTIVE_MONITORS=[]

def check_disk():
    if shutil.disk_usage(WORK).free < 20*2**30:
        raise RuntimeError('Study filesystem reached the 20 GiB free-space floor')


class Lease:
    """Caller releases at a joint-update boundary, at most 30 s between yields."""
    def __init__(self, purpose, required_gib=4.):
        self.purpose, self.required = purpose, required_gib

    def __enter__(self):
        for monitor in ACTIVE_MONITORS:monitor.check()
        self.handle = GPU_LOCK.open('a+')
        wait = time.monotonic()
        while True:
            # Let the kernel queue a waiter. Polling every 200 ms can miss the
            # short release between training blocks and starve validation or
            # another cooperating job despite nominal 30-second leases.
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            for monitor in ACTIVE_MONITORS:monitor.check()
            if gpu_snapshot()['total_gib'] + self.required < 20:
                break
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            time.sleep(1)
        self.wait_seconds = time.monotonic()-wait
        self.started = time.monotonic()
        self.token = f'{os.getpid()}-{time.time_ns()}'
        ledger = WORK / 'resources/leases.jsonl'
        total = 0.
        if ledger.exists():
            import json
            events = [json.loads(line) for line in ledger.read_text().splitlines()]
            total = sum(x.get('lease_seconds', 0.) for x in events)
            closed = {x['token'] for x in events if x['event'] == 'end'}
            if any(x['event'] == 'begin' and x['token'] not in closed for x in events):
                self.handle.close()
                raise RuntimeError('Unclosed GPU lease: reconcile interrupted accounting before continuing')
        if total >= 72*3600:
            self.handle.close()
            raise RuntimeError('Study 72 GPU lease-hour cap reached; resumable incomplete')
        append_json(ledger, dict(event='begin', token=self.token, purpose=self.purpose,
                                utc=now(), wait_seconds=self.wait_seconds))
        return self

    def __exit__(self, *exc):
        self.seconds = time.monotonic()-self.started
        append_json(WORK / 'resources/leases.jsonl', dict(event='end', token=self.token,
            purpose=self.purpose, utc=now(), lease_seconds=self.seconds,
            wait_seconds=self.wait_seconds, exception=str(exc[1]) if exc[1] else None))
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


class Monitor:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.peak_gpu = self.peak_rss = 0.
        self.error = None
        self.samples = 0
        self.cache_bytes=0
        self.last_disk_scan=0.

    def sample(self):
        process = psutil.Process()
        rss = 0
        processes={p.pid:p for p in [process,*process.children(recursive=True)]}
        for p in psutil.process_iter(['cmdline']):
            with contextlib.suppress(psutil.NoSuchProcess,psutil.AccessDenied):
                if any(a.startswith('division_generalization_v2') for a in (p.info['cmdline'] or [])):
                    processes[p.pid]=p
                    processes.update({q.pid:q for q in p.children(recursive=True)})
        # Short fixture/calibration helpers may be launched from stdin. Register
        # their PID and creation time so the total includes them without relying
        # on command-name matching or trusting stale PID files.
        for path in (WORK/'resources/workers').glob('*.json'):
            with contextlib.suppress(psutil.NoSuchProcess,psutil.AccessDenied,FileNotFoundError):
                marker=read_json(path);p=psutil.Process(marker['pid'])
                if p.create_time()==marker['created']:
                    processes[p.pid]=p
                    processes.update({q.pid:q for q in p.children(recursive=True)})
        for p in processes.values():
            with contextlib.suppress(psutil.NoSuchProcess):
                rss += p.memory_info().rss
        gpu = gpu_snapshot()['total_gib']
        self.peak_gpu = max(gpu, self.peak_gpu)
        self.peak_rss = max(rss/2**30, self.peak_rss)
        self.samples += 1
        if gpu >= 20 or rss >= 50*2**30:
            self.error = f'Resource envelope: GPU={gpu:.3f} GiB tree RSS={rss/2**30:.3f} GiB'
        if time.monotonic()-self.last_disk_scan>=60:
            size=0
            for p in WORK.rglob('*'):
                with contextlib.suppress(FileNotFoundError):
                    if p.is_file():size+=p.stat().st_size
            self.cache_bytes=size
            self.last_disk_scan=time.monotonic()
            if self.cache_bytes>=80*2**30:
                self.error='Study artifacts reached the 80 GiB cache ceiling'

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        check_disk()

    def loop(self):
        while not self.stop.wait(2):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)

    def __enter__(self):
        process=psutil.Process()
        write_json(WORK/'resources/workers'/(str(process.pid)+'.json'),
            dict(pid=process.pid,created=process.create_time(),command=process.cmdline()))
        self.sample()
        self.check()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        ACTIVE_MONITORS.append(self)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        ACTIVE_MONITORS.remove(self)
        write_json(self.path, dict(peak_total_gpu_gib=self.peak_gpu,
            peak_process_tree_rss_gib=self.peak_rss, samples=self.samples,
            includes_all_visible_study_workers=True,
            wall_seconds=time.monotonic()-self.started, error=self.error,
            study_disk_gib=self.cache_bytes/2**30,
            cuda_kernel_time_is_not_wall_time=True))
        if exc[0] is None:self.check()
