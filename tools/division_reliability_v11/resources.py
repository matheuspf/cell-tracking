"""Shared GPU lease accounting, admission, and measured resource limits."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import threading
import time
from .common import WORK, Blocked, now, read, write

ROOT=WORK/'resources'
GPU_LOCK=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock')
GIB=2**30


class Resources:
    """Construct before the read guard; include ROOT among permitted outputs."""
    def __init__(self,stage,source,seed):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.lock=GPU_LOCK.open('a+');self.stage=stage;self.source=source;self.seed=seed
        self.seconds=0.

    def study_rss(self):
        import psutil
        members={os.getpid():psutil.Process().memory_info().rss}
        for process in psutil.process_iter(['pid','cmdline','memory_info']):
            try:
                if 'division_reliability_v11' in ' '.join(process.info['cmdline'] or []):
                    members[process.pid]=process.info['memory_info'].rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        self.members=tuple(members)
        return sum(members.values())

    def sampled_rss(self):
        # Discover study workers before admission. Query only those PIDs in the
        # short monitor interval; scanning unrelated command lines while holding
        # a tiny inference lease otherwise dominates the actual GPU operation.
        import psutil
        value=0
        for pid in self.members:
            try:value+=psutil.Process(pid).memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return value

    def totals(self):
        """Replay only journal entries not yet reflected by the durable cursor."""
        journal=ROOT/'gpu-leases.jsonl';path=ROOT/'totals.json'
        total=read(path) if path.exists() else dict(journal_bytes=0,seconds=0.,by_stage={})
        if journal.exists():
            if journal.stat().st_size<total['journal_bytes']:raise Blocked('GPU accounting journal was truncated')
            with journal.open() as f:
                f.seek(total['journal_bytes'])
                for line in f:
                    if not line.strip():continue
                    row=json.loads(line);total['seconds']+=row['seconds']
                    total['by_stage'][row['stage']]=total['by_stage'].get(row['stage'],0.)+row['seconds']
                total['journal_bytes']=f.tell()
        write(path,total);return total

    def host(self):
        import psutil
        rss=self.study_rss()
        available=psutil.virtual_memory().available;disk=shutil.disk_usage(WORK).free
        # Two GiB cover an atomic replacement checkpoint plus the next bounded
        # graph/cache write, in addition to the registered ten-GiB free floor.
        if rss>44*GIB or available<10*GIB or disk<12*GIB:
            raise Blocked(f'Host resource limit: process-tree RSS={rss}, available={available}, durable_free={disk}')
        return dict(rss_bytes=rss,host_available_bytes=available,durable_free_bytes=disk)

    @contextmanager
    def lease(self,minimum_bytes=GIB):
        import torch
        self.host();fcntl.flock(self.lock,fcntl.LOCK_EX);started=time.monotonic()
        row=dict(stage=self.stage,source=self.source,seed=self.seed,pid=os.getpid(),started_utc=now(),status='started')
        pending=ROOT/'active_lease.json';journal=ROOT/'gpu-leases.jsonl'
        stopped=threading.Event();samples=[];thread=None
        try:
            if pending.exists():
                stale=read(pending)
                # Holding the sole lock proves the previous owner released/died.
                # Conservatively charge its full maximum permitted work unit.
                with journal.open('a') as f:f.write(json.dumps({**stale,'status':'unclosed_lease_conservatively_charged','seconds':60.})+'\n')
                pending.unlink()
            prior=read(ROOT/'prior_accounting.json')['conservative_gpu_lease_seconds'] if (ROOT/'prior_accounting.json').exists() else 0.
            totals=self.totals();prior+=totals['seconds']
            # Every lease reserves its maximum duration, including failed work.
            limit=(72 if self.stage in ('inference','cold') else 56)*3600
            if prior+60>limit:raise Blocked('Study GPU budget exhausted; inference reserve protected')
            if (ROOT/'allocation.json').exists():
                allocation=read(ROOT/'allocation.json')
                category='inference_cold' if self.stage in ('inference','cold') else self.stage
                spent=sum(seconds for stage,seconds in totals['by_stage'].items()
                          if ('inference_cold' if stage in ('inference','cold') else stage)==category)
                if category=='pilot':spent+=read(ROOT/'prior_accounting.json')['conservative_gpu_lease_seconds']
                if spent+60>allocation['hours'][category]*3600:raise Blocked('Locked stage GPU allocation exhausted: '+category)
            write(pending,row)
            free,total=torch.cuda.mem_get_info();own_reserved=torch.cuda.memory_reserved()
            other=total-free-own_reserved
            ceiling=min(20*GIB-other,free+own_reserved-2*GIB)
            if ceiling-torch.cuda.memory_allocated()<minimum_bytes:
                raise Blocked(f'GPU admission cannot fit measured work unit: process_ceiling={ceiling}, own_allocated={torch.cuda.memory_allocated()}, other={other}')
            torch.cuda.set_per_process_memory_fraction(max(0,ceiling)/total)
            torch.cuda.reset_peak_memory_stats()
            def sample():
                while not stopped.is_set():
                    f,t=torch.cuda.mem_get_info()
                    import psutil
                    samples.append((time.monotonic(),t-f,f,self.sampled_rss(),psutil.virtual_memory().available))
                    stopped.wait(.5)
            thread=threading.Thread(target=sample,daemon=True);thread.start()
            yield row
            torch.cuda.synchronize()
            row['status']='complete'
        except BaseException as exc:
            row.update(status='failed',failure_type=type(exc).__name__,reason=str(exc))
            raise
        finally:
            stopped.set()
            if thread:thread.join(timeout=2)
            elapsed=time.monotonic()-started
            self.seconds+=elapsed
            row.update(seconds=elapsed,finished_utc=now(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if samples:
                row.update(total_device_peak_bytes=max(s[1] for s in samples),device_free_min_bytes=min(s[2] for s in samples),
                           rss_peak_bytes=max(s[3] for s in samples),host_available_min_bytes=min(s[4] for s in samples))
            exceeded=elapsed>60 or any(s[1]>20*GIB or s[2]<2*GIB or s[3]>44*GIB or s[4]<10*GIB for s in samples)
            if exceeded:row.update(status='resource_limit_exceeded')
            with journal.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            if pending.exists():pending.unlink()
            self.totals()
            fcntl.flock(self.lock,fcntl.LOCK_UN)
            if exceeded:raise Blocked('Measured lease exceeded a registered resource bound')

    def close(self):self.lock.close()
