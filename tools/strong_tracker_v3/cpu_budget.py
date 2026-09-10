"""Apply a shared CPU affinity budget to current V3 workers and support threads."""
import os
from pathlib import Path
import psutil
from .common import now,write_json

def run(ctx):
    cpus=sorted(os.sched_getaffinity(0))[:32]
    processes={}
    for p in psutil.process_iter(['cmdline']):
        try:
            if 'strong_tracker_v3' in ' '.join(p.info['cmdline'] or []):
                processes[p.pid]=p
                processes.update({c.pid:c for c in p.children(recursive=True)})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    count=0
    for p in processes.values():
        try:
            for thread in p.threads():
                try:os.sched_setaffinity(thread.id,cpus);count+=1
                except ProcessLookupError:pass
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    receipt=dict(created=now(),allowed_cpu_count=len(cpus),cpu_ids=cpus,
        process_count=len(processes),thread_affinities_set=count,
        scope='Current V3 workers and descendants, including idle CUDA/Zarr support threads; unrelated user processes excluded',
        history='BLAS/OpenMP compute pools were restricted from launch; this additionally bounds aggregate CPU execution after launch')
    write_json(ctx.out/'cpu_affinity_receipt.json',receipt)
    return receipt

if __name__=='__main__':
    from .context import RunContext
    print(run(RunContext.default()))
