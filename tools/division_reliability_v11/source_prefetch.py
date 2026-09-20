"""Overlap source observation preparation with an earlier cell's C11 fit.

The normal queue waits at its C00 package entry while this owner is active.
No trainer, target prediction, or calibration selection is launched here.
Existing guarded source workers retain their original recipes and diagnostics.
"""
from contextlib import contextmanager
import argparse
import fcntl
import json
import os
import shutil
import time
from .common import WORK,RESULTS,Blocked,read,write,now


def folder(source,seed):
    return WORK/'controller/source-prefetch'/source/str(seed)


@contextmanager
def ownership(source,seed,*,exclusive=False):
    root=folder(source,seed);root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a+') as owner:
        mode=fcntl.LOCK_EX|fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH
        try:fcntl.flock(owner,mode)
        except BlockingIOError as exc:raise Blocked('A source preparation owner or C00 package reader is already active') from exc
        yield


def archive(jobs,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        name='-'.join(str(job[k]) for k in ('stage','source','seed','arm','part','clip') if job.get(k) is not None)
        for suffix in ('.json','.resources.json'):
            path=WORK/'controller/jobs'/(name+suffix)
            if path.exists():shutil.copy2(path,destination/path.name)


def await_workers(source,seed):
    """A dead prefetch owner may leave finite source workers still running."""
    from .report import alive
    while True:
        busy=False
        for path in (WORK/'controller/jobs').glob('*.json'):
            r=read(path)
            if r.get('source')!=source or r.get('seed')!=seed or r.get('status')!='running':continue
            relevant=r.get('stage') in ('prepare-actions','source-diagnostics') or (
                r.get('stage')=='predict' and r.get('arm')=='C00' and r.get('part') in ('fit','calibration'))
            if relevant and alive(r.get('pid')):busy=True;break
        if not busy:return
        time.sleep(1)


def run(source,seed,*,workers=3):
    from .readiness import require_production
    from .report import alive
    from .controller import batch,worker
    from .populations import clips
    from .packaging import _build
    require_production('source-prefetch')
    if not 1<=workers<=8:raise Blocked('Invalid source preparation concurrency')
    schedule=[(c['source'],c['seed']) for c in read(RESULTS/'execution_lock.json')['schedule']]
    if (source,seed) not in schedule:raise Blocked('Unregistered source/seed cell')
    final=WORK/'fits'/source/str(seed)/'upstream/final.json'
    if not final.exists():raise Blocked('Source preparation requires completed C00 weights; it never starts training')
    root=folder(source,seed)
    with ownership(source,seed,exclusive=True):
        state=read(WORK/'controller/pipeline.json') if (WORK/'controller/pipeline.json').exists() else {}
        if alive(state.get('controller_pid')):
            cell=state.get('cell',state);current=(cell.get('source'),cell.get('seed'))
            if current in schedule and schedule.index(current)>=schedule.index((source,seed)):
                raise Blocked('The normal queue has reached this cell; let its existing owner continue')
        for path in (WORK/'controller/jobs').glob('*.json'):
            r=read(path)
            if r.get('source')==source and r.get('seed')==seed and r.get('status')=='running' and alive(r.get('pid')):
                raise Blocked('A source worker for this cell is already active')
        receipt=root/'progress.json'
        history=WORK/'controller/jobs/history/source-prefetch'/source/str(seed)/str(time.time_ns())
        write(receipt,dict(status='running',source=source,seed=seed,pid=os.getpid(),started_utc=now()))
        try:
            # This process holds the exclusive barrier. Calling the private
            # package builder avoids trying to acquire its own shared barrier.
            _build(source,seed,'C00')
            package_identity=read(WORK/'packages'/source/str(seed)/'C00/manifest.json')['identity']
            for part in ('fit','calibration'):
                for stage in ('predict','prepare-actions'):
                    jobs=[dict(stage=stage,source=source,seed=seed,clip=n,part=part,
                               **({'arm':'C00'} if stage=='predict' else {})) for n in clips(source,part)]
                    write(receipt,dict(status='running',source=source,seed=seed,pid=os.getpid(),stage=stage,part=part,updated_utc=now()))
                    try:batch(jobs,workers)
                    finally:archive(jobs,history)
            job=dict(stage='source-diagnostics',source=source,seed=seed)
            try:worker(**job)
            finally:archive([job],history)
            result=dict(status='complete',source=source,seed=seed,package_identity=package_identity,
                source_only=True,training_launched=False,target_predictions_launched=False,finished_utc=now())
            write(receipt,result);return result
        except Exception as exc:
            write(receipt,dict(status='failed',source=source,seed=seed,reason=str(exc),finished_utc=now()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['44b6','6bba'],required=True)
    p.add_argument('--seed',type=int,choices=[20260918,314159],required=True)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    try:print(json.dumps(run(a.source,a.seed,workers=a.workers),indent=2))
    except Blocked as exc:raise SystemExit(str(exc))
