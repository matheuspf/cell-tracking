"""Fit a later cell's registered heads through C11's fixed mining boundary.

The C00 package ownership barrier makes the main queue wait before entering
this cell. Full source preparation must already be complete. The ordinary
workers retain their source guards, optimizer recipe and durable resume state.
This helper never mines, calibrates, predicts targets or changes a horizon.
"""
import argparse
import json
import os
from .common import WORK,RESULTS,Blocked,read,write,now


def run(source,seed):
    from .readiness import require_production
    from .report import alive
    from .source_prefetch import ownership
    from .controller import worker
    require_production('head-prefetch')
    lock=read(RESULTS/'execution_lock.json');schedule=[(c['source'],c['seed']) for c in lock['schedule']]
    if (source,seed) not in schedule:raise Blocked('Unregistered source/seed cell')
    prepared=WORK/'controller/source-prefetch'/source/str(seed)/'progress.json'
    if not prepared.exists() or read(prepared).get('status')!='complete':
        raise Blocked('Complete source preparation is required before early head fitting')
    root=WORK/'controller/head-prefetch'/source/str(seed);progress=root/'progress.json'
    with ownership(source,seed,exclusive=True):
        state=read(WORK/'controller/pipeline.json')
        if alive(state.get('controller_pid')):
            cell=state.get('cell',state);current=(cell.get('source'),cell.get('seed'))
            if current in schedule and schedule.index(current)>=schedule.index((source,seed)):
                raise Blocked('The main queue has reached this cell; let its existing owner continue')
        for path in (WORK/'controller/jobs').glob('*.json'):
            r=read(path)
            if r.get('source')==source and r.get('seed')==seed and r.get('status')=='running' and alive(r.get('pid')):
                raise Blocked('A worker for this cell is already active')
        census=read(WORK/'source_diagnostics'/source/str(seed)/'summary.json')['census']
        if not all(census.get(k) for k in ('positive','negative','identity_groups')):
            raise Blocked('Complete source census does not support both registered heads')
        try:
            for stage in ('fit-linear','train-compact'):
                write(progress,dict(status='running',source=source,seed=seed,stage=stage,pid=os.getpid(),updated_utc=now()))
                worker(stage,source,seed)
            compact=read(WORK/'fits'/source/str(seed)/'compact/progress.json')
            if compact['status'] not in ('waiting_for_mining','trained'):
                raise Blocked('Early head worker did not reach a durable registered boundary')
            step=compact.get('step',compact.get('updates'))
            expected=lock['event_updates']//2 if compact['status']=='waiting_for_mining' else lock['event_updates']
            if step!=expected:raise Blocked('Early head worker changed the registered horizon boundary')
            result=dict(status='complete',source=source,seed=seed,compact_status=compact['status'],
                compact_step=step,source_only=True,
                mining_started=False,target_predictions_started=False,finished_utc=now())
            write(progress,result);return result
        except Exception as exc:
            write(progress,dict(status='failed',source=source,seed=seed,reason=str(exc),finished_utc=now()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['44b6','6bba'],required=True)
    p.add_argument('--seed',type=int,choices=[20260918,314159],required=True)
    a=p.parse_args()
    try:print(json.dumps(run(a.source,a.seed),indent=2))
    except Blocked as exc:raise SystemExit(str(exc))
