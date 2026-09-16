"""Evaluation-only imports and fresh pinned official matching."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager,ExitStack
import time
import pandas as pd
from .common import *

def one(task):
    variant,row,expected=task;name=row['dataset'];start=time.monotonic()
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v3.common import validate,graph_hash
    g=graph(name,variant);h=graph_hash(g['nodes'],g['edges']);assert h==expected
    dest=OUT/'evaluation'/variant/f'{name}.json'
    if dest.exists():
        r=read(dest);assert r['graph_hash']==h;return r
    gtpath=V1/'evaluation/gt'/f'{name}.npz';gt=arrays(gtpath)
    validate(g['nodes'],g['edges'],row['image_shape'])
    r,m,tp=evaluate_graph(name,g['nodes'],g['edges'],gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    r.update(variant=variant,embryo=row['embryo'],graph_hash=h,gt_sha256=sha(gtpath),seconds=time.monotonic()-start)
    save(OUT/'evaluation_matches'/variant/f'{name}.npz',matches=np.array(sorted(m.items()),np.int64).reshape(-1,2),
         recovered=np.array(sorted({(m[int(a)],m[int(b)]) for a,b in tp}),np.int64).reshape(-1,2),tp=np.array(sorted(tp),np.int64).reshape(-1,2))
    write(dest,r);return r

def aggregate_complete():
    from annotation_selection.metric_adapter import aggregate
    rows=inventory()
    import fcntl
    handle=(OUT/'aggregation.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX)
    allrows=[];summary=[]
    for folder in sorted((OUT/'evaluation').iterdir()):
        if len(list(folder.glob('*.json')))!=199:continue
        rr=[read(folder/f"{r['dataset']}.json") for r in rows];allrows+=rr
        for em in ['44b6','6bba','pooled']:
            sub=[r for r in rr if em=='pooled' or r['embryo']==em];a=aggregate(sub,[r['dataset'] for r in sub])
            summary.append(dict(variant=folder.name,embryo=em,samples=len(sub),**{**{k:v for k,v in a.items() if k!='counts'},**a['counts']},delta_C0=a['score']-BASE[em]))
    pd.DataFrame(allrows).to_csv(OUT/'score_rows.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'ablation_scores.csv',index=False)
    for r in summary:
        if r['variant']=='C0':assert abs(r['delta_C0'])<1e-12,r
    print([r for r in summary if r['embryo']=='pooled'],flush=True)
    fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


@contextmanager
def variant_locks(variants,blocking=True):
    """Coordinate overlapping registered batches without racing atomic receipts."""
    import fcntl
    directory=OUT/'evaluation_variant_locks';directory.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for variant in sorted(set(variants)):
            assert variant.replace('_','').isalnum(),variant
            handle=stack.enter_context((directory/f'{variant}.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        yield


def _run(variants,workers=6,serial=False):
    from strong_tracker_v3.common import graph_hash
    from annotation_selection.metric_adapter import aggregate
    rows=inventory();assert len(rows)==199
    locks={v:{r['dataset']:graph_hash(**{k:graph(r['dataset'],v)[k] for k in ('nodes','edges')}) for r in rows} for v in variants}
    lock=OUT/'round_locks'/f'{digest(locks)[:16]}.json'
    if not lock.exists():write(lock,dict(frozen=now(),graphs=locks,both_directions_frozen=True,models={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').rglob('*.pt') if '.resume' not in p.name and '_tiny' not in p.name}))
    tasks=[(v,r,locks[v][r['dataset']]) for v in variants for r in rows]
    if serial:
        # One serialized scorer can overlap a bounded decoder pool. Admission
        # reserves at least 4 GiB below the summed RSS cap for the next clip.
        assert len(variants)==1
        import psutil
        for i,task in enumerate(tasks):
            while True:
                worker_ids=set()
                for p in psutil.process_iter(['cmdline']):
                    try:
                        cmd=p.info['cmdline'] or []
                        if any(x in cmd for x in ['image_native_tracking_v5.predict_batch','image_native_tracking_v5.evaluate',
                                'image_native_tracking_v5.headroom','image_native_tracking_v5.regret','image_native_tracking_v5.division_review']):
                            worker_ids.update(c.pid for c in p.children() if c.status()!=psutil.STATUS_ZOMBIE)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                current=read(OUT/'resource_current.json')
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(current['at'])).total_seconds()
                if len(worker_ids)<=10 and current['summed_process_rss_gib']<24 and 0<=age<90:break
                time.sleep(15)
            r=one(task)
            if i%25==0:print('official',i+1,len(tasks),r['variant'],flush=True)
    else:
        with cpu_batch(),ProcessPoolExecutor(max_workers=min(workers,6)) as pool:
            for i,r in enumerate(pool.map(one,tasks)):
                if i%25==0:print('official',i+1,len(tasks),r['variant'],flush=True)
    aggregate_complete()


def run(variants,workers=6,serial=False,nonblocking=False):
    import fcntl
    # Variant locks precede pool admission. An opportunistic serial caller can
    # skip a variant owned by a queued regular batch and score another ready one.
    with variant_locks(variants,blocking=not nonblocking):
        if serial:
            with (OUT/'serial_scorer.lock').open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX)
                return _run(variants,workers,serial=True)
        return _run(variants,workers)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('variants',nargs='*',default=['C0']);p.add_argument('--serial',action='store_true');a=p.parse_args()
    run(a.variants or ['C0'],serial=a.serial)
