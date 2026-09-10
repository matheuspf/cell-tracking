"""Evaluation-only imports and fresh pinned official matching."""
from concurrent.futures import ProcessPoolExecutor
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

def run(variants,workers=6):
    from strong_tracker_v3.common import graph_hash
    from annotation_selection.metric_adapter import aggregate
    rows=inventory();assert len(rows)==199
    locks={v:{r['dataset']:graph_hash(**{k:graph(r['dataset'],v)[k] for k in ('nodes','edges')}) for r in rows} for v in variants}
    lock=OUT/'round_locks'/f'{digest(locks)[:16]}.json'
    if not lock.exists():write(lock,dict(frozen=now(),graphs=locks,both_directions_frozen=True,models={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').rglob('*.pt') if '.resume' not in p.name and '_tiny' not in p.name}))
    tasks=[(v,r,locks[v][r['dataset']]) for v in variants for r in rows]
    with cpu_batch(),ProcessPoolExecutor(max_workers=min(workers,6)) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            if i%25==0:print('official',i+1,len(tasks),r['variant'],flush=True)
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

if __name__=='__main__':
    import sys
    run(sys.argv[1:] or ['C0'])
