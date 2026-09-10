"""Fresh complete graphs, pinned official node/division matching and aggregation."""
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from .common import *

def path(variant,name):
    return (V3/'selected_predictions' if variant=='C0' else OUT/'predictions'/variant)/f'{name}.npz'

def one(args):
    variant,row,locked=args; name=row['dataset']; p=path(variant,name)
    dest=OUT/'evaluation'/variant/f'{name}.json'
    assert sha(p)==locked
    if dest.exists():
        d=read(dest); assert d['prediction_sha256']==locked
        if 'match_distance_sum_um' in d:return d
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v3.common import validate
    start=time.monotonic();g=arrays(p);gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    validate(g['nodes'],g['edges'],row['image_shape'])
    result,matched,tp=evaluate_graph(name,g['nodes'],g['edges'],gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    child={}
    for aa,bb in gt['edges']:child.setdefault(int(aa),[]).append(int(bb))
    forks=[cc for cc in child.values() if len(cc)==2];observed=set(matched.values())
    pred_points={int(n[0]):n[2:] for n in g['nodes']};gt_points={int(n[0]):n[2:] for n in gt['nodes']}
    distance=np.array([np.linalg.norm((pred_points[a]-gt_points[b])*row['physical_scale']) for a,b in matched.items()])
    result.update(variant=variant,embryo=row['embryo'],prediction_sha256=locked,seconds=time.monotonic()-start,
        count_ratio=float(len(g['nodes'])/row['estimated_total']),
        gt_node_count=len(gt['nodes']),
        exact_duplicate_centers=int(len(g['nodes'])-len(np.unique(g['nodes'][:,1:],axis=0))),
        match_distance_sum_um=float(distance.sum()),match_distance_squared_sum_um2=float((distance**2).sum()),
        mean_match_distance_um=float(distance.mean()) if len(distance) else None,
        gt_division_parents=len(forks),division_daughters_matched=sum(set(cc)<=observed for cc in forks))
    recovered=np.array(sorted({(matched[int(a)],matched[int(b)]) for a,b in tp}),np.int64).reshape(-1,2)
    save(OUT/'evaluation_matches'/variant/f'{name}.npz',matches=np.array(sorted(matched.items()),np.int64).reshape(-1,2),recovered=recovered)
    write(dest,result);return result

def freeze(variants):
    names=sorted(r['dataset'] for r in inputs())
    locks={v:{n:sha(path(v,n)) for n in names} for v in variants}
    for v in variants:
        assert sorted(p.stem for p in path(v,names[0]).parent.glob('*.npz'))==names
    p=OUT/'prediction_lock.json'
    if p.exists():
        old=read(p)
        for v in old['graphs']: assert locks.get(v,old['graphs'][v])==old['graphs'][v]
        locks={**old['graphs'],**locks}
    assert len(locks)<=24
    metric_root=REPO/'work/annotation-selection-v1/official/src/tracking_cellmot'
    models=read(OUT/'checkpoint_manifest.json') if (OUT/'checkpoint_manifest.json').exists() else {}
    if (OUT/'checkpoint_manifest_secondary.json').exists():models.update(read(OUT/'checkpoint_manifest_secondary.json'))
    if (OUT/'checkpoint_manifest_rendering.json').exists():models.update(read(OUT/'checkpoint_manifest_rendering.json'))
    frozen=dict(created=now(),graphs=locks,models=models,
        protocol_sha256=sha(OUT/'protocol.json'),both_directions_frozen=True,
        metric_files={p.name:sha(p) for p in sorted(metric_root.glob('*.py'))})
    write(p,frozen)
    round_id=hashlib.sha256(json.dumps(locks,sort_keys=True).encode()).hexdigest()[:16]
    round_path=OUT/'round_locks'/f'{len(locks):02d}_{round_id}.json'
    if not round_path.exists():write(round_path,frozen)
    return locks

def run(variants, workers=8):
    from annotation_selection.metric_adapter import aggregate
    locks=freeze(variants); rows=read(V1/'inventory.json')
    tasks=[(v,r,locks[v][r['dataset']]) for v in variants for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(one,tasks,chunksize=1)):
            if i%40==0:print('official score',i+1,'/',len(tasks),r['variant'],flush=True)
    all_rows=[];summ=[]
    for v in locks:
        ps=sorted((OUT/'evaluation'/v).glob('*.json'))
        if len(ps)!=199:continue
        rr=[read(p) for p in ps];all_rows+=rr
        for em in ['44b6','6bba','pooled']:
            sub=[r for r in rr if em=='pooled' or r['embryo']==em]
            a=aggregate(sub,[r['dataset'] for r in sub]);summ.append(dict(variant=v,embryo=em,samples=len(sub),**{**{k:v for k,v in a.items() if k!='counts'},**a['counts']},delta_v3=a['score']-BASE[em]))
    pd.DataFrame(all_rows).to_csv(OUT/'score_rows.csv',index=False)
    pd.DataFrame(summ).to_csv(OUT/'ablation_scores.csv',index=False)
    for r in summ:
        if r['variant']=='C0':assert abs(r['score']-BASE[r['embryo']])<1e-12,r
    write(OUT/'baseline_verification.json',{'fresh_official_rescore':True,'rows':[r for r in summ if r['variant']=='C0']})
    print([(r['variant'],r['score'],r['delta_v3']) for r in summ if r['embryo']=='pooled'],flush=True)
