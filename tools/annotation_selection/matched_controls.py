"""Exact realized-cost controls for the frozen primary coherent selector."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

import numpy as np
import pandas as pd

from .common import OUT,SEED,load_graph,read_json,write_json
from .evaluate import confusion
from .filter_graph import PreparedFilter,filtered
from .metric_adapter import evaluate_graph


def one(task):
    row,source=task;name=row['dataset'];path=OUT/'evaluation/matched_budget_samples'/f'{name}.json'
    if path.exists():return name
    b=load_graph(OUT/'baseline/clean'/f'{name}.npz');gt=load_graph(OUT/'evaluation/gt'/f'{name}.npz')
    with np.load(OUT/'predictions'/source/f'{name}.npz') as f:membership=f['hgb_all_leaf7'];quality=f['quality']
    with np.load(OUT/'evaluation/membership'/f'{name}.npz') as f:labels=f['annotation_label']
    prepare=PreparedFilter(b['nodes'],b['edges']);records=[]
    for r in [.9,.8,.7,.5,.3,.1]:
        primary=prepare.mask(membership,r,'membership_tracklets',key='primary')
        target=int(primary.sum())
        controls=[('quality_exact',SEED,False)]
        if r==.9:controls.extend(('random_exact',seed,True) for seed in range(20))
        for model,seed,random in controls:
            keep=prepare.exact_tracklet_cost(quality,target,seed=seed,random=random)
            n,e=filtered(b['nodes'],b['edges'],keep)
            er,_,_=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
            er.update(confusion(labels,keep));er.update(embryo=row['embryo'],source=source,model_id=model,seed=seed,
                requested_keep=r,target_primary_nodes=target,exact_realized_budget=True,policy='lexicographic_whole_tracklet_exact_cost')
            records.append(er)
    write_json(path,dict(dataset=name,records=records))
    return name


def run(args):
    if not (OUT/'prediction_lock.json').exists():raise ValueError('Primary predictions must be locked')
    manifest=read_json(OUT/'fold_manifest.json');inv=read_json(OUT/'inventory.json')
    tasks=[(r,next(d['source'] for d in manifest['directions'] if d['outer']==r['embryo'])) for r in inv]
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):print(f'Exact-cost control {i}/{len(tasks)} {f.result()}',flush=True)
    records=[]
    for r,_ in tasks:records.extend(read_json(OUT/'evaluation/matched_budget_samples'/f'{r["dataset"]}.json')['records'])
    pd.DataFrame(records).to_csv(OUT/'matched_budget_controls.csv',index=False)
    pd.DataFrame(records).to_parquet(OUT/'matched_budget_controls.parquet',index=False)
