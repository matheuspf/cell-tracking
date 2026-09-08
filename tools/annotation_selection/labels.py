"""Evaluation-only labels. Never imported by deployment inference."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import OUT,METRIC_REV,load_graph,read_json,sha,stage,write_json
from .metric_adapter import match_nodes


def label_one(row):
    name=row['dataset'];path=OUT/'evaluation/membership'/f'{name}.npz'
    base=OUT/'baseline/clean'/f'{name}.npz'
    stamp=dict(baseline_sha256=sha(base),gt_sha256=sha(OUT/'evaluation/gt'/f'{name}.npz'),metric_revision=METRIC_REV)
    if path.exists() and path.with_suffix('.json').exists():
        if read_json(path.with_suffix('.json'))!=stamp:raise ValueError('Membership cache drift')
        return name
    b=load_graph(base);gt=load_graph(OUT/'evaluation/gt'/f'{name}.npz')
    nodes=b['nodes'];gn=gt['nodes'];scale=np.array(row['physical_scale'])
    matches=match_nodes(nodes,b['edges'],gn,gt['edges'],scale)
    labels=np.array([int(i in matches) for i in nodes[:,0]],np.uint8)
    match=np.array([matches.get(int(i),-1) for i in nodes[:,0]],np.int64)
    distance=np.full(len(nodes),np.inf,np.float32)
    for t in np.unique(nodes[:,1]):
        idx=np.flatnonzero(nodes[:,1]==t);g=gn[gn[:,1]==t]
        if len(g):distance[idx]=cKDTree(g[:,2:]*scale).query(nodes[idx,2:]*scale)[0]
    ambiguous=(labels==0)&(distance<=7.)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,annotation_label=labels,matched_gt_id=match,ambiguous=ambiguous,nearest_gt_um=distance)
    write_json(path.with_suffix('.json'),stamp)
    return name


def run(args):
    rows=read_json(OUT/'inventory.json')
    if args.limit:rows=rows[:args.limit]
    stage('S030','running',samples=len(rows),embargo='Labels are source-training inputs; outer outcomes not reported')
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        for i,n in enumerate(pool.map(label_one,rows),1):
            print(f'Labels sealed {i}/{len(rows)} {n}',flush=True)
    stage('S030','complete',samples=len(rows),one_to_one=True,outer_results_revealed=False)
