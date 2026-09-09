"""Score cumulative repair phases and measured bypass interventions."""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd

from annotation_selection.metric_adapter import aggregate,evaluate_graph

from .common import OUT,V1,export_nodes,inventory,load_graph,read_json,run_pool,save_arrays,validate,write_json
from .replay import as_arrays,original_namespace


def bypass_after_gaps(name):
    root=OUT/'replay/identity'/name
    a=load_graph(root/'gaps.npz');ns=original_namespace(load_detector=False)
    nodes={int(r[0]):dict(node_id=int(r[0]),t=int(r[1]),z=float(r[2]),y=float(r[3]),x=float(r[4])) for r in a['nodes']}
    edges=[dict(source_id=int(i),target_id=int(j),edge_prob=None if not np.isfinite(p) else float(p)) for (i,j),p in zip(a['edges'],a['edge_prob'])]
    for edge in edges:
        edge['distance_um']=ns['edge_distance_um'](nodes[edge['source_id']],nodes[edge['target_id']])
    # Run exactly the downstream isolated/short-track pruning and smoothing,
    # bypassing safe division addition. Geometry filtering is disabled in preset.
    assert not ns['OUTPUT_DIVISION_GEOMETRY_FILTER']
    if ns['OUTPUT_PRUNE_ISOLATED']:
        used={int(x[k]) for x in edges for k in ['source_id','target_id']}
        if used:nodes={i:r for i,r in nodes.items() if i in used}
    from collections import defaultdict
    stats=defaultdict(int)
    nodes,edges=ns['filter_short_track_components'](nodes,edges,stats)
    nodes=ns['linefit_smooth_output_graph'](nodes,edges,stats)
    return as_arrays(nodes,edges)


def one(row):
    name=row['dataset'];dest=OUT/'evaluation/stages'/f'{name}.json'
    if dest.exists():return name
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');base=load_graph(V1/'baseline/public'/f'{name}.npz')
    results=[];root=OUT/'replay/identity'/name
    phases={p:load_graph(root/f'{p}.npz') for p in ['motion','gaps','divisions','pruned','final']}
    phases['bypass_safe_divisions']=bypass_after_gaps(name)
    # With smoothing last, its bypass is the saved pre-smoothing graph exactly.
    phases['bypass_smoothing']=phases['pruned']
    corrected=base['nodes'].copy();shape=np.array(row['image_shape'][1:])
    corrected[:,2:]=np.clip(corrected[:,2:],0,shape-1)
    phases['bounds_corrected']=dict(nodes=corrected,edges=base['edges'])
    for variant,a in phases.items():
        n=export_nodes(a['nodes']);e=a['edges'];start=time.perf_counter()
        validate(n,e,row['image_shape'],reference=n,allow_legacy_bounds=True)
        if variant in ['bypass_safe_divisions','bounds_corrected']:
            save_arrays(OUT/'stage_graphs'/variant/f'{name}.npz',nodes=n,edges=e)
        r,_,_=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        r.update(variant='stage_'+variant,embryo=row['embryo'],seconds=time.perf_counter()-start)
        results.append(r)
    write_json(dest,dict(dataset=name,results=results))
    return name


def run(args):
    inv=inventory();rows=inv[:args.limit] if args.limit else inv
    list(run_pool(one,rows,args.workers))
    results=[r for sample in rows for r in read_json(OUT/'evaluation/stages'/f"{sample['dataset']}.json")['results']]
    results += [read_json(OUT/'evaluation/census'/f"{sample['dataset']}.json")[k] for sample in rows for k in ['base','raw']]
    pd.DataFrame(results).to_csv(OUT/'stage_scores.csv',index=False)
    summaries=[]
    for v in sorted({r['variant'] for r in results}):
        for embryo in ['44b6','6bba','pooled']:
            chosen=[r for r in results if r['variant']==v and (embryo=='pooled' or r['embryo']==embryo)]
            expected=[r['dataset'] for r in rows if embryo=='pooled' or r['embryo']==embryo]
            if expected:summaries.append(dict(variant=v,embryo=embryo,**aggregate(chosen,expected)))
    write_json(OUT/'stage_summary.json',summaries)
    print({r['variant']:r['score'] for r in summaries if r['embryo']=='pooled'},flush=True)
