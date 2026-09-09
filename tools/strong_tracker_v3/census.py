"""Evaluation-only complete residual census on the frozen v2 incumbent."""
from collections import Counter
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .common import *

def one(task):
    ctx,row=task;name=row['dataset'];path=ctx.out/'evaluation/census'/f'{name}.json'
    graph=load_graph(ctx.incumbent(name));n,e=graph['nodes'],graph['edges']
    gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz');gn,ge=gt['nodes'],gt['edges']
    stamp=dict(graph=graph_hash(n,e),gt=sha(ctx.v1/'evaluation/gt'/f'{name}.npz'),code=sha(__file__))
    if path.exists():
        if read_json(path)['inputs']!=stamp:raise RuntimeError('Census cache drift')
        return name+' cached'
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v2.census import division_census
    # This reused function is pure and receives every graph/scale explicitly.
    start=time.perf_counter()
    base,matches,tp=evaluate_graph(name,n,e,gn,ge,row['physical_scale'],row['estimated_total'])
    divisions=division_census(name,n,e,gn,ge,row['physical_scale'])
    for phase in ['divisions','pruned']:
        g=load_graph(ctx.v2/'replay/no_motion'/name/f'{phase}.npz')
        nn=g['nodes'].copy();nn[:,2:]=np.clip(np.rint(nn[:,2:]),0,np.array(row['image_shape'][1:])-1)
        dr=division_census(name,nn.astype(np.int64),g['edges'],gn,ge,row['physical_scale'])
        recovered={r['event_id']:r['recovered'] for r in dr if r['kind']=='gt_division'}
        for r in divisions:
            if r['kind']=='gt_division':r[phase+'_recovered']=recovered[r['event_id']]
    for r in divisions:
        if r['kind']=='gt_division' and not r['recovered'] and (r['divisions_recovered'] or r['pruned_recovered']):
            r['initial_reason']=r['primary_reason'];r['primary_reason']='lost_by_later_phase'
    reverse={v:k for k,v in matches.items()};es=set(map(tuple,e));gout=set(ge[:,0]);gin=set(ge[:,1])
    ix={int(v):i for i,v in enumerate(n[:,0])};pos=n[:,2:]*row['physical_scale']
    raw=load_graph(ctx.v2/'raw'/f'{name}.npz');raw_edges=set(map(tuple,raw['edges']))
    native=load_graph(ctx.full/'inputs'/f'pre_ilp_{name}.npz')
    native_edges={tuple(map(int,r[:2])) for r in native['edge_scores']}
    edges=[]
    by_t={int(t):np.flatnonzero(n[:,1]==t) for t in np.unique(n[:,1])}
    trees={t:cKDTree(pos[i]) for t,i in by_t.items()}
    for a,b in ge:
        p,q=reverse.get(int(a),-1),reverse.get(int(b),-1)
        if (p,q) in es:continue
        distance=None;rank=None
        if p>=0 and q>=0:
            i,j=ix[p],ix[q];distance=float(np.linalg.norm(pos[j]-pos[i]));t=int(n[j,1])
            rank=len(trees[t].query_ball_point(pos[i],distance-1e-7))
        available=p>=0 and q>=0 and ((p,q) in native_edges or (distance<=16 and rank<6))
        row0=dict(dataset=name,embryo=row['embryo'],kind='edge_fn',gt_source=int(a),gt_target=int(b),
            pred_source=p,pred_target=q,source_matched=p>=0,target_matched=q>=0,distance_um=distance,forward_rank=rank,
            native_available=(p,q) in native_edges,raw_edge_present=(p,q) in raw_edges,candidate_available=available,
            primary_reason='missing_points_or_assignment' if min(p,q)<0 else 'alternative_available_but_rejected' if available else 'alternative_link_absent')
        edges.append(row0)
    for a,b in e:
        if (a,b) in tp:continue
        ga,gb=matches.get(int(a),-1),matches.get(int(b),-1)
        if ga not in gout and gb not in gin:continue
        edges.append(dict(dataset=name,embryo=row['embryo'],kind='edge_fp',gt_source=ga,gt_target=gb,
            pred_source=int(a),pred_target=int(b),source_matched=ga>=0,target_matched=gb>=0,
            raw_edge_present=(a,b) in raw_edges,native_available=(a,b) in native_edges,
            primary_reason='wrong_matched_association' if min(ga,gb)>=0 else 'one_endpoint_unmatched'))
    assert sum(x['kind']=='edge_fn' for x in edges)==base['edge_fn']
    assert sum(x['kind']=='edge_fp' for x in edges)==base['edge_fp']
    assert sum(x['kind']=='gt_division' and x['recovered'] for x in divisions)==base['division_tp']
    assert sum(x['kind']=='predicted_division_fp' for x in divisions)==base['division_fp']
    write_json(path,dict(inputs=stamp,base=base,divisions=divisions,edges=edges,seconds=time.perf_counter()-start,
        candidate_gate='Source-independent diagnostic pool: pre-ILP union six nearest <=16um; inference expanded pool measured separately'))
    return f'{name} census {time.perf_counter()-start:.1f}s'

def run(ctx,args):
    rows=ctx.eval_rows();list(run_pool(one,[(ctx,r) for r in rows],args.workers))
    records=[read_json(ctx.out/'evaluation/census'/f'{r["dataset"]}.json') for r in rows]
    div=[d for r in records for d in r['divisions']];edges=[e for r in records for e in r['edges']]
    pd.DataFrame(div).to_parquet(ctx.out/'evaluation/division_census.parquet',index=False)
    pd.DataFrame(edges).to_parquet(ctx.out/'evaluation/edge_census.parquet',index=False)
    summary=dict(samples=len(rows),division_observations=sum(x['kind']=='gt_division' for x in div),
        division_fn=sum(x['kind']=='gt_division' and not x['recovered'] for x in div),
        division_fp=sum(x['kind']=='predicted_division_fp' for x in div),
        division_reasons=Counter(x['primary_reason'] for x in div if x['kind']=='gt_division'),
        division_fp_reasons=Counter(x['primary_reason'] for x in div if x['kind']=='predicted_division_fp'),
        edge_reasons=Counter(x['kind']+':'+x['primary_reason'] for x in edges),
        fn_candidate_available=sum(x['kind']=='edge_fn' and x['candidate_available'] for x in edges),
        fn_native_available=sum(x['kind']=='edge_fn' and x['native_available'] for x in edges),
        exact_counts_reconciled=True,scorer='Fresh official local division assignments per no-motion phase')
    write_json(ctx.out/'census_summary.json',summary);print(summary,flush=True)
