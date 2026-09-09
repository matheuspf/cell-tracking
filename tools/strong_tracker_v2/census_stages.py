"""Evaluation-only stage attribution of all final division and link failures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .census import division_census
from .common import OUT,V1,export_nodes,inventory,load_graph,read_json,run_pool,write_json


def one(row):
    name=row['dataset'];dest=OUT/'evaluation/census_stages'/f'{name}.json'
    if dest.exists():return name
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz')
    census=read_json(OUT/'evaluation/census'/f'{name}.json')
    graphs={v:load_graph(OUT/'replay/identity'/name/f'{v}.npz') for v in ['motion','gaps','divisions','pruned','final']}
    stage_div={}
    for phase,g in graphs.items():
        # Each phase uses its own official local assignment, never global-match
        # bookkeeping projected backwards through the pipeline.
        stage_div[phase]=division_census(name,export_nodes(g['nodes']),g['edges'],gt['nodes'],gt['edges'],row['physical_scale'])
    mapped={phase:{r['event_id']:r for r in rr if r['kind']=='gt_division'} for phase,rr in stage_div.items()}
    for r in census['divisions']:
        if r['kind']=='gt_division':
            for phase,events in mapped.items():r[phase+'_recovered']=events[r['event_id']]['recovered']
            if not r['recovered'] and any(r[p+'_recovered'] for p in ['divisions','pruned']):
                r['initial_primary_reason']=r['primary_reason'];r['primary_reason']='correct_fork_suppressed_or_relocalized_by_repair'
                r['repair_suppressed_correct_fork']=True
        else:
            for phase,rr in stage_div.items():
                r[phase+'_fp']=any(x['kind']=='predicted_division_fp' and x['event_id']==r['event_id'] for x in rr)
                r[phase+'_fork_exists']=np.sum(graphs[phase]['edges'][:,0]==r['event_id'])>=2
    sets={phase:set(map(tuple,g['edges'])) for phase,g in graphs.items()}
    for r in census['edges']:
        edge=(r['pred_source'],r['pred_target'])
        for phase,s in sets.items():r[phase+'_same_id_edge_exists']=edge in s
    write_json(dest,dict(dataset=name,final_divisions=census['divisions'],final_edges=census['edges'],
        division_phase_rows=[dict(phase=phase,**r) for phase,rr in stage_div.items() for r in rr],
        edge_stage_attribution='Stable-ID edge presence; each phase independently rescored. Coordinate matching may differ by phase.'))
    return name


def run(args):
    inv=inventory();rows=inv[:args.limit] if args.limit else inv
    list(run_pool(one,rows,args.workers))
    out=[read_json(OUT/'evaluation/census_stages'/f"{r['dataset']}.json") for r in rows]
    pd.DataFrame([r for x in out for r in x['final_divisions']]).to_csv(OUT/'division_failure_census.csv',index=False)
    pd.DataFrame([r for x in out for r in x['final_edges']]).to_csv(OUT/'edge_failure_census.csv',index=False)
    pd.DataFrame([r for x in out for r in x['division_phase_rows']]).to_csv(OUT/'division_stage_census.csv',index=False)
    write_json(OUT/'census_stage_receipt.json',dict(samples=len(rows),official_local_match_per_phase=True,
        gt_division_observations=sum(r['kind']=='gt_division' for x in out for r in x['final_divisions']),
        final_division_fp=sum(r['kind']=='predicted_division_fp' for x in out for r in x['final_divisions']),
        final_edge_fn=sum(r['kind']=='edge_fn' for x in out for r in x['final_edges']),
        final_edge_fp=sum(r['kind']=='edge_fp' for x in out for r in x['final_edges'])))
