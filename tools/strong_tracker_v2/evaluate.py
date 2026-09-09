"""Exact official rescoring, with complete sample sets and fresh graph objects."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from annotation_selection.filter_graph import filtered
from annotation_selection.metric_adapter import aggregate,evaluate_graph

from .common import (OUT,V1,METRIC_REV,graph_hash,inventory,load_graph,read_json,
    run_pool,sha,stage,write_json)


def baseline_fp_set(n,e,gt,m):
    gs=set(map(tuple,gt['edges']));out=set(gt['edges'][:,0]);inc=set(gt['edges'][:,1])
    return {(int(a),int(b)) for a,b in e if (m.get(int(a)),m.get(int(b))) not in gs and
            (m.get(int(a)) in out or m.get(int(b)) in inc)}


def one(task):
    row,variant_names=task;name=row['dataset'];start=time.perf_counter()
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');gn,ge=gt['nodes'],gt['edges']
    pred=load_graph(OUT/'predictions'/f'{name}.npz');meta=read_json(OUT/'predictions'/f'{name}.json')
    cfg=read_json(OUT/'config_locks.json')['variants']
    before=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
    bm={int(i):int(j) for i,j in zip(n[:,0],before['matched_gt_id']) if j>=0}
    base_tp=set(map(tuple,before['tp_edges']));base_fp=baseline_fp_set(n,e,gt,bm)
    stamp=dict(predictions_sha256=sha(OUT/'predictions'/f'{name}.npz'),baseline_sha256=sha(V1/'baseline/public'/f'{name}.npz'),
        gt_sha256=sha(V1/'evaluation/gt'/f'{name}.npz'),metric_revision=METRIC_REV)
    for v in variant_names:
        dest=OUT/'evaluation/scores'/v/f'{name}.json'
        if dest.exists():
            assert read_json(dest)['inputs']==stamp,'Scored input drift'
            continue
        tic=time.perf_counter();keep=pred[v+'__keep'];ee=pred.get(v+'__edges',e)
        nn,ee=filtered(n,ee,keep)
        assert graph_hash(nn,ee)==meta['variants'][v]['graph_hash']
        result,matches,tp=evaluate_graph(name,nn,ee,gn,ge,row['physical_scale'],row['estimated_total'])
        fp=baseline_fp_set(nn,ee,gt,matches)
        conf=cfg[v];source=meta['source']
        result.update(variant=v,embryo=row['embryo'],source=source,fold=f'{source}->{row["embryo"]}',
            kind=conf['kind'],seconds=time.perf_counter()-tic,requested_keep=conf.get('keep',.9 if conf['kind']=='transfer' else 1.),
            realized_keep=len(nn)/len(n),baseline_num_pred_nodes=len(n),
            baseline_tp=len(base_tp),baseline_tp_rematched_surviving=len(base_tp&tp),
            newly_recovered_tp=len(tp-base_tp),lost_tp=len(base_tp-tp),
            removed_fp=len(base_fp-fp),introduced_fp=len(fp-base_fp),
            baseline_matched_nodes=len(bm),removed_baseline_matches=sum(not k and int(i) in bm for i,k in zip(n[:,0],keep)),
            lane='exploratory_public_upstream_contaminated')
        write_json(dest,dict(inputs=stamp,result=result))
    return f'{name} {len(variant_names)} variants {time.perf_counter()-start:.1f}s'


def collect():
    inv=inventory();expected_all=[r['dataset'] for r in inv]
    rows=[];summaries=[]
    for path in sorted((OUT/'evaluation/scores').iterdir()):
        if not path.is_dir():continue
        records=[read_json(p)['result'] for p in sorted(path.glob('*.json'))]
        if {r['dataset'] for r in records}!=set(expected_all):continue
        rows.extend(records)
        for embryo in ['44b6','6bba','pooled']:
            expected=[r['dataset'] for r in inv if embryo=='pooled' or r['embryo']==embryo]
            selected=[r for r in records if embryo=='pooled' or r['embryo']==embryo]
            a=aggregate(selected,expected)
            summaries.append(dict(variant=path.name,embryo=embryo,samples=len(selected),
                **{k:v for k,v in a.items() if k!='counts'},**{k:v for k,v in a['counts'].items() if k not in a},
                seconds=sum(r['seconds'] for r in selected),
                requested_keep=selected[0]['requested_keep'],
                realized_keep=sum(r['num_pred_nodes'] for r in selected)/sum(r['baseline_num_pred_nodes'] for r in selected),
                baseline_tp_rematched_surviving=sum(r['baseline_tp_rematched_surviving'] for r in selected),
                newly_recovered_tp=sum(r['newly_recovered_tp'] for r in selected),
                lost_tp=sum(r['lost_tp'] for r in selected),removed_fp=sum(r['removed_fp'] for r in selected),introduced_fp=sum(r['introduced_fp'] for r in selected)))
    pd.DataFrame(rows).to_csv(OUT/'score_rows.csv',index=False)
    frame=pd.DataFrame(summaries)
    if len(frame) and (frame.variant=='identity').any():
        base=frame[frame.variant=='identity'].set_index('embryo').score
        frame['delta']=frame.score-frame.embryo.map(base)
    frame.to_csv(OUT/'operating_points.csv',index=False)
    write_json(OUT/'score_completeness.json',dict(expected_samples=199,complete_variants=sorted({r['variant'] for r in rows}),
        score_rows=len(rows),exact_official_aggregation=True))
    if len(frame):print(frame[frame.embryo=='pooled'].sort_values('score',ascending=False)[['variant','score','delta']].head(15).to_string(index=False),flush=True)


def run(args):
    if args.variant=='collect':collect();return
    configs=read_json(OUT/'config_locks.json')['variants']
    names=list(configs)
    if args.variant=='repair':names=[v for v in names if configs[v]['kind'] in ['identity','D','E','DE']]
    elif args.variant=='filter':names=[v for v in names if configs[v]['kind'] in ['identity','F','transfer','DEF']]
    elif args.variant:names=args.variant.split(',')
    rows=inventory();rows=rows[:args.limit] if args.limit else rows
    stage('V260','scoring',variants=names,samples=len(rows))
    list(run_pool(one,[(r,names) for r in rows],args.workers))
    collect()
