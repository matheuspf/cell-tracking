"""Evaluation-only teacher coordinate/topology reconciliation, never inference."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .common import (graph_hash,load_graph,now,read_json,run_pool,save_graph,sha,write_json)
from .disagreements import teachers
from .context import RunContext


def edge_evidence(edges,match,truth):
    gt_edges=set(map(tuple,truth));out=set(truth[:,0]);inc=set(truth[:,1])
    tp={(int(a),int(b)) for a,b in edges if (match.get(int(a),-1),match.get(int(b),-1)) in gt_edges}
    fp={(int(a),int(b)) for a,b in edges if (int(a),int(b)) not in tp and
        (match.get(int(a),-1) in out or match.get(int(b),-1) in inc)}
    recovered={(int(match[a]),int(match[b])) for a,b in tp}
    return tp,fp,recovered


def one(task):
    from annotation_selection.metric_adapter import match_nodes
    ctx,s=task;name=s['dataset'];dest=ctx.out/'evaluation/teacher_reconciliation_partitioned'/f'{name}.json'
    base=load_graph(ctx.incumbent(name));raw=load_graph(ctx.v2/'raw'/f'{name}.npz')
    old=load_graph(ctx.v1/'baseline/public'/f'{name}.npz');gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
    inputs=dict(incumbent_sha256=sha(ctx.incumbent(name)),raw_sha256=sha(ctx.v2/'raw'/f'{name}.npz'),
        old_sha256=sha(ctx.v1/'baseline/public'/f'{name}.npz'),predictions_sha256=sha(ctx.v2/'predictions'/f'{name}.npz'),
        gt_sha256=sha(ctx.v1/'evaluation/gt'/f'{name}.npz'),code_sha256=sha(Path(__file__)))
    if dest.exists():
        saved=read_json(dest)
        if saved['inputs']!=inputs:raise ValueError('Teacher reconciliation fingerprint drift')
        return saved['aggregate']
    votes,audits=teachers(ctx,s,base['nodes'],base['edges'],raw)
    # Original raw follows the notebook integer export boundary; no image remapping.
    rn=raw['nodes'].copy();rn[:,2:]=np.maximum(0,np.rint(rn[:,2:]));rn=rn.astype(np.int64)
    original_nodes={'raw_neural':rn,'old_final':old['nodes'],'v2_E_native':old['nodes'],'v2_E_hgb':old['nodes']}
    with np.load(ctx.v2/'predictions'/f'{name}.npz',allow_pickle=False) as f:
        original_edges=dict(raw_neural=raw['edges'],old_final=old['edges'],
            v2_E_native=f['E_native_m1.5__edges'],v2_E_hgb=f['E_hgb_m1.5__edges'])
    # Matching depends on positions only. Shared old-center matches are an edge-census
    # diagnostic, not a substitute for fresh whole-graph scoring of variants.
    bm_path=ctx.out/'evaluation/matches/incumbent'/f'{name}.npz'
    if bm_path.exists():bm=dict(map(tuple,load_graph(bm_path)['matched_ids']))
    else:bm=match_nodes(base['nodes'],base['edges'],gt['nodes'],gt['edges'],s['physical_scale'])
    rm=match_nodes(rn,raw['edges'],gt['nodes'],gt['edges'],s['physical_scale'])
    om=match_nodes(old['nodes'],old['edges'],gt['nodes'],gt['edges'],s['physical_scale'])
    native_ids=set(map(int,raw['nodes'][:,0]));projected_center_matches={}
    for key,donor in [('raw_neural',rn),('old_final',old['nodes'])]:
        donor_map={int(r[0]):r for r in donor};n=base['nodes'].copy()
        for i,r in enumerate(n):
            k=int(r[0])
            if k in native_ids and k in donor_map and r[1]==donor_map[k][1]:n[i,2:]=donor_map[k][2:]
        ee=np.asarray(sorted(votes[key]),np.int64).reshape(-1,2)
        projected_center_matches[key]=match_nodes(n,ee,gt['nodes'],gt['edges'],s['physical_scale'])
    btp,bfp,brec=edge_evidence(base['edges'],bm,gt['edges']);records={};aggregate=[]
    for label in original_edges:
        original=original_edges[label];projected=np.asarray(sorted(votes[label]),np.int64).reshape(-1,2)
        ot,of,og=edge_evidence(original,rm if label=='raw_neural' else om,gt['edges'])
        ct,cf,cg=edge_evidence(projected,bm,gt['edges'])
        dm=projected_center_matches['raw_neural' if label=='raw_neural' else 'old_final']
        dt,df,dg=edge_evidence(projected,dm,gt['edges'])
        graph_path=ctx.out/'candidate_graphs'/f'C_{label}_canonical'/f'{name}.npz'
        if graph_path.exists():
            existing=load_graph(graph_path)
            if graph_hash(existing['nodes'],existing['edges'])!=graph_hash(base['nodes'],projected):raise ValueError('Canonical teacher drift')
        else:save_graph(graph_path,base['nodes'],projected)
        original_match=rm if label=='raw_neural' else om
        record=dict(dataset=name,embryo=s['embryo'],teacher=label,original_tp=len(ot),original_fp=len(of),
            canonical_tp=len(ct),canonical_fp=len(cf),original_recovered_gt_edges=len(og),canonical_recovered_gt_edges=len(cg),
            canonical_truth_lost_from_original=len(og-cg),canonical_truth_gained_from_original=len(cg-og),
            projected_at_donor_centers_tp=len(dt),projected_at_donor_centers_fp=len(df),
            mapping_universe_gt_gained=len(dg-og),mapping_universe_gt_lost=len(og-dg),
            coordinate_only_gt_gained=len(cg-dg),coordinate_only_gt_lost=len(dg-cg),
            original_gt_gains_vs_v2=len(og-brec),canonical_gt_gains_vs_v2=len(cg-brec),
            original_gt_losses_vs_v2=len(brec-og),canonical_gt_losses_vs_v2=len(brec-cg),
            original_predicted_tp_gains_vs_v2=len(ot-btp),canonical_predicted_tp_gains_vs_v2=len(ct-btp),
            original_predicted_tp_losses_vs_v2=len(btp-ot),canonical_predicted_tp_losses_vs_v2=len(btp-ct),
            original_teacher_tp_pairs_lost_but_truth_recovered_by_v2=sum(
                (original_match.get(a,-1),original_match.get(b,-1)) in brec for a,b in ot-btp),
            original_teacher_tp_pairs_lost_and_truth_missing_in_v2=sum(
                (original_match.get(a,-1),original_match.get(b,-1)) not in brec for a,b in ot-btp),
            old_tp_predicted_pairs_lost_but_truth_equivalently_recovered=sum((bm.get(a,-1),bm.get(b,-1)) in cg for a,b in btp-ct),
            **audits[label])
        records[label]=dict(original_gt_gains=sorted(og-brec),canonical_gt_gains=sorted(cg-brec),
            original_gt_losses=sorted(brec-og),canonical_gt_losses=sorted(brec-cg))
        aggregate.append(record)
    write_json(dest,dict(inputs=inputs,aggregate=aggregate,evaluation_only_truth=records,
        scope='fresh position-only matching for edge census; full variant scorer required for division and combined scores'))
    return aggregate


def run(ctx=None,workers=2):
    import pandas as pd
    ctx=ctx or RunContext.default();tic=time.perf_counter()
    records=[r for rows in run_pool(one,[(ctx,s) for s in ctx.samples()],workers) for r in rows]
    pd.DataFrame(records).to_csv(ctx.out/'teacher_reconciliation.csv',index=False)
    numeric=['original_tp','original_fp','canonical_tp','canonical_fp','original_recovered_gt_edges','canonical_recovered_gt_edges',
        'canonical_truth_lost_from_original','canonical_truth_gained_from_original','original_gt_gains_vs_v2','canonical_gt_gains_vs_v2',
        'projected_at_donor_centers_tp','projected_at_donor_centers_fp','mapping_universe_gt_gained','mapping_universe_gt_lost',
        'coordinate_only_gt_gained','coordinate_only_gt_lost',
        'original_gt_losses_vs_v2','canonical_gt_losses_vs_v2','original_predicted_tp_gains_vs_v2','canonical_predicted_tp_gains_vs_v2',
        'original_predicted_tp_losses_vs_v2','canonical_predicted_tp_losses_vs_v2','old_tp_predicted_pairs_lost_but_truth_equivalently_recovered',
        'original_teacher_tp_pairs_lost_but_truth_recovered_by_v2','original_teacher_tp_pairs_lost_and_truth_missing_in_v2',
        'donor_nodes','mapped_nodes','donor_edges','mapped_edges','different_centers']
    aggregate=[]
    for teacher in ['raw_neural','old_final','v2_E_native','v2_E_hgb']:
        for embryo in ['44b6','6bba','pooled']:
            rr=[r for r in records if r['teacher']==teacher and (embryo=='pooled' or r['embryo']==embryo)]
            a=dict(teacher=teacher,embryo=embryo,samples=len(rr),**{k:sum(r[k] for r in rr) for k in numeric})
            a['unmapped_node_fraction']=1-a['mapped_nodes']/max(a['donor_nodes'],1)
            a['unmapped_edge_fraction']=1-a['mapped_edges']/max(a['donor_edges'],1);aggregate.append(a)
    write_json(ctx.out/'teacher_reconciliation_summary.json',dict(created=now(),seconds=time.perf_counter()-tic,aggregate=aggregate,
        scope='edge matching diagnostics, no official combined-score claim',canonical_graph_variants=[f'C_{k}_canonical' for k in ['raw_neural','old_final','v2_E_native','v2_E_hgb']]))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(workers=a.workers)
