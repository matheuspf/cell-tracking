"""Evaluation-only failure census using the pinned local division matcher."""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from annotation_selection.metric_adapter import aggregate, evaluate_graph, make_graph
from tracking_cellmot import division_metrics as dm

from .common import (OUT, V1, SCALE, adjacency, export_nodes, graph_hash, inventory,
    load_graph, preservation_snapshot, raw_graph, read_json, run_pool, save_arrays,
    sha, stage, validate, write_json)
from . import native


def division_census(name, n, e, gn, ge, scale, candidate_pairs=None):
    g, rev = make_graph(n, e)
    gt, gr = make_graph(gn, ge)
    result = dm.score_divisions(g, gt, tuple(scale), 7.)
    windows = dm.extract_divisions(gt)
    matched = dm.match_divisions(g, gt, tuple(scale), 7.)
    evaluable, cross, malformed = dm._pred_division_fork_sets(g, gt, tuple(scale), 7.)
    invalid = cross | malformed
    forks = set(g.dividing_nodes())
    rows = []
    ni = {int(r[0]): r for r in n}
    gi = {int(r[0]): r for r in gn}
    for divider, window in windows.items():
        mg = matched[divider]
        attrs = dm._matched_node_attrs(mg)
        pairs = dict(attrs.iter_rows())
        parent_gt = {divider, *window.predecessors(divider)}
        children = window.successors(divider)
        daughter_gt = [{c, *window.successors(c)} for c in children]
        parent = {a for a,b in pairs.items() if b in parent_gt}
        daughters = [{a for a,b in pairs.items() if b in d} for d in daughter_gt]
        local_forks = (parent | {c for a in parent for c in mg.successors(a)}) & forks
        topology = {f for f in local_forks if dm._is_strongly_connected_division(mg, f, parent, daughters)}
        candidates_per_role = []
        for ids in [parent_gt, *daughter_gt]:
            hits = set()
            for j in ids:
                point = gi[gr[j]]
                ix = np.flatnonzero(n[:, 1] == point[1])
                near = ix[np.linalg.norm((n[ix, 2:]-point[2:])*scale, axis=1) <= 7.]
                hits.update(int(n[k, 0]) for k in near)
            candidates_per_role.append(len(hits))
        if result.scores[divider]: reason = 'recovered'
        elif not parent: reason = 'parent_side_evidence_absent'
        elif any(not d for d in daughters):
            reason = ('daughter_lineage_candidate_absent' if any(k == 0 for k in candidates_per_role[1:])
                      else 'daughter_assignment_competition')
        elif topology & invalid: reason = 'cross_component_or_shared_branch'
        elif topology: reason = 'division_pairing_collision'
        elif not local_forks: reason = 'two_daughters_present_single_branch_survives'
        else: reason = 'fork_wrong_local_timing_or_topology'
        rows.append(dict(dataset=name, embryo=name.split('_')[0], kind='gt_division',
            event_id=gr[divider], time=int(gi[gr[divider]][1]), recovered=result.scores[divider],
            primary_reason=reason, parent_matches=len(parent), daughter1_matches=len(daughters[0]),
            daughter2_matches=len(daughters[1]), parent_candidate_count=candidates_per_role[0],
            daughter1_candidate_count=candidates_per_role[1], daughter2_candidate_count=candidates_per_role[2],
            local_forks=len(local_forks), topology_valid_forks=len(topology),
            invalid_forks=len(local_forks & invalid),
            local_match_pairs=[[rev[a], gr[b]] for a,b in pairs.items()],
            gt_window_nodes=[gr[j] for j in window.node_ids()]))
    for f in sorted(result.fp_forks):
        rows.append(dict(dataset=name, embryo=name.split('_')[0], kind='predicted_division_fp',
            event_id=rev[f], time=int(ni[rev[f]][1]), recovered=0,
            primary_reason=('shared_or_merged_branch' if f in malformed else
                'cross_component_branches' if f in cross else 'evaluable_spurious_or_local_topology_reject'),
            evaluable_global=f in evaluable, local_match_pairs=[], gt_window_nodes=[]))
    return rows


def edge_census(name, n, e, gn, ge, matches, tp, candidates, raw):
    es = set(map(tuple, e)); gs = set(map(tuple, ge))
    reverse = {j:i for i,j in matches.items()}
    gt_out = set(ge[:, 0]); gt_in = set(ge[:, 1])
    ix = {int(i): j for j, i in enumerate(n[:, 0])}
    native_by_ids = {(int(n[a, 0]), int(n[b, 0])): j for j,(a,b) in enumerate(candidates['pairs'])}
    raw_edges = set(map(tuple, raw['edges']))
    rows = []
    for a,b in ge:
        p,q = reverse.get(int(a),-1),reverse.get(int(b),-1)
        if (p,q) in es: continue
        idx = native_by_ids.get((p,q))
        rows.append(dict(dataset=name,embryo=name.split('_')[0],kind='edge_fn',gt_source=int(a),gt_target=int(b),
            pred_source=p,pred_target=q,source_matched=p!=-1,target_matched=q!=-1,
            candidate_available=idx is not None,pre_ilp_available=idx is not None and not bool(candidates['edge_features'][idx,1]),
            raw_edge_present=(p,q) in raw_edges,
            primary_reason='endpoint_missing' if min(p,q)<0 else 'candidate_pruned_or_relinked' if (p,q) in raw_edges else
                'available_alternative_not_selected' if idx is not None else 'outside_candidate_pool'))
    for a,b in e:
        if (a,b) in tp: continue
        ga,gb=matches.get(int(a),-1),matches.get(int(b),-1)
        if ga not in gt_out and gb not in gt_in: continue
        idx=native_by_ids.get((int(a),int(b)))
        rows.append(dict(dataset=name,embryo=name.split('_')[0],kind='edge_fp',gt_source=ga,gt_target=gb,
            pred_source=int(a),pred_target=int(b),source_matched=ga!=-1,target_matched=gb!=-1,
            candidate_available=True,pre_ilp_available=idx is not None and not bool(candidates['edge_features'][idx,1]),
            raw_edge_present=(a,b) in raw_edges,
            primary_reason='wrong_matched_association' if min(ga,gb)>=0 else 'one_endpoint_unmatched'))
    return rows


def one(row):
    name=row['dataset']; dest=OUT/'evaluation/census'/f'{name}.json'
    if dest.exists(): return name
    start=time.perf_counter()
    b=load_graph(V1/'baseline/public'/f'{name}.npz'); n,e=b['nodes'],b['edges']
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz'); gn,ge=gt['nodes'],gt['edges']
    meta=read_json(V1/'baseline/public'/f'{name}.json')
    assert sha(V1/'baseline/public'/f'{name}.npz')==meta['sha256']
    assert graph_hash(n,e)==meta['graph_hash']
    validity=validate(n,e,row['image_shape'],allow_legacy_bounds=True)
    raw=raw_graph(name)
    c=native.build(name,b,raw);save_arrays(OUT/'native'/f'{name}.npz',**c)
    base,matches,tp=evaluate_graph(name,n,e,gn,ge,row['physical_scale'],row['estimated_total'])
    base.update(variant='identity',embryo=row['embryo'])
    rr,_,_=evaluate_graph(name,export_nodes(raw['nodes']),raw['edges'],gn,ge,row['physical_scale'],row['estimated_total'])
    rr.update(variant='raw_neural',embryo=row['embryo'])
    labels=np.array([int(i in matches) for i in n[:,0]],np.uint8)
    mids=np.array([matches.get(int(i),-1) for i in n[:,0]],np.int64)
    save_arrays(OUT/'evaluation/membership'/f'{name}.npz',annotation_label=labels,matched_gt_id=mids,
        tp_edges=np.array(sorted(tp),np.int64).reshape(-1,2))
    div=division_census(name,n,e,gn,ge,row['physical_scale'],c['pairs'])
    edges=edge_census(name,n,e,gn,ge,matches,tp,c,raw)
    assert sum(x['kind']=='edge_fn' for x in edges)==base['edge_fn']
    assert sum(x['kind']=='edge_fp' for x in edges)==base['edge_fp']
    assert sum(x['kind']=='gt_division' and x['recovered'] for x in div)==base['division_tp']
    assert sum(x['kind']=='predicted_division_fp' for x in div)==base['division_fp']
    write_json(dest,dict(dataset=name,seconds=time.perf_counter()-start,base=base,raw=rr,divisions=div,edges=edges,
        validity=validity,origin_inserted=int((c['native_index']<0).sum()),
        provenance='Stable raw node IDs; exact raw tzyx equals pre-ILP row; no nearest-neighbor score reassignment'))
    return name


def run(args):
    preservation_snapshot();stage('V200','running');stage('V210','running')
    inv=inventory();rows=inv[:args.limit] if args.limit else inv
    list(run_pool(one,rows,args.workers))
    results=[read_json(OUT/'evaluation/census'/f"{r['dataset']}.json") for r in rows]
    div=[d for r in results for d in r['divisions']];edges=[d for r in results for d in r['edges']]
    pd.DataFrame(div).to_csv(OUT/'division_failure_census.csv',index=False)
    pd.DataFrame(edges).to_csv(OUT/'edge_failure_census.csv',index=False)
    score=[r[k] for r in results for k in ['base','raw']]
    pd.DataFrame(score).to_csv(OUT/'stage_scores.csv',index=False)
    expected=[r['dataset'] for r in rows]
    summaries={v:aggregate([s for s in score if s['variant']==v],expected) for v in ['identity','raw_neural']}
    if not args.limit:
        expected_baseline=read_json('handover/strong-tracker-v2/measured_baseline.json')['public_pooled']
        assert abs(summaries['identity']['score']-expected_baseline['score'])<1e-12
        for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']:
            assert sum(s[k] for s in score if s['variant']=='identity')==expected_baseline[k]
    write_json(OUT/'baseline_verification.json',dict(samples=len(rows),fresh_official_scoring=True,
        summaries=summaries,identity_parity=True,origin_inserted=sum(r['origin_inserted'] for r in results),
        six_legacy_coordinates_preserved=sum(r['validity']['out_of_bounds'] for r in results)))
    write_json(OUT/'candidate_coverage.json',dict(samples=len(rows),division_reasons=Counter(x['primary_reason'] for x in div if x['kind']=='gt_division'),
        division_fp_reasons=Counter(x['primary_reason'] for x in div if x['kind']=='predicted_division_fp'),
        edge_reasons=Counter(x['kind']+':'+x['primary_reason'] for x in edges),
        fn_candidate_available=sum(x['kind']=='edge_fn' and x['candidate_available'] for x in edges),
        fn_pre_ilp_available=sum(x['kind']=='edge_fn' and x['pre_ilp_available'] for x in edges)))
    stage('V200','baseline_verified_replay_pending',samples=len(rows),score=summaries['identity']['score'])
    stage('V210','census_complete_oracles_and_stages_pending',samples=len(rows))
    print(summaries,flush=True)
