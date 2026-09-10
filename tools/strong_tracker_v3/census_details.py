"""Evaluation-only endpoint ambiguity and no-motion phase attribution."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from .common import *

def run(ctx,args=None):
    totals=Counter();gate=Counter();examples=[]
    for s in ctx.samples():
        name=s['dataset'];d=read_json(ctx.out/'evaluation/census'/f'{name}.json')
        b=load_graph(ctx.incumbent(name));n=b['nodes'];gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
        gn={int(r[0]):r for r in gt['nodes']};scale=np.asarray(s['physical_scale'])
        trees={int(t):cKDTree(n[n[:,1]==t,2:]*scale) for t in np.unique(n[:,1])}
        stage_edges={phase:set(map(tuple,load_graph(ctx.v2/'replay/no_motion'/name/f'{phase}.npz')['edges']))
            for phase in ['motion','gaps','divisions','pruned','final']}
        for r in d['edges']:
            if r['kind']!='edge_fn':continue
            available=[]
            for key in ['gt_source','gt_target']:
                p=gn[r[key]];tree=trees.get(int(p[1]))
                available.append(len(tree.query_ball_point(p[2:]*scale,7.)) if tree else 0)
            reason=r['primary_reason']
            if not r['source_matched'] or not r['target_matched']:
                reason='missing_points' if min(available)==0 else 'ambiguous_node_assignment'
            totals[reason]+=1
            pair=(r['pred_source'],r['pred_target'])
            present={phase:pair in edges for phase,edges in stage_edges.items()}
            if not present['final'] and any(present[p] for p in ['motion','gaps','divisions','pruned']):gate['lost_by_later_phase_same_ids']+=1
            if r['source_matched'] and r['target_matched']:
                gate['matched_endpoints_before_geometry']+=1
                if r['distance_um']<=16:gate['within_16um']+=1
                if r['forward_rank']<6:gate['top_six_before_distance']+=1
                if r['distance_um']<=16 and r['forward_rank']<6:gate['six_and_16um']+=1
            examples.append(dict(**r,endpoint_candidate_counts=available,refined_reason=reason,phase_edge_presence=present))
    write_json(ctx.out/'evaluation/edge_residual_details.json',dict(rows=examples,inputs=sha(ctx.out/'incumbent_manifest.json')))
    summary=dict(samples=199,total_edge_fn=sum(totals.values()),reasons=totals,gate_coverage=gate,
        attribution='Stable predicted IDs trace edge presence; stage coordinate matches are separate and no removal benefit is inferred from this trace.')
    write_json(ctx.out/'census_endpoint_summary.json',summary);print(summary,flush=True)
