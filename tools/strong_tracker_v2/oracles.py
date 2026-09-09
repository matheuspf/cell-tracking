"""EVALUATION ONLY. Impossible inference interventions; never imported by repair."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from annotation_selection.metric_adapter import aggregate, evaluate_graph, match_nodes

from .common import (OUT, V1, adjacency, inventory, load_graph, read_json, run_pool,
    save_graph, stage, validate, write_json)


def legal_links(n,e,gn,ge,matches,allowed=None):
    """Correct annotated transitions only, leaving unknown graph regions intact.

    Search is restricted to the global one-to-one assignment and then a single
    pass over local division windows. This is not a global optimal upper bound.
    """
    gt_edges=set(map(tuple,ge));rev={j:i for i,j in matches.items()}
    gout=set(ge[:,0]);gin=set(ge[:,1])
    kept={(int(a),int(b)) for a,b in e if
          (matches.get(int(a)) not in gout and matches.get(int(b)) not in gin)
          or (matches.get(int(a)),matches.get(int(b))) in gt_edges}
    desired={(rev[int(a)],rev[int(b)]) for a,b in ge if int(a) in rev and int(b) in rev}
    if allowed is not None:desired &= allowed
    # All chosen edges come from legal GT transitions and a bijection. Remove
    # competing unknown boundary edges at constrained endpoints jointly.
    for a,b in sorted(desired):
        kept={x for x in kept if x[1]!=b or x[0]==a}
        existing=sorted(x for x in kept if x[0]==a and x not in desired)
        while sum(x[0]==a for x in kept)>=2 and (a,b) not in kept and existing:
            kept.remove(existing.pop())
        if sum(x[0]==a for x in kept)<2 or (a,b) in kept:kept.add((a,b))
    return np.array(sorted(kept),np.int64).reshape(-1,2)


def one(row):
    name=row['dataset'];p=OUT/'evaluation/oracles'/f'{name}.json'
    if p.exists():return name
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');gn,ge=gt['nodes'],gt['edges']
    m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
    matches={int(i):int(j) for i,j in zip(n[:,0],m['matched_gt_id']) if j>=0}
    c=load_graph(OUT/'native'/f'{name}.npz')
    allowed={(int(n[i,0]),int(n[j,0])) for i,j in c['pairs']}
    topology=legal_links(n,e,gn,ge,matches,allowed)
    # Local-window mapping supplies time-shifted daughter evidence that global
    # matching can assign differently. Apply direct local GT transitions only.
    windows=read_json(OUT/'evaluation/census'/f'{name}.json')['divisions']
    for w in windows:
        if w['kind']!='gt_division':continue
        local=dict(w['local_match_pairs'])
        subset=set(w['gt_window_nodes']);ge_local=np.array([x for x in ge if x[0] in subset and x[1] in subset],np.int64).reshape(-1,2)
        if len(ge_local):topology=legal_links(n,topology,gn,ge_local,local,allowed)
    missing=[r for r in gn if int(r[0]) not in set(matches.values())]
    injected=n.copy();injected_matches=dict(matches)
    if missing:
        extra=[];next_id=int(n[:,0].max())+1
        for k,r in enumerate(missing):
            extra.append([next_id+k,*r[1:]])
            injected_matches[next_id+k]=int(r[0])
        injected=np.concatenate([n,np.array(extra,np.int64)])
    ie=legal_links(injected,e,gn,ge,injected_matches)
    # Distinguish localization failures by doing one fresh assignment after
    # impossible center injection before reconnecting known transitions.
    mm=match_nodes(injected,ie,gn,ge,row['physical_scale'])
    ie=legal_links(injected,ie,gn,ge,mm)
    keep=np.isin(n[:,0],list(matches))
    pn=n[keep];pe=e[np.isin(e[:,0],pn[:,0])&np.isin(e[:,1],pn[:,0])]
    variants={'oracle_fixed_nodes_links':(n,topology),'oracle_gt_center_injection':(injected,ie),
              'oracle_gt_pruning':(pn,pe)}
    results=[]
    for v,(nn,ee) in variants.items():
        start=time.perf_counter();valid=validate(nn,ee,row['image_shape'],reference=n,allow_legacy_bounds=True)
        save_graph(OUT/'evaluation/oracle_graphs'/v/f'{name}.npz',nn,ee)
        r,_,_=evaluate_graph(name,nn,ee,gn,ge,row['physical_scale'],row['estimated_total'])
        r.update(variant=v,embryo=row['embryo'],seconds=time.perf_counter()-start,evaluation_only=True)
        results.append(r)
    write_json(p,dict(dataset=name,results=results,missing_gt_centers_injected=len(missing)))
    return name


def run(args):
    rows=inventory();rows=rows[:args.limit] if args.limit else rows
    list(run_pool(one,rows,args.workers))
    scores=[s for r in rows for s in read_json(OUT/'evaluation/oracles'/f"{r['dataset']}.json")['results']]
    scores += [read_json(OUT/'evaluation/census'/f"{r['dataset']}.json")['base'] for r in rows]
    pd.DataFrame(scores).to_csv(OUT/'oracle_score_rows.csv',index=False)
    summaries={}
    for v in sorted({s['variant'] for s in scores}):
        summaries[v]={}
        for embryo in ['44b6','6bba','pooled']:
            subset=[s for s in scores if s['variant']==v and (embryo=='pooled' or s['embryo']==embryo)]
            expected=[r['dataset'] for r in rows if embryo=='pooled' or r['embryo']==embryo]
            if expected:summaries[v][embryo]=aggregate(subset,expected)
    write_json(OUT/'oracle_diagnostics.json',dict(samples=len(rows),summaries=summaries,
        evaluation_only=True,global_upper_bound=False,
        fixed_nodes_search='Known GT transitions under global assignment; one pass local division-window transitions; existing native/physical candidate edges only; legal degrees',
        injection_search='One missing center per globally unmatched GT node; fresh assignment and known-transition rewiring',
        pruning_search='Delete globally unmatched candidate nodes; leave surviving baseline links',
        completeness='Heuristic feasible interventions, not exhaustive or globally optimized; GT outputs isolated from deployment features and models'))
    print({v:s['pooled']['score'] for v,s in summaries.items()},flush=True)
