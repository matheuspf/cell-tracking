"""Source-training labels. Kept separate from prediction-only code."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .common import (OUT,V1,SCALE,SEED,adjacency,inventory,load_graph,read_json,
    run_pool,save_arrays,write_json)
from .hypotheses import build


def one(row):
    name=row['dataset'];out=OUT/'evaluation/training'/f'{name}.npz'
    if out.exists() and out.with_suffix('.json').exists():return name
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    c=load_graph(OUT/'native'/f'{name}.npz')
    hp=OUT/'hypotheses'/f'{name}.npz'
    h=load_graph(hp) if hp.exists() else build(b,c)
    if not hp.exists():save_arrays(hp,**h)
    m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')['matched_gt_id']
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');gn,ge=gt['nodes'],gt['edges']
    gs=set(map(tuple,ge));gix,gpred,gsucc=adjacency(gn,ge)
    gout=set(ge[:,0]);gin=set(ge[:,1])
    pairs=c['pairs'];a,d=pairs.T
    y=np.array([1 if (int(m[i]),int(m[j])) in gs else
        0 if m[i] in gout or m[j] in gin else -1 for i,j in pairs],np.int8)
    # Association negatives are contradictory at a locally annotated endpoint,
    # not generic unknown/unannotated associations.
    units=b['tracklet']
    positive=y==1;negative=np.flatnonzero(y==0)
    rng=np.random.default_rng(SEED+sum(name.encode()))
    # Cluster-aware negative sampling: uniformly sample at most 48 alternatives
    # per predicted source tracklet. Inclusion probabilities are recorded.
    sampled=[];inclusion=[]
    groups=defaultdict(list)
    for k in negative:groups[int(units[a[k]])].append(k)
    for members in groups.values():
        prob=min(1.,48/len(members));chosen=rng.choice(members,min(48,len(members)),replace=False)
        sampled.extend(chosen);inclusion.extend([prob]*len(chosen))
    edge_index=np.r_[np.flatnonzero(positive),np.array(sampled,int)]
    edge_probability=np.r_[np.ones(positive.sum()),inclusion].astype(np.float32)
    triples=h['triples'];fy=np.full(len(triples),-1,np.int8);event=np.full(len(triples),-1,np.int64)
    _,pred,succ=adjacency(n,e);node_ix={int(r[0]):i for i,r in enumerate(n)}
    by_parent=defaultdict(list)
    for k,(i,a,d) in enumerate(triples):by_parent[int(i)].append(k)
    windows=read_json(OUT/'evaluation/census'/f'{name}.json')['divisions']
    coverage=[]
    for w in windows:
        if w['kind']!='gt_division':continue
        divider=gix[w['event_id']];children=gsucc[divider]
        parent_gt={w['event_id'],*[int(gn[j,0]) for j in gpred[divider]]}
        daughter_gt=[{int(gn[ch,0]),*[int(gn[j,0]) for j in gsucc[ch]]} for ch in children]
        local={node_ix[int(p)]:int(g) for p,g in w['local_match_pairs']}
        parents={p for p,g in local.items() if g in parent_gt}
        daughters=[{p for p,g in local.items() if g in group} for group in daughter_gt]
        parent_options=parents|{j for p in parents for j in succ[p]}
        recovered=[]
        for i in parent_options:
            if not ({i,*pred[i]}&parents):continue
            for k in by_parent[i]:
                _,a,d=triples[k]
                la,lb={a,*succ[a]},{d,*succ[d]}
                if (la&daughters[0] and lb&daughters[1]) or (la&daughters[1] and lb&daughters[0]):
                    fy[k]=1;event[k]=w['event_id'];recovered.append(k)
        coverage.append(dict(dataset=name,embryo=row['embryo'],event_id=w['event_id'],
            positive_hypotheses=len(recovered),candidate_covered=bool(recovered)))
    div_positions=gn[[len(ch)>=2 for ch in gsucc]]
    for k,(i,a,d) in enumerate(triples):
        if fy[k]==1 or m[i]<0:continue
        gi=gix[int(m[i])]
        # Require a fully observed linear GT context on both sides. Mask all
        # candidate windows near an annotated division, including timing shifts.
        valid=True
        for direction in [gpred,gsucc]:
            cur=gi
            for _ in range(2):
                if len(direction[cur])!=1 or len(gsucc[cur])>1:valid=False;break
                cur=direction[cur][0]
        if len(div_positions):
            near=(np.abs(div_positions[:,1]-n[i,1])<=3)&(np.linalg.norm((div_positions[:,2:]-n[i,2:])*SCALE,axis=1)<20)
            if near.any():valid=False
        if valid:fy[k]=0
    # All positives retained; bounded contradictory negatives per tracklet.
    fp=np.flatnonzero(fy==1);neg=np.flatnonzero(fy==0);groups=defaultdict(list)
    for k in neg:groups[int(units[triples[k,0]])].append(k)
    sampled=[];inclusion=[]
    for members in groups.values():
        prob=min(1.,24/len(members));chosen=rng.choice(members,min(24,len(members)),replace=False)
        sampled.extend(chosen);inclusion.extend([prob]*len(chosen))
    fi=np.r_[fp,np.array(sampled,int)];fprob=np.r_[np.ones(len(fp)),inclusion].astype(np.float32)
    save_arrays(out,edge_index=edge_index,edge_y=y[edge_index],edge_probability=edge_probability,
        fork_index=fi,fork_y=fy[fi],fork_probability=fprob,fork_event=event[fi])
    write_json(out.with_suffix('.json'),dict(dataset=name,edge_positive=int(positive.sum()),
        edge_negative_population=len(negative),edge_negative_sampled=len(edge_index)-int(positive.sum()),
        edge_positive_tracklet_groups=len(np.unique(units[pairs[positive,0]])),
        division_observations=coverage,fork_positive=len(fp),fork_negative_population=len(neg),
        fork_negative_sampled=len(fi)-len(fp),
        unique_positive_event_observations=len(set(event[fp])),
        note='event IDs are observation groups, not guaranteed unique biological events across unregistered crops'))
    return name


def run(args):
    inv=inventory();rows=inv[:args.limit] if args.limit else inv
    list(run_pool(one,rows,args.workers))
    meta=[read_json(OUT/'evaluation/training'/f"{r['dataset']}.json") for r in rows]
    write_json(OUT/'training_candidate_coverage.json',dict(samples=len(rows),
        divisions=[x for r in meta for x in r['division_observations']],
        totals={k:sum(r[k] for r in meta) for k in ['edge_positive','edge_negative_population','edge_negative_sampled',
            'fork_positive','fork_negative_population','fork_negative_sampled','unique_positive_event_observations']}))
    print({k:sum(r[k] for r in meta) for k in ['edge_positive','fork_positive','unique_positive_event_observations']},flush=True)
