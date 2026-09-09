"""Prediction-only constrained link/fork decoding on frozen nodes."""
from __future__ import annotations

from collections import defaultdict
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

from .common import adjacency


def logit(p):
    p=np.clip(p,1e-5,1-1e-5)
    return np.log(p/(1-p))


def associations(nodes,edges,native,probability,margin=1.5):
    ix,pred,succ=adjacency(nodes,edges);pairs=native['pairs'];features=native['edge_features']
    old={(ix[int(a)],ix[int(b)]) for a,b in edges}
    scores={tuple(p):float(s) for p,s in zip(pairs,logit(probability))}
    # Existing forks and high-confidence external links are fixed boundaries.
    fixed={x for x in old if len(succ[x[0]])==2}
    for k,(a,b) in enumerate(pairs):
        if (a,b) in old and features[k,0]>=.99 and features[k,3]<=3. and probability[k]>=.98:
            fixed.add((int(a),int(b)))
    fixed_sources={a for a,b in fixed};fixed_targets={b for a,b in fixed}
    by_frame=defaultdict(list)
    for k,(a,b) in enumerate(pairs):
        if a not in fixed_sources and b not in fixed_targets:
            by_frame[int(nodes[a,1])].append(k)
    selected=set(fixed)
    for t,ks in by_frame.items():
        src=sorted({int(pairs[k,0]) for k in ks});dst=sorted({int(pairs[k,1]) for k in ks})
        ai={a:i for i,a in enumerate(src)};bi={b:i for i,b in enumerate(dst)}
        cost=np.full((len(src),len(dst)+len(src)),1e6)
        for i,a in enumerate(src):
            # Termination requires compelling evidence against the existing link.
            cost[i,len(dst)+i]=0. if not succ[a] else 5.
        for k in ks:
            a,b=map(int,pairs[k]);s=scores[a,b]
            if (a,b) not in old:
                if probability[k]<.2:continue
                s-=margin
                # Avoid treating a confident birth as a continuation on weak cues.
                if not succ[a]:s-=2.
            cost[ai[a],bi[b]]=-s
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if j<len(dst) and cost[i,j]<1e5:selected.add((src[i],dst[j]))
    # Sources with no allowed alternatives retain their valid baseline links.
    modeled={int(pairs[k,0]) for ks in by_frame.values() for k in ks}|fixed_sources
    selected.update(x for x in old if x[0] not in modeled)
    return np.array(sorted((int(nodes[a,0]),int(nodes[b,0])) for a,b in selected),np.int64).reshape(-1,2)


def divisions(nodes,edges,hypotheses,probability,threshold=.05):
    """Joint local fork-vs-continuation choices, including competing incoming links.

    Connected components of interacting edited sources/targets are solved as a
    bounded binary program. Existing high-confidence external links are boundary
    conditions, not silently stolen. Infeasible/timeout components abstain.
    """
    ix,pred,succ=adjacency(nodes,edges);triples=hypotheses['triples'];features=hypotheses['fork_features']
    scores=logit(probability)-float(logit(threshold));candidates=[]
    old={(ix[int(a)],ix[int(b)]) for a,b in edges}
    for k,(i,a,b) in enumerate(triples):
        if scores[k]<=0:continue
        if len(succ[i])==2:continue  # Original forks remain fixed in the add-only arm.
        # One predicted daughter persistence frame is mandatory, not a GT check.
        if features[k,16]>=3:continue
        owned=[pred[x][0] for x in [a,b] if pred[x] and pred[x][0]!=i]
        # Strong external continuations do not become second daughters.
        if any(len(succ[o])==2 for o in owned):continue
        if any(features[k,25]>0 and features[k,25]<2.0 for _ in owned):continue
        value=float(scores[k])-.8*len(owned)
        if value>0:candidates.append((int(i),int(a),int(b),value))
    # Sources connected by shared targets or displaced incoming owners must be
    # reconciled together. A continuation option is always available.
    parent={}
    def root(a):
        parent.setdefault(a,a)
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[b]=a
    target_sources=defaultdict(list)
    for i,a,b,s in candidates:
        root(i)
        for c in [*succ[i],a,b]:
            target_sources[c].append(i)
            if pred[c]:union(i,pred[c][0])
    for sources in target_sources.values():
        for i in sources[1:]:union(sources[0],i)
    comps=defaultdict(list)
    for x in candidates:comps[root(x[0])].append(x)
    sources_by_comp=defaultdict(set)
    for a in list(parent):sources_by_comp[root(a)].add(a)
    selected=set(old);receipt=dict(proposals=len(candidates),components=len(comps),solved=0,abstained=0,divisions_added=0)
    for comp,choices in comps.items():
        src=sources_by_comp[comp]
        # Bound the solve to small ambiguous neighborhoods. Large components
        # abstain, preserving the original graph exactly.
        if len(src)>40 or len(choices)>160:
            receipt['abstained']+=1;continue
        options=[]
        for i in sorted(src):
            options.append((i,tuple(succ[i]),0.))
            # A displaced one-child continuation can terminate at a fixed cost.
            if len(succ[i])==1:options.append((i,(),-2.))
        options.extend((i,(a,b),s) for i,a,b,s in choices)
        targets=sorted({j for _,children,_ in options for j in children})
        rows={('s',i):k for k,i in enumerate(sorted(src))}
        rows.update({('t',j):len(src)+k for k,j in enumerate(targets)})
        rr=[];cc=[]
        for k,(i,children,_) in enumerate(options):
            rr.append(rows['s',i]);cc.append(k)
            for j in children:rr.append(rows['t',j]);cc.append(k)
        matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(rows),len(options))).tocsc()
        lower=np.r_[np.ones(len(src)),np.zeros(len(targets))]
        upper=np.ones(len(rows))
        # Any parent outside this component remains a fixed boundary.
        for j in targets:
            if pred[j] and pred[j][0] not in src:upper[rows['t',j]]=0.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            result=milp(c=-np.array([x[2] for x in options]),integrality=np.ones(len(options)),
                bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,lower,upper),
                options={'time_limit':2.,'mip_rel_gap':0.,'presolve':True,'threads':1})
        if result.x is None or not result.success:
            receipt['abstained']+=1;continue
        selected={x for x in selected if x[0] not in src}
        for k,(i,children,_) in enumerate(options):
            if result.x[k]>.5:
                selected.update((i,j) for j in children)
                if len(children)==2:receipt['divisions_added']+=1
        receipt['solved']+=1
    out=np.array(sorted((int(nodes[a,0]),int(nodes[b,0])) for a,b in selected),np.int64).reshape(-1,2)
    return out,receipt


def score_only_divisions(h):
    f=h['fork_features']
    # Prespecified lower division-penalty ablation, with soft temporal/image cues.
    log_odds=-5.+1.2*f[:,24]-.15*f[:,7]-.1*f[:,10]+.5*np.clip(f[:,13],-4,4)+.3*f[:,0]
    return 1/(1+np.exp(-np.clip(log_odds,-20,20)))
