"""Deterministic policies using only predicted graphs and scores."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def tracklets(nodes,edges):
    ids=nodes[:,0]
    index={int(v):i for i,v in enumerate(ids)}
    pred=[[] for _ in ids]; succ=[[] for _ in ids]
    for a,b in edges:
        i,j=index[int(a)],index[int(b)]
        succ[i].append(j); pred[j].append(i)
    units=np.full(len(ids),-1,np.int64)
    u=0
    for i in range(len(ids)):
        if units[i]>=0:
            continue
        cur=i
        while len(pred[cur])==1 and len(succ[pred[cur][0]])==1 and units[pred[cur][0]]<0:
            cur=pred[cur][0]
        while units[cur]<0:
            units[cur]=u
            if len(succ[cur])!=1 or len(pred[succ[cur][0]])!=1:
                break
            cur=succ[cur][0]
        u+=1
    return units,pred,succ


def keep_mask(nodes,edges,scores,fraction,policy,seed=20260908,quantile=.9):
    if not 0<=fraction<=1:
        raise ValueError('Invalid keep fraction')
    n=len(nodes)
    if fraction==1 or policy=='identity':
        return np.ones(n,bool)
    if fraction==0:
        return np.zeros(n,bool)
    score=np.asarray(scores,dtype=float)
    if len(score)!=n or not np.isfinite(score).all():
        raise ValueError('Missing/nonfinite scores')
    rng=np.random.default_rng(seed)
    tie=rng.random(n)
    if policy.startswith('random'):
        score=rng.random(n)
    budget=math.ceil(fraction*n)
    keep=np.zeros(n,bool)
    if 'tracklet' not in policy:
        order=np.lexsort((tie,-score))
        keep[order[:budget]]=True
        return keep
    units,pred,succ=tracklets(nodes,edges)
    groups=[np.flatnonzero(units==u) for u in np.unique(units)]
    scores_u=np.array([np.quantile(score[g],quantile) for g in groups])
    if policy=='random_tracklets':
        scores_u=rng.random(len(groups))
    order=np.lexsort((rng.random(len(groups)),-scores_u))
    total=0
    for u in order:
        # Include the unit crossing the requested budget; report overshoot.
        if total>=budget: break
        keep[groups[u]]=True; total+=len(groups[u])
    if 'fork_protected' in policy:
        # Protect the 2-hop context of predicted forks touched by selected units.
        for i,children in enumerate(succ):
            if len(children)<2: continue
            context={i,*pred[i],*children}
            context.update(j for c in children for j in succ[c])
            if any(keep[j] for j in context):
                keep[list(context)]=True
    return keep


def filtered(nodes,edges,keep):
    if np.asarray(keep).dtype!=bool or len(keep)!=len(nodes):
        raise ValueError('Invalid node mask')
    ids=nodes[keep,0]
    mask=np.isin(edges[:,0],ids)&np.isin(edges[:,1],ids)
    return nodes[keep].copy(),edges[mask].copy()


class PreparedFilter:
    """Reuse label-free topology and rankings across the preregistered budgets."""
    def __init__(self,nodes,edges):
        self.nodes=nodes;self.edges=edges;self.n=len(nodes)
        self.units,self.pred,self.succ=tracklets(nodes,edges)
        order=np.argsort(self.units,kind='stable')
        self.groups=np.split(order,np.flatnonzero(np.diff(self.units[order]))+1) if self.n else []
        self.cost=np.array([len(g) for g in self.groups])
        self.rankings={}

    def mask(self,scores,fraction,policy,seed=20260908,key='default'):
        if fraction==1 or policy=='identity':return np.ones(self.n,bool)
        if fraction==0:return np.zeros(self.n,bool)
        ranking_key=(key,policy.replace('_fork_protected',''),seed)
        if ranking_key not in self.rankings:
            rng=np.random.default_rng(seed);tie=rng.random(self.n)
            score=rng.random(self.n) if policy.startswith('random') else np.asarray(scores)
            if not np.isfinite(score).all():raise ValueError('Nonfinite filter scores')
            if 'tracklet' in policy:
                su=pd.Series(score).groupby(self.units,sort=True).quantile(.9).to_numpy()
                if policy=='random_tracklets':su=rng.random(len(self.groups))
                order=np.lexsort((rng.random(len(self.groups)),-su))
                costs=np.cumsum(self.cost[order])
            else:
                order=np.lexsort((tie,-score));costs=None
            self.rankings[ranking_key]=(order,costs)
        order,costs=self.rankings[ranking_key]
        budget=math.ceil(fraction*self.n);keep=np.zeros(self.n,bool)
        if 'tracklet' in policy:
            k=np.searchsorted(costs,budget,side='left')+1
            for u in order[:k]:keep[self.groups[u]]=True
        else:keep[order[:budget]]=True
        if 'fork_protected' in policy:
            for i,children in enumerate(self.succ):
                if len(children)<2:continue
                context={i,*self.pred[i],*children}
                context.update(j for c in children for j in self.succ[c])
                if any(keep[j] for j in context):keep[list(context)]=True
        return keep

    def exact_tracklet_cost(self,scores,target,seed=20260908,random=False):
        """Lexicographically best ranked whole-tracklet subset of exact node cost.

        The primary prefix mask proves that its realized cost is reachable. A
        suffix bitset supplies an exact feasibility check when matching controls
        to that cost; no annotation labels or GT structure are used.
        """
        if not isinstance(target,(int,np.integer)) or not 0<=target<=self.n:
            raise ValueError('Invalid exact node budget')
        if target==0:return np.zeros(self.n,bool)
        if target==self.n:return np.ones(self.n,bool)
        rng=np.random.default_rng(seed);rng.random(self.n)
        q=pd.Series(scores).groupby(self.units,sort=True).quantile(.9).to_numpy()
        if random:q=rng.random(len(self.groups))
        order=np.lexsort((rng.random(len(self.groups)),-q))
        cap=(1<<(target+1))-1;suffix=[0]*(len(order)+1);suffix[-1]=1
        for i in range(len(order)-1,-1,-1):
            suffix[i]=(suffix[i+1]|(suffix[i+1]<<int(self.cost[order[i]])))&cap
        if not ((suffix[0]>>target)&1):raise ValueError('Exact coherent budget is not reachable')
        keep=np.zeros(self.n,bool);remaining=target
        for i,u in enumerate(order):
            cost=int(self.cost[u])
            if cost<=remaining and ((suffix[i+1]>>(remaining-cost))&1):
                keep[self.groups[u]]=True;remaining-=cost
        if remaining or keep.sum()!=target:raise AssertionError('Exact budget construction failed')
        return keep
