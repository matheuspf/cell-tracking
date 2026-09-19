"""Complete fork edits from the inherited prediction-only neighbor/action bank.

No event feature emitted by the exposed historical adapter enters v11 models.
Missing clean edge measurements invalidate an action instead of inventing zeros.
"""
from itertools import combinations,product
import math
import numpy as np
from scipy.special import expit,logsumexp
from strong_tracker_v3.event_proposals import ProposalConfig,candidate_neighbors
from pipeline_error_training.actions import Alternatives,apply_decisions

PROPOSAL=ProposalConfig()


class Bank:
    def __init__(self,nodes,edges,edge_scores,frames):
        self.nodes,self.edges=nodes,edges
        self.config=PROPOSAL
        scores=np.asarray(edge_scores).copy().reshape(-1,3)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite clean edge evidence')
        # Serialized edge evidence always uses persisted IDs. All inherited
        # local action/adjacency routines use row indices internally.
        index={int(n[0]):i for i,n in enumerate(nodes)}
        for row in scores:
            row[:2]=[index[int(row[0])],index[int(row[1])]]
        self.logits={tuple(map(int,p[:2])):float(p[2]) for p in scores}
        native=dict(pairs=scores[:,:2].astype(np.int64),edge_features=np.column_stack([expit(scores[:,2]),np.zeros(len(scores))]))
        self.near,self.probability,(_,self.pred,self.succ),self.counts=candidate_neighbors(nodes,edges,native,[1.625,.40625,.40625],self.config)
        self.pos=nodes[:,2:]*np.array([1.625,.40625,.40625])
        from scipy.spatial import cKDTree
        self.density=np.zeros(len(nodes),np.float32)
        for t in np.unique(nodes[:,1]):
            members=np.flatnonzero(nodes[:,1]==t)
            self.density[members]=cKDTree(self.pos[members]).query_ball_point(self.pos[members],10.,return_length=True)-1
        self.frames=frames
        self.birth=[any(not self.pred[j] for j in ns) for ns in self.near]
        anchors=set()
        for i,ns in enumerate(self.near):
            # Anchor decisions require actual clean scores for every neighbor.
            if any((i,j) not in self.logits for j in ns):raise ValueError('Missing neighbor edge score')
            pp=sorted((self.probability[(i,j)] for j in ns),reverse=True)
            if len(self.succ[i])!=1 or self.birth[i] or (len(pp)>1 and pp[1]>=self.config.second_probability_trigger) or int(nodes[i,0])%self.config.uniform_modulus==0:
                anchors.add(i)
        self.expanded=set(anchors)
        for i in anchors:self.expanded.update(self.pred[i]);self.expanded.update(self.near[i][:2])
        self.paths=[list(ns[:self.config.paths_per_daughter]) or [-1] for ns in self.near]
        for i in range(len(nodes)):
            for j in self.succ[i]:
                if j not in self.paths[i]:self.paths[i]=([j]+self.paths[i])[:self.config.paths_per_daughter]
        self.alternatives=Alternatives(self)

    def events(self,parent):
        if parent not in self.expanded:return
        for a,b in combinations(sorted(set(self.near[parent])|set(self.succ[parent])),2):
            if max(np.linalg.norm(self.pos[d]-self.pos[parent]) for d in (a,b))>self.config.parent_gate_um:continue
            sister=np.linalg.norm(self.pos[a]-self.pos[b])
            if not 1<=sister<=self.config.sister_gate_um:continue
            for qa,qb in product(self.paths[a],self.paths[b]):
                if qa>=0 and qa==qb:continue
                yield parent,a,b,qa,qb

    def parent(self,parent):
        forks={};nofork={};missing=0;requests={}
        def retain(decision,destination):
            nonlocal missing
            if any(pair not in self.logits for pair in decision.add|decision.remove):
                missing+=1;return
            key=(int(self.nodes[parent,0]),tuple(sorted(decision.add)),tuple(sorted(decision.remove)))
            if key not in destination or decision.event<destination[key].event:destination[key]=decision
        for event in self.events(parent):
            p,a,b,qa,qb=event
            families=[('continuation_a',{p:(a,)}),('continuation_b',{p:(b,)})]
            for d,other in ((a,b),(b,a)):
                if self.pred[other] and self.pred[other][0]!=p:
                    families.append(('two_parents',{p:(d,),self.pred[other][0]:(other,)}))
            for kind,desired in families:
                key=(kind,tuple(sorted(desired.items())))
                if key not in requests or event<requests[key][0]:requests[key]=(event,desired,kind)
            desired={p:(a,b)}
            if qa>=0:desired[a]=(qa,)
            if qb>=0:desired[b]=(qb,)
            for decision in self.alternatives.complete(event,desired,'division'):
                retain(decision,forks)
                if len(forks)>4096:
                    return dict(parent=parent,forks=[],nofork=[],complete=False,reason='parent_unique_action_cap',
                                unique_forks_lower_bound=4097,missing_scores=missing)
        # No-fork ownership edits do not depend on the two daughter-path choices.
        # Enumerate each desired ownership map once, preserving its lexicographic
        # canonical event representative and the original complete-edit closure.
        for event,desired,kind in requests.values():
            for decision in self.alternatives.complete(event,desired,kind):retain(decision,nofork)
        return dict(parent=parent,forks=[forks[k] for k in sorted(forks)],nofork=[nofork[k] for k in sorted(nofork)],
                    complete=missing==0,reason='missing_score' if missing else None,missing_scores=missing)

    def structural(self,decision):
        added=sum(self.logits[pair] for pair in decision.add)
        removed=sum(self.logits[pair] for pair in decision.remove)
        births=sum(self.nodes[n,1]>0 for n in decision.births)
        terminations=sum(self.nodes[n,1]<self.frames-1 for n in decision.terminations)
        return added-removed-2*births-2*terminations


def utilities(bank,group,occurrence,conditional,margin):
    if not group['complete']:return np.empty(0,np.float64)
    if len(conditional)!=len(group['forks']):raise ValueError('Conditional denominator does not cover every unique legal fork')
    if not len(conditional):return np.empty(0,np.float64)
    if not np.isfinite(conditional).all() or not math.isfinite(occurrence):raise ValueError('Nonfinite learned utility')
    competitor=max([0.,*[bank.structural(d) for d in group['nofork']]])
    return occurrence+np.asarray(conditional)-logsumexp(conditional)+np.array([bank.structural(d) for d in group['forks']])-competitor-margin
