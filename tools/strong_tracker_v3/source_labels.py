"""Training-only division labels using the pinned official local matcher.

No inference module imports this file. Coordinates and proposal selection are
fixed before matching. Missing sparse evidence stays unknown.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .common import adjacency


class LocalTopology:
    """Minimal adjacency protocol consumed by the unmodified official rule."""
    def __init__(self,pred,succ,parent,a,b,qa,qb):
        self.pred=pred;self.succ=succ;self.i=parent
        self.over={parent:[a,b]}
        self.incoming={a:[parent],b:[parent]}
        for d,q in [(a,qa),(b,qb)]:
            if q>=0:self.over[d]=[q];self.incoming[q]=[d]
    def predecessors(self,i):return self.incoming.get(i,self.pred[i])
    def successors(self,i):return self.over.get(i,self.succ[i])


def labels(nodes,edges,events,gt_nodes,gt_edges,scale,seed=20260909):
    # Deliberately deferred training-only imports; module import reads no files.
    from annotation_selection.metric_adapter import make_graph
    from tracking_cellmot import division_metrics as dm
    _,pred,succ=adjacency(nodes,edges)
    g,reverse=make_graph(nodes,edges);gt,greverse=make_graph(gt_nodes,gt_edges)
    # graph IDs from make_graph enumerate the array rows; assert rather than guess.
    assert all(reverse[i]==int(nodes[i,0]) for i in range(len(nodes)))
    full=dm._match_full(g,gt,tuple(scale),7.)
    mapping=dict(dm._matched_node_attrs(full).iter_rows())
    components=dm._gt_weak_component_ids(gt)
    windows=dm.extract_divisions(gt)
    matched=dm.match_divisions(g,gt,tuple(scale),7.)
    role_windows=[]
    for event_id,window in windows.items():
        roles=dm._matched_division_nodes(dm._matched_node_attrs(matched[event_id]),window,event_id)
        if roles is None:role_windows.append((event_id,None));continue
        role_windows.append((event_id,roles))
    n=len(events);y=np.full(n,-1,np.int8);bag=np.full(n,-1,np.int64)
    reason=np.zeros(n,np.uint8) # 1 positive, 2 quiet observed, 3 contradictory pair
    by_parent=defaultdict(list)
    for event_id,roles in role_windows:
        if roles is None:continue
        ps,ds=roles
        potential=ps|{j for p in ps for j in succ[p]}
        # Parent-side matching can extend through a candidate predecessor only
        # when that edge is explicitly present in the current event action.
        for i in potential:by_parent[i].append((event_id,ps,ds))
    def component(d,q):
        if d in mapping:return components[mapping[d]]
        if q>=0 and q in mapping:return components[mapping[q]]
        ss=succ[d] if q<0 else [q]
        found={components[mapping[s]] for s in ss if s in mapping}
        return next(iter(found)) if len(found)==1 else None
    for k,(i,a,b,qa,qb) in enumerate(events):
        i,a,b,qa,qb=map(int,(i,a,b,qa,qb))
        ca,cb=component(a,qa),component(b,qb)
        cross=ca is not None and cb is not None and ca!=cb
        valid=[];considered=False
        for event_id,ps,ds in by_parent.get(i,[]):
            topology=LocalTopology(pred,succ,i,a,b,qa,qb)
            if not ({i,*topology.predecessors(i)}&ps):continue
            considered=True
            if not cross and dm._is_strongly_connected_division(topology,i,ps,ds):valid.append(event_id)
        if valid:
            y[k]=1;bag[k]=int(min(valid));reason[k]=1;continue
        # Different reliable GT components directly refute the proposed pair.
        if cross:y[k]=0;reason[k]=3;bag[k]=-2-i;continue
        if i not in mapping:continue
        gi=mapping[i]
        # Fully observed continuation on each side, with no local fork, is a
        # supported nondivision. End-of-annotation contexts are ignored.
        quiet=True
        for direction in [gt.predecessors,gt.successors]:
            cur=gi
            for _ in range(2):
                ns=direction(cur)
                if len(ns)!=1 or len(gt.successors(cur))>1:quiet=False;break
                cur=ns[0]
        if quiet:y[k]=0;reason[k]=2;bag[k]=-2-i
        elif considered and a in mapping and b in mapping and len(gt.successors(gi))>=1:
            # Near-event wrong pairs need both direct branches observed. A lone
            # unmatched daughter is not fabricated negative supervision.
            y[k]=0;reason[k]=3;bag[k]=-2-i
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    rng=np.random.default_rng(seed)
    sampled=[];sampling=[]
    groups=defaultdict(list)
    for k in negative:groups[int(bag[k])].append(int(k))
    for members in groups.values():
        # Uniformly sampled nondivision groups, plus contradictory pair examples.
        hard=[k for k in members if reason[k]==3]
        ordinary=[k for k in members if reason[k]!=3]
        chosen=list(rng.choice(hard,min(4,len(hard)),replace=False))+list(rng.choice(ordinary,min(4,len(ordinary)),replace=False))
        sampled.extend(chosen);sampling.extend([len(chosen)/len(members)]*len(chosen))
    index=np.r_[positive,np.asarray(sampled,np.int64)]
    counts=Counter(bag[index]);weight=np.array([1./counts[b] for b in bag[index]],np.float32)
    coverage=[dict(event_id=int(greverse[e]),covered=bool(np.any(bag[positive]==e)),positive_alternatives=int(np.sum(bag[positive]==e)),
                   local_roles_available=r is not None) for e,r in role_windows]
    receipt=dict(proposals=n,positive_alternatives=len(positive),negative_population=len(negative),unknown=int(np.sum(y<0)),
                 sampled_negative=len(sampled),positive_event_groups=len(set(bag[positive])),negative_groups=len(groups),
                 coverage=coverage,reason_counts=dict(Counter(map(int,reason))),
                 labels='official local matching + directed branch rule + global direct-child component precedence; sparse incomplete contexts masked',
                 positive_weight='one total weight per event observation; crops do not create events',
                 provenance='source embryo only during fitting; reused embryo observations are not independent biological events')
    return dict(index=index,y=y[index],bag=bag[index],weight=weight,inclusion=np.r_[np.ones(len(positive)),sampling].astype(np.float32),
                all_y=y,all_bag=bag),receipt
