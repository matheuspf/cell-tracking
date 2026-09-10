"""Rolling five-frame binary explanation objective with shared committed boundaries."""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .event_paths import unique_edges,equivalence_classes
from .event_index import WindowEvents

DEFAULT=dict(birth_cost=2.,termination_cost=1.,split_cost=4.,incumbent_edge_bonus=.75,
             new_node_cost=2.,context_frames=5,time_limit=2.,mip_rel_gap=.001,
             spatial_shape=[64,256,256],spacing_um=[1.625,.40625,.40625],spatial_border_um=3.25)

def protected_context(edges):
    parents={};children={}
    for a,b in edges:children.setdefault(int(a),[]).append(int(b));parents[int(b)]=int(a)
    protected=set();nodes=set()
    for a,cc in children.items():
        if len(cc)<2:continue
        # Full scorer topology: predecessor, mother, daughters and grandchildren.
        if a in parents:protected.add((parents[a],a))
        for b in cc:
            protected.add((a,b));protected.update((b,c) for c in children.get(b,[]))
    for a,b in protected:nodes.update([a,b])
    return protected,nodes

def solve_window(nodes,pairs,scores,old_ids,incumbent,confidence,groups,first_selected=None,first_incoming=None,
                 global_last=None,protected=(),config=None,event_classes=None,committed_events=()):
    cfg={**DEFAULT,**(config or {})};n=len(nodes);m=len(pairs);ids={int(r[0]):i for i,r in enumerate(nodes)}
    if not n:return set(),set(),dict(status='empty',seconds=0.)
    times=nodes[:,1].astype(int);t0=int(times.min());t1=int(times.max());global_last=t1 if global_last is None else global_last
    # Variables [y nodes, e edges, b births, d terminations, s splits].
    pair_index={tuple(map(int,p)):j for j,p in enumerate(pairs)}
    event_choices=[];event_variables={}
    for identity,alternatives in (event_classes or {}).items():
        present=[(pair_index[a],pair_index[b]) for a,b in alternatives if a in pair_index and b in pair_index]
        if len(present)<2 and identity not in committed_events:continue
        for a,b in present:
            event_variables.setdefault(identity,[]).append(4*n+m+len(event_choices));event_choices.append((a,b))
    y=0;e=n;b=n+m;d=2*n+m;s=3*n+m;size=4*n+m+len(event_choices)
    cost=np.zeros(size);lb=np.zeros(size);ub=np.ones(size)
    known=np.array([int(r[0]) in old_ids for r in nodes]);cost[:n]=cfg['new_node_cost']-np.log(np.clip(confidence,1e-4,1-1e-4)/np.clip(1-confidence,1e-4,1.))
    cost[:n][known]=0
    cost[e:e+m]=-np.nan_to_num(scores,nan=-40)-np.array([cfg['incumbent_edge_bonus'] if tuple(p) in incumbent else 0 for p in pairs])
    # Boundary births/terminations are cheap because the required context is not observed.
    distance_to_border=np.minimum(nodes[:,2:],np.asarray(cfg['spatial_shape'])-1-nodes[:,2:])*cfg['spacing_um']
    spatial_boundary=distance_to_border.min(1)<cfg['spatial_border_um']
    cost[b:b+n]=np.where(times==0,.1,np.where(spatial_boundary,.25,cfg['birth_cost']))
    cost[d:d+n]=np.where(times==global_last,.1,np.where(spatial_boundary,.25,cfg['termination_cost']))
    cost[s:s+n]=cfg['split_cost']
    incompatible=set()
    for original,alternatives in groups:
        incompatible.add(original)
    for i,node in enumerate(nodes[:,0].astype(int)):
        if known[i] and node not in incompatible:lb[i]=ub[i]=1
        if times[i]==t0 and first_selected is not None:lb[i]=ub[i]=int(node in first_selected)
    rows=[];cols=[];values=[];lower=[];upper=[]
    def constraint(terms,low,high):
        ri=len(lower)
        for col,value in terms:rows.append(ri);cols.append(col);values.append(value)
        lower.append(low);upper.append(high)
    incoming=[[] for _ in range(n)];outgoing=[[] for _ in range(n)]
    for j,(aa,bb) in enumerate(pairs):
        ai,bi=ids[int(aa)],ids[int(bb)];assert times[bi]==times[ai]+1
        incoming[bi].append(e+j);outgoing[ai].append(e+j)
        if tuple(map(int,(aa,bb))) in protected:lb[e+j]=ub[e+j]=1
    for i,node in enumerate(nodes[:,0].astype(int)):
        past=first_incoming.get(node,0) if times[i]==t0 and first_incoming else 0
        constraint([(j,1) for j in incoming[i]]+[(b+i,1),(i,-1)],-past,-past)
        if times[i]<t1:
            constraint([(j,1) for j in outgoing[i]]+[(d+i,1),(i,-1),(s+i,-1)],0,0)
        else:
            # The far halo boundary has missing future context, not a death label.
            ub[s+i]=0;cost[d+i]=0
        constraint([(s+i,1),(i,-1)],-np.inf,0)
        constraint([(d+i,1),(s+i,1),(i,-1)],-np.inf,0)
        if not known[i] and t0<times[i]<t1:
            constraint([(j,1) for j in incoming[i]+outgoing[i]]+[(i,-2)],0,np.inf)
    for original,alternatives in groups:
        if original not in ids:continue
        represented=[v for v in alternatives if v in ids]
        if len(represented)!=2:continue
        for v in represented:constraint([(ids[original],1),(ids[v],1)],-np.inf,1)
        constraint([(ids[represented[0]],1),(ids[represented[1]],-1)],0,0)
    for j,(a,b) in enumerate(event_choices):
        q=4*n+m+j
        constraint([(q,1),(e+a,-1)],-np.inf,0)
        constraint([(q,1),(e+b,-1)],-np.inf,0)
        constraint([(q,1),(e+a,-1),(e+b,-1)],-1,np.inf)
    for identity,variables in event_variables.items():
        constraint([(q,1) for q in variables],-np.inf,0 if identity in committed_events else 1)
    matrix=coo_matrix((values,(rows,cols)),shape=(len(lower),size)).tocsc()
    start=time.monotonic()
    result=milp(cost,integrality=np.ones(size),bounds=Bounds(lb,ub),constraints=LinearConstraint(matrix,lower,upper),
        options=dict(time_limit=cfg['time_limit'],mip_rel_gap=cfg['mip_rel_gap']))
    if result.x is None:return None,None,dict(status=int(result.status),seconds=time.monotonic()-start,variables=size,constraints=len(lower),fallback=True)
    selected=set(nodes[result.x[:n]>.5,0].astype(int));chosen={tuple(map(int,p)) for p in pairs[result.x[e:e+m]>.5]}
    return selected,chosen,dict(status=int(result.status),seconds=time.monotonic()-start,variables=size,constraints=len(lower),
        objective=float(result.fun),fallback=False,optimal=result.status==0)

def decode(nodes,pairs,scores,base_nodes,base_edges,confidence=None,split_owner=None,config=None,no_new_evidence=False):
    if no_new_evidence:return base_nodes.copy(),base_edges.copy(),dict(identity_no_evidence=True,changed_edges=0)
    cfg={**DEFAULT,**(config or {})};old_ids=set(base_nodes[:,0].astype(int));incumbent=set(map(tuple,base_edges));lookup={int(n[0]):i for i,n in enumerate(nodes)}
    pairs,scores=unique_edges(pairs,scores)
    confidence=np.ones(len(nodes)) if confidence is None else confidence
    protected,protected_nodes=protected_context(base_edges);groups=[]
    missing_features=int((~np.isfinite(scores)).sum())
    missing_incumbent={tuple(map(int,p)) for p,v in zip(pairs,scores) if not np.isfinite(v) and tuple(p) in incumbent}
    protected.update(missing_incumbent)
    for a,b in missing_incumbent:protected_nodes.update([a,b])
    if split_owner is not None:
        for owner in sorted(set(split_owner)-{-1}):
            group=np.flatnonzero(split_owner==owner)
            if owner in protected_nodes or len(group)<2:continue
            selected=sorted(group,key=lambda i:(-confidence[i],int(nodes[i,0])))[:2]
            groups.append((int(owner),list(nodes[selected,0].astype(int))))
    tpair=nodes[[lookup[int(p[0])] for p in pairs],1].astype(int);alltimes=nodes[:,1].astype(int);last=int(alltimes.max())
    # An edge below this bound cannot pay even avoided birth/death/split costs. Retain C0 regardless.
    enabled=np.isfinite(scores)&(scores>-(cfg['birth_cost']+cfg['termination_cost']+cfg['incumbent_edge_bonus']+4))
    enabled |= np.array([tuple(p) in incumbent for p in pairs])
    pairs=pairs[enabled];scores=scores[enabled];tpair=tpair[enabled]
    event_classes=equivalence_classes(nodes,pairs,scores,base_edges,old_ids)
    window_events=WindowEvents(event_classes,{int(n[0]):int(n[1]) for n in nodes})
    event_lookup={alternative:identity for identity,alternatives in event_classes.items() for alternative in alternatives}
    committed_events=set()
    chosen_all=set();selected_all=set();first_selected=None;first_incoming={};receipts=[]
    for t in range(last):
        end=min(last,t+cfg['context_frames']-1);ni=np.flatnonzero((alltimes>=t)&(alltimes<=end));pi=np.flatnonzero((tpair>=t)&(tpair<end))
        wn=nodes[ni];wp=pairs[pi];ws=scores[pi]
        sel,edges,receipt=solve_window(wn,wp,ws,old_ids,incumbent,confidence[ni],groups,
            first_selected,first_incoming,last,protected,cfg,window_events.window(t,end),committed_events)
        if sel is None:
            # Shared boundary may differ from C0. Keep committed IDs, then choose a legal C0 continuation.
            sel=(old_ids&set(wn[:,0].astype(int)))|(first_selected or set());edges=set()
            owners=set();degree={}
            for a,b in sorted(incumbent):
                if a not in sel or b not in sel or a not in lookup or nodes[lookup[a],1]!=t:continue
                if first_selected is not None and a not in first_selected:continue
                if b in owners or degree.get(a,0)>=2:continue
                edges.add((a,b));owners.add(b);degree[a]=degree.get(a,0)+1
        core_edges={(a,b) for a,b in edges if nodes[lookup[a],1]==t}
        by_parent={}
        for a,b in core_edges:by_parent.setdefault(a,[]).append(b)
        for a,bs in by_parent.items():
            if len(bs)==2:
                identity=event_lookup.get(tuple((a,b) for b in sorted(bs)))
                if identity is not None:committed_events.add(identity)
        first_selected={int(n[0]) for n in wn if n[1]==t+1 and int(n[0]) in sel}
        first_incoming={b:1 for a,b in core_edges};chosen_all.update(core_edges)
        selected_all.update(int(n[0]) for n in wn if n[1] in [t,t+1] and int(n[0]) in sel)
        receipts.append(dict(t=t,**receipt))
    n=nodes[np.array([int(r[0]) in selected_all for r in nodes])];e=np.array(sorted(chosen_all),np.int64).reshape(-1,2)
    return n,e,dict(windows=receipts,fallback_windows=sum(r.get('fallback',False) for r in receipts),
        changed_edges=len(chosen_all.symmetric_difference(incumbent)),added_nodes=len(selected_all-old_ids),
        removed_nodes=len(old_ids-selected_all),protected_edges=len(protected),exclusive_split_groups=len(groups),
        event_equivalence_classes=len(event_classes),committed_event_classes=len(committed_events),event_reduction='max compatible complete explanation; duplicate edge representations deduplicated by max',
        model_feature_missing=missing_features,missing_feature_C0_edges_preserved=len(missing_incumbent),config=cfg)
