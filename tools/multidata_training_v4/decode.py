"""Explicit event gain repairs fork dominance without a blanket penalty change."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def apply(nodes,edges,events,gains,enabled=True):
    if not enabled or len(events)==0:return edges.copy(),dict(proposed=len(events),selected=0)
    ix={int(n):i for i,n in enumerate(nodes[:,0])};pred={};succ={i:[] for i in range(len(nodes))}
    for aa,bb in edges:
        a,b=ix[int(aa)],ix[int(bb)];pred[b]=a;succ[a].append(b)
    protected=set()
    for p,ch in succ.items():
        if len(ch)!=2:continue
        protected.add(p);front=[p]
        for _ in range(2):
            front=[j for i in front for j in succ[i]];protected.update(front)
        q=p
        for _ in range(2):
            if q in pred:q=pred[q];protected.add(q)
    allowed=[];touch=[];reject=0
    for j,(p,a,b) in enumerate(events):
        p,a,b=map(int,(p,a,b));owners={pred[d] for d in [a,b] if d in pred}
        touched={p,a,b,*owners,*succ[p],*succ[a],*succ[b]}
        if gains[j]<=0 or touched&protected or len(succ[p])>1:reject+=1;continue
        if nodes[a,1]!=nodes[p,1]+1 or nodes[b,1]!=nodes[p,1]+1 or a==b:raise ValueError('Invalid fork')
        allowed.append(j);touch.append(touched)
    if not allowed:return edges.copy(),dict(proposed=len(events),selected=0,protected_or_nonpositive=reject)
    used=sorted(set.union(*touch));rows={n:i for i,n in enumerate(used)}
    matrix=lil_matrix((len(used),len(allowed)),dtype=float)
    for j,touched in enumerate(touch):
        for n in touched:matrix[rows[n],j]=1
    sol=milp(-np.asarray(gains)[allowed],integrality=np.ones(len(allowed)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix.tocsr(),0,1),options={'time_limit':30.,'mip_rel_gap':0.})
    if sol.x is None:raise RuntimeError('Event solver failed: '+sol.message)
    picked=[allowed[i] for i in np.flatnonzero(sol.x>.5)]
    chosen=set(map(tuple,edges.tolist()));removed=0
    for j in picked:
        p,a,b=map(int,events[j])
        old={(int(nodes[s,0]),int(nodes[d,0])) for s,d in [(p,k) for k in succ[p]]+[(pred[k],k) for k in [a,b] if k in pred]}
        removed+=len(old&chosen);chosen-=old
        chosen|={(int(nodes[p,0]),int(nodes[a,0])),(int(nodes[p,0]),int(nodes[b,0]))}
    return np.array(sorted(chosen),np.int64).reshape(-1,2),dict(proposed=len(events),selected=len(picked),removed_edges=removed,
        protected_or_nonpositive=reject,solver_optimal=bool(sol.success),solver_gap=float(sol.mip_gap))

def solver_fixtures():
    import contextlib,io
    import polars as pl
    import tracksdata as td
    result=[]
    for gain in [0.,2.,-2.]:
        g=td.graph.InMemoryGraph();g.add_edge_attr_key('prob',pl.Float64,0.);g.add_node_attr_key('gain',pl.Float64,0.)
        n=[dict(t=0,gain=0.),dict(t=1,gain=gain)];e=[(0,1)]
        for branch in range(2):
            prev=1
            for t in range(2,10):
                i=len(n);n.append(dict(t=t,gain=0.));e.append((prev,i));prev=i
        ids=g.bulk_add_nodes(n);g.bulk_add_edges([dict(source_id=ids[a],target_id=ids[b],prob=.99) for a,b in e])
        solver=td.solvers.ILPSolver(edge_weight=-td.EdgeAttr('prob'),appearance_weight=0.,disappearance_weight=2.,
            division_weight=1.2-td.NodeAttr('gain'))
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):out=solver.solve(g)
        _,degree=np.unique(out.edge_attrs()['source_id'].to_numpy(),return_counts=True)
        result.append(dict(learned_gain=gain,forks=int((degree==2).sum()),edges=out.num_edges()))
    assert [r['forks'] for r in result]==[0,1,0]
    nodes=np.array([[0,0,0,0,0],[1,1,1,0,0],[2,1,0,1,0]])
    edges=np.array([[0,1]]);events=np.array([[0,1,2]])
    assert np.array_equal(apply(nodes,edges,events,[2],enabled=False)[0],edges)
    assert len(apply(nodes,edges,events,[2])[0])==2
    assert np.array_equal(apply(nodes,edges,events,[-2])[0],edges)
    return dict(raw_actual_solver=result,event_solver_positive=True,event_solver_negative=True,zero_head_identity=True)
