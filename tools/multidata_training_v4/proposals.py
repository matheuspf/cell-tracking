"""Label-blind, bounded daughter observations. No adjacency or GT arguments."""
import itertools
import numpy as np
from scipy.spatial import cKDTree

PAIRS=np.array(list(itertools.combinations(range(6),2)),np.int64)

def build(t,points,anchors=None):
    t=np.asarray(t);points=np.asarray(points,np.float32)
    if not len(t):return np.empty(0,np.int64),np.empty((0,6),np.int64),np.empty((0,6,16),np.float32)
    anchors=np.flatnonzero(t<t.max()) if anchors is None else np.asarray(anchors)
    # Per-axis normalization is mandatory for uncalibrated Zoo coordinates.
    extent=np.maximum(np.quantile(points,.75,axis=0)-np.quantile(points,.25,axis=0),1e-5)
    q=points/extent
    frames={int(f):np.flatnonzero(t==f) for f in np.unique(t)}
    trees={f:cKDTree(q[ids]) for f,ids in frames.items()}
    cand=np.full((len(anchors),6),-1,np.int64);x=np.zeros((len(anchors),6,16),np.float32)
    for f in np.unique(t[anchors]):
        rows=np.flatnonzero(t[anchors]==f);a=anchors[rows];nxt=frames.get(int(f)+1)
        if nxt is None or len(nxt)==0:continue
        same=frames[int(f)];tree=trees[int(f)]
        ds,_=tree.query(q[same],k=min(2,len(same)))
        scale=max(float(np.median(ds[:,-1])) if ds.ndim==2 else 1.,1e-5)
        d,j=trees[int(f)+1].query(q[a],k=min(6,len(nxt)))
        if d.ndim==1:d=d[:,None];j=j[:,None]
        m=d.shape[1];c=nxt[j];cand[rows,:m]=c
        delta=(q[c]-q[a,None])/scale
        hist=np.zeros((len(a),3),np.float32);hv=int(f)-1 in trees
        if hv:
            _,back=trees[int(f)-1].query(q[a]);hist=(q[a]-q[frames[int(f)-1][back]])/scale
        fut=np.zeros_like(delta);fv=int(f)+2 in trees
        if fv:
            _,ff=trees[int(f)+2].query(q[c]);fut=(q[frames[int(f)+2][ff]]-q[c])/scale
        x[rows,:m,:3]=delta;x[rows,:m,3:6]=hist[:,None];x[rows,:m,6:9]=fut
        x[rows,:m,9]=d/scale;x[rows,:m,10]=np.arange(m)/5
        x[rows,:m,11]=1;x[rows,:m,12]=hv;x[rows,:m,13]=fv
        # Relative neighborhood competition, no source physical units or topology.
        x[rows,:m,14]=np.linalg.norm(delta-hist[:,None],axis=-1)
        x[rows,:m,15]=np.linalg.norm(fut-delta,axis=-1)
    return anchors,cand,np.clip(x,-8,8)

def pair_mask(x):return (x[:,PAIRS[:,0],11]>0)&(x[:,PAIRS[:,1],11]>0)
