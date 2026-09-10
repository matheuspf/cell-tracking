"""Frozen observation corruption; clean biological topology stays a target only."""
import numpy as np

def observations(t,points,seed):
    rng=np.random.default_rng(seed)
    scale=np.maximum(np.quantile(points,.75,axis=0)-np.quantile(points,.25,axis=0),1e-5)
    keep=rng.random(len(t))>.06
    ids=np.flatnonzero(keep);p=points[keep].copy()+rng.normal(0,.002, (len(ids),3))*scale
    dup=ids[rng.random(len(ids))<.02]
    # Duplicate observations do not inherit a second truth identity.
    return np.r_[t[keep],t[dup]],np.vstack([p,points[dup]+rng.normal(0,.004,(len(dup),3))*scale]),np.r_[ids,np.full(len(dup),-1)]
