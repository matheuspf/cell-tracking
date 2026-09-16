"""Bidirectional physical candidate bank retaining every incumbent edge."""
import numpy as np
from scipy.spatial import cKDTree
from .common import *

def motion_config():
    p=OUT/'motion_config.json'
    if p.exists():return read(p)
    out={}
    for source in ['44b6','6bba']:
        distances=[]
        for r in inventory():
            if r['embryo']!=source:continue
            g=graph(r['dataset']);lookup={int(n[0]):n[2:] for n in g['nodes']}
            distances.extend(np.linalg.norm((lookup[int(a)]-lookup[int(b)])*r['physical_scale']) for a,b in g['edges'])
        q=float(np.quantile(distances,.995));out[source]=dict(distance_um=max(10.,2*q),quantile995_um=q,
            estimator='source C0 predicted displacement; no matching tolerance or target labels',neighbors=8)
    write(p,out);return out

def bank(nodes,edges,max_distance=10.,k=8):
    scale=np.array([1.625,.40625,.40625]);idx={int(n[0]):i for i,n in enumerate(nodes)}
    pairs={tuple(map(int,e)) for e in edges};counts=[]
    for t in sorted(set(nodes[:,1])):
        a=nodes[nodes[:,1]==t];b=nodes[nodes[:,1]==t+1]
        if not len(a) or not len(b):continue
        for src,tgt,reverse in [(a,b,False),(b,a,True)]:
            d,j=cKDTree(tgt[:,2:]*scale).query(src[:,2:]*scale,k=min(k,len(tgt)),distance_upper_bound=max_distance)
            d=np.asarray(d).reshape(len(src),-1);j=np.asarray(j).reshape(len(src),-1)
            for i in range(len(src)):
                for distance,other in zip(d[i],j[i]):
                    if np.isfinite(distance):
                        p=(int(src[i,0]),int(tgt[other,0]));pairs.add(p[::-1] if reverse else p)
    return np.asarray(sorted(pairs),np.int64).reshape(-1,2)
