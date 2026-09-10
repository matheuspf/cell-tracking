"""Explicit fresh evidence adapters; no annotation or legacy-global mutation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree

from .common import adjacency,save_arrays,save_graph


def native_v2(nodes,edges,node_features,raw_nodes,pre_ilp,scale):
    """Pure extraction of v2 native.build, retaining its exact teacher semantics.

    This compatibility function is only for recreating frozen v2 E teachers.
    The v3 selector itself always rebuilds its own incumbent-native features.
    """
    scale=np.asarray(scale);pos=nodes[:,2:]*scale;ix,pred,succ=adjacency(nodes,edges)
    coords=pre_ilp['coords'];probs=pre_ilp['node_probabilities'];es=pre_ilp['edge_scores']
    orig={int(r[0]):r for r in raw_nodes};mapping=np.full(len(nodes),-1,np.int64);displacement=np.zeros(len(nodes),np.float32)
    for i,r in enumerate(nodes):
        j=int(r[0])
        if j in orig:
            if not (0<=j<len(coords) and np.array_equal(orig[j][1:],coords[j])):raise ValueError('Raw ID provenance drift')
            mapping[i]=j;displacement[i]=np.linalg.norm((r[2:]-orig[j][2:])*scale)
    confidence=np.zeros(len(nodes),np.float32);confidence[mapping>=0]=probs[mapping[mapping>=0]]
    native_to_i={int(j):i for i,j in enumerate(mapping) if j>=0};native_prob={}
    for a,b,p,*_ in es:
        if int(a) in native_to_i and int(b) in native_to_i:
            i,j=native_to_i[int(a)],native_to_i[int(b)]
            if nodes[j,1]==nodes[i,1]+1:native_prob[i,j]=max(native_prob.get((i,j),0.),float(p))
    pairs=set(native_prob);existing={(ix[int(a)],ix[int(b)]) for a,b in edges};pairs.update(existing)
    for t in np.unique(nodes[:,1])[:-1]:
        a,b=np.flatnonzero(nodes[:,1]==t),np.flatnonzero(nodes[:,1]==t+1)
        if not len(a) or not len(b):continue
        d,j=cKDTree(pos[b]).query(pos[a],k=min(4,len(b)))
        for ii,dd,jj in zip(a,np.asarray(d).reshape(len(a),-1),np.asarray(j).reshape(len(a),-1)):
            pairs.update((int(ii),int(b[k])) for v,k in zip(dd,jj) if v<=14.)
        d,j=cKDTree(pos[a]).query(pos[b],k=min(2,len(a)))
        for jj,dd,ii in zip(b,np.asarray(d).reshape(len(b),-1),np.asarray(j).reshape(len(b),-1)):
            pairs.update((int(a[k]),int(jj)) for v,k in zip(dd,ii) if v<=14.)
    pairs=np.asarray(sorted(pairs),np.int64).reshape(-1,2);a,b=pairs.T
    vp=np.zeros_like(pos);vn=np.zeros_like(pos);hm=np.ones(len(nodes),bool);fm=np.ones(len(nodes),bool)
    for i in range(len(nodes)):
        if len(pred[i])==1:vp[i]=pos[i]-pos[pred[i][0]];hm[i]=False
        if len(succ[i])==1:vn[i]=pos[succ[i][0]]-pos[i];fm[i]=False
    delta=pos[b]-pos[a];distance=np.linalg.norm(delta,axis=1)
    pp=np.array([native_prob.get(tuple(p),0.) for p in pairs],np.float32)
    has=np.array([tuple(p) in native_prob for p in pairs]);ex=np.array([tuple(p) in existing for p in pairs])
    fr=np.zeros(len(pairs));rr=np.zeros(len(pairs))
    for dest,ids in [(fr,a),(rr,b)]:
        order=np.lexsort((np.arange(len(ids)),distance,ids));split=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1]
        dest[order]=np.arange(len(order))-np.repeat(split,np.diff(np.r_[split,len(order)]))
    f=node_features
    x=np.column_stack([pp,~has,ex,distance,np.linalg.norm(delta-.5*vp[a],axis=1),
        np.linalg.norm(delta-.5*vn[b],axis=1),np.linalg.norm(delta-.25*(vp[a]+vn[b]),axis=1),fr,rr,
        f[a,5],f[b,5],np.abs(f[a,6]-f[b,6]),np.minimum(f[a,6],f[b,6])/(np.maximum(f[a,6],f[b,6])+.01),
        f[a,8],f[b,8],confidence[a],confidence[b],mapping[a]<0,mapping[b]<0,hm[a],fm[b],f[a,12],f[b,11]]).astype(np.float32)
    node=np.zeros((len(nodes),10),np.float32);node[:,:4]=np.column_stack([confidence,mapping<0,mapping<0,displacement]);node[:,8:]=1.
    for j,(i,k) in enumerate(pairs):
        if has[j]:
            if pp[j]>node[i,4]:node[i,5]=node[i,4];node[i,4]=pp[j]
            elif pp[j]>node[i,5]:node[i,5]=pp[j]
            node[k,7]=max(node[k,7],pp[j]);node[k,8],node[i,9]=0.,0.
    node[:,6]=node[:,4]-node[:,5]
    return dict(pairs=pairs,edge_features=x,node_extra=node,native_index=mapping)


def build_from_inputs(ctx,sample,nodes,edges,raw,pre_ilp,old_final,e_native_edges,e_hgb_edges,node_features=None):
    """Build the exact frozen v3 feature schema from supplied fresh neural evidence.

    A private temporary path view adapts the proven path-based feature builder;
    all scientific inputs are explicit arrays. No old store is written or read.
    Images remain the sample's explicit raw Zarr input. No GT key is accepted.
    """
    from .features import build
    name=sample['dataset'];work=ctx.work/'fresh_association';work.mkdir(parents=True,exist_ok=True)
    ss=dict(sample,image_path=str(sample.get('image_path',ctx.data/'train'/f'{name}.zarr')))
    with TemporaryDirectory(prefix='evidence-',dir=work) as tmp:
        root=Path(tmp);view=replace(ctx,v1=root/'v1',v2=root/'v2')
        save_arrays(view.v2/'raw'/f'{name}.npz',nodes=raw['nodes'],edges=raw['edges'],
            edge_prob=raw.get('edge_prob',np.full(len(raw['edges']),np.nan)))
        save_arrays(view.full/'inputs'/f'pre_ilp_{name}.npz',**{k:pre_ilp[k] for k in ['coords','node_probabilities','edge_scores']})
        save_graph(view.v1/'baseline/public'/f'{name}.npz',old_final['nodes'],old_final['edges'])
        save_arrays(view.v2/'predictions'/f'{name}.npz',**{'E_native_m1.5__edges':e_native_edges,'E_hgb_m1.5__edges':e_hgb_edges})
        return build(view,ss,nodes,edges,node_features=node_features)


def apply_fresh(ctx,sample,nodes,edges,raw,pre_ilp,old_final,e_native_edges,e_hgb_edges,
                model_family='residual',margin=1.5,node_features=None):
    from .association import score_features,decode
    from .common import validate
    c,provenance=build_from_inputs(ctx,sample,nodes,edges,raw,pre_ilp,old_final,e_native_edges,e_hgb_edges,node_features)
    source='6bba' if sample['embryo']=='44b6' else '44b6';scores=score_features(ctx,source,c,model_family)
    ee,ledger=decode(nodes,edges,c,scores,margin);validate(nodes,ee,sample['image_shape'])
    return nodes.copy(),ee,dict(fresh_explicit_inputs=True,features=provenance,**ledger)
