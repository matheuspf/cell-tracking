"""Incumbent-native image, adjacency and teacher features; annotation-free."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .common import adjacency, digest, graph_hash, load_graph, read_json, save_arrays, sha, write_json
from .disagreements import TEACHERS, teachers

OLD_EDGE_FEATURES=['native_probability','native_missing','existing_edge','distance_um',
 'forward_residual_um','backward_residual_um','mutual_motion_residual_um','forward_distance_rank',
 'reverse_distance_rank','source_response','target_response','intensity_difference','intensity_ratio',
 'source_density','target_density','source_native_confidence','target_native_confidence',
 'source_origin_inserted','target_origin_inserted','source_history_missing','target_future_missing',
 'source_track_remaining','target_track_age']
EDGE_FEATURES=OLD_EDGE_FEATURES+['native_logit','native_logit_missing',
 *['vote_'+x for x in TEACHERS[1:]],'teacher_vote_fraction','source_relocation_um',
 'target_relocation_um','source_native_margin','target_native_margin',
 'source_in_degree','source_out_degree','target_in_degree','target_out_degree']
CONFIG=dict(version='incumbent_native_v3_1',forward_neighbors=2,reverse_neighbors=1,neighbor_gate_um=10.,
 image_response='DoG 1.2/1.8 um, downsample 1x2x2, fresh current centers',feature_names=EDGE_FEATURES)


def temporal(nodes,edges,f,scale):
    from annotation_selection.filter_graph import tracklets
    units,pred,succ=tracklets(nodes,edges)
    # Vectorized grouping avoids the quadratic old per-tracklet scan.
    count=np.bincount(units);lo=np.full(len(count),np.inf);hi=np.full(len(count),-np.inf)
    np.minimum.at(lo,units,nodes[:,1]);np.maximum.at(hi,units,nodes[:,1])
    f[:,10]=count[units];f[:,11]=nodes[:,1]-lo[units];f[:,12]=hi[units]-nodes[:,1]
    f[:,13:]=0
    for i in range(len(nodes)):
        f[i,13]=min((np.linalg.norm((nodes[i,2:]-nodes[j,2:])*scale) for j in pred[i]),default=0.)
        f[i,14]=min((np.linalg.norm((nodes[i,2:]-nodes[j,2:])*scale) for j in succ[i]),default=0.)
        f[i,15]=len(succ[i])==2
    return units


def node_image_features(ctx,sample,nodes,edges):
    import zarr
    from annotation_selection.features import FEATURES,frame_features
    path=Path(sample.get('image_path',ctx.data/'train'/f"{sample['dataset']}.zarr"))
    meta=read_json(path/'zarr.json');shape=read_json(path/'0/zarr.json')['shape']
    scale=np.asarray(sample['physical_scale']);q=float(meta['attributes']['image_statistics']['quantiles']['0.999'])
    if not q>0: raise ValueError('Invalid image quantile')
    arr=zarr.open_group(path,mode='r')['0'];f=np.zeros((len(nodes),len(FEATURES)),np.float32)
    for t in np.unique(nodes[:,1]):
        ids=np.flatnonzero(nodes[:,1]==t);xyz=nodes[ids,2:].astype(np.int64)
        im=np.asarray(arr[int(t)]);z,y,x=im.shape
        a=im.reshape(z,y//2,2,x//2,2).mean(axis=(2,4),dtype=np.float32);ds=scale*[1,2,2]
        responses=[]
        for s in [1.2,1.8]:
            bg=gaussian_filter(a,2*s/ds,mode='reflect')
            responses.append((gaussian_filter(a,s/ds,mode='reflect')-bg)/q)
        response=np.maximum(*responses);sample_xyz=np.clip(xyz,0,np.asarray(im.shape)-1)
        coords=np.clip(xyz//[1,2,2],0,np.asarray(response.shape)-1)
        f[ids]=frame_features(int(t),xyz,response[tuple(coords.T)],im,bg,shape,scale,q,sampling_xyz=sample_xyz)
    units=temporal(nodes,edges,f,scale)
    return f,units


def build(ctx,sample,nodes,edges,node_features=None):
    name=sample['dataset'];scale=np.asarray(sample['physical_scale']);pos=nodes[:,2:]*scale
    raw=load_graph(ctx.v2/'raw'/f'{name}.npz')
    with np.load(ctx.full/'inputs'/f'pre_ilp_{name}.npz',allow_pickle=False) as p:
        coords=p['coords'];probs=p['node_probabilities'];es=p['edge_scores']
    raw_map={int(r[0]):r for r in raw['nodes']};ix,pred,succ=adjacency(nodes,edges)
    mapping=np.full(len(nodes),-1,np.int64);conf=np.zeros(len(nodes),np.float32);reloc=np.zeros(len(nodes),np.float32)
    for i,r in enumerate(nodes):
        k=int(r[0])
        if k in raw_map:
            if not (0<=k<len(coords) and np.array_equal(raw_map[k][1:],coords[k]) and r[1]==coords[k,0]):
                raise ValueError('Raw stable-ID provenance drift')
            mapping[i]=k;conf[i]=probs[k];reloc[i]=np.linalg.norm((r[2:]-coords[k,1:])*scale)
    native={}
    for a,b,p,*_ in es:
        a,b=int(a),int(b)
        if a in ix and b in ix and mapping[ix[a]]>=0 and mapping[ix[b]]>=0 and nodes[ix[b],1]==nodes[ix[a],1]+1:
            key=(ix[a],ix[b]);native[key]=max(native.get(key,0.),float(p))
    votes,audit=teachers(ctx,sample,nodes,edges,raw)
    pairs=set(native)
    for es in votes.values():pairs.update((ix[int(a)],ix[int(b)]) for a,b in es)
    before=len(pairs)
    for t in np.unique(nodes[:,1]):
        src=np.flatnonzero(nodes[:,1]==t);dst=np.flatnonzero(nodes[:,1]==t+1)
        if not len(src) or not len(dst):continue
        for aa,bb,k,reverse in [(src,dst,2,False),(dst,src,1,True)]:
            dd,jj=cKDTree(pos[bb]).query(pos[aa],k=min(k,len(bb)))
            for i,ds,js in zip(aa,np.asarray(dd).reshape(len(aa),-1),np.asarray(jj).reshape(len(aa),-1)):
                for d,j in zip(ds,js):
                    if d<=CONFIG['neighbor_gate_um']:pairs.add((int(bb[j]),int(i)) if reverse else (int(i),int(bb[j])))
    pairs=np.asarray(sorted(pairs),np.int64).reshape(-1,2);a,b=pairs.T
    existing={(ix[int(a)],ix[int(b)]) for a,b in edges};delta=pos[b]-pos[a];distance=np.linalg.norm(delta,axis=1)
    vp=np.zeros_like(pos);vn=np.zeros_like(pos);hm=np.ones(len(nodes),bool);fm=np.ones(len(nodes),bool)
    for i in range(len(nodes)):
        if len(pred[i])==1:vp[i]=pos[i]-pos[pred[i][0]];hm[i]=False
        if len(succ[i])==1:vn[i]=pos[succ[i][0]]-pos[i];fm[i]=False
    pp=np.array([native.get(tuple(p),0.) for p in pairs],np.float32)
    has=np.array([tuple(p) in native for p in pairs]);ex=np.array([tuple(p) in existing for p in pairs])
    fr=np.zeros(len(pairs));rr=np.zeros(len(pairs))
    for dest,ids in [(fr,a),(rr,b)]:
        order=np.lexsort((np.arange(len(ids)),distance,ids));starts=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1]
        dest[order]=np.arange(len(order))-np.repeat(starts,np.diff(np.r_[starts,len(order)]))
    if node_features is None:f,units=node_image_features(ctx,sample,nodes,edges)
    else:f=np.asarray(node_features).copy();units=temporal(nodes,edges,f,scale)
    native_logit=np.log(np.clip(pp,1e-5,1-1e-5)/(1-np.clip(pp,1e-5,1-1e-5)))
    # Missing native evidence uses geometric continuity as an explicit, flagged offset.
    native_logit=np.where(has,native_logit,3-np.linalg.norm(delta-.25*(vp[a]+vn[b]),axis=1))
    vv=np.array([[int((int(nodes[i,0]),int(nodes[j,0])) in votes[k]) for k in TEACHERS[1:]] for i,j in pairs],np.float32)
    out_probs=[[] for _ in nodes];in_probs=[[] for _ in nodes]
    for (i,j),p in native.items():out_probs[i].append(p);in_probs[j].append(p)
    def margins(xs):
        return np.array([sorted(x,reverse=True)[0]-sorted(x,reverse=True)[1] if len(x)>1 else x[0] if x else 0 for x in xs])
    x=np.column_stack([pp,~has,ex,distance,np.linalg.norm(delta-.5*vp[a],axis=1),
        np.linalg.norm(delta-.5*vn[b],axis=1),np.linalg.norm(delta-.25*(vp[a]+vn[b]),axis=1),fr,rr,
        f[a,5],f[b,5],np.abs(f[a,6]-f[b,6]),np.minimum(f[a,6],f[b,6])/(np.maximum(f[a,6],f[b,6])+.01),
        f[a,8],f[b,8],conf[a],conf[b],mapping[a]<0,mapping[b]<0,hm[a],fm[b],f[a,12],f[b,11],
        native_logit,~has,vv,vv.mean(axis=1),reloc[a],reloc[b],margins(out_probs)[a],margins(in_probs)[b],
        np.array([len(x) for x in pred])[a],np.array([len(x) for x in succ])[a],
        np.array([len(x) for x in pred])[b],np.array([len(x) for x in succ])[b]]).astype(np.float32)
    assert x.shape[1]==len(EDGE_FEATURES) and np.isfinite(x).all()
    arrays=dict(pairs=pairs,edge_features=x,node_features=f,tracklet=units,native_index=mapping,
        native_confidence=conf,node_relocation_um=reloc)
    return arrays,dict(teacher_mapping=audit,teacher_native_union=before,total_candidates=len(pairs),
        image_neighbor_added=len(pairs)-before,inserted_nodes=int(np.sum(mapping<0)),
        relocated_native_nodes=int(np.sum(reloc>0)),nodes=len(nodes))


def stamp(ctx,sample,nodes,edges):
    name=sample['dataset'];path=Path(sample.get('image_path',ctx.data/'train'/f'{name}.zarr'))
    return dict(graph_hash=graph_hash(nodes,edges),config_sha256=digest(CONFIG),
        code_sha256=sha(Path(__file__)),disagreements_code_sha256=sha(Path(__file__).with_name('disagreements.py')),
        raw_sha256=sha(ctx.v2/'raw'/f'{name}.npz'),native_sha256=sha(ctx.full/'inputs'/f'pre_ilp_{name}.npz'),
        old_graph_sha256=sha(ctx.v1/'baseline/public'/f'{name}.npz'),teacher_sha256=sha(ctx.v2/'predictions'/f'{name}.npz'),
        image_metadata_sha256=sha(path/'zarr.json'))


def cached(ctx,sample,nodes=None,edges=None):
    if nodes is None:
        b=load_graph(ctx.incumbent(sample['dataset']));nodes,edges=b['nodes'],b['edges']
    path=ctx.out/'features'/f"{sample['dataset']}.npz";expected=stamp(ctx,sample,nodes,edges)
    if path.with_suffix('.json').exists():
        meta=read_json(path.with_suffix('.json'))
        if meta['inputs']!=expected or meta['sha256']!=sha(path):raise ValueError('Stale incumbent feature cache')
        return load_graph(path)
    tic=time.perf_counter();a,meta=build(ctx,sample,nodes,edges);save_arrays(path,**a)
    write_json(path.with_suffix('.json'),dict(inputs=expected,sha256=sha(path),seconds=time.perf_counter()-tic,**meta))
    return a


def prepare_one(task):
    ctx,sample=task;a=cached(ctx,sample)
    return dict(dataset=sample['dataset'],candidates=len(a['pairs']))
