"""Fixed classical image candidates; no annotation imports or reads."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter,maximum_filter
from scipy.spatial import cKDTree

from .common import DATA,OUT,SEED,digest,graph_hash,load_graph,now,read_json,save_graph,sha,stage,write_json
from .features import FEATURES,VERSION,frame_features,patches
from .filter_graph import tracklets
from .inventory import image_metadata,validate_graph

CONFIG=dict(generator='classical_dog_v1',xy_downsample=2,sigma_um=[1.2,1.8],background_ratio=2,
            low_threshold=.025,high_threshold=.05,nms_um=3.25,max_candidates_per_frame=1200,
            link_max_um=7.,division_rule='disabled_fixed_no_source_independent_groups_yet',
            patch=dict(shape=[3,32,32],pixel_um=.8125,normalization='image_metadata_0.999_quantile',
                       previous_next=False),fingerprints_per_frame=32)


def nms(points,scores,scale,radius):
    if len(points)==0: return np.empty(0,int)
    tree=cKDTree(points*scale)
    order=np.lexsort((np.arange(len(scores)),-scores))
    disabled=np.zeros(len(points),bool); keep=[]
    for i in order:
        if disabled[i]: continue
        keep.append(i)
        disabled[tree.query_ball_point(points[i]*scale,radius)]=True
    return np.asarray(keep)


def detect(image,scale,q):
    a=image.reshape(image.shape[0],image.shape[1]//2,2,image.shape[2]//2,2).mean(axis=(2,4),dtype=np.float32)
    ds=np.asarray(scale)*[1,2,2]
    responses=[]
    for sigma in CONFIG['sigma_um']:
        small=gaussian_filter(a,sigma/ds,mode='reflect')
        background=gaussian_filter(a,2*sigma/ds,mode='reflect')
        responses.append((small-background)/q)
    response=np.maximum(*responses)
    peak=(response==maximum_filter(response,size=(3,5,5),mode='reflect'))&(response>=CONFIG['low_threshold'])
    xyz=np.column_stack(np.nonzero(peak))
    scores=response[tuple(xyz.T)]
    order=np.argsort(-scores,kind='stable')[:CONFIG['max_candidates_per_frame']*3]
    xyz=xyz[order]*[1,2,2]; scores=scores[order]
    # Local refinement uses the image only; both classes get identical centers.
    for i,point in enumerate(xyz):
        low=np.maximum(point-[1,2,2],0); high=np.minimum(point+[2,3,3],image.shape)
        cut=image[tuple(slice(l,h) for l,h in zip(low,high))]
        xyz[i]=low+np.array(np.unravel_index(np.argmax(cut),cut.shape))
    keep=nms(xyz,scores,np.asarray(scale),CONFIG['nms_um'])[:CONFIG['max_candidates_per_frame']]
    xyz=xyz[keep]; scores=scores[keep]
    # Stable coordinate ordering determines candidate IDs, independently of scores/GT.
    order=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0])) if len(xyz) else np.array([],int)
    return xyz[order],scores[order],background


def link(nodes,scale):
    edges=[]
    for t in np.unique(nodes[:,1])[:-1]:
        aa=nodes[nodes[:,1]==t]; bb=nodes[nodes[:,1]==t+1]
        if not len(aa) or not len(bb): continue
        tree=cKDTree(bb[:,2:]*scale)
        pairs=tree.query_ball_point(aa[:,2:]*scale,CONFIG['link_max_um'])
        proposals=[]
        for i,js in enumerate(pairs):
            for j in js:
                distance=float(np.linalg.norm((aa[i,2:]-bb[j,2:])*scale))
                proposals.append((distance,int(aa[i,0]),int(bb[j,0])))
        used_a=set();used_b=set()
        for d,a,b in sorted(proposals):
            if a not in used_a and b not in used_b:
                edges.append((a,b));used_a.add(a);used_b.add(b)
    return np.asarray(edges,np.int64).reshape(-1,2)


def temporal_features(nodes,edges,features,scale):
    units,pred,succ=tracklets(nodes,edges)
    for u in np.unique(units):
        ix=np.flatnonzero(units==u);times=nodes[ix,1]
        features[ix,10]=len(ix)
        features[ix,11]=times-times.min()
        features[ix,12]=times.max()-times
    for i in range(len(nodes)):
        features[i,13]=min((np.linalg.norm((nodes[i,2:]-nodes[j,2:])*scale) for j in pred[i]),default=0.)
        features[i,14]=min((np.linalg.norm((nodes[i,2:]-nodes[j,2:])*scale) for j in succ[i]),default=0.)
        features[i,15]=len(succ[i])>=2
    return units


def generate(name):
    target=OUT/'baseline/clean'/f'{name}.npz'
    meta_target=OUT/'baseline/clean'/f'{name}.json'
    if target.exists() and meta_target.exists():
        meta=read_json(meta_target)
        if meta['config_hash']!=digest(CONFIG) or meta['sha256']!=sha(target):
            raise RuntimeError(f'Immutable baseline changed: {name}')
        return dict(dataset=name,resumed=True,**{k:meta[k] for k in ['nodes','seconds']})
    start=time.perf_counter()
    path=DATA/'train'/f'{name}.zarr'
    shape,scale,transforms,statistics=image_metadata(path)
    scale=np.asarray(scale[1:]);q=float(statistics['quantiles']['0.999'])
    if q<=0: raise ValueError('Invalid image quantile')
    arr=zarr.open_group(path,mode='r')['0']
    ns=[];fs=[];ps=[];fingerprints=[];read_seconds=0.;detect_seconds=0.;patch_seconds=0.
    next_id=0
    for t in range(shape[0]):
        tic=time.perf_counter();im=np.asarray(arr[t]);read_seconds+=time.perf_counter()-tic
        tic=time.perf_counter();xyz,scores,bg=detect(im,scale,q);detect_seconds+=time.perf_counter()-tic
        nodes=np.column_stack([np.arange(next_id,next_id+len(xyz)),np.full(len(xyz),t),xyz]);next_id+=len(xyz)
        ns.append(nodes);fs.append(frame_features(t,xyz,scores,im,bg,shape,scale,q))
        tic=time.perf_counter();ps.append(patches(im,xyz,q));patch_seconds+=time.perf_counter()-tic
        for i in np.argsort(-scores)[:CONFIG['fingerprints_per_frame']]:
            point=xyz[i]
            if np.any(point<[1,2,2]) or np.any(point>=np.asarray(im.shape)-[1,2,2]):continue
            cut=im[point[0]-1:point[0]+2,point[1]-2:point[1]+3,point[2]-2:point[2]+3]
            fingerprints.append([hashlib.sha256(cut.tobytes()).hexdigest()[:32],t,*point.tolist()])
    nodes=np.concatenate(ns).astype(np.int64);features=np.concatenate(fs);patch=np.concatenate(ps)
    edges=link(nodes,scale);units=temporal_features(nodes,edges,features,scale)
    validate_graph(nodes,edges,shape,prediction=True)
    save_graph(target,nodes,edges,features=features,tracklet=units)
    pp=OUT/'patches'/f'{name}.npy';pp.parent.mkdir(parents=True,exist_ok=True)
    np.save(pp,patch,allow_pickle=False)
    write_json(OUT/'fingerprints'/f'{name}.json',fingerprints)
    result=dict(dataset=name,nodes=len(nodes),edges=len(edges),graph_hash=graph_hash(nodes,edges),sha256=sha(target),
                config_hash=digest(CONFIG),feature_schema=FEATURES,feature_version=VERSION,scale=scale,
                created=now(),seconds=time.perf_counter()-start,read_seconds=read_seconds,detect_seconds=detect_seconds,
                patch_seconds=patch_seconds,patch_bytes=patch.nbytes,high_threshold_nodes=int(np.sum(features[:,5]>=CONFIG['high_threshold'])))
    write_json(meta_target,result)
    return result


def run(args):
    write_json(OUT/'candidate_config.json',CONFIG,immutable=True)
    write_json(OUT/'feature_schema.json',dict(version=VERSION,columns=FEATURES),immutable=True)
    names=[r['dataset'] for r in read_json(OUT/'inventory.json')]
    # Image-only low/high quantiles choose the pilot; no target labels are involved.
    if args.limit:
        by_q=sorted(names,key=lambda n: image_metadata(DATA/'train'/f'{n}.zarr')[3]['quantiles']['0.999'])
        names=[by_q[0],by_q[-1]][:args.limit] if args.limit<=2 else names[:args.limit]
    stage('S020','running',lane='clean',expected=len(names),config_hash=digest(CONFIG))
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        for i,result in enumerate(pool.map(generate,names),1):
            print(f'{i}/{len(names)} {result["dataset"]}: {result["nodes"]} nodes, {result["seconds"]:.1f}s',flush=True)
    metas=[read_json(OUT/'baseline/clean'/f'{n}.json') for n in names]
    pd.DataFrame(metas).to_csv(OUT/('candidate_pilot.csv' if args.limit else 'candidate_inventory.csv'),index=False)
    stage('S020','pilot_complete' if args.limit else 'clean_complete',samples=len(names),nodes=sum(x['nodes'] for x in metas))
