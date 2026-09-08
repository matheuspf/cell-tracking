"""Diagnostic transfer onto frozen Harmonic Fusion graphs; known contamination."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor,as_completed

import joblib
import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .common import DATA,OUT,SEED,digest,graph_hash,load_graph,now,read_json,save_graph,sha,write_json
from .candidates import temporal_features
from .features import FEATURES,frame_features,patches
from .filter_graph import PreparedFilter,filtered
from .image_model import predict_image
from .inventory import image_metadata,validate_graph


def reflected_coordinates(xyz,shape):
    """Integer half-sample reflection, matching the existing patch sampler.

    This indexes image features only. Original graph coordinates remain intact.
    """
    shape=np.asarray(shape)
    folded=np.asarray(xyz)%(2*shape)
    return np.where(folded<shape,folded,2*shape-1-folded)


def prepare_one(name,nodes,edges):
    target=OUT/'baseline/public'/f'{name}.npz'
    patch_path=OUT/'public_patches'/f'{name}.npy'
    raw=OUT/'public_harmonic_full/inputs'/f'pre_ilp_{name}.npz'
    stamp=dict(graph_hash=graph_hash(nodes,edges),pre_ilp_sha256=sha(raw),
               image_metadata_sha256=sha(DATA/'train'/f'{name}.zarr'/'zarr.json'))
    if target.with_suffix('.json').exists():
        cached=read_json(target.with_suffix('.json'))
        if (cached['inputs']!=stamp or sha(target)!=cached['sha256']
                or sha(patch_path)!=cached['patch_sha256']):
            raise ValueError('Public feature/patch cache changed')
        return
    start=time.perf_counter();shape,scale,_,statistics=image_metadata(DATA/'train'/f'{name}.zarr')
    scale=np.array(scale[1:]);q=statistics['quantiles']['0.999']
    # The original notebook exported six z=64 centers. Preserve those graphs for
    # diagnostic official scoring; image sampling follows patch reflection.
    validate_graph(nodes,edges,shape,prediction=True,spatial_bounds=False)
    outside=np.any((nodes[:,2:]<0)|(nodes[:,2:]>=np.asarray(shape[1:])),axis=1)
    arr=zarr.open_group(DATA/'train'/f'{name}.zarr',mode='r')['0']
    fs=np.zeros((len(nodes),len(FEATURES)),np.float32);pp=np.empty((len(nodes),3,32,32),np.uint8)
    for t in np.unique(nodes[:,1]):
        ix=np.flatnonzero(nodes[:,1]==t);xyz=nodes[ix,2:];im=arr[int(t)]
        z,y,x=im.shape
        a=im.reshape(z,y//2,2,x//2,2).mean(axis=(2,4),dtype=np.float32);ds=scale*[1,2,2]
        response=[]
        for s in [1.2,1.8]:
            bg=gaussian_filter(a,2*s/ds,mode='reflect')
            response.append((gaussian_filter(a,s/ds,mode='reflect')-bg)/q)
        response=np.maximum(*response)
        sample_xyz=reflected_coordinates(xyz,im.shape)
        coords=reflected_coordinates(xyz//[1,2,2],response.shape)
        fs[ix]=frame_features(int(t),xyz,response[tuple(coords.T)],im,bg,shape,scale,q,sampling_xyz=sample_xyz)
        pp[ix]=patches(im,xyz,q)
    units=temporal_features(nodes,edges,fs,scale)
    with np.load(raw) as f:det=f['coords'];prob=f['node_probabilities']
    confidence=np.zeros(len(nodes),np.float32);missing=np.ones(len(nodes),bool)
    for t in np.unique(nodes[:,1]):
        ix=np.flatnonzero(nodes[:,1]==t);di=np.flatnonzero(det[:,0]==t)
        if not len(di):continue
        dist,j=cKDTree(det[di,1:]*scale).query(nodes[ix,2:]*scale)
        found=dist<=2.
        confidence[ix[found]]=prob[di[j[found]]];missing[ix[found]]=False
    save_graph(target,nodes,edges,features=fs,tracklet=units,detector_confidence=confidence,confidence_missing=missing)
    patch_path.parent.mkdir(parents=True,exist_ok=True)
    np.save(patch_path,pp)
    write_json(target.with_suffix('.json'),dict(dataset=name,inputs=stamp,graph_hash=graph_hash(nodes,edges),sha256=sha(target),
               patch_sha256=sha(patch_path),seconds=time.perf_counter()-start,
               spatial_out_of_bounds=int(outside.sum()),image_boundary_sampling='half-sample reflection; original graph coordinates retained',
               confidence_missing=int(missing.sum()),confidence_rule='Nearest same-frame pre-ILP center within 2 um; otherwise zero with missing flag',lane='diagnostic_contaminated'))


def verify_prediction_lock():
    lock=read_json(OUT/'public_prediction_lock.json')
    manifest=read_json(OUT/'fold_manifest.json')
    if set(lock['graph_hashes'])!=set(manifest['expected_samples']):
        raise ValueError('Incomplete public graph lock')
    expected={str((OUT/'public_predictions'/d['source']/(n+'.npz')).relative_to(OUT))
              for d in manifest['directions'] for n in d['outer_samples']}
    if set(lock['predictions'])!=expected:raise ValueError('Incomplete public prediction lock')
    for path,h in lock['predictions'].items():
        if sha(OUT/path)!=h:raise ValueError('Public prediction changed')
    for name,h in lock['graph_hashes'].items():
        if sha(OUT/'baseline/public'/f'{name}.npz')!=h:raise ValueError('Public baseline changed')
    if sha(OUT/'public_selector_preregistration.json')!=lock['config_hash']:
        raise ValueError('Public configuration changed')
    if sha(OUT/'model_lock.json')!=lock['source_model_lock_sha256']:
        raise ValueError('Source model lock changed')
    if sha(OUT/'public_harmonic_full/submission.csv')!=lock['notebook_graph_csv_sha256']:
        raise ValueError('Original exported notebook graphs changed')
    return lock


def run(args):
    root=OUT/'public_harmonic_full'
    receipt=read_json(root/'adapter_manifest.json')
    if receipt['status']!='inference_complete':raise ValueError('Full public inference is not complete')
    inv=read_json(OUT/'inventory.json');manifest=read_json(OUT/'fold_manifest.json');config=read_json(OUT/'public_selector_preregistration.json')
    if set(receipt['expected_samples'])!=set(manifest['expected_samples']):raise ValueError('Public inference sample set changed')
    if (OUT/'public_prediction_lock.json').exists():
        verify_prediction_lock()
        return
    df=pd.read_csv(root/'submission.csv')
    if set(df.dataset)!=set(manifest['expected_samples']):raise ValueError('Public notebook skipped samples')
    jobs=[]
    for name,g in df.groupby('dataset'):
        nodes=g[g.row_type=='node'][['node_id','t','z','y','x']].to_numpy(np.int64)
        edges=g[g.row_type=='edge'][['source_id','target_id']].to_numpy(np.int64)
        jobs.append((name,nodes,edges))
    outside_rows=[]
    for name,nodes,edges in jobs:
        shape=image_metadata(DATA/'train'/f'{name}.zarr')[0]
        bad=np.any((nodes[:,2:]<0)|(nodes[:,2:]>=np.asarray(shape[1:])),axis=1)
        outside_rows.extend(dict(dataset=name,node_id=int(i),t=int(t),z=int(z),y=int(y),x=int(x)) for i,t,z,y,x in nodes[bad])
    write_json(OUT/'public_coordinate_audit.json',dict(created=now(),total_nodes=sum(len(n) for _,n,_ in jobs),
               out_of_bounds_nodes=len(outside_rows),rows=outside_rows,
               scope='Original public notebook graph; spatial integrity exception, diagnostic scoring only',
               handling='No graph coordinates changed or nodes dropped. Image features use half-sample reflection; geometry uses original coordinates.',
               official_metric_behavior='Pinned graph metric accepts spatial coordinates without image bounds; no image shape argument.',
               clean_lane_affected=False))
    # Original pilot vs export-hook full run: exact final graph reproduction.
    pilot=pd.read_csv(OUT/'public_harmonic_pilot/submission.csv');parity=[]
    for name,g in pilot.groupby('dataset'):
        old_n=g[g.row_type=='node'][['node_id','t','z','y','x']].to_numpy(np.int64)
        old_e=g[g.row_type=='edge'][['source_id','target_id']].to_numpy(np.int64)
        _,n,e=next(x for x in jobs if x[0]==name)
        parity.append(dict(dataset=name,exact_graph_parity=graph_hash(old_n,old_e)==graph_hash(n,e),pilot_hash=graph_hash(old_n,old_e),full_hash=graph_hash(n,e)))
    write_json(OUT/'public_graph_parity.json',parity)
    if not all(p['exact_graph_parity'] for p in parity):raise ValueError('Public baseline reproduction mismatch')
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        futures={pool.submit(prepare_one,*job):job[0] for job in jobs}
        for i,f in enumerate(as_completed(futures),1):f.result();print(f'Public image features {i}/{len(jobs)} {futures[f]}',flush=True)
    hashes={};torch.set_num_threads(2);model_lock=read_json(OUT/'model_lock.json')
    for d in manifest['directions']:
        models={}
        for mid in config['model_ids']:
            path=OUT/'models'/d['source']/(mid+('.pt' if mid.startswith('image_') else '.joblib'))
            if sha(path)!=model_lock['models'][str(path.relative_to(OUT))]:
                raise ValueError('Frozen source model changed')
            models[mid]=torch.load(path,map_location='cpu',weights_only=True) if mid.startswith('image_') else joblib.load(path)
        for name in d['outer_samples']:
            base=load_graph(OUT/'baseline/public'/f'{name}.npz')
            scores={'quality':base['detector_confidence']}
            pp=np.load(OUT/'public_patches'/f'{name}.npy',mmap_mode='r')
            for mid,m in models.items():
                if mid.startswith('image_'):scores[mid]=predict_image(m,pp,base['features'])
                else:scores[mid]=m['model'].predict_proba(base['features'][:,[FEATURES.index(c) for c in m['columns']]])[:,1].astype(np.float32)
            dest=OUT/'public_predictions'/d['source']/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,**scores);hashes[str(dest.relative_to(OUT))]=sha(dest)
    write_json(OUT/'public_prediction_lock.json',dict(created=now(),predictions=hashes,graph_hashes={n:sha(OUT/'baseline/public'/f'{n}.npz') for n in manifest['expected_samples']},
                config_hash=sha(OUT/'public_selector_preregistration.json'),source_model_lock_sha256=sha(OUT/'model_lock.json'),
                notebook_graph_csv_sha256=sha(root/'submission.csv'),
                provenance_lane='diagnostic_contaminated',outer_public_outcomes_revealed=False),immutable=True)


def evaluate_one(task):
    from .evaluate import confusion,variants
    from .metric_adapter import evaluate_graph
    row,source=task;name=row['dataset'];dest=OUT/'evaluation/public_score_samples'/f'{name}.json'
    bp=OUT/'baseline/public'/f'{name}.npz';gp=OUT/'evaluation/gt'/f'{name}.npz'
    stamp=dict(baseline_sha=sha(bp),gt_sha=sha(gp),prediction_sha=sha(OUT/'public_predictions'/source/f'{name}.npz'),
               estimate=row['estimated_total'],physical_scale=row['physical_scale'],lock_sha=sha(OUT/'public_prediction_lock.json'))
    if stamp['gt_sha']!=read_json(OUT/'evaluation/membership'/f'{name}.json')['gt_sha256']:
        raise ValueError('Public evaluation GT changed since source labels were created')
    if dest.exists():
        if read_json(dest)['inputs']!=stamp:raise ValueError('Public scored sample drift')
        return name,True
    start=time.perf_counter();b=load_graph(bp);gt=load_graph(gp)
    with np.load(OUT/'public_predictions'/source/f'{name}.npz') as f:scores={k:f[k] for k in f.files}
    n,e=b['nodes'],b['edges'];prepare=PreparedFilter(n,e)
    base,matches,base_tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    labels=np.array([int(i in matches) for i in n[:,0]],np.uint8);index={int(i):j for j,i in enumerate(n[:,0])}
    results=[]
    for v in variants(list(scores)):
        policy=v['policy'];mid=v['model_id']
        if policy in ['absolute_response_0.05','isolated_node_cleanup']:continue
        if policy=='identity':keep=np.ones(len(n),bool)
        else:
            s=labels.astype(float) if mid=='oracle' else np.zeros(len(n)) if mid=='random' else scores[mid]
            keep=prepare.mask(s,v['requested_keep'],policy.replace('oracle','membership'),v['seed'],key=mid)
        nn,ee=filtered(n,e,keep)
        if policy=='identity':r=base.copy();tp=base_tp
        else:r,_,tp=evaluate_graph(name,nn,ee,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        r.update(v);r.update(confusion(labels,keep));r.update(embryo=row['embryo'],fold=f'{source}->{row["embryo"]}',source=source,lane='public_contaminated_transfer',
                 baseline_tp=len(base_tp),baseline_tp_endpoints_kept=int(sum(keep[index[a]] and keep[index[c]] for a,c in base_tp)),
                 baseline_tp_rematched_surviving=len(base_tp&tp),newly_recovered_tp=len(tp-base_tp),baseline_count_ratio=len(n)/row['estimated_total'],
                 baseline_num_pred_nodes=len(n),count_ratio=len(nn)/row['estimated_total'],baseline_edge_fp=base['edge_fp'])
        results.append(r)
    write_json(dest,dict(dataset=name,inputs=stamp,seconds=time.perf_counter()-start,results=results))
    return name,False


def evaluate(args):
    verify_prediction_lock()
    manifest=read_json(OUT/'fold_manifest.json');inv=read_json(OUT/'inventory.json')
    if sorted(r['dataset'] for r in inv)!=sorted(manifest['expected_samples']):
        raise ValueError('Public expected sample set changed')
    tasks=[(r,next(d['source'] for d in manifest['directions'] if d['outer']==r['embryo'])) for r in inv]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):print(f'Public official evaluation {i}/{len(tasks)} {f.result()}',flush=True)
    rows=[]
    for r,source in tasks:rows.extend(read_json(OUT/'evaluation/public_score_samples'/f'{r["dataset"]}.json')['results'])
    pd.DataFrame(rows).to_parquet(OUT/'public_score_rows.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'public_score_rows.csv',index=False)
    write_json(OUT/'public_score_manifest.json',dict(complete=True,samples=manifest['expected_samples'],
                rows=len(rows),variants=len({r['variant_id'] for r in rows}),created=now(),
                prediction_lock_sha256=sha(OUT/'public_prediction_lock.json'),lane='diagnostic_contaminated_transfer'))
