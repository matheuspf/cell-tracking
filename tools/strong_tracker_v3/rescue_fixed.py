"""Persistent foreground-peak rescue, correcting the sealed plateau control.

The original rescue.py and its measured graph outputs are preserved. This new
policy rejects flat/background maxima and requires three positive image events.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
from scipy.ndimage import maximum_filter

from .common import (adjacency,digest,graph_hash,load_graph,now,read_json,run_pool,
                     save_arrays,save_graph,sha,validate,write_json)
from .context import RunContext
from .rescue import CONFIG as ORIGINAL_CONFIG,Frames,candidate_restore,refine

CONFIG=dict(**ORIGINAL_CONFIG,peak_mad_multiplier=5.,mad_normalization=1.4826,
    minimum_noise_sigma_raw=1.,derived_id_start=1<<60,
    plateau_policy='Strict positive foreground contrast and dynamic range at every frame',
    corrected_version='rescue_v3_2_persistent_foreground',
    refinement_limit='0.75um continuous proposal before integer quantization; not a strict final lattice bound')
CONFIG['version']=CONFIG.pop('corrected_version')
VARIANT='R_image_persistent'


def foreground_peaks(patch):
    """Return true foreground peaks; a constant patch contains no object evidence."""
    patch=np.asarray(patch,dtype=np.float64)
    if not patch.size or not np.isfinite(patch).all():return np.empty((0,3),np.int64),dict(reason='invalid_patch')
    median=float(np.median(patch));dynamic=float(np.max(patch)-np.min(patch))
    mad=float(np.median(np.abs(patch-median)))
    sigma=max(CONFIG['minimum_noise_sigma_raw'],CONFIG['mad_normalization']*mad)
    threshold=median+CONFIG['peak_mad_multiplier']*sigma
    if dynamic<=0:return np.empty((0,3),np.int64),dict(reason='flat_patch',median=median,mad=mad,dynamic_range=dynamic,threshold=threshold)
    q=float(np.quantile(patch,CONFIG['secondary_maxima_quantile']))
    mask=(patch==maximum_filter(patch,size=3,mode='nearest'))&(patch>=q)&(patch>threshold)
    peaks=np.argwhere(mask)
    return peaks,dict(reason='foreground_peaks' if len(peaks) else 'insufficient_contrast',median=median,
        mad=mad,dynamic_range=dynamic,threshold=threshold,quantile_threshold=q,noise_sigma=sigma)


def maxima_rescue(ctx,sample,nodes,edges):
    """Append three-frame foreground trajectories, keeping every external edge."""
    n=nodes.copy();e=edges.copy();_,pred,succ=adjacency(n,e)
    shape=sample['image_shape'];scale=np.asarray(sample['physical_scale']);frames=Frames(ctx,sample)
    endpoints=[i for i in range(len(n)) if not succ[i] and pred[i] and n[i,1]+3<shape[0] and int(n[i,0])%16==0]
    cap=max(1,int(len(n)*CONFIG['point_cap_fraction']));ledger=[];added=[];added_edges=[];tested=0;reasons=Counter()
    by_t={int(t):n[n[:,1]==t,2:]*scale for t in np.unique(n[:,1])};new_by_t={}
    nextid=max(CONFIG['derived_id_start'],int(n[:,0].max(initial=-1))+1)
    for i in sorted(endpoints,key=lambda i:(n[i,1],n[i,0])):
        if len(added)+3>cap:break
        tested+=1;pos=n[i,2:].astype(float);chain=[];appearance=[]
        for dt in [1,2,3]:
            t=int(n[i,1])+dt;patch,lo=frames.patch(t,pos,(2,8,8));peaks,quality=foreground_peaks(patch)
            if not len(peaks):reasons[quality['reason']]+=1;break
            known=np.vstack([by_t.get(t,np.empty((0,3))),np.asarray(new_by_t.get(t,[])).reshape(-1,3)])
            accepted=[]
            for point in peaks:
                coord=(point+lo).astype(np.int64);distance=float(np.linalg.norm((coord-pos)*scale))
                if distance>CONFIG['secondary_persistence_radius_um']:reasons['outside_persistence_gate']+=1;continue
                if len(known) and np.linalg.norm(known-coord*scale,axis=1).min()<CONFIG['secondary_min_distance_um']:
                    reasons['existing_or_inserted_neighbor']+=1;continue
                value=float(patch[tuple(point)])
                accepted.append((value,-distance,tuple(map(int,coord)),coord))
            if not accepted:break
            # Deterministic choice among positive peaks; no plateau/background tie can qualify.
            value,_,_,pos=max(accepted,key=lambda x:(x[0],x[1],tuple(-z for z in x[2])))
            chain.append((t,pos.copy()));appearance.append(dict(time=t,peak=value,
                robust_snr=(value-quality['median'])/quality['noise_sigma'],**quality))
        if len(chain)!=3:continue
        ids=list(range(nextid,nextid+3));nextid+=3
        for k,(t,point) in zip(ids,chain):
            added.append([k,t,*point]);new_by_t.setdefault(t,[]).append(point*scale)
        links=[(int(n[i,0]),ids[0]),(ids[0],ids[1]),(ids[1],ids[2])];added_edges.extend(links)
        ledger.append(dict(kind='persistent_foreground_maxima',anchor_id=int(n[i,0]),node_ids=ids,
            canonical_nodes=[[k,t,*map(int,point)] for k,(t,point) in zip(ids,chain)],
            added_edges=links,removed_edges=[],persistent_frames=3,appearance=appearance,
            boundary_policy='Append at a previously terminated source; all prior nodes and edges preserved'))
    if added:n=np.vstack([n,np.asarray(added,np.int64)]);e=np.vstack([e,np.asarray(added_edges,np.int64)])
    assert np.array_equal(n[:len(nodes)],nodes) and np.array_equal(e[:len(edges)],edges)
    validate(n,e,shape)
    return n,e,ledger,dict(triggered_endpoints=len(endpoints),tested=tested,secondary_inserted_nodes=len(added),
        point_cap=cap,foreground_rejections=dict(reasons),fixed_foreground_gate=True)


def _apply_local(ctx,sample,nodes,edges):
    n,e,ledger,stats=candidate_restore(ctx,sample,nodes,edges)
    n,ls,st=refine(ctx,sample,n,e);ledger+=ls;stats.update(st)
    n,e,ls,st=maxima_rescue(ctx,sample,n,e);ledger+=ls;stats.update(st)
    validate(n,e,sample['image_shape'])
    return n,e,ledger,stats


def apply(ctx,sample,nodes,edges,raw=None,pre_ilp=None):
    """Annotation-free regular or explicit-fresh-evidence API for V360/V370."""
    if (raw is None)!=(pre_ilp is None):raise ValueError('Supply both fresh raw graph and pre-ILP arrays')
    if raw is None:return _apply_local(ctx,sample,nodes,edges)
    work=ctx.work/'fresh_rescue_fixed';work.mkdir(parents=True,exist_ok=True);name=sample['dataset']
    with TemporaryDirectory(prefix='evidence-',dir=work) as tmp:
        root=Path(tmp);view=replace(ctx,v1=root/'v1',v2=root/'v2')
        save_arrays(view.v2/'raw'/f'{name}.npz',nodes=raw['nodes'],edges=raw['edges'],edge_prob=raw['edge_prob'])
        save_arrays(view.full/'inputs'/f'pre_ilp_{name}.npz',**{k:pre_ilp[k] for k in ['coords','node_probabilities','edge_scores']})
        return _apply_local(view,sample,nodes,edges)


def one(task):
    from .association import deny_annotations
    deny_annotations();ctx,sample=task;name=sample['dataset'];base=load_graph(ctx.incumbent(name))
    dest=ctx.out/'rescue_fixed'/f'{name}.json';path=ctx.out/'candidate_graphs'/VARIANT/f'{name}.npz'
    stamp=dict(graph_hash=graph_hash(base['nodes'],base['edges']),incumbent_sha256=sha(ctx.incumbent(name)),
        native_sha256=sha(ctx.full/'inputs'/f'pre_ilp_{name}.npz'),raw_sha256=sha(ctx.v2/'raw'/f'{name}.npz'),
        image_metadata_sha256=sha(Path(sample['image_path'])/'zarr.json'),
        code_sha256=sha(Path(__file__)),original_helpers_sha256=sha(Path(__file__).with_name('rescue.py')),config_sha256=digest(CONFIG))
    if dest.exists():
        record=read_json(dest)
        if record['inputs']!=stamp or record['sha256']!=sha(path):raise ValueError('Fixed rescue fingerprint drift')
        return name+' exact cache'
    tic=time.perf_counter();n,e,ledger,stats=apply(ctx,sample,base['nodes'],base['edges'])
    save_graph(path,n,e)
    write_json(dest,dict(inputs=stamp,variant=VARIANT,graph_hash=graph_hash(n,e),sha256=sha(path),
        seconds=time.perf_counter()-tic,stats=stats,ledger=ledger,annotation_reads_blocked=True))
    return dict(dataset=name,secondary_inserted_nodes=stats['secondary_inserted_nodes'],restored_nodes=stats['restored_nodes'],relocated=stats['relocated'])


def run(ctx=None,args=None,workers=2):
    ctx=ctx or RunContext.default();ctx.check_outputs()
    if args is not None:workers=args.workers
    path=ctx.out/'rescue_fixed_config_lock.json';previous=read_json(path) if path.exists() else {}
    write_json(path,dict(created=previous.get('created',now()),config=CONFIG,variant=VARIANT,code_sha256=sha(Path(__file__)),
        original_helpers_sha256=sha(Path(__file__).with_name('rescue.py')),
        scope='Adaptive correctness fix after a synthetic plateau counterexample; fixed image-only thresholds, no target-label calibration',
        old_invalid_control_preserved='R_image_local'),immutable=True)
    samples=ctx.samples();tic=time.perf_counter();list(run_pool(one,[(ctx,s) for s in samples],workers))
    records=[read_json(ctx.out/'rescue_fixed'/f"{s['dataset']}.json") for s in samples];stats=Counter();rejections=Counter()
    for r in records:
        for key,value in r['stats'].items():
            if isinstance(value,(int,float)):stats[key]+=value
        rejections.update(r['stats']['foreground_rejections'])
    write_json(ctx.out/'rescue_fixed_summary.json',dict(created=now(),variant=VARIANT,samples=len(samples),seconds=time.perf_counter()-tic,
        stats=dict(stats),foreground_rejections=dict(rejections),config=CONFIG,
        original_control_preserved=True,annotation_reads_blocked=True))
    write_json(ctx.out/'rescue_fixed_prediction_lock.json',dict(created=now(),variant=VARIANT,
        config_lock_sha256=sha(path),predictions={s['dataset']:sha(ctx.out/'candidate_graphs'/VARIANT/f"{s['dataset']}.npz") for s in samples},
        both_directions_complete_before_comparative_scoring=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);a=parser.parse_args();run(workers=a.workers)
