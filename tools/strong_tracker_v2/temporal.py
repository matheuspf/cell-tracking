"""Full positive-coverage temporal image study, conditional on cheap-model results."""
from __future__ import annotations

import gc
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .common import (OUT,V1,SEED,inventory,load_graph,now,read_json,save_arrays,sha,stage,write_json)
from .temporal_model import Encoder,make_batch,temporal_indices

CONFIG=dict(frames=[-2,-1,0,1,2],panels=['XY','XZ','YZ'],full_context_um=26.,local_context_um=13.,
    feature_geometry=False,positive_group_fractions=[.1,.3,1.],steps=5000,batch_size=256,
    optimizer='AdamW',learning_rate=.0003,weight_decay=.01,seed=SEED,
    augmentation='Shared intensity gain only; no independent panel transforms or rolled borders',
    positive_sampling='Every positive in chosen source tracklet groups; deterministic shuffled cycles',
    negative_sampling='Uniform predicted tracklet strata; inverse selection weights restore natural negative population',
    checkpoint='Fixed final step; stop early only for nonfinite loss, resource cap, or GPU failure',
    max_gpu_allocated_gib=20,max_nonfocus_gpu_hours=16,scope='Exploratory native annotation membership, not biological cell truth')


def source_data(source):
    rows=[r for r in inventory() if r['embryo']==source]
    total=sum(len(load_graph(V1/'baseline/public'/f"{r['dataset']}.npz")['nodes']) for r in rows)
    patches=np.empty((total,3,32,32),np.uint8);features=[];labels=[];indices=[];valid=[];groups=[]
    cursor=0;group_start=0
    for r in rows:
        name=r['dataset'];b=load_graph(V1/'baseline/public'/f'{name}.npz');c=load_graph(OUT/'native'/f'{name}.npz')
        length=len(b['nodes']);patch=np.load(V1/'public_patches'/f'{name}.npy',mmap_mode='r')
        assert patch.shape==(length,3,32,32)
        patches[cursor:cursor+length]=patch
        index,mask=temporal_indices(b['nodes'],b['edges'])
        save_arrays(OUT/'temporal_index'/f'{name}.npz',index=index,valid=mask)
        indices.append(index+cursor);valid.append(mask)
        features.append(np.column_stack([b['features'][:,5:],c['node_extra']]).astype(np.float32))
        labels.append(load_graph(OUT/'evaluation/membership'/f'{name}.npz')['annotation_label'])
        u=np.unique(b['tracklet'],return_inverse=True)[1]+group_start;groups.append(u);group_start=int(u.max())+1
        cursor+=length
    return dict(patches=patches,features=np.concatenate(features),labels=np.concatenate(labels),index=np.concatenate(indices),
                valid=np.concatenate(valid),groups=np.concatenate(groups))


def fit_one(source,fraction,data,steps,prior_gpu_seconds=0.):
    path=OUT/'temporal_models'/source/f'fraction_{fraction}.pt';receipt=path.with_suffix('.json')
    if path.exists() and receipt.exists():return read_json(receipt)
    rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
    y=data['labels'];positive_groups=np.unique(data['groups'][y==1]);rng.shuffle(positive_groups)
    selected_groups=positive_groups[:max(1,int(np.ceil(fraction*len(positive_groups))))]
    positives=np.flatnonzero((y==1)&np.isin(data['groups'],selected_groups))
    negative=np.flatnonzero(y==0);ng=data['groups'][negative]
    order=np.argsort(ng,kind='stable');neg_order=negative[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ng[order]))+1];lengths=np.diff(np.r_[starts,len(negative)])
    means=data['features'].mean(axis=0);std=data['features'].std(axis=0)+1e-3
    model=Encoder(data['features'].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    parameter_count=sum(p.numel() for p in model.parameters())
    assert 1_000_000<=parameter_count<=5_000_000
    seen=np.zeros(len(y),bool);losses=[];curves=[];pos_order=rng.permutation(positives);pos_cursor=0
    batch=CONFIG['batch_size'];half=batch//2;start_step=1;elapsed_before=0.
    checkpoint=path.with_name(path.stem+'.checkpoint.pt')
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=True)
        if 'numpy_rng_state' in ck:
            assert ck['source']==source and ck['fraction']==fraction and ck['config']==CONFIG
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer_state'])
            rng.bit_generator.state=ck['numpy_rng_state'];torch.set_rng_state(ck['torch_rng_state'].cpu())
            torch.cuda.set_rng_state(ck['cuda_rng_state'].cpu())
            pos_order=ck['pos_order'].cpu().numpy();pos_cursor=ck['pos_cursor'];seen=ck['seen'].cpu().numpy()
            curves=ck['curves'];losses=ck['losses'];elapsed_before=ck['elapsed_seconds'];start_step=ck['step']+1
            print(f'Resuming {source} fraction {fraction} at step {start_step}',flush=True)
    start=time.perf_counter();torch.cuda.synchronize()
    for step in range(start_step,steps+1):
        pi=[]
        while len(pi)<half:
            size=min(half-len(pi),len(pos_order)-pos_cursor)
            pi.extend(pos_order[pos_cursor:pos_cursor+size]);pos_cursor+=size
            if pos_cursor==len(pos_order):pos_order=rng.permutation(positives);pos_cursor=0
        # Uniform groups avoid repeated long negative tracks; weights restore
        # each negative node's natural source-population contribution.
        gi=rng.integers(0,len(starts),size=half)
        ni=neg_order[starts[gi]+(rng.random(half)*lengths[gi]).astype(int)]
        ids=np.r_[np.array(pi,int),ni];seen[ids]=True
        w=np.r_[np.ones(half),lengths[gi]*len(starts)/len(negative)].astype(np.float32)
        xx,ff=make_batch(data['patches'],data['index'],data['valid'],data['features'],ids,means,std,device,augment=True)
        target=torch.from_numpy(y[ids].astype(np.float32)).to(device);weight=torch.from_numpy(w).to(device)
        lr=CONFIG['learning_rate']*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(xx,ff);loss=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite temporal loss {source} {fraction} step {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        losses.append(float(loss.detach()))
        if torch.cuda.max_memory_allocated()>20*1024**3:raise RuntimeError('Temporal fit exceeded 20 GiB allocated cap')
        if step%100==0 or step==steps:
            torch.cuda.synchronize();elapsed=elapsed_before+time.perf_counter()-start
            curve=dict(source=source,model='temporal_image',positive_fraction=fraction,steps=step,
                positives=int((seen&(y==1)).sum()),positive_groups=len(np.unique(data['groups'][seen&(y==1)])),
                available_positives=len(positives),loss=float(np.mean(losses[-100:])),seconds=elapsed,
                gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated(),scope='source_training_balanced_weighted_BCE')
            curves.append(curve);print(curve,flush=True)
            write_json(path.with_suffix('.progress.json'),curve)
            if elapsed+prior_gpu_seconds>16*3600:raise RuntimeError('Aggregate temporal fit resource budget exhausted')
        if step%1000==0 or step==steps:
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),step=step,
                source=source,fraction=fraction,mean=torch.from_numpy(means),std=torch.from_numpy(std),
                source_prevalence=float(y.mean()),features=data['features'].shape[1],config=CONFIG,
                numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
                pos_order=torch.from_numpy(pos_order),pos_cursor=pos_cursor,seen=torch.from_numpy(seen),curves=curves,losses=losses,
                elapsed_seconds=elapsed_before+time.perf_counter()-start),checkpoint)
    torch.cuda.synchronize();elapsed=elapsed_before+time.perf_counter()-start
    assert (seen[positives]).all(),'Not all selected positive observations were used'
    torch.save(dict(state_dict=model.state_dict(),step=steps,source=source,fraction=fraction,
        mean=torch.from_numpy(means),std=torch.from_numpy(std),source_prevalence=float(y.mean()),
        features=data['features'].shape[1],config=CONFIG),path)
    checkpoint=path.with_name(path.stem+'.checkpoint.pt')
    # Keep the resumable checkpoint and compact inference checkpoint, both owned v2.
    record=dict(source=source,fraction=fraction,steps=steps,parameter_count=parameter_count,
        source_candidates=len(y),selected_positive_groups=len(selected_groups),all_positive_groups=len(positive_groups),
        selected_positives=len(positives),unique_positive_observations_seen=int((seen&(y==1)).sum()),
        unique_negative_observations_seen=int((seen&(y==0)).sum()),all_selected_positives_used=True,
        gpu_synchronized_seconds=elapsed,gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        source_prevalence=float(y.mean()),training_curves=curves,sha256=sha(path),config=CONFIG)
    write_json(receipt,record);del model,optimizer;torch.cuda.empty_cache();return record


def run(args):
    assert torch.cuda.is_available();print(torch.__version__,torch.cuda.get_device_name(0),flush=True)
    CONFIG['steps']=args.steps
    if (OUT/'temporal_config.json').exists():
        saved=read_json(OUT/'temporal_config.json')
        assert {k:v for k,v in saved.items() if k!='created'}==CONFIG,'Locked temporal fit configuration changed'
    if (OUT/'temporal_model_lock.json').exists():
        lock=read_json(OUT/'temporal_model_lock.json')
        assert lock['full_steps']==args.steps
        assert all(sha(OUT/p)==h for p,h in lock['models'].items()),'Locked temporal model changed'
        print('Temporal fits already complete; model/config hashes verified',flush=True)
        return
    write_json(OUT/'temporal_config.json',dict(created=now(),**CONFIG),immutable=True) if not (OUT/'temporal_config.json').exists() else None
    stage('V250','running',scope='Full temporal native-selector study',steps_per_fit=args.steps)
    all_results=[]
    for source in ['44b6','6bba']:
        data=source_data(source)
        for fraction in CONFIG['positive_group_fractions']:
            all_results.append(fit_one(source,fraction,data,args.steps,sum(r['gpu_synchronized_seconds'] for r in all_results)))
        del data;gc.collect()
    curves=pd.DataFrame([c for r in all_results for c in r['training_curves']])
    curves.to_csv(OUT/'temporal_learning_curves.csv',index=False)
    old=pd.read_csv(OUT/'learning_curves.csv');old=old[old.model!='temporal_image']
    pd.concat([old,curves],ignore_index=True).to_csv(OUT/'learning_curves.csv',index=False)
    write_json(OUT/'temporal_model_lock.json',dict(created=now(),models={str(p.relative_to(OUT)):sha(p) for p in (OUT/'temporal_models').glob('*/*.pt') if 'checkpoint' not in p.name},
        both_directional_full_models_frozen_before_comparative_scoring=True,full_steps=args.steps,
        gpu_synchronized_hours=sum(r['gpu_synchronized_seconds'] for r in all_results)/3600,
        full_fits=[{k:v for k,v in r.items() if k not in ['training_curves','config']} for r in all_results]),immutable=True)
    stage('V250','models_frozen',fits=len(all_results),gpu_hours=sum(r['gpu_synchronized_seconds'] for r in all_results)/3600)
