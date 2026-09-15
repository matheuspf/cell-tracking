"""Resumable scratch native training with one embryo excluded from every read."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from .common import REPO,WORK,OUT,GPU_LOCK,read,write,save,sha,now,modules,source_guard,source_hashes,verify_sources
from .data import dataset
from .frames import CachedDataset


def prepare():
    fits=[]
    for source in ('44b6','6bba'):
        index=read(WORK/'metadata'/source/'index.json')
        fits.append({'source':source,'target':{'44b6':'6bba','6bba':'44b6'}[source],
                     'clips':[r['dataset'] for r in index['clips']],'windows':sum(r['windows'] for r in index['clips']),
                     'metadata_index_sha256':sha(WORK/'metadata'/source/'index.json'),
                     'frames_index_sha256':sha(WORK/'frames'/source/'index.json')})
    plan={'created_utc':now(),'fits':fits,'seed':314159,'epochs':400,'batch_size':8,'learning_rate':1e-4,'weight_decay':.01,
          'det_loss_weight':1.,'det_neg_weight':.01,'pool_kernel_um':5.,'initialization':'Random entire native encoder, detector and association head; no released weights, cached incumbent proposals, teachers or embeddings.',
          'architecture':{'unet_out_channels':32,'unet_layers':[32,64,128],'downsample':[1,4,4],'window_size':2},
          'selection':'Fixed final epoch400 for each direction. No target evaluation until both final checkpoints exist. Intermediate source-only loss and checkpoints are operational diagnostics.',
          'precision':'Original FP16 image-storage round trip into FP32 training; original batch8 including BatchNorm. No microbatch substitution.',
          'augmentations':'Original additive brightness and three-axis flips; seeded by(seed,epoch,window_index) so resume reproduces augmentations. Original default_rng was unseeded.',
          'recipe_differences':['Source-only data instead of all199 clips.','Fixed final epoch rather than best training-overlapping monitor.','Explicit reproducible augmentation RNG.','Exact-value FP32 normalized memmap cache replaces repeated Zarr decompression.','Eight-update GPU leases and checkpoint/resume wrapper; same upstream per-batch loss and updates.'],
          'eligibility':'All original upstream-eligible source windows; requires GT nodes in both frames. This differs from the refiner five-timepoint query sample and is reported explicitly.',
          'native_source':source_hashes(),'implementation':{str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('data.py'),Path(__file__).with_name('frames.py'))},
          'gpu_lease_updates':8,'process_memory_limit_gib':20.,'checkpoint_every_updates':50,
          'milestone_epochs':[1,10,50,100,200,400],'target_label_access':False}
    path=OUT/'native-training-plan.json'
    if path.exists():
        previous=read(path);assert {k:v for k,v in previous.items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
    else:write(path,plan)
    print(path,flush=True)


def optimizer_device(optimizer,device):
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor) and k!='step':state[k]=v.to(device)


def _run(source):
    training,_=modules();source_guard(source)
    from tools.detector_screen.cellpose_adapter import gpu_lock
    plan=read(OUT/'native-training-plan.json');fit=next(r for r in plan['fits'] if r['source']==source)
    for p,h in plan['implementation'].items():assert sha(p)==h,p
    verify_sources(plan['native_source'])
    assert sha(WORK/'metadata'/source/'index.json')==fit['metadata_index_sha256']
    assert sha(WORK/'frames'/source/'index.json')==fit['frames_index_sha256']
    for record in read(WORK/'frames'/source/'index.json')['records']:assert sha(record['path'])==record['sha256']
    original=dataset(fit['clips']);assert len(original)==fit['windows']
    seed=plan['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(min(1.,plan['process_memory_limit_gib']*2**30/torch.cuda.get_device_properties(0).total_memory))
    model=training.UNetNodeTransformer(training.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),32,32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    folder=WORK/'models/native'/f'source-{source}-seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    state_path=folder/'resume.pt';plan_sha=sha(OUT/'native-training-plan.json')
    epoch=1;offset=0;updates=0;history=[];accumulated_seconds=0.;samples=0;edge_sum=det_sum=0.;last_saved=0
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=True);assert state['plan_sha256']==plan_sha and state['source']==source
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        epoch=state['epoch'];offset=state['offset'];updates=state['updates'];history=state['history'];accumulated_seconds=state['seconds']
        samples=state['epoch_samples'];edge_sum=state['epoch_edge_sum'];det_sum=state['epoch_det_sum']
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng']);last_saved=updates
    else:
        save(folder/'initial.pt',{'model':model.state_dict(),'seed':seed,'source':source,'initialization':'random','plan_sha256':plan_sha})
    if (folder/'final.json').exists():
        final=read(folder/'final.json');assert final['sha256']==sha(folder/'final.pt') and final['plan_sha256']==plan_sha;print('Already complete',source,flush=True);return
    started=time.perf_counter();queue_seconds=0.;peak=0
    def checkpoint():
        save(state_path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'source':source,'epoch':epoch,'offset':offset,
                        'updates':updates,'history':history,'seconds':accumulated_seconds+time.perf_counter()-started-queue_seconds,
                        'epoch_samples':samples,'epoch_edge_sum':edge_sum,'epoch_det_sum':det_sum,'plan_sha256':plan_sha,
                        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    while epoch<=plan['epochs']:
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch,911])).permutation(len(original)).tolist()
        cached=CachedDataset(original,source,seed,epoch)
        # Worker seeding uses a separate generator so resuming a loader cannot change model dropout RNG.
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(Subset(cached,order[offset:]),batch_size=8,shuffle=False,num_workers=2,prefetch_factor=2,generator=generator)
        iterator=iter(loader)
        while offset<len(original):
            batches=[]
            for _ in range(plan['gpu_lease_updates']):
                try:batches.append(next(iterator))
                except StopIteration:break
            if not batches:break
            lease_started=time.perf_counter()
            with gpu_lock(GPU_LOCK) as queued:
                queue_seconds+=queued;torch.cuda.reset_peak_memory_stats();model.to('cuda');optimizer_device(optimizer,'cuda')
                try:
                    edge,det=training.train_epoch(model,batches,optimizer,torch.device('cuda'),det_loss_weight=1.,det_neg_weight=.01,pool_kernel_um=5.)
                    assert np.isfinite(edge) and np.isfinite(det)
                    torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_reserved())
                finally:
                    model.to('cpu');optimizer_device(optimizer,'cpu');torch.cuda.empty_cache()
            n=sum(len(b['imgs']) for b in batches);samples+=n;edge_sum+=edge*n;det_sum+=det*n;offset+=n;updates+=len(batches)
            progress={'updated_utc':now(),'source':source,'target':fit['target'],'seed':seed,'epoch':epoch,'target_epochs':plan['epochs'],
                      'epoch_windows':offset,'windows_per_epoch':len(original),'updates':updates,'latest_edge_loss':edge,'latest_detection_loss':det,
                      'seconds_excluding_queue':accumulated_seconds+time.perf_counter()-started-queue_seconds,'gpu_queue_seconds_this_process':queue_seconds,
                      'latest_seconds_per_update':(time.perf_counter()-lease_started-queued)/len(batches),
                      'peak_reserved_bytes':peak,'target_evaluated':False,'plan_sha256':plan_sha}
            write(WORK/f'native-{source}-progress.json',progress)
            with (folder/'lease-history.jsonl').open('a') as history_file:history_file.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
            if updates-last_saved>=plan['checkpoint_every_updates'] or last_saved==0:
                checkpoint();last_saved=updates
        history.append({'epoch':epoch,'windows':samples,'edge_loss':edge_sum/max(samples,1),'detection_loss':det_sum/max(samples,1),'updates':updates})
        if epoch in plan['milestone_epochs']:
            save(folder/f'epoch-{epoch:03}.pt',{'model':model.state_dict(),'config':plan['architecture'],'source':source,'target':fit['target'],'epoch':epoch,'plan_sha256':plan_sha})
        epoch+=1;offset=0;samples=0;edge_sum=det_sum=0.;checkpoint();last_saved=updates
    save(folder/'final.pt',{'model':model.state_dict(),'config':plan['architecture'],'source':source,'target':fit['target'],'epochs':plan['epochs'],'plan_sha256':plan_sha})
    receipt={'created_utc':now(),'source':source,'target':fit['target'],'seed':seed,'epochs':plan['epochs'],'updates':updates,
             'sha256':sha(folder/'final.pt'),'plan_sha256':plan_sha,'history':history,'target_labels_read':False,'initializer':'random entire model'}
    write(folder/'final.json',receipt);write(OUT/f'native-{source}-final.json',receipt)


def run(source):
    import fcntl
    lock=WORK/'locks'/f'native-{source}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This source already has an active training worker: '+source)
        return _run(source)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('--source',choices=('44b6','6bba'));a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        if a.source is None:p.error('--source required')
        run(a.source)
