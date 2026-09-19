"""Fresh source-fit C00 optimizer, equal supported query pools, resumable state.

Admission requires a completed scientific execution lock. This module never
opens target images, calibration inputs, or historical checkpoints.
"""
from pathlib import Path
import fcntl
import os
import random
import sys
import tempfile
import time
import gc
from .common import ARCH, DATA, WORK, RESULTS, Blocked, CONFIG, now, read, write, sha


def run(source,seed,*,resume=True,proof_name=None,stop_after=None):
    lock_manifest=RESULTS/'execution_lock.json'
    if proof_name is not None:
        if proof_name not in ('continuous','resumed'):raise Blocked('Invalid validation fixture')
        # Deliberately tiny sampler/optimizer correctness fixture, never a study fit.
        lock_spec=dict(upstream_updates=20,identity='v11-production-resume-fixture-v1',
                       implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    elif not lock_manifest.exists() or read(lock_manifest).get('status')!='locked':
        raise Blocked('Production training requires passed source readiness and a measured execution lock')
    else:lock_spec=read(lock_manifest)
    if proof_name is None:
        from .provenance import digest
        if lock_spec['identity']!=digest({k:v for k,v in lock_spec.items() if k!='identity'}):raise Blocked('Execution lock identity changed')
        if sha(WORK/'source_partitions.json')!=lock_spec['source_partitions_sha256']:raise Blocked('Source partition changed')
    horizon=lock_spec['upstream_updates']
    if horizon not in (24000,16000,8000) and proof_name is None:raise Blocked('Unregistered update horizon')
    for name,digest in lock_spec['implementation_sha256'].items():
        if sha(Path(__file__).parent/name)!=digest:raise Blocked('Implementation differs from lock: '+name)
    split=read(WORK/'source_partitions.json')[source]
    folder=(WORK/'resume_proof'/source/str(seed)/proof_name if proof_name else WORK/'fits'/source/str(seed)/'upstream')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.json').exists():
        final=read(folder/'final.json')
        if final['weights_sha256']!=sha(folder/'final.pt'):raise Blocked('Final weight integrity failure')
        return final
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    ownership=(folder/'owner.lock').open('a+')
    try:fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as e:raise Blocked('Another process owns this cell') from e
    from .resources import Resources,ROOT
    resources=Resources('upstream' if proof_name is None else 'pilot',source,seed)
    cache=WORK/'preprocessed'/source
    if not (cache/'manifest.json').exists():raise Blocked('Complete parity-approved source preprocessing required')
    if proof_name is None and sha(cache/'manifest.json')!=lock_spec['source_cache_manifest_sha256'][source]:raise Blocked('Source cache provenance changed')
    input_paths=[cache]
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=input_paths,outputs=[folder,ROOT],
                  code_roots=[Path(__file__).parent,ARCH/'src',Path(sys.prefix),Path(sys.base_prefix),
                              *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    import psutil
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from .data import pair,supported_incoming,SPACING
    from .graphs import peaks
    from .upstream import Upstream,detection_loss,incoming_loss,learning_rate
    from .preprocess import SourcePairs
    reader=SourcePairs(cache,split['fit'])
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    # CUDA's lazy initialization replays pending manual_seed calls. Initialize
    # before restoring a saved generator state so those callbacks cannot reset it.
    torch.cuda.init()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    rng=np.random.default_rng(seed)
    model=Upstream(ARCH);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    step=0;charged=0.;last_saved=0;last_checkpoint=time.monotonic();lock_hash=lock_spec['identity']
    if resume and (folder/'resume.pt').exists():
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        if state['lock_identity']!=lock_hash or state['source']!=source or state['seed']!=seed:raise Blocked('Resume lineage mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];charged=state['charged_gpu_lease_seconds']
        rng.bit_generator.state=state['sampler'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng'])
        last_saved=step
    elif (folder/'resume.pt').exists():raise Blocked('Existing state may not be overwritten without resume')
    history=folder/'history.jsonl'
    if history.exists():
        import json
        rows=[json.loads(line) for line in history.read_text().splitlines() if line]
        uncommitted=[row for row in rows if row['step']>step]
        if uncommitted:
            write(folder/'interrupted_updates'/f'{time.time_ns()}.json',uncommitted,immutable=True)
            history.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step']<=step))
    def save():
        nonlocal last_saved,last_checkpoint
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,source=source,seed=seed,
                        lock_identity=lock_hash,charged_gpu_lease_seconds=charged,sampler=rng.bit_generator.state,
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),
                        scheduler=dict(one_based_step=step,horizon=horizon),scaler=None,precision='bfloat16')
        torch.save(checkpoint,folder/'resume.tmp.pt');(folder/'resume.tmp.pt').replace(folder/'resume.pt')
        last_saved=step;last_checkpoint=time.monotonic()
    def matched(points,gt):
        if not len(points) or not len(gt):return np.empty((0,5),np.float64)
        distance=cdist(points*SPACING,gt[:,2:]*SPACING)
        # Big null cost prioritizes match cardinality, then minimizes distance.
        cost=np.concatenate([np.where(distance<=7,distance,1e12),np.full((len(points),len(points)),1e6)],axis=1)
        a,b=linear_sum_assignment(cost);keep=(b<len(gt))&(cost[a,b]<1e6)
        return np.column_stack([gt[b[keep],0],gt[b[keep],1],points[a[keep]]])
    try:
        if not (folder/'resume.pt').exists():save()
        while step<horizon:
            # Prepare one real batch outside the GPU lease; no normalization fit.
            specs=[(split['fit'][int(rng.integers(len(split['fit'])))],int(rng.integers(99))) for _ in range(8)]
            io_start=time.monotonic();batches=[reader.pair(n,t) for n,t in specs];io_seconds=time.monotonic()-io_start
            tensors={k:torch.from_numpy(np.stack([b[k] for b in batches])) for k in ('images','targets','positives','backgrounds')}
            if psutil.virtual_memory().available<10*2**30:raise Blocked('Host available memory below floor')
            manager=resources.lease(9*2**30);lease_record=manager.__enter__();lease=time.monotonic()
            try:
                model.cuda()
                for state in optimizer.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor) and k!='step':state[k]=v.cuda()
                tensors={k:v.cuda() for k,v in tensors.items()}
                optimizer.zero_grad(set_to_none=True);lr=learning_rate(step+1,horizon)
                for g in optimizer.param_groups:g['lr']=lr
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,features=model(tensors['images'])
                    det=detection_loss(logits,tensors['targets'],tensors['positives'],tensors['backgrounds'])
                    all_terms=[];quotas=[]
                    for i,batch in enumerate(batches):
                        ga,gb=batch['query'];y=supported_incoming(ga[:,0],gb[:,0],batch['edges'])
                        coords=[]
                        for gt in (ga,gb):
                            jitter=rng.normal(size=(len(gt),3));jitter/=np.maximum(np.linalg.norm(jitter,axis=1,keepdims=True),1e-8)
                            jitter*=rng.uniform(0,1.5,(len(gt),1))
                            coords.append(np.clip(gt[:,2:]+jitter/SPACING,[0,0,0],[63,255,255]).astype(np.float32))
                        supported=np.flatnonzero((y>=0).any(0)) if len(ga) and len(gb) else np.empty(0,int)
                        predicted=None
                        if step>=horizon//10:
                            pa=matched(peaks(logits[i,0].detach().float().cpu().numpy())[0],ga)
                            pb=matched(peaks(logits[i,1].detach().float().cpu().numpy())[0],gb)
                            py=supported_incoming(pa[:,0],pb[:,0],batch['edges'])
                            pg=np.flatnonzero((py>=0).any(0)) if len(pa) and len(pb) else np.empty(0,int)
                            count=min(len(pg),len(supported))
                            if count:
                                chosen=rng.choice(pg,count,replace=False);selected=rng.choice(supported,count,replace=False)
                                y[:,np.setdiff1d(supported,selected)]=-1;py[:,np.setdiff1d(pg,chosen)]=-1
                                predicted=(pa[:,2:].astype(np.float32),pb[:,2:].astype(np.float32),py)
                                quotas.append(dict(supervised=count,proposal=count,fallback=False))
                        if predicted is None:quotas.append(dict(supervised=len(supported),proposal=0,fallback=True))
                        pools=[(*coords,y)]+([predicted] if predicted is not None else [])
                        for a,b,labels in pools:
                            if len(a) and len(b) and (labels>=0).any():
                                a=torch.from_numpy(a).cuda();b=torch.from_numpy(b).cuda();labels=torch.from_numpy(labels).cuda()
                                z=model.association(features[i,0],features[i,1],a,b,batch['time'])
                                for j in range(labels.shape[1]):
                                    if (labels[:,j]>=0).any():all_terms.append(incoming_loss(z[:,j:j+1],labels[:,j:j+1]))
                    ass=torch.stack(all_terms).mean() if all_terms else features.sum()*0
                    loss=det+ass
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise Blocked('Nonfinite full upstream update')
                optimizer.step();torch.cuda.synchronize();step+=1
                row=dict(step=step,lr=lr,loss=float(loss.detach()),detection=float(det.detach()),association=float(ass.detach()),
                         gradient_norm=float(norm),groups=quotas,samples=specs,io_seconds=io_seconds,
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=source,seed=seed)
                # Loop-local tensors can retain a feature/attention autograd
                # graph after backward. Drop them before returning the lease.
                z=a=b=labels=None
                del logits,features,loss,det,ass,all_terms,tensors
            finally:
                model.cpu()
                for state in optimizer.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor):state[k]=v.cpu()
                gc.collect();torch.cuda.empty_cache()
                manager.__exit__(*sys.exc_info());elapsed=lease_record['seconds'];charged+=elapsed
            if elapsed>60:raise Blocked('One upstream update exceeded the lease bound')
            row.update(gpu_lease_seconds=elapsed,cumulative_gpu_lease_seconds=charged,
                       released_allocated_bytes=torch.cuda.memory_allocated(),released_reserved_bytes=torch.cuda.memory_reserved())
            with (folder/'history.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
            write(folder/'progress.json',dict(status='running',step=step,horizon=horizon,charged_gpu_lease_seconds=charged,pid=os.getpid(),updated_utc=now()))
            if step-last_saved>=250 or time.monotonic()-last_checkpoint>=300 or step in (horizon//10,horizon):save()
            if stop_after is not None and step>=stop_after and step<horizon:
                save();return dict(status='validation_checkpoint_saved',step=step,source=source,seed=seed)
        torch.save(dict(model=model.state_dict(),source=source,seed=seed,step=step,lock_identity=lock_hash),folder/'final.pt')
        final=dict(status='validation_fit' if proof_name else 'trained',source=source,seed=seed,updates=step,weights_sha256=sha(folder/'final.pt'),
                   charged_gpu_lease_seconds=charged,guard=guard,lock_identity=lock_hash)
        write(folder/'final.json',final);write(folder/'progress.json',{**final,'status':'complete','updated_utc':now()});return final
    except Exception as exc:
        write(folder/'progress.json',dict(status='failed',step=step,horizon=horizon,
              durable_step=last_saved,charged_gpu_lease_seconds=charged,
              failure_type=type(exc).__name__,reason=str(exc),updated_utc=now()))
        raise
    finally:
        ownership.close();resources.close()
