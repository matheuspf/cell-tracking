"""Real batch-eight sparse-safe upstream optimizer and exact-resume pilot.

This pilot is never a retained model and cannot unlock target inference/scoring.
"""
import fcntl
import os
from pathlib import Path
import sys
import time
from .common import ARCH, DATA, WORK, RESULTS, Blocked, now, read, write, sha


def run(source, seed, data=DATA, architecture=ARCH, overfit_updates=0):
    split = read(WORK/'source_partitions.json')[source]
    folder = WORK/('overfit' if overfit_updates else 'pilot')/source/str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'receipt.json').exists():
        previous = read(folder/'receipt.json')
        failures = folder/'attempts'
        failures.mkdir(exist_ok=True)
        attempt=failures/str(time.time_ns());attempt.mkdir()
        write(attempt/'receipt.json', previous, immutable=True)
        import shutil
        for old_file in ('resume.pt','overfit-resume.pt'):
            if (folder/old_file).exists():shutil.copy2(folder/old_file,attempt/old_file)
    import tempfile
    (folder/'tmp').mkdir(exist_ok=True)
    tempfile.tempdir = str((folder/'tmp').resolve())
    sys.dont_write_bytecode = True
    from .guard import install
    permitted = [data/'train'/f'{n}{suffix}' for n in split['fit'] for suffix in ('.zarr','.geff')]
    code = [Path(__file__).parent, architecture/'src', Path(sys.prefix), Path(sys.base_prefix)]
    code += [Path(p) for p in sys.path if 'site-packages' in p]
    lockpath = Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock')
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        lock.close()
        raise Blocked('Another cooperative GPU lease is active') from e
    # Numerical imports occur only after manifest and filesystem read guards.
    report = RESULTS/f'{"overfit" if overfit_updates else "pilot"}-{source}-{seed}.json'
    guard = install(inputs=permitted, outputs=[folder, report, report.with_name(report.name+f'.{os.getpid()}.tmp')], code_roots=code)
    import numpy as np
    import psutil
    import torch
    from .data import pair, supported_incoming, SPACING
    from .upstream import Upstream, detection_loss, incoming_loss, learning_rate
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(seed)
    batches = []
    started = time.monotonic()
    record = dict(source=source, seed=seed, status='started', started_utc=now(),
                  initialization='random entire encoder/detector/association', batch=8,
                  production_updates=0, target_labels_read=False, guard=guard,
                  real_source_pairs=[], steps=[])
    write(folder/'receipt.json', record)
    lease_start = None
    charged = 0.
    try:
        for _ in range(8):
            name = split['fit'][int(rng.integers(len(split['fit'])))]
            t = int(rng.integers(99))
            if overfit_updates:
                # Fixed lexicographically first source-fit division fixture,
                # selected before any v11 production fit or target evaluation.
                name,t={'44b6':('44b6_12dfb391',66),'6bba':('6bba_05db0fb1',24)}[source]
                if name not in split['fit']:raise Blocked('Overfit fixture is not source-fit')
            batches.append(pair(data, name, t))
            record['real_source_pairs'].append([name,t])
        record['data_seconds'] = time.monotonic()-started
        cpu = {k:torch.from_numpy(np.stack([b[k] for b in batches])) for k in
               ('images','targets','positives','backgrounds')}
        unknown = ~(cpu['positives']|cpu['backgrounds'])
        probe = torch.zeros_like(cpu['targets'], requires_grad=True)
        detection_loss(probe,cpu['targets'],cpu['positives'],cpu['backgrounds']).backward()
        record['unknown_detection_gradient_max'] = float(probe.grad[unknown].abs().max())
        record['positive_voxels'] = int(cpu['positives'].sum())
        record['background_voxels'] = int(cpu['backgrounds'].sum())
        free,total = torch.cuda.mem_get_info()
        allowed = min(20*2**30-(total-free), free-2*2**30)
        if allowed <= 0 or psutil.virtual_memory().available < 10*2**30:
            raise Blocked('Insufficient device or host memory at admission')
        torch.cuda.set_per_process_memory_fraction(allowed/total)
        model = Upstream(architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        lease_start = time.monotonic()
        model.cuda()
        gpu = {k:v.cuda() for k,v in cpu.items()}
        torch.cuda.reset_peak_memory_stats()
        queries = []
        positive = negative = unobserved = 0
        for batch in batches:
            a,b = batch['query']
            y = supported_incoming(a[:,0],b[:,0],batch['edges'])
            positive += int((y==1).sum()); negative += int((y==0).sum()); unobserved += int((y==-1).sum())
            coords = []
            for nodes in (a,b):
                jitter = rng.normal(size=(len(nodes),3))
                jitter /= np.maximum(np.linalg.norm(jitter,axis=1,keepdims=True),1e-8)
                jitter *= rng.uniform(0,1.5,size=(len(nodes),1))
                coords.append(torch.from_numpy(np.clip(nodes[:,2:]+jitter/SPACING,[0,0,0],[63,255,255]).astype(np.float32)).cuda())
            queries.append((*coords,torch.from_numpy(y).cuda(),batch['time']))
        record['association_labels'] = dict(positive=positive,supported_negative=negative,unknown=unobserved)
        if positive == 0 or negative == 0:
            raise Blocked('Uniform real pilot batch lacks positive or supported-negative association supervision')
        def step(index):
            tic = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(index,overfit_updates) if overfit_updates else learning_rate(index,8000)
            for group in optimizer.param_groups: group['lr'] = lr
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits, features = model(gpu['images'])
                det = detection_loss(logits,gpu['targets'],gpu['positives'],gpu['backgrounds'])
                terms=[]
                for i,(a,b,y,t) in enumerate(queries):
                    if len(a) and len(b) and (y>=0).any():
                        terms.append(incoming_loss(model.association(features[i,0],features[i,1],a,b,t),y))
                ass = torch.stack(terms).mean() if terms else features.sum()*0
                loss=det+ass
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise Blocked('Nonfinite real optimizer loss/gradient')
            optimizer.step()
            torch.cuda.synchronize()
            free,total = torch.cuda.mem_get_info()
            rss=psutil.Process().memory_info().rss
            row=dict(step=index,loss=float(loss.detach()),detection=float(det.detach()),association=float(ass.detach()),
                     gradient_norm=float(grad),lr=lr,seconds=time.monotonic()-tic,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),total_device_used_bytes=total-free,
                     rss_bytes=rss,host_available_bytes=psutil.virtual_memory().available)
            record['steps'].append(row)
            if total-free > 20*2**30 or free < 2*2**30 or rss > 44*2**30 or psutil.virtual_memory().available < 10*2**30:
                raise Blocked('Measured pilot exceeds registered resource floor/ceiling')
            if time.monotonic()-lease_start > 55:
                raise Blocked('Pilot needs a shorter work unit to preserve 60-second GPU leases')
            return row
        step(1)
        # Torch state dicts contain tensors only; local checkpoint owns all parents.
        checkpoint=folder/'resume.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),step=1,
                        source=source,seed=seed,initialization='random',sampler=rng.bit_generator.state),checkpoint)
        first=step(2)
        saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state=torch.load(checkpoint,map_location='cuda',weights_only=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        replay=step(2)
        record['resume'] = dict(checkpoint_sha256=sha(checkpoint),
                               exact_weights=all(torch.equal(saved[k],v.cpu()) for k,v in model.state_dict().items()),
                               exact_loss=first['loss']==replay['loss'],replayed_step=2,
                               maximum_parameter_difference=max(float((saved[k]-v.cpu()).abs().max()) for k,v in model.state_dict().items()),
                               unequal_tensors=[k for k,v in model.state_dict().items() if not torch.equal(saved[k],v.cpu())])
        record['status']='real_optimizer_resume_passed' if all(record['resume'][k] for k in ['exact_weights','exact_loss']) else 'resume_failed'
        if overfit_updates and record['status']=='real_optimizer_resume_passed':
            record['overfit_horizon']=overfit_updates
            record['overfit_sampling']='same source-fit division pair repeated eight times; diagnostic only'
            for index in range(3,overfit_updates+1):
                if time.monotonic()-lease_start>35:
                    model.cpu()
                    for state in optimizer.state.values():
                        for key,value in state.items():
                            if isinstance(value,torch.Tensor):state[key]=value.cpu()
                    gpu={k:v.cpu() for k,v in gpu.items()}
                    queries=[(a.cpu(),b.cpu(),y.cpu(),t) for a,b,y,t in queries]
                    torch.cuda.empty_cache();charged+=time.monotonic()-lease_start;lease_start=None
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    # Persist full resumable state at every short lease boundary.
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                                    step=index-1,source=source,seed=seed),folder/'overfit-resume.pt')
                    fcntl.flock(lock,fcntl.LOCK_EX);lease_start=time.monotonic()
                    model.cuda();gpu={k:v.cuda() for k,v in gpu.items()}
                    queries=[(a.cuda(),b.cuda(),y.cuda(),t) for a,b,y,t in queries]
                    for state in optimizer.state.values():
                        for key,value in state.items():
                            if isinstance(value,torch.Tensor) and key!='step':state[key]=value.cuda()
                step(index)
                if index%25==0:print(f'{source} diagnostic overfit {index}/{overfit_updates}: {record["steps"][-1]["loss"]:.6f}',flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                            step=overfit_updates,source=source,seed=seed),folder/'resume.pt')
            record['final_checkpoint_sha256']=sha(folder/'resume.pt')
            record['status']='overfit_complete'
    except Exception as e:
        record['status']='blocked'
        record['failure_type']=type(e).__name__
        record['failure']=str(e).replace(str(Path.home()),'<home>')
        raise
    finally:
        # Charge even failed work and restore the cooperative lease.
        if lease_start is not None:
            torch.cuda.synchronize()
            record['charged_gpu_lease_seconds']=charged+time.monotonic()-lease_start
            record['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        else:
            record['charged_gpu_lease_seconds']=charged
        record['wall_seconds']=time.monotonic()-started
        record['finished_utc']=now()
        write(folder/'receipt.json',record)
        write(report,record)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return record
