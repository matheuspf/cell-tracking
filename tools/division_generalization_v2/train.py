"""Fixed 4096 joint-update fits, exact common prefix, deterministic replay."""
from collections import Counter
import copy
import json
from pathlib import Path
import signal
import time
import contextlib

import numpy as np
import torch

from .common import (WORK, RESULTS, read_json, write_json, append_json, atomic_torch_save,
                     sha, digest, code_hashes)
from .contracts import UPDATES, PREFIX, LR, lr_factor, scientific_recipe
from .dataset import SourceDataset
from .model import ActionModel, loss
from .resources import Lease, Monitor


def optimizer_to(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):
                state[key]=value.to(device)


def state(model,optimizer,step,dataset,recipe):
    return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer=copy.deepcopy(optimizer.state_dict()),step=step,recipe=recipe,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if model.image else None,
                numpy_rng=np.random.get_state(),visits=dict(dataset.visits),anchor_visits=dict(dataset.anchor_visits),
                mined=list(dataset.mined),data_order='Stateless seed/step/slot plus explicit mined list')


def restore(saved,model,optimizer,dataset):
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'])
    np.random.set_state(saved['numpy_rng'])
    dataset.visits=Counter(saved['visits'])
    dataset.anchor_visits=Counter(saved['anchor_visits'])
    dataset.mined=list(saved['mined'])
    return saved['step']


@contextlib.contextmanager
def emergency_checkpoint(get_state,folder):
    try:
        yield
    except BaseException:
        # Preserve the last fully completed update after any cooperative
        # resource/implementation interruption; the previous milestone survives.
        atomic_torch_save(get_state(),folder/'interrupted.pt')
        raise


def diagnostic_keys(dataset):
    # Fixed whole-group panels, same across steps, arms and seeds.
    groups=sorted(dataset.groups,key=lambda k:digest(['fixed-diagnostic',k]))[:32]
    return [min(dataset.groups[g],key=lambda k:digest(k)) for g in groups]


@torch.no_grad()
def diagnose(model,dataset,keys,device,amp):
    model.eval();totals=Counter()
    for key in keys:
        b,s=dataset.batch(dict(key=key),device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
            out=model([b],s[None] if s is not None else None)[0]
        _,terms=loss(out,b)
        totals.update({k:float(v) for k,v in terms.items()})
    model.train()
    return {k:v/max(1,len(keys)) for k,v in totals.items()}


def train_step(model,optimizer,dataset,step,seed,arm,device,amp,microbatch=4):
    started=time.monotonic();loader=forward=backward=transfer=0.
    optimizer.zero_grad(set_to_none=True)
    samples=dataset.samples(step,seed,arm)
    samples.sort(key=lambda s:(dataset.anchors[s['key']]['dataset'],dataset.anchors[s['key']]['time'],s['key']))
    totals=Counter()
    loader_parts=Counter()
    # G30 is CPU-vectorized within each decision group. The image arm encodes
    # multiple scenes at once and shares each encoding across every action.
    for start in range(0,32,microbatch):
        ts=time.monotonic()
        chunk=samples[start:start+microbatch]
        batches=[];scenes=[]
        for sample in chunk:
            b,s=dataset.batch(sample,device,training=True)
            loader_parts.update(dataset.last_batch_timing)
            batches.append(b)
            if s is not None:scenes.append(s)
        scene=torch.stack(scenes) if scenes else None
        if device=='cuda':torch.cuda.synchronize()
        loader+=time.monotonic()-ts
        ts=time.monotonic()
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
            outputs=model(batches,scene)
        terms_all=[]
        for out,b,sample in zip(outputs,batches,chunk):
            # Risk is a representative supported-background estimate only in
            # random slots. Positive/mined streams supply ranking/identity.
            weight=2.*sample['risk_design_weight']
            total,terms=loss(out,b,risk_weight=weight)
            terms_all.append(total/32)
            totals.update({k:float(v.detach())/32 for k,v in terms.items()})
        if device=='cuda':torch.cuda.synchronize()
        forward+=time.monotonic()-ts
        ts=time.monotonic()
        torch.stack(terms_all).sum().backward()
        if device=='cuda':torch.cuda.synchronize()
        backward+=time.monotonic()-ts
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for group in optimizer.param_groups:
        group['lr']=LR*lr_factor(step)
    ts=time.monotonic()
    optimizer.step()
    if device=='cuda':torch.cuda.synchronize()
    return dict(step=step+1,lr=LR*lr_factor(step),gradient_norm=float(norm),
        loader_transfer_seconds=loader,encoder_head_seconds=forward,backward_seconds=backward,
        optimizer_seconds=time.monotonic()-ts,compute_seconds=time.monotonic()-started,
        **loader_parts,
        **totals)


def run(args):
    source,arm,seed=args.source,args.arm,args.seed
    if seed not in (20260916,314159):
        raise ValueError('Unregistered seed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    image=arm!='G30';device='cuda' if image else 'cpu'
    profile=read_json(RESULTS/'profile.json')
    validation=read_json(RESULTS/'validation.json')
    sampling=read_json(RESULTS/'sampling_audit.json')[source]['fit']
    parity=read_json(RESULTS/'action_label_parity.json')
    if not validation['literal_zero_path_test'] or not validation['unit_tests_passed']:
        raise ValueError('Production training requires measured active zero and correctness gates')
    if not sampling['negative_only_groups'] or not sampling['positive_groups']:
        raise ValueError('Production training requires representative audited groups')
    if parity['status']!='measured':raise ValueError('Complete-action full scorer parity not established')
    amp=bool(image and profile['amp_enabled'])
    if image:
        image_gate=read_json(RESULTS/'corrected_image_validation.json')
        model_hash=sha(Path(__file__).with_name('model.py'))
        if image_gate['model_code_sha256']!=model_hash or not all(r['full_clip'] and r['exact_active_zero'] for r in image_gate['clips']):
            raise ValueError('Current image implementation needs full active zero validation')
        if read_json(RESULTS/'profile_provenance.json')['model_code_sha256']!=model_hash:
            raise ValueError('Image implementation changed after the measured source profile')
    dataset=SourceDataset(source,image=image)
    held=SourceDataset(source,'calibration',image=image)
    if not dataset.positive_keys or not dataset.ordinary_keys:
        raise ValueError('Representative positive and negative-only pools are required')
    folder=WORK/'training'/arm/source/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    recipe=dict(scientific_recipe(),source=source,arm=arm,seed=seed,image=image,amp=amp,
                prepared_manifest_sha256=sha(WORK/'source'/source/'manifest.json'),
                input_sha256=sha(RESULTS/'input_manifest.json'),
                model_code_sha256=sha(Path(__file__).with_name('model.py')),
                training_code_sha256=sha(Path(__file__)),
                features_code_sha256=sha(Path(__file__).with_name('features.py')),
                dataset_code_sha256=sha(Path(__file__).with_name('dataset.py')),
                actions_code_sha256=sha(Path(__file__).with_name('actions.py')),
                mining_code_sha256=sha(Path(__file__).with_name('mining.py')),
                scenes_code_sha256=sha(Path(__file__).with_name('scenes.py')))
    write_json(folder/'recipe.json',recipe,immutable=True)
    model=ActionModel(image=image)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    step=0;prefix_sha=None
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        if saved['recipe']!=recipe:raise ValueError('Resume recipe drift; preserve invalid attempt before repair')
        step=restore(saved,model,optimizer,dataset)
    elif arm in ('J_uniform','J_mined'):
        prefix=WORK/'training/prefix'/source/str(seed)/'checkpoint-2048.pt'
        saved=torch.load(prefix,map_location='cpu',weights_only=False)
        if saved['step']!=PREFIX or saved['recipe']['source']!=source or saved['recipe']['seed']!=seed:
            raise ValueError('Common prefix provenance mismatch')
        for key in ('model_code_sha256','training_code_sha256','features_code_sha256','dataset_code_sha256','scenes_code_sha256','actions_code_sha256','mining_code_sha256','prepared_manifest_sha256'):
            if recipe[key]!=saved['recipe'][key]:raise ValueError('Prefix implementation/input drift')
        step=restore(saved,model,optimizer,dataset);prefix_sha=sha(prefix)
        write_json(folder/'common_prefix.json',dict(path=str(prefix),sha256=prefix_sha,
            model_optimizer_rng_data_order_shared=True,shared_updates=PREFIX,
            independent_experiment=False),immutable=True)
    stop=PREFIX if arm=='prefix' else UPDATES
    if args.stop_at is not None:
        # An interruption boundary never changes the locked fit floor.
        if not step<args.stop_at<=8192:raise ValueError('Invalid interruption boundary')
        stop=args.stop_at
    interrupted=[]
    def request_stop(signum,frame):interrupted.append(signum)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,request_stop)
    fixed_fit=diagnostic_keys(dataset);fixed_held=diagnostic_keys(held)
    write_json(folder/'diagnostic_panel.json',dict(fit=fixed_fit,calibration=fixed_held),immutable=True)
    started=time.monotonic();compute=lease_time=wait=checkpoint_time=0.
    with emergency_checkpoint(lambda:state(model,optimizer,step,dataset,recipe),folder),Monitor(folder/'resources.json') as monitor:
        while step<stop and not interrupted:
            if step in (2048,3072) and arm in ('J_uniform','J_mined'):
                from .mining import refresh
                mined=refresh(model,dataset,device,amp,folder/f'mining-{step}.json')
                if arm=='J_mined':dataset.mined=mined
            # A fresh lease block retains model and optimizer on device. It ends
            # at the next update boundary after 30 s, or at a 128-update checkpoint.
            context=Lease(f'train/{arm}/{source}/{seed}',required_gib=4.) if image else __import__('contextlib').nullcontext()
            with context as lease:
                model.to(device);optimizer_to(optimizer,device);model.train()
                block=time.monotonic()
                while step<stop and not interrupted:
                    record=train_step(model,optimizer,dataset,step,seed,arm,device,amp,
                                      microbatch=profile['microbatch_groups'] if image else 8)
                    step+=1;compute+=record['compute_seconds']
                    append_json(folder/'history.jsonl',record)
                    if step%128==0 or step in (1,3072,4096,8192):
                        fit=diagnose(model,dataset,fixed_fit,device,amp)
                        cal=diagnose(model,held,fixed_held,device,amp)
                        append_json(folder/'diagnostics.jsonl',dict(step=step,fit=fit,calibration=cal))
                        print(f'{arm}/{source}/{seed} {step}/{UPDATES}: fit={fit["total"]:.5f} held={cal["total"]:.5f} step={record["compute_seconds"]:.2f}s',flush=True)
                    monitor.check()
                    if step%128==0 or time.monotonic()-block>=30:break
                model.cpu();optimizer_to(optimizer,'cpu')
                if image:torch.cuda.empty_cache()
            if image:
                lease_time+=lease.seconds;wait+=lease.wait_seconds
            if step%128==0 or step==stop or interrupted:
                ts=time.monotonic()
                checkpoint=state(model,optimizer,step,dataset,recipe)
                atomic_torch_save(checkpoint,folder/'resume.pt')
                if step in (2048,3072,4096,8192):atomic_torch_save(checkpoint,folder/f'checkpoint-{step}.pt')
                checkpoint_time+=time.monotonic()-ts
            write_json(folder/'progress.json',dict(status='running',joint_optimizer_updates=step,
                required_updates=UPDATES,compute_seconds=compute,lease_seconds=lease_time,wait_seconds=wait))
    complete=(arm=='prefix' and step==PREFIX) or (arm!='prefix' and step>=UPDATES)
    receipt=dict(source=source,arm=arm,seed=seed,status='complete' if complete else 'incomplete_resumable',
        joint_optimizer_updates=step,required_updates=PREFIX if arm=='prefix' else UPDATES,
        prefix_sha256=prefix_sha,model_parameters=sum(p.numel() for p in model.parameters()),
        model_sha256=sha(folder/'resume.pt'),compute_seconds=compute,lease_seconds=lease_time,
        wait_seconds=wait,checkpoint_seconds=checkpoint_time,wall_seconds=time.monotonic()-started,
        exposure=dataset.audit(),new_target_used_for_training=False,unknown_as_negative_count=0,
        positive_groups_seen=len(set(dataset.visits)&set(dataset.positive_keys)),
        negative_only_groups_seen=len(set(dataset.visits)&set(dataset.ordinary_keys)),
        literal_zero_path_test=validation['literal_zero_path_test'],
        counterfactual_label_parity=parity['status']=='measured',
        checkpoint_resume_parity=profile['checkpoint_resume_exact'],random_background_stream=True,
        full_source_screen=False,finite_gradient_checks=True)
    write_json(folder/'training_receipt.json',receipt)
    write_json(folder/'progress.json',receipt)
    return receipt
