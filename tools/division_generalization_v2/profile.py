"""Measured source loader/encoder/backward and BF16 admission, never a budget tuner."""
import copy
import time
import numpy as np
import torch

from .common import RESULTS, WORK, write_json, read_json
from .dataset import SourceDataset
from .model import ActionModel, loss
from .resources import Lease, Monitor
from .contracts import scientific_recipe


def run(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260916)
    dataset=SourceDataset(args.source,image=True,allow_partial=True)
    positive=[a for g in dataset.positive_groups.values() for a in g]
    ordinary=[a for g in dataset.ordinary.values() for a in g]
    if not positive or not ordinary:
        raise ValueError('Source profile needs a real positive and negative-only example')
    keys=[positive[0],*ordinary[:3]]
    model=ActionModel(image=True)
    # Exercise trainable encoder backward, not only zero-initialized output heads.
    for h in (model.score,model.risk,model.identity):torch.nn.init.normal_(h.weight,std=.01)
    timings={}
    with Monitor(WORK/'profile/resources.json'),Lease('source_profile',required_gib=4.) as lease:
        model.cuda()
        started=time.monotonic()
        batches,scenes=zip(*(dataset.batch(dict(key=k),'cuda') for k in keys))
        scenes=torch.stack(scenes)
        torch.cuda.synchronize();timings['cold_loader_seconds']=time.monotonic()-started
        started=time.monotonic()
        for k in keys:dataset.batch(dict(key=k),'cuda')
        torch.cuda.synchronize();timings['warm_loader_seconds']=time.monotonic()-started
        initial=copy.deepcopy(model.state_dict())
        outcomes=[]
        for amp in (False,True):
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            started=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):out=model(batches,scenes)
            losses=[loss(o,b)[0] for o,b in zip(out,batches)]
            total=torch.stack(losses).mean()
            torch.cuda.synchronize();forward=time.monotonic()-started
            started=time.monotonic();total.backward();torch.cuda.synchronize()
            gradient=torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])
            outcomes.append(dict(loss=float(total.detach()),gradient=gradient.cpu(),
                finite=bool(torch.isfinite(gradient).all()),forward_seconds=forward,
                backward_seconds=time.monotonic()-started))
        full,mixed=outcomes
        cos=float(torch.nn.functional.cosine_similarity(full['gradient'],mixed['gradient'],dim=0))
        relative=abs(full['loss']-mixed['loss'])/max(abs(full['loss']),1e-6)
        amp_ok=all(o['finite'] for o in outcomes) and cos>=.99 and relative<=.02
        # Resume fixture executes two identical continuations from saved complete
        # state. It compares all trainable tensors and optimizer state exactly.
        model.load_state_dict(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        def update():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp_ok):out=model(batches,scenes)
            value=torch.stack([loss(o,b)[0] for o,b in zip(out,batches)]).mean()
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            return float(value.detach())
        update()
        ms,os=copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict())
        cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        update();expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        model.load_state_dict(ms);opt.load_state_dict(os);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        update()
        resume=all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
        resume &= all(torch.equal(v,opt.state_dict()['state'][k][name]) if torch.is_tensor(v)
                      else v==opt.state_dict()['state'][k][name]
                      for k,state in expected_opt['state'].items() for name,v in state.items())
        if not resume:raise ValueError('GPU optimizer/RNG resume fixture failed exact parity')
        started=time.monotonic();curve=[]
        for step in range(96):
            value=update()
            if step in (0,15,31,63,95):curve.append(dict(step=step+1,total=value))
        torch.cuda.synchronize()
        timings['warm_fixture_update_seconds']=(time.monotonic()-started)/96
        overfit=curve[-1]['total']<curve[0]['total']*.8
        if not overfit:raise ValueError('Tiny supported real-source fixture did not reduce loss by 20%; debug implementation')
        result=dict(status='measured',source=args.source,keys=keys,microbatch_groups=4,
            amp_enabled=amp_ok,amp_loss_relative_error=relative,amp_gradient_cosine=cos,
            fp32={k:v for k,v in full.items() if k!='gradient'},
            bf16={k:v for k,v in mixed.items() if k!='gradient'},
            timings=timings,checkpoint_resume_exact=True,overfit_curve=curve,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            trainable_embedding_cache=False,context_frames=list(range(-3,4)))
        model.cpu();torch.cuda.empty_cache()
    result.update(lease_seconds=lease.seconds,lease_wait_seconds=lease.wait_seconds)
    write_json(RESULTS/'profile.json',result)
    write_json(RESULTS/'execution_lock.json',dict(training=scientific_recipe(),
        model='Shared two-scale scene Conv3D 16/32/64; spatial query; two ordered 128-wide four-head attention layers',
        precision='BF16 autocast with FP32 reductions' if amp_ok else 'FP32',
        microbatch_groups=4,source_profile=result,training_floor_unchanged=True,
        new_target_model_results_read=False,checkpoint_interval=128,
        resources=dict(gpu_total_gib=20,rss_gib=50,cpu_threads=16,cache_gib=80,disk_floor_gib=20,lease_hours=72)),immutable=True)
    print(result,flush=True)
    return result
