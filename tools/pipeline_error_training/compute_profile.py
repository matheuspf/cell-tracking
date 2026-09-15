"""Measure registered encoder FLOP formulas; never change model parameters.

These shape probes run on CPU. They are not full-training FLOP totals, hardware
instructions or inference timing measurements. Convolution/matrix formulas omit
unsupported elementwise, normalization, augmentation, loss and decoder work.
"""
import gc
import hashlib
import inspect
from pathlib import Path
import time


def tensor_hash(model):
    result = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def run():
    from .guard import install
    install(source='44b6')
    import torch
    from torch.utils.checkpoint import checkpoint
    from torch.utils.flop_counter import FlopCounterMode
    from .common import RESULTS, WORK, sha, write_json
    from .evaluate import write_csv
    from .resources import Monitor
    from .scoring import load_model
    torch.set_num_threads(2)
    # Verify the installed counter's convention on one analytic matrix product.
    simple = torch.nn.Linear(3,5,bias=False)
    with FlopCounterMode(display=False) as check:
        simple(torch.zeros(16,3)).sum().backward()
    if check.get_total_flops()!=2*(2*16*3*5):
        raise ValueError('Installed FLOP counter matrix-product convention changed')
    rows=[]
    started=time.monotonic()
    with Monitor(WORK/'resources/encoder-compute-profile.json') as monitor:
        for arm in ['D10_frozen','D10_adapted','D20_compact','D20_temporal']:
            package=WORK/'training'/arm/'44b6/20260915'
            model,spec=load_model(package)
            if arm.startswith('D10'):
                model.encoder.adapt(arm=='D10_adapted')
            before=tensor_hash(model)
            family=model.family
            shape={'organoid':(16,12,64,64,3),'compact':(16,3,3,12,12),
                   'temporal':(16,7,2,16,64,64)}[family]
            for support in ['interior','first_frame']:
                patch=torch.zeros(shape)
                valid=torch.ones((16,3 if family=='compact' else 7,4))
                if support=='first_frame':
                    valid[:,:1 if family=='compact' else 2]=0
                for recompute in [False,True]:
                    model.zero_grad(set_to_none=True)
                    begin=time.monotonic()
                    with FlopCounterMode(display=False) as counter:
                        value=(checkpoint(model.encoder,patch,valid,use_reentrant=False) if recompute
                               else model.encoder(patch,valid))
                        forward=counter.get_total_flops()
                        value.sum().backward()
                    total=counter.get_total_flops()
                    if forward<=0 or total<forward or tensor_hash(model)!=before:
                        raise ValueError('Invalid operation profile or model-state mutation')
                    rows.append(dict(arm=arm,family=family,support=support,node_tokens=16,
                        activation_checkpointing=recompute,registered_forward_flops=forward,
                        registered_forward_backward_flops=total,
                        registered_forward_backward_flops_per_node=total/16,
                        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                        encoder_trainable_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                        total_model_parameters=sum(p.numel() for p in model.parameters()),
                        operator_counts={str(k):v for k,v in counter.get_flop_counts()['Global'].items()},
                        CPU_probe_seconds=time.monotonic()-begin,source_model_sha256=spec['weights_sha256'],
                        unchanged_model_tensor_sha256=before,status='measured'))
                    monitor.check()
                del patch,valid,value
            del model
            gc.collect()
            print(f'Encoder operation profile: {arm} complete',flush=True)
    receipt=dict(status='measured',torch_version=torch.__version__,source='44b6',rows=rows,
        seconds=time.monotonic()-started,batch_nodes=16,source_contract_shapes_only=True,
        real_image_or_annotation_reads=False,optimizer_steps=0,model_parameters_changed=False,
        new_target_metrics_read=False,hardware_device='CPU',
        counter_source_sha256=sha(inspect.getfile(FlopCounterMode)),implementation_sha256=sha(Path(__file__)),
        method_reference='https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py',
        scope='PyTorch registered shape-based operation formulas for one encoder view, forward/backward, with and without the actual activation checkpointing wrapper.',
        exclusions=['Decision heads','Augmentation','Loss functions','Optimizer','Decoder',
            'Unsupported elementwise and normalization operations','Variable biological-group sizes','Second augmented view'],
        temporal_encoder_also_used_by=['D20_no_pretrain','A10','O10_swap'],
        whole_training_FLOPs_measured=False,equal_compute_claim=False,
        interpretation='Compare these fixed-shape encoder costs with the separately measured update counts and wall times. They do not imply equal compute or a hardware FLOP/s result.')
    write_json(RESULTS/'encoder_compute_profile.json',receipt,immutable=True)
    write_csv(RESULTS/'encoder_compute_profile.csv',[{k:v for k,v in row.items() if k!='operator_counts'} for row in rows])
    return receipt


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
