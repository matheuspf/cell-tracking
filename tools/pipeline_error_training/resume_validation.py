"""Actual source-group optimizer/RNG replay after a separate-process restart.

Validation weights are never eligible for calibration, nomination or inference.
"""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time


def worker(mode):
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .common import WORK, write_json
    from .dataset import SourceDataset
    from .resources import Monitor, cpu_budget
    from .train import loss_for, make_model, save_checkpoint
    cpu_budget(4)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1701)
    rng = np.random.default_rng(1701)
    dataset = SourceDataset('44b6')
    model, family = make_model('D20_compact')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    root = WORK/'resume_validation'
    recipe = dict(arm='D20_compact', source='44b6', seed=1701, updates=2,
                  effective_batch_groups=32, pretrain_updates=1, validation_only=True)
    start = 0
    if mode=='replay':
        state = torch.load(root/'step1.pt', map_location='cpu', weights_only=False)
        assert state['recipe']==recipe
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        dataset.visits.update(state['visits'])
        start = state['step']
    draws = []
    with Monitor(root/f'{mode}.resources.json') as monitor:
        for step in range(start, 2):
            model.cuda()
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cuda()
            optimizer.zero_grad(set_to_none=True)
            step_draws = []
            for micro in range(32):
                pretrain = step==0
                sample = dataset.sample(rng, event=not pretrain and micro%2==0)
                batch = dataset.batch(sample, family)
                loss, _ = loss_for(model, batch, pretrain=pretrain)
                (loss/32).backward()
                step_draws.append(dict(group=sample['key'], dataset=sample['name'],
                    decisions=len(sample['decisions']), pair_rows=sample['pair_rows']))
                del batch, loss
                monitor.check()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.param_groups[0]['lr'] = .0003*min(1.,(step+1)/2)*.5*(1+math.cos(math.pi*step/2))
            optimizer.step()
            torch.cuda.synchronize()
            model.cpu()
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
            torch.cuda.empty_cache()
            draws = step_draws
            if step==0:
                save_checkpoint(root/'step1.pt', model, optimizer, 1, rng, recipe, dataset)
        save_checkpoint(root/f'{mode}-step2.pt', model, optimizer, 2, rng, recipe, dataset)
        write_json(root/f'{mode}-draws.json', draws)


def run():
    from .common import RESULTS, WORK, read_json, sha, write_json
    from .resources import Lease, Monitor, cpu_budget
    cpu_budget(4)
    root = WORK/'resume_validation'
    root.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    with Monitor(root/'resources.json'), Lease(required_gib=8.):
        for mode in ['reference', 'replay']:
            with (root/f'{mode}.log').open('a') as log:
                subprocess.run([sys.executable, '-m', 'pipeline_error_training.resume_validation', '--worker', mode],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch
    left = torch.load(root/'reference-step2.pt', map_location='cpu', weights_only=False)
    right = torch.load(root/'replay-step2.pt', map_location='cpu', weights_only=False)
    maximum = 0.
    def compare(a, b):
        nonlocal maximum
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
            if a.numel():
                maximum = max(maximum, float((a.double()-b.double()).abs().max()))
        elif isinstance(a, dict):
            assert a.keys()==b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a)==len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a==b
    for key in ['model', 'optimizer', 'rng', 'torch_rng', 'cuda_rng', 'visits']:
        compare(left[key], right[key])
    assert read_json(root/'reference-draws.json')==read_json(root/'replay-draws.json')
    result = dict(status='measured', source='44b6', real_source_groups_per_update=32,
        reference_updates=2, restarted_from_update=1, separate_worker_process=True,
        model_optimizer_rng_and_group_visits_match=True, max_abs_tensor_difference=maximum,
        rtol=1e-6, atol=1e-7, checkpoint_sha256=sha(root/'step1.pt'),
        seconds=time.monotonic()-begin, new_target_metrics_read=False,
        validation_weights_eligible_for_selection=False)
    write_json(RESULTS/'resume_validation.json', result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference','replay'])
    args = parser.parse_args()
    worker(args.worker) if args.worker else print(run())
