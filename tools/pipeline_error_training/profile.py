"""Source-image-only throughput measurement before any execution-budget choice."""
import time

import numpy as np
import torch

from .common import WORK, inputs, verified_evidence, verified_graph, write_json
from .crops import Images, compact_view, sample_native
from .models import DecisionModel
from .resources import Lease, Monitor


def run():
    from .bank import EventBank
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row = max((r for r in inputs() if r['embryo'] == '44b6'), key=lambda r: r['baselines']['P0']['nodes'])
    graph = verified_graph(row)
    bank = EventBank(graph['nodes'], graph['edges'], verified_evidence(row), row['physical_scale'])
    event, features, _ = next(e for e in bank if int(graph['nodes'][e[0][0], 1]) == 20)
    indices = list(event[:3])
    images = Images(row['image_path'])
    start = time.monotonic()
    crops = [sample_native(images, graph['nodes'], bank.pred, bank.succ, i) for i in indices]
    native = np.stack([p for p, _ in crops])
    valid = np.stack([v for _, v in crops])
    io = time.monotonic()-start
    records = []
    with Monitor(WORK / 'resources/profile.json') as monitor:
        for family in ['compact', 'temporal']:
            patch = native if family == 'temporal' else np.stack([compact_view(p) for p in native])
            mask = valid if family == 'temporal' else valid[:, 1:4]
            model = DecisionModel(family)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
            with Lease(required_gib=6.):
                model.cuda()
                x = torch.from_numpy(patch).cuda().float()/255
                m = torch.from_numpy(mask).cuda()
                ev = torch.tensor([[0, 1, 2]], device='cuda')
                f = torch.from_numpy(features[None]).cuda()
                times = []
                torch.cuda.reset_peak_memory_stats()
                for step in range(8):
                    monitor.check()
                    begin = time.monotonic()
                    optimizer.zero_grad()
                    z = model.encoder(x, m)
                    event_score, risk = model.event_scores(z, ev, f)
                    loss = torch.nn.functional.softplus(-event_score).mean()+risk.square().mean()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append(time.monotonic()-begin)
                record = dict(family=family, microbatch_groups=1, node_tokens_per_group=len(indices),
                    seconds_per_microbatch=float(np.median(times[2:])), observed_steps=8,
                    estimated_seconds_per_effective_batch32=float(np.median(times[2:])*32),
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    parameters=sum(p.numel() for p in model.parameters()))
                records.append(record)
                model.cpu()
                del optimizer, model, x, m, loss, z, event_score, risk
                torch.cuda.empty_cache()
    receipt = dict(source='44b6', target_labels_opened=False, image_crop_seconds=io,
                   real_node_tokens=len(indices), results=records,
                   note='Pilot optimizer state is discarded. Measured three-node groups are a lower bound on complete multi-anchor/owner group compute.')
    write_json(WORK / 'profile.json', receipt)
    print(receipt, flush=True)
