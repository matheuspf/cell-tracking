"""Source-only group training, equal-update controls, resumable optimizer/RNG state."""
from collections import Counter
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import RESULTS, STUDY, WORK, digest, read_json, sha, write_json
from .dataset import SourceDataset
from .models import DecisionModel, compatible_set_loss
from .resources import Lease, Monitor

EVENT_SCALE = np.array([1, 1, 2, 1, 2, 2, 20, 20, 20, 20, 20, 2, 1, 20, 20, 20,
                       2, 1, 1, 1, 1, 4, 1, 30, 2, 20, 20, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1], np.float32)


def make_model(arm):
    if arm.startswith('D10'):
        from .organoid_adapter import OrganoidEncoder
        family = 'organoid'
        model = DecisionModel(family, OrganoidEncoder(random=arm == 'D10_random'))
    else:
        family = 'compact' if arm == 'D20_compact' else 'temporal'
        model = DecisionModel(family)
    return model, family


def encode(model, batch, *, training):
    parts = []
    for start in range(0, len(batch['patch']), 16):
        p, v = batch['patch'][start:start+16], batch['valid'][start:start+16]
        # CPU inputs are checkpointed, bounding GPU activations without changing groups.
        fn = lambda a, b: model.encoder(a.to('cuda'), b.to('cuda'))
        parts.append(checkpoint(fn, p, v, use_reentrant=False) if training else fn(p, v))
    return torch.cat(parts)


def decision_scores(model, batch, z, link):
    decisions = batch['decisions']
    if not decisions:
        return link.new_empty(0), link.new_empty(0)
    events = torch.tensor([[batch['node_map'][i] for i in d.event[:3]] for d in decisions], device=z.device)
    features = torch.tensor(np.clip(np.asarray([r['event_features'] for r in batch['records']])/EVENT_SCALE, -10, 10),
                            dtype=torch.float32, device=z.device)
    event_score, risk = model.event_scores(z, events, features)
    anchor = 1 if model.family == 'compact' else 2
    boundary = model.boundary(torch.cat([z, batch['valid'][:, anchor].to(z.device)], -1))
    cost = F.softplus(boundary)
    scores = []
    for k, d in enumerate(decisions):
        if d.kind == 'keep':
            scores.append(link.sum()*0)
            continue
        value = link.new_zeros(())
        for pair in d.add:
            value = value + link[batch['pair_map'][pair]]
        for pair in d.remove:
            value = value - link[batch['pair_map'][pair]]
        for node in d.births:
            value = value - cost[batch['node_map'][node], 0]
        for node in d.terminations:
            value = value - cost[batch['node_map'][node], 1]
        if d.kind == 'division':
            value = value + event_score[k] + risk[k]
        scores.append(value)
    return torch.stack(scores), risk


def loss_for(model, batch, *, training=True, pretrain=False):
    z = encode(model, batch, training=training)
    pairs = batch['pairs'].to('cuda')
    link = model.link_scores(z, pairs, batch['edge_features'].to('cuda')) + batch['native_offset'].to('cuda')
    zero = z.sum()*0
    identity, event_loss, metric_loss = zero, zero, zero
    if len(batch['pair_labels']):
        ii = torch.tensor(batch['supervised_pair_indices'], device='cuda')
        y = torch.tensor(batch['pair_labels'], dtype=torch.float32, device='cuda')
        identity = F.binary_cross_entropy_with_logits(link[ii], y)
    if batch['decisions'] and not pretrain:
        scores, risk = decision_scores(model, batch, z, link)
        known = []
        positive = []
        event_anchors = {r['anchor'] for r in batch['records'] if r['labels']['biological'] == 1}
        for r in batch['records']:
            label = r['labels']
            good = label['biological'] == 1
            bad = label['identity'] == 0 or label['metric_fork_target'] == 0
            if r['anchor'] in event_anchors:
                # All fully supported compatible alternatives (including time anchors)
                # are positive as a set; keep/continuation cannot explain both known daughters.
                bad |= not good and r['decision']['kind'] in ['keep', 'continuation_a', 'continuation_b', 'two_parents']
            else:
                good |= r['decision']['kind'] != 'division' and label['identity'] == 1
            positive.append(good)
            known.append(good or bad)
        event_loss = compatible_set_loss(scores, torch.tensor(positive, device='cuda'),
            torch.tensor(known, device='cuda'), torch.zeros(len(scores), dtype=torch.int64, device='cuda'))
        risk_targets = torch.tensor([r['labels']['metric_fork_target'] for r in batch['records']], device='cuda')
        valid = risk_targets >= 0
        if valid.any():
            metric_loss = F.binary_cross_entropy_with_logits(risk[valid], risk_targets[valid].float())
    contrastive = zero
    consistency = zero
    if training:
        # Same-input views only teach invariance; they do not declare lineage identity.
        from .augment import view as augment_view
        augmentation = augment_view(batch, model.family)
        view = encode(model, augmentation, training=True)
        contrastive = (1-F.cosine_similarity(z, view, dim=-1)).mean()
        augmented_links = model.link_scores(view, pairs, batch['edge_features'].to('cuda')) + batch['native_offset'].to('cuda')
        if len(link):
            consistency = F.mse_loss(augmented_links.sigmoid(), link.sigmoid())
    total = identity + event_loss + metric_loss + .1*contrastive + .1*consistency
    return total, dict(identity=float(identity.detach()), decision=float(event_loss.detach()),
                      metric_risk=float(metric_loss.detach()), contrastive=float(contrastive.detach()),
                      consistency=float(consistency.detach()), total=float(total.detach()), node_tokens=len(z))


def save_checkpoint(path, model, optimizer, step, rng, recipe, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), recipe=recipe,
                    mean=dataset.mean, scale=dataset.scale, visits=dict(dataset.visits)), tmp)
    tmp.replace(path)


def run(source, arm, seed=20260915, pilot=False):
    from .guard import install
    install(source=source)
    if arm == 'O10_swap':
        raise ValueError('O10 uses the independent observation training entry point, not division labels')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    dataset = SourceDataset(source)
    model, family = make_model(arm)
    folder = WORK / ('training_pilot' if pilot else 'training') / arm / source / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if pilot:
        recipe = dict(updates=4, effective_batch_groups=32, pretrain_updates=2, learning_rate=.0003)
    else:
        execution_path = RESULTS / 'execution_repair_lock.json'
        if not execution_path.exists():
            execution_path = RESULTS / 'execution_lock.json'
        execution = read_json(execution_path)
        recipe = execution['training']
    recipe = dict(recipe, arm=arm, source=source, seed=seed, family=family,
                  input_manifest_sha256=sha(RESULTS / 'input_manifest.json'),
                  source_manifest_sha256=sha(WORK / 'source' / source / 'manifest.json'),
                  code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('models.py'),
                      Path(__file__).with_name('augment.py'), Path(__file__).with_name('dataset.py'),
                      Path(__file__).with_name('fast_crops.py'), Path(__file__).with_name('organoid_adapter.py')]})
    write_json(folder / 'recipe.json', recipe, immutable=True)
    if family == 'organoid':
        pretrained = list(model.encoder.backbone.parameters())
        inherited_ids = {id(p) for p in pretrained}
        optimizer = torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in inherited_ids], lr=.0003),
                                     dict(params=pretrained, lr=.00003)], weight_decay=.0001)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    step = 0
    path = folder / 'resume.pt'
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['recipe'] != recipe:
            raise ValueError('Resume recipe drift')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        dataset.visits.update(state['visits'])
    measurements = []
    leased_seconds = 0.
    with Monitor(folder / 'resources.json') as monitor:
        while step < recipe['updates']:
            # GPU leases cover at most one optimizer step, released during checkpointing.
            lease_start = time.monotonic()
            with Lease(required_gib=8.):
                model.cuda()
                for value in optimizer.state.values():
                    for key, tensor in value.items():
                        if isinstance(tensor, torch.Tensor):
                            value[key] = tensor.cuda()
                if family == 'organoid':
                    model.encoder.adapt(arm in ['D10_adapted', 'D10_random'] and step >= max(1, recipe['updates']//8))
                optimizer.zero_grad(set_to_none=True)
                losses = Counter()
                peak_group_nodes = 0
                for micro in range(recipe['effective_batch_groups']):
                    pretrain = step < recipe['pretrain_updates'] and arm != 'D20_no_pretrain'
                    event = not pretrain and arm != 'A10' and micro % 2 == 0
                    sample = dataset.sample(rng, event=event)
                    batch = dataset.batch(sample, family)
                    loss, terms = loss_for(model, batch, pretrain=pretrain)
                    (loss/recipe['effective_batch_groups']).backward()
                    peak_group_nodes = max(peak_group_nodes, terms.pop('node_tokens'))
                    losses.update(terms)
                    del loss, batch
                    monitor.check()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                warmup = min(1., (step+1)/min(500, recipe['updates']))
                cosine = .5*(1+math.cos(math.pi*step/recipe['updates']))
                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = (.00003 if family == 'organoid' and i == 1 else .0003)*warmup*cosine
                optimizer.step()
                torch.cuda.synchronize()
                model.cpu()
                for value in optimizer.state.values():
                    for key, tensor in value.items():
                        if isinstance(tensor, torch.Tensor):
                            value[key] = tensor.cpu()
                torch.cuda.empty_cache()
            step += 1
            elapsed = time.monotonic()-lease_start
            leased_seconds += elapsed
            measurement = dict(step=step, seconds=elapsed, peak_group_nodes=peak_group_nodes,
                               **{k: v/recipe['effective_batch_groups'] for k, v in losses.items()})
            measurements.append(measurement)
            with (folder / 'history.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(measurement)+'\n')
            save_checkpoint(path, model, optimizer, step, rng, recipe, dataset)
            write_json(folder / 'progress.json', dict(step=step, total=recipe['updates'], latest=measurement,
                leased_seconds=leased_seconds, source_groups_visited=len(dataset.visits), status='running'))
            print(f'{arm} source {source} seed {seed}: {step}/{recipe["updates"]} updates, '
                  f'{elapsed:.2f}s, loss {measurement["total"]:.4f}', flush=True)
    weights = folder / 'model.pt'
    torch.save(model.state_dict(), weights)
    write_json(folder / 'package.json', dict(recipe=recipe, weights_sha256=sha(weights),
        completed_updates=step, mean=dataset.mean, scale=dataset.scale, status='measured',
        seconds=time.monotonic()-start, measured_leased_seconds=leased_seconds,
        group_visits=dict(dataset.visits), source_event_groups=len(dataset.event_keys),
        calibration_status='pending source-only calibration',
        claim='source-only head; inherited upstream exposure'))
    write_json(folder / 'progress.json', dict(step=step, total=recipe['updates'], status='complete'))
    return folder


def lock_budget():
    # A source-only end-to-end pilot (including crop I/O) is required; no target scores enter.
    pilot = WORK / 'training_pilot/D20_temporal/44b6/20260915'
    import json
    measurements = [json.loads(line) for line in (pilot / 'history.jsonl').read_text().splitlines()]
    seconds = max(m['seconds'] for m in measurements)
    # Fourteen directional first-seed fits, with headroom for group-size variability.
    common_updates = min(16000, max(2, int((24*3600)/(14*seconds*1.5))))
    common_updates -= common_updates % 2
    training = dict(read_json(STUDY)['training'], updates=common_updates,
                    pretrain_updates=common_updates//2, joint_updates=common_updates//2)
    from .augment import CONFIG as augmentations
    lock = dict(status='source_only_frozen_before_target_comparisons', training=training,
                pilot_sha256=sha(pilot / 'history.jsonl'), seconds_per_update_bound=seconds,
                inference_gpu_hours_reserved=8, replication_gpu_hours_reserved=12,
                primary_gpu_hours_reserved=24, first_seed_directional_fit_budget=14,
                temporal_operator='GRUCell', normalization='fixed 1/99% per-frame robust crop normalization',
                source_checkpoint='fixed final checkpoint; no target selection',
                augmentations=augmentations,
                precision='FP32; TF32 disabled; activation checkpoint chunks of 16 tokens',
                deviation='Budget selected from measured source end-to-end throughput; matched arms retain equal optimizer updates.',
                new_target_scores_read=False, recipe_sha256=digest(training))
    write_json(RESULTS / 'execution_lock.json', lock, immutable=True)
    print(f'Locked {common_updates} total updates per matched directional fit from source-only throughput', flush=True)
