"""Independent source-trained keep/substitute/retain-both temporal selector."""
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import RESULTS, WORK, digest, read_json, save_arrays, sha, write_json
from .dataset import SourceDataset
from .fast_crops import sample_native
from .models import DecisionModel
from .observations import CONFIG, ObservationBank, raw_graph
from .resources import Lease, Monitor
from .train import encode, loss_for, save_checkpoint


class ObservationDataset(SourceDataset):
    def __init__(self, source, partition='fit'):
        super().__init__(source, partition)
        self.observation_groups = defaultdict(list)
        self.synthetic_groups = defaultdict(list)
        self.observation_arrays = {}
        self.banks = OrderedDict()
        self.raw_crops = OrderedDict()
        self.cache_bytes += sum(p.stat().st_size for p in (WORK / 'observation_crop_cache').rglob('*.npz'))
        for name in self.rows:
            path = WORK / 'observation_source' / source / f'{name}.npz'
            receipt = read_json(path.with_suffix('.json'))
            if sha(path) != receipt['sha256']:
                raise ValueError('Observation source labels changed')
            with np.load(path, allow_pickle=False) as data:
                self.observation_arrays[name] = {k: data[k] for k in data.files}
            a = self.observation_arrays[name]
            for k in np.flatnonzero((a['labels'] >= 0) & (a['group'] >= 0)):
                self.observation_groups[f'{name}:{a["group"][k]}'].append((name, int(k)))
            for index, group in zip(a['synthetic_nodes'], a['synthetic_groups']):
                self.synthetic_groups[f'{name}:{group}'].append((name, int(index)))
        self.observation_keys = sorted(self.observation_groups)
        self.synthetic_keys = sorted(self.synthetic_groups)

    def bank(self, name):
        if name not in self.banks:
            graph = self.graph(name)[0]
            raw = raw_graph(self.rows[name])
            self.banks[name] = ObservationBank(self.rows[name], graph, raw)
            while len(self.banks) > 4:
                self.banks.popitem(last=False)
        self.banks.move_to_end(name)
        return self.banks[name]

    def raw_crop(self, name, index):
        key = (name, int(index))
        if key not in self.raw_crops:
            bank = self.bank(name)
            path = WORK / 'observation_crop_cache' / name / f'{index}.npz'
            fingerprint = self.fingerprint+bank.hash
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data['fingerprint']) != fingerprint:
                        raise ValueError('Raw observation crop provenance changed')
                    patch, valid = data['patch'], data['valid']
            else:
                patch, valid = sample_native(self.images(name), bank.raw_nodes, bank.raw_pred, bank.raw_succ, index)
                # A conservative ten-GiB additional observation cache budget.
                if self.cache_bytes < 70*2**30:
                    save_arrays(path, patch=patch, valid=valid, fingerprint=np.array(fingerprint))
                    self.cache_bytes += path.stat().st_size
            self.raw_crops[key] = (patch, valid)
            while len(self.raw_crops) > 256:
                self.raw_crops.popitem(last=False)
        self.raw_crops.move_to_end(key)
        return self.raw_crops[key]

    def observation_sample(self, rng):
        synthetic = float(rng.random()) < .25
        keys = self.synthetic_keys if synthetic else self.observation_keys
        group = keys[int(rng.integers(len(keys)))]
        self.visits['observation:'+group] += 1
        choices = self.synthetic_groups[group] if synthetic else self.observation_groups[group]
        name, index = choices[int(rng.integers(len(choices)))]
        if synthetic:
            shift = np.array([0, -2 if rng.random() < .5 else 2, -2 if rng.random() < .5 else 2])
            return dict(name=name, old=index, rows=[], synthetic_shift=shift, group=group)
        arrays = self.observation_arrays[name]
        old = int(arrays['pairs'][index, 0])
        rows = [k for n, k in choices if n == name and arrays['pairs'][k, 0] == old]
        return dict(name=name, old=old, rows=rows, synthetic_shift=None, group=group)

    def observation_batch(self, sample):
        name, old = sample['name'], sample['old']
        bank = self.bank(name)
        context = sorted({old, *bank.pred[old], *bank.succ[old]})
        old_nodes = sorted(set(context) | {old})
        crop = [self.crop(name, i, 'temporal') for i in old_nodes]
        pairs, evidence, labels = [], [], []
        old_index = old_nodes.index(old)
        arrays = self.observation_arrays[name]
        if sample['synthetic_shift'] is not None:
            shifted = bank.nodes.copy()
            shifted[:, 2:] += sample['synthetic_shift']
            crop.append(sample_native(self.images(name), shifted, bank.pred, bank.succ, old))
            pairs.append((old_index, len(crop)-1))
            displacement = np.linalg.norm(sample['synthetic_shift']*self.rows[name]['physical_scale'])
            evidence.append([displacement/7, .5, 0., not bank.pred[old], not bank.succ[old],
                             not bank.pred[old], not bank.succ[old], 1.])
            labels.append(0)
        else:
            for k in sample['rows']:
                _, raw = arrays['pairs'][k]
                crop.append(self.raw_crop(name, int(raw)))
                pairs.append((old_index, len(crop)-1))
                evidence.append(bank.evidence(old, int(raw)))
                labels.append(int(arrays['labels'][k]))
        return dict(patch=torch.from_numpy(np.stack([p for p, _ in crop])).float()/255,
                    valid=torch.from_numpy(np.stack([v for _, v in crop])),
                    observation_pairs=np.asarray(pairs), evidence=np.asarray(evidence, np.float32),
                    labels=np.asarray(labels), context_indices=[old_nodes.index(i) for i in context],
                    group=sample['group'], synthetic=sample['synthetic_shift'] is not None)


def logits(model, batch, z):
    pair = torch.as_tensor(batch['observation_pairs'], device=z.device)
    context = z[batch['context_indices']].mean(0, keepdim=True).expand(len(pair), -1)
    return model.selection_scores(z[pair[:, 0]], z[pair[:, 1]], context,
                                  torch.as_tensor(batch['evidence'], device=z.device))


def observation_loss(model, batch):
    from .augment import view
    z = encode(model, batch, training=True)
    pred = logits(model, batch, z)
    label = torch.as_tensor(batch['labels'], device='cuda')
    known = label >= 0
    supervised = F.cross_entropy(pred[known], label[known]) if known.any() else pred.sum()*0
    augmented_z = encode(model, view(batch, 'temporal'), training=True)
    augmented = logits(model, batch, augmented_z)
    contrastive = (1-F.cosine_similarity(z, augmented_z)).mean()
    consistency = F.mse_loss(pred.softmax(-1), augmented.softmax(-1))
    total = supervised+.1*contrastive+.1*consistency
    return total, dict(total=float(total.detach()), selection=float(supervised.detach()),
        contrastive=float(contrastive.detach()), consistency=float(consistency.detach()),
        synthetic=float(batch['synthetic']), node_tokens=len(z))


def run(source, seed=20260915):
    from .guard import install
    install(source=source)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dataset = ObservationDataset(source)
    model = DecisionModel('temporal')
    folder = WORK / 'training/O10_swap' / source / str(seed)
    recipe = dict(read_json(RESULTS / 'execution_repair_lock.json')['training'], arm='O10_swap', source=source,
        seed=seed, family='temporal', config=CONFIG, synthetic_fraction_in_joint_observation_groups=.25,
        source_manifest_sha256=sha(WORK / 'observation_source' / source / 'manifest.json'),
        input_manifest_sha256=sha(RESULTS / 'input_manifest.json'),
        code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('observations.py'),
            Path(__file__).with_name('augment.py'), Path(__file__).with_name('models.py')]})
    write_json(folder / 'recipe.json', recipe, immutable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    step = 0
    resume = folder / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['recipe'] != recipe:
            raise ValueError('Observation resume recipe drift')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        dataset.visits.update(state['visits'])
    begin, leased = time.monotonic(), 0.
    with Monitor(folder / 'resources.json') as monitor:
        while step < recipe['updates']:
            start = time.monotonic()
            with Lease(required_gib=8.):
                model.cuda()
                for state in optimizer.state.values():
                    for key, tensor in state.items():
                        if isinstance(tensor, torch.Tensor):
                            state[key] = tensor.cuda()
                optimizer.zero_grad(set_to_none=True)
                terms = Counter()
                for micro in range(32):
                    pretrain = step < recipe['pretrain_updates']
                    if pretrain or micro % 2:
                        batch = dataset.batch(dataset.sample(rng), 'temporal')
                        loss, record = loss_for(model, batch, pretrain=True)
                    else:
                        batch = dataset.observation_batch(dataset.observation_sample(rng))
                        loss, record = observation_loss(model, batch)
                    (loss/32).backward()
                    terms.update(record)
                    del loss, batch
                    monitor.check()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                warmup = min(1., (step+1)/min(500, recipe['updates']))
                for group in optimizer.param_groups:
                    group['lr'] = .0003*warmup*.5*(1+math.cos(math.pi*step/recipe['updates']))
                optimizer.step()
                torch.cuda.synchronize()
                model.cpu()
                for state in optimizer.state.values():
                    for key, tensor in state.items():
                        if isinstance(tensor, torch.Tensor):
                            state[key] = tensor.cpu()
                torch.cuda.empty_cache()
            step += 1
            elapsed = time.monotonic()-start
            leased += elapsed
            record = dict(step=step, seconds=elapsed, **{k: v/32 for k, v in terms.items()})
            import json
            with (folder / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            save_checkpoint(resume, model, optimizer, step, rng, recipe, dataset)
            write_json(folder / 'progress.json', dict(status='running', step=step, total=recipe['updates'], latest=record))
            print(f'O10 source {source} seed {seed}: {step}/{recipe["updates"]}, {elapsed:.2f}s, loss {record["total"]:.4f}', flush=True)
    torch.save(model.state_dict(), folder / 'model.pt')
    write_json(folder / 'package.json', dict(status='measured', recipe=recipe, weights_sha256=sha(folder / 'model.pt'),
        completed_updates=step, mean=dataset.mean, scale=dataset.scale, group_visits=dict(dataset.visits),
        seconds=time.monotonic()-begin, measured_leased_seconds=leased, calibration_status='pending source-only calibration',
        claim='Source-only observation head; inherited upstream exposure; supplementary synthetic duplicate corruption only'))
    write_json(folder / 'progress.json', dict(status='complete', step=step, total=recipe['updates']))
