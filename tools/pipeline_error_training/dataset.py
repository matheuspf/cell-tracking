"""Group sampling and bounded prediction-centered crop reuse for source fitting."""
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_evidence, verified_graph
from .crops import Images, compact_view, prediction_tracklet
from .fast_crops import sample_native
from .feasibility import records


class SourceDataset:
    def __init__(self, source, partition='fit'):
        self.source, self.partition = source, partition
        split = read_json(RESULTS / 'split_manifest.json')['directions'][source]
        self.rows = {r['dataset']: r for r in inputs() if r['embryo'] == source and
                     split[r['dataset']]['partition'] == partition}
        self.graphs = OrderedDict()
        self.image_cache = OrderedDict()
        self.crop_cache = OrderedDict()
        self.cache_bytes = sum(p.stat().st_size for p in (WORK / 'crop_cache').rglob('*.npz'))
        self.fingerprint = sha(RESULTS / 'input_manifest.json')[:16] + sha(Path(__file__).with_name('crops.py'))[:16]
        self.groups = defaultdict(lambda: dict(decisions=[], pairs=[]))
        feature_sum = np.zeros(38, np.float64)
        feature_sum2 = np.zeros(38, np.float64)
        count = 0
        self.pair_arrays = {}
        for name, row in self.rows.items():
            path = WORK / 'source' / source / 'pairs' / f'{name}.npz'
            with np.load(path, allow_pickle=False) as loaded:
                pair_labels = {k: loaded[k] for k in loaded.files}
            self.pair_arrays[name] = pair_labels
            for k, group in enumerate(pair_labels['group']):
                self.groups[f'{name}:{group}']['pairs'].append((name, k))
            for record in records(source, name):
                self.groups[record['group']]['decisions'].append((name, record))
            native = verified_evidence(row)
            x = native['edge_features'][pair_labels['index']].astype(np.float64)
            feature_sum += x.sum(0)
            feature_sum2 += np.square(x).sum(0)
            count += len(x)
        self.mean = feature_sum/max(count, 1)
        self.scale = np.sqrt(np.maximum(feature_sum2/max(count, 1)-self.mean**2, 1e-4))
        self.keys = sorted(self.groups)
        self.event_keys = [k for k in self.keys if any(r['labels']['biological'] == 1 for _, r in self.groups[k]['decisions'])]
        self.pair_keys = [k for k in self.keys if self.groups[k]['pairs']]
        self.visits = defaultdict(int)

    def graph(self, name):
        if name not in self.graphs:
            from .common import adjacency
            row = self.rows[name]
            graph, native = verified_graph(row), verified_evidence(row)
            _, pred, succ = adjacency(graph['nodes'], graph['edges'])
            feature_map = {tuple(map(int, pair)): k for k, pair in enumerate(native['pairs'])}
            self.graphs[name] = (graph, native, pred, succ, feature_map)
            while len(self.graphs) > 4:
                self.graphs.popitem(last=False)
        self.graphs.move_to_end(name)
        return self.graphs[name]

    def images(self, name):
        if name not in self.image_cache:
            self.image_cache[name] = Images(self.rows[name]['image_path'])
            while len(self.image_cache) > 2:
                self.image_cache.popitem(last=False)
        self.image_cache.move_to_end(name)
        return self.image_cache[name]

    def crop(self, name, index, family):
        graph, _, pred, succ, _ = self.graph(name)
        row = self.rows[name]
        kind = 'organoid' if family.startswith('organoid') else 'native'
        key = (name, int(index), kind)
        if key not in self.crop_cache:
            node_id = int(graph['nodes'][index, 0])
            path = WORK / 'crop_cache' / kind / name / f'{node_id}.npz'
            fingerprint = self.fingerprint
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data['fingerprint']) != fingerprint:
                        raise ValueError('Crop input fingerprint drift')
                    patch, valid = data['patch'], data['valid']
            else:
                image = self.images(name)
                if kind == 'organoid':
                    from .organoid_adapter import patches
                    _, tracked = prediction_tracklet(graph['nodes'], pred, succ, index)
                    t0 = int(graph['nodes'][index, 1])
                    valid = np.zeros((7, 4), np.float32)
                    for k, dt in enumerate(range(-2, 5)):
                        if 0 <= t0+dt < image.shape[0]:
                            valid[k] = [1., float(tracked[dt]), 1., 1.]
                    patch = patches(image, graph['nodes'], [index])[0]
                else:
                    patch, valid = sample_native(image, graph['nodes'], pred, succ, index)
                # Store a bounded reusable training cache; never copy historical crops.
                if self.cache_bytes < 60 * 2**30:
                    save_arrays(path, patch=patch, valid=valid, fingerprint=np.array(fingerprint),
                                metadata_sha256=np.array(row['metadata_sha256']))
                    self.cache_bytes += path.stat().st_size
            self.crop_cache[key] = (patch, valid)
            while len(self.crop_cache) > 512:
                self.crop_cache.popitem(last=False)
        self.crop_cache.move_to_end(key)
        patch, valid = self.crop_cache[key]
        if family == 'compact':
            return compact_view(patch), valid[1:4]
        return patch, valid

    def sample(self, rng, *, event=False):
        # Record group sampling probabilities, not a fabricated row propensity.
        pool = self.event_keys if event and self.event_keys else self.pair_keys
        if not pool:
            raise ValueError('Source partition has no supported training groups')
        key = pool[int(rng.integers(len(pool)))]
        self.visits[key] += 1
        group = self.groups[key]
        if event and group['decisions']:
            available_events = sorted({e for _, r in group['decisions'] for e in r['labels']['compatible_events']})
            if available_events:
                event_id = available_events[int(rng.integers(len(available_events)))]
                anchors = {r['anchor'] for _, r in group['decisions'] if event_id in r['labels']['compatible_events']}
                selected = [(name, r) for name, r in group['decisions'] if r['anchor'] in anchors]
            else:
                anchors = sorted({r['anchor'] for _, r in group['decisions']})
                anchor = anchors[int(rng.integers(len(anchors)))]
                selected = [(name, r) for name, r in group['decisions'] if r['anchor'] == anchor]
            name = selected[0][0]
            return dict(key=key, name=name, decisions=[r for _, r in selected], pair_rows=[],
                        sampling_probability=1/len(pool))
        # One daughter incoming-choice group, sampled inside a biological trajectory group.
        name, k = group['pairs'][int(rng.integers(len(group['pairs'])))]
        _, native, _, _, _ = self.graph(name)
        index = self.pair_arrays[name]['index'][k]
        target = native['pairs'][index, 1]
        selected = [j for n, j in group['pairs'] if n == name and
                    native['pairs'][self.pair_arrays[name]['index'][j], 1] == target]
        return dict(key=key, name=name, decisions=[], pair_rows=selected,
                    sampling_probability=1/len(pool))

    def batch(self, sample, family, norm=None):
        import torch
        from .feasibility import from_record
        name = sample['name']
        graph, native, _, _, fmap = self.graph(name)
        decisions = [from_record(r) for r in sample['decisions']]
        edge_set = {e for d in decisions for e in d.remove | d.add}
        if sample['pair_rows']:
            pair_index = self.pair_arrays[name]['index'][sample['pair_rows']]
            edge_set.update(map(tuple, native['pairs'][pair_index]))
        node_set = {n for e in edge_set for n in e}
        node_set.update(n for d in decisions for n in d.event[:3] if n >= 0)
        nodes = sorted(node_set)
        if not nodes:
            raise ValueError('Training group has no prediction-centered tokens')
        mapping = {n: i for i, n in enumerate(nodes)}
        cropped = [self.crop(name, n, family) for n in nodes]
        patches = np.stack([c[0] for c in cropped])
        valid = np.stack([c[1] for c in cropped])
        pairs = sorted(edge_set)
        pair_map = {p: k for k, p in enumerate(pairs)}
        features = []
        for a, b in pairs:
            if (a, b) in fmap:
                features.append(native['edge_features'][fmap[a, b]])
            else:
                # Missing native evidence remains explicitly flagged, never a zero-confidence label.
                f = np.zeros(38, np.float32)
                f[1] = f[24] = 1.
                f[3] = np.linalg.norm((graph['nodes'][a, 2:]-graph['nodes'][b, 2:])*self.rows[name]['physical_scale'])
                features.append(f)
        features = np.asarray(features, np.float32).reshape(-1, 38)
        mean, scale = norm if norm is not None else (self.mean, self.scale)
        normalized = np.clip((features-mean)/scale, -10, 10).astype(np.float32)
        tensor = torch.from_numpy(patches).float()
        if not family.startswith('organoid'):
            tensor = tensor/255.
        return dict(patch=tensor, valid=torch.from_numpy(valid),
                    pairs=torch.tensor([[mapping[a], mapping[b]] for a, b in pairs], dtype=torch.long).reshape(-1, 2),
                    edge_features=torch.from_numpy(normalized), native_offset=torch.from_numpy(features[:, 23]),
                    decisions=decisions, records=sample['decisions'], pair_map=pair_map, node_map=mapping,
                    pair_labels=(self.pair_arrays[name]['labels'][sample['pair_rows']] if sample['pair_rows'] else np.array([], np.int8)),
                    supervised_pair_indices=[pair_map[tuple(native['pairs'][self.pair_arrays[name]['index'][j]])] for j in sample['pair_rows']],
                    sampling_probability=sample['sampling_probability'], group=sample['key'])
