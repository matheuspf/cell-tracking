"""Closed raw/P0 observation hypotheses with explicit identities and provenance."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .common import DATA, RESULTS, ROOT, WORK, adjacency, digest, inputs, load_graph, read_json, save_arrays, sha, verified_graph, write_json

CONFIG = dict(raw_neighbors=3, candidate_distance_um=7., source_localization_tie_um=.25,
              max_changed_node_fraction=.02, tracklet_offsets=[-1, 0, 1],
              synthetic_jitter_voxels_zyx=[0, 2, 2], keep=0, substitute=1, retain_both=2,
              unknown=-1, duplicate_radius=None)


def raw_graph(row):
    path = Path(row['raw']['path'])
    if sha(path) != row['raw']['sha256']:
        raise ValueError('Closed raw input hash mismatch')
    raw = load_graph(path)
    coords = raw['coords'].astype(np.int64)
    nodes = np.column_stack([np.arange(len(coords)), coords])
    original = ROOT / 'strong-tracker-v2/raw' / f'{row["dataset"]}.npz'
    evidence_receipt = Path(row['evidence']['path']).with_suffix('.json')
    if sha(evidence_receipt) != row['evidence']['receipt_sha256']:
        raise ValueError('Pinned incumbent provenance receipt changed')
    expected = read_json(evidence_receipt)['inputs']['raw_sha256']
    if sha(original) != expected:
        raise ValueError('Original raw graph differs from the exact incumbent provenance hash')
    graph = load_graph(original)
    if not np.array_equal(graph['nodes'][:, 1:], coords[graph['nodes'][:, 0].astype(int)]):
        raise ValueError('Original raw graph IDs no longer address the closed raw bank')
    return dict(nodes=nodes, edges=graph['edges'], node_probabilities=raw['node_probabilities'],
                edge_scores=raw['edge_scores'], original_graph_sha256=sha(original))


class ObservationBank:
    def __init__(self, row, graph, raw):
        self.row, self.graph, self.raw = row, graph, raw
        self.nodes, self.raw_nodes = graph['nodes'], raw['nodes']
        _, self.pred, self.succ = adjacency(self.nodes, graph['edges'])
        _, self.raw_pred, self.raw_succ = adjacency(self.raw_nodes, raw['edges'])
        self.exact_alias = {}
        existing = {tuple(map(int, n[1:])): int(i) for i, n in enumerate(self.nodes)}
        for j, n in enumerate(self.raw_nodes):
            if tuple(map(int, n[1:])) in existing:
                self.exact_alias[j] = existing[tuple(map(int, n[1:]))]
        self.pairs = []
        scale = np.asarray(row['physical_scale'])
        for t in range(row['image_shape'][0]):
            ii, jj = np.flatnonzero(self.nodes[:, 1] == t), np.flatnonzero(self.raw_nodes[:, 1] == t)
            if not len(ii) or not len(jj):
                continue
            distance, neighbor = cKDTree(self.raw_nodes[jj, 2:]*scale).query(self.nodes[ii, 2:]*scale,
                k=min(len(jj), CONFIG['raw_neighbors']+1), distance_upper_bound=CONFIG['candidate_distance_um'])
            for i, ds, ns in zip(ii, np.asarray(distance).reshape(len(ii), -1), np.asarray(neighbor).reshape(len(ii), -1)):
                accepted = 0
                for d, k in zip(ds, ns):
                    if not np.isfinite(d) or k >= len(jj):
                        continue
                    j = int(jj[k])
                    if self.exact_alias.get(j) == i:
                        continue
                    self.pairs.append((int(i), j))
                    accepted += 1
                    if accepted == CONFIG['raw_neighbors']:
                        break
        self.pairs = np.asarray(self.pairs, np.int64).reshape(-1, 2)
        self.hash = digest(dict(P0=row['baselines']['P0']['sha256'], raw=row['raw']['sha256'],
                                raw_graph=raw['original_graph_sha256'], config=CONFIG, pairs=self.pairs))

    def tube(self, old, raw):
        """Prediction-only three-frame correspondence; no nearest-ID copying."""
        pairs = [(int(old), int(raw))]
        for old_adj, raw_adj in [(self.pred, self.raw_pred), (self.succ, self.raw_succ)]:
            if len(old_adj[old]) == len(raw_adj[raw]) == 1:
                pairs.append((old_adj[old][0], raw_adj[raw][0]))
        return sorted(pairs, key=lambda p: self.nodes[p[0], 1])

    def evidence(self, old, raw):
        distance = np.linalg.norm((self.nodes[old, 2:]-self.raw_nodes[raw, 2:])*self.row['physical_scale'])
        return np.array([distance/7, self.raw['node_probabilities'][raw],
            raw in self.exact_alias, not self.pred[old], not self.succ[old],
            not self.raw_pred[raw], not self.raw_succ[raw], len(self.tube(old, raw))/3], np.float32)


def prepare(source):
    from .guard import install
    install(source=source)
    from annotation_selection.metric_adapter import match_nodes
    from center_comparison.pipeline import read_gt
    from .labels import SourceLabels
    from .resources import Monitor
    folder = WORK / 'observation_source' / source
    totals = Counter()
    receipts = []
    with Monitor(folder / 'resources.json') as monitor:
        for row in [r for r in inputs() if r['embryo'] == source]:
            path = folder / f'{row["dataset"]}.npz'
            stamp = dict(code_sha256=sha(Path(__file__)), input_manifest_sha256=sha(RESULTS / 'input_manifest.json'))
            if path.exists():
                receipt = read_json(path.with_suffix('.json'))
                if receipt['inputs'] != stamp or sha(path) != receipt['sha256']:
                    raise ValueError('Observation preparation drift')
                receipts.append(receipt)
                totals.update(receipt['counts'])
                continue
            graph, raw = verified_graph(row), raw_graph(row)
            bank = ObservationBank(row, graph, raw)
            gn, ge = read_gt(DATA, row['dataset'], row['physical_scale'])
            labels = SourceLabels(graph['nodes'], graph['edges'], gn, ge, row['physical_scale'])
            old_match = labels.persistent_matches
            raw_match = match_nodes(raw['nodes'], raw['edges'], gn, ge, row['physical_scale'])
            gt = {int(n[0]): n for n in gn}
            annotated_tracks = {int(n) for edge in ge for n in edge}
            already_matched = set(old_match.values())
            targets, groups = [], []
            for old, r in bank.pairs:
                old_id, raw_id = int(graph['nodes'][old, 0]), int(raw['nodes'][r, 0])
                a, b = old_match.get(old_id), raw_match.get(raw_id)
                target = -1
                if a in annotated_tracks and b in annotated_tracks:
                    if a == b:
                        da = np.linalg.norm((graph['nodes'][old, 2:]-gt[a][2:])*row['physical_scale'])
                        db = np.linalg.norm((raw['nodes'][r, 2:]-gt[b][2:])*row['physical_scale'])
                        target = 1 if db+CONFIG['source_localization_tie_um'] < da else 0
                    else:
                        # A different supported real identity is never merged.
                        # Restore is supervised only when that identity is absent
                        # from the incumbent's full one-to-one assignment.
                        target = 2 if b not in already_matched and r not in bank.exact_alias else 0
                targets.append(target)
                groups.append(labels.groups.get(int(old), -1))
            target = np.asarray(targets, np.int8)
            group = np.asarray(groups, np.int64)
            known = (target >= 0) & (group >= 0)
            synthetic = np.array([i for i, n in enumerate(graph['nodes']) if old_match.get(int(n[0])) in annotated_tracks], np.int64)
            synthetic_groups = np.array([labels.groups[i] for i in synthetic], np.int64)
            save_arrays(path, pairs=bank.pairs, labels=target, group=group, synthetic_nodes=synthetic,
                        synthetic_groups=synthetic_groups)
            counts = dict(keep=int((target == 0).sum()), substitute=int((target == 1).sum()),
                          retain_both=int((target == 2).sum()), unknown=int((target < 0).sum()),
                          synthetic_supported_nodes=len(synthetic), supported_real_rows=int(known.sum()))
            receipt = dict(dataset=row['dataset'], source=source, inputs=stamp, counts=counts,
                bank_sha256=bank.hash, sha256=sha(path), raw_graph_sha256=raw['original_graph_sha256'],
                real_unmatched_negative=False, synthetic_identity='Explicit corruption of a source-supported incumbent trajectory',
                direct_target_label_reads=0)
            write_json(path.with_suffix('.json'), receipt)
            receipts.append(receipt)
            totals.update(counts)
            monitor.check()
            print(f'Observation source {source}: {len(receipts)} clips, {counts}', flush=True)
    write_json(folder / 'manifest.json', dict(status='measured', source=source, config=CONFIG, counts=totals,
        clips=receipts, input_manifest_sha256=sha(RESULTS / 'input_manifest.json')))
