"""Build source supervision over the frozen prediction-only bank and full edits."""
from collections import Counter
from dataclasses import asdict
import gzip
import json
import time

import numpy as np

from .actions import Alternatives
from .bank import EventBank
from .common import (
    DATA, RESULTS, WORK, digest, inputs, read_json, save_arrays, sha,
    verified_evidence, verified_graph, write_json,
)
from .labels import SourceLabels, supported_incoming


def decision_dict(decision):
    result = asdict(decision)
    for key in ['remove', 'add', 'resources']:
        result[key] = sorted(result[key])
    return result


def one(row, source):
    if row['embryo'] != source:
        raise PermissionError('Target row passed to source preparation')
    from center_comparison.pipeline import read_gt
    name = row['dataset']
    folder = WORK / 'source' / source
    receipt_path = folder / 'receipts' / f'{name}.json'
    stamp = dict(input_manifest_sha256=sha(RESULTS / 'input_manifest.json'),
                 code={name: sha(__import__('pathlib').Path(__file__).with_name(name+'.py'))
                       for name in ['prepare', 'labels', 'bank', 'actions']})
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['inputs'] != stamp:
            raise ValueError('Source preparation implementation changed; preserve and archive invalid artifacts first')
        for path, expected in receipt['files'].items():
            if sha(path) != expected:
                raise ValueError('Source preparation artifact changed')
        return receipt
    begin = time.monotonic()
    graph, native = verified_graph(row), verified_evidence(row)
    gn, ge = read_gt(DATA, name, row['physical_scale'])
    labels = SourceLabels(graph['nodes'], graph['edges'], gn, ge, row['physical_scale'])
    bank = EventBank(graph['nodes'], graph['edges'], native, row['physical_scale'])
    alternatives = Alternatives(bank)
    pair_y = supported_incoming(graph['nodes'][native['pairs'], 0], labels.persistent_matches, ge)
    pair_groups = np.array([labels.groups.get(int(a), labels.groups.get(int(b), -1)) for a, b in native['pairs']], np.int64)
    pair_index = np.flatnonzero(pair_y >= 0)
    pair_file = folder / 'pairs' / f'{name}.npz'
    save_arrays(pair_file, index=pair_index, labels=pair_y[pair_index], group=pair_groups[pair_index])
    # All supported event time anchors are retained. Ordinary trajectories use
    # one deterministic temporal residue, with inclusion 1/9 recorded explicitly.
    event_parents = set()
    for roles in labels.roles.values():
        if roles is not None:
            ps, _ = roles
            event_parents.update(ps)
            event_parents.update(j for p in ps for j in bank.near[p])
    parents = event_parents & bank.expanded
    parents.update(p for p, group in labels.groups.items() if p in bank.expanded
                   and int(graph['nodes'][p, 1]) % 9 == group % 9)
    dest = folder / 'decisions' / f'{name}.jsonl.gz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    positive_groups = set()
    seen = set()
    with gzip.open(dest, 'wt') as stream:
        for parent in sorted(parents):
            for event, features, _ in bank.iter_parent(parent):
                for decision in alternatives.event(event):
                    # Multiple event descriptions of the same complete graph are one alternative.
                    key = (parent, decision.key)
                    if key in seen:
                        continue
                    seen.add(key)
                    label = labels.decision(decision)
                    if all(label[k] < 0 for k in ['identity', 'biological', 'metric_fork_target']):
                        counts['fully_unknown'] += 1
                        continue
                    group = labels.groups.get(parent)
                    if label['compatible_events']:
                        positive_groups.update(label['compatible_events'])
                        group = min(labels.event_groups[e] for e in labels.roles
                                    if labels.gt_reverse[e] in label['compatible_events'])
                    if group is None:
                        # Unsupported group identity cannot acquire a fabricated group label.
                        counts['group_unknown'] += 1
                        continue
                    record = dict(decision=decision_dict(decision), event_features=features.tolist(), labels=label,
                                  group=f'{name}:{group}', anchor=parent,
                                  sampling_probability=1. if parent in event_parents else 1/9,
                                  anchor_selection='all_event_compatible_anchors' if parent in event_parents else 'fixed_temporal_residue')
                    stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
                    counts['decisions'] += 1
                    counts[f'biological_{label["biological"]}'] += 1
                    counts[f'risk_{label["metric_fork_target"]}'] += 1
                    counts[decision.kind] += 1
    receipt = dict(dataset=name, source=source, inputs=stamp, seconds=time.monotonic()-begin,
                   files={str(p): sha(p) for p in [pair_file, dest]}, bank_sha256=bank.hash,
                   supported_pair_positive=int((pair_y == 1).sum()), supported_pair_negative=int((pair_y == 0).sum()),
                   unknown_pairs=int((pair_y < 0).sum()), counts=dict(counts),
                   covered_division_groups=len(positive_groups), annotated_divisions=len(labels.roles),
                   parent_anchors=len(parents), all_prediction_anchors=len(bank.expanded),
                   structural_rejections=dict(alternatives.rejections),
                   biological_negatives='Supported contradictory identities only; quiet chains remain unknown.',
                   metric_negatives='Officially evaluable forks recorded separately from biological targets.')
    write_json(receipt_path, receipt)
    return receipt


def run(source):
    from .guard import install
    from .resources import Monitor
    guard = install(source=source)
    rows = [r for r in inputs() if r['embryo'] == source]
    receipts = []
    with Monitor(WORK / 'resources' / f'prepare-{source}.json') as monitor:
        for i, row in enumerate(rows, 1):
            receipt = one(row, source)
            receipts.append(receipt)
            monitor.check()
            print(f'Source {source}: {i}/{len(rows)} clips; {receipt["counts"].get("decisions", 0)} complete decisions; '
                  f'{receipt["covered_division_groups"]}/{receipt["annotated_divisions"]} available events', flush=True)
    write_json(WORK / 'source' / source / 'manifest.json', dict(source=source, guard=guard, clips=receipts,
        split_manifest_sha256=sha(RESULTS / 'split_manifest.json'),
        direct_target_label_reads=0, biological_event_grouping='GT weak lineage component within each clip; whole overlap groups partitioned together.'))
