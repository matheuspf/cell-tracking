"""Source-label heuristic witnesses on frozen banks, serialized and rescored."""
from collections import Counter
import gzip
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from .actions import Decision, apply_decisions
from .common import (
    DATA, WORK, inputs, load_graph, read_json, save_graph, sha,
    verified_evidence, verified_graph, write_json,
)


def from_record(record):
    d = record['decision']
    return Decision(d['kind'], tuple(d['event']), frozenset(map(tuple, d['remove'])),
                    frozenset(map(tuple, d['add'])), tuple(d['owners']), tuple(d['births']),
                    tuple(d['terminations']), frozenset(map(tuple, d['resources'])))


def records(source, name):
    path = WORK / 'source' / source / 'decisions' / f'{name}.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def raw_swap_witness(nodes, raw, gt_nodes, matches, scale, max_fraction=.02):
    """A localization feasibility witness. Proximity is not proven biological identity."""
    selected = nodes.copy()
    gt = {int(n[0]): n for n in gt_nodes}
    choices = []
    for t in np.unique(nodes[:, 1]):
        ri = np.flatnonzero(raw[:, 0] == t)
        if not len(ri):
            continue
        tree = cKDTree(raw[ri, 1:] * scale)
        for i in np.flatnonzero(nodes[:, 1] == t):
            gid = matches.get(int(nodes[i, 0]))
            if gid is None:
                continue
            target = gt[gid][2:]
            distance, j = tree.query(target * scale)
            before = np.linalg.norm((nodes[i, 2:]-target)*scale)
            if distance + 1e-9 < before:
                choices.append((before-distance, int(i), int(ri[int(j)])))
    used_raw = set()
    ledger = []
    existing = {tuple(n[1:]) for n in nodes}
    for gain, i, j in sorted(choices, key=lambda x: (-x[0], int(nodes[x[1], 0]), x[2])):
        if len(ledger) >= int(np.floor(max_fraction*len(nodes))):
            break
        candidate = tuple(map(int, raw[j]))
        if j in used_raw or candidate in existing:
            continue
        existing.remove(tuple(selected[i, 1:]))
        selected[i, 1:] = raw[j]
        existing.add(candidate)
        used_raw.add(j)
        ledger.append(dict(node_id=int(nodes[i, 0]), raw_id=j, old=nodes[i].tolist(),
                           new=selected[i].tolist(), source_distance_gain_um=float(gain)))
    return selected, ledger


def one(row, source):
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    from .association import decode
    from .labels import supported_incoming
    if row['embryo'] != source:
        raise PermissionError('Source witness attempted target labels')
    graph, native = verified_graph(row), verified_evidence(row)
    gn, ge = read_gt(DATA, row['dataset'], row['physical_scale'])
    _, matches, _ = evaluate_graph(row['dataset'], graph['nodes'], graph['edges'], gn, ge,
                                   row['physical_scale'], row['estimated_total'])
    examples = list(records(source, row['dataset']))
    positives = [r for r in examples if r['decision']['kind'] == 'division' and r['labels']['biological'] == 1]
    decisions = [from_record(r) for r in positives]
    values = [100.-len(r['decision']['remove'])-len(r['decision']['add']) for r in positives]
    division_edges, dl = apply_decisions(graph['nodes'], graph['edges'], decisions, values)
    y = supported_incoming(graph['nodes'][native['pairs'], 0], matches, ge)
    scores = np.where(y == 1, 20., np.where(y == 0, -20., native['edge_features'][:, 23]))
    association_edges, al = decode(graph['nodes'], graph['edges'], native, scores)
    with np.load(row['raw']['path'], allow_pickle=False) as raw:
        if sha(row['raw']['path']) != row['raw']['sha256']:
            raise ValueError('Closed raw bank changed')
        swap_nodes, ol = raw_swap_witness(graph['nodes'], raw['coords'], gn, matches, np.asarray(row['physical_scale']))
    output = []
    for lane, nodes, edges, ledger in [('division', graph['nodes'], division_edges, dl),
                                       ('continuation', graph['nodes'], association_edges, al),
                                       ('observation_swap', swap_nodes, graph['edges'], ol)]:
        start = time.monotonic()
        path = WORK / 'source_feasibility' / source / lane / f'{row["dataset"]}.npz'
        save_graph(path, nodes, edges)
        # Score the serialized arrays, not pre-serialization count arithmetic.
        actual = load_graph(path)
        result, _, _ = evaluate_graph(row['dataset'], actual['nodes'], actual['edges'], gn, ge,
                                      row['physical_scale'], row['estimated_total'])
        result.update(arm=f'source_witness_{lane}', embryo=source, source=source, status='measured',
                      graph_sha256=sha(path), seconds=time.monotonic()-start,
                      scope='Source-label heuristic feasibility witness; not a trained result or upper bound.',
                      gt_injected_proposals=False)
        write_json(path.with_suffix('.json'), dict(result=result, ledger=ledger))
        output.append(result)
    return output


def run(source):
    from annotation_selection.metric_adapter import aggregate
    from .guard import install
    from .resources import Monitor
    install(source=source)
    result = []
    rows = [r for r in inputs() if r['embryo'] == source]
    with Monitor(WORK / 'resources' / f'feasibility-{source}.json') as monitor:
        for i, row in enumerate(rows, 1):
            result.extend(one(row, source))
            monitor.check()
            if i % 10 == 0 or i == len(rows):
                print(f'Rescored source {source} witnesses: {i}/{len(rows)} clips', flush=True)
    summaries = {}
    for arm in sorted({r['arm'] for r in result}):
        chosen = [r for r in result if r['arm'] == arm]
        summaries[arm] = aggregate(chosen, sorted(r['dataset'] for r in chosen))
    write_json(WORK / 'source_feasibility' / source / 'summary.json', dict(source=source,
        full_clip_graph_scores=True, summaries=summaries, experiments=Counter(r['arm'] for r in result)))
