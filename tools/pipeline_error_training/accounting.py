"""Separate complete-graph scoring and identity-based changed-error accounting."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing as mp
from pathlib import Path
import time
import warnings

import numpy as np

from .common import DATA, RESULTS, WORK, graph_hash, inputs, load_graph, read_json, save_arrays, sha, validate, verified_graph, write_json
from .evaluate import summary, write_csv
from .resources import Monitor, cpu_budget


def require_target_freeze(arm):
    if arm in {'P0', 'C4_m6', 'C0', 'D00'}:
        return
    path = RESULTS / 'target_freeze.json'
    if not path.exists():
        raise PermissionError('New target comparisons require the complete directional/replication freeze')
    freeze = read_json(path)
    if arm not in freeze['experiments']:
        raise PermissionError('Candidate was not registered before target scoring')
    for package in freeze['packages']:
        if sha(package['manifest_path']) != package['manifest_sha256']:
            raise ValueError('A frozen model recipe or calibration changed')
        if sha(package['weights_path']) != package['weights_sha256']:
            raise ValueError('A frozen model checkpoint changed')


def raw_centers(row, gn, ge):
    from annotation_selection.metric_adapter import match_nodes
    from center_comparison.detection_index import spatial_evidence
    if sha(row['raw']['path']) != row['raw']['sha256']:
        raise ValueError('Raw input used for stage attribution changed')
    raw = load_graph(row['raw']['path'])['coords'].astype(np.int64)
    raw = np.column_stack([np.arange(len(raw)), raw])
    # Use the same raw-center diagnostic as the September 15 index, without
    # inventing topology for pre-ILP observations.
    matches = match_nodes(raw, np.empty((0, 2), np.int64), gn, ge, row['physical_scale'])
    return spatial_evidence(gn, raw, matches, np.asarray(row['physical_scale']))


def audit(row, graph, score, gn, ge, raw):
    from annotation_selection.metric_adapter import make_graph
    from center_comparison.detection_index import audit_graph, classify_center, match_array, spatial_evidence
    from center_comparison.tracking_index import division_flags, edge_flags
    from tracking_cellmot import division_metrics as dm
    spacing = np.asarray(row['physical_scale'])
    matches, status = audit_graph(graph['nodes'], graph['edges'], gn, ge, spacing, score)
    spatial = spatial_evidence(gn, graph['nodes'], matches, spacing)
    centers = {}
    for g in gn:
        gid = int(g[0])
        p, r = spatial[gid], raw[gid]
        crowded = p['count'] > 1 or p['competitors'] > 0 or p['assigned_not_nearest'] or r['count'] > 1 or r['competitors'] > 0
        kind = classify_center(p['assigned'] is not None, r['assigned'] is not None, r['nearest'], p['nearest'],
                               p['distance'] or 0., crowded)
        centers[gid] = dict(kind=kind, raw_count=r['count'], final_count=p['count'])
    final = dict(graph, matches=match_array(matches, graph['nodes'], gn, spacing), edge_status=status)
    common = dict(gt=gn, gt_edges=ge)
    flags = edge_flags(common, final, centers, spacing)+division_flags(common, final, spacing, score)
    pred, pmap = make_graph(graph['nodes'], graph['edges'])
    truth, gmap = make_graph(gn, ge)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        divisions = dm.score_divisions(pred, truth, scale=tuple(spacing), max_distance=7.)
    lookup = {int(n[0]): tuple(map(int, n)) for n in graph['nodes']}
    tp_edges = {(matches[int(a)], matches[int(b)]) for a, b, s in status if s == 1}
    fp_edges = {(lookup[int(a)], lookup[int(b)]) for a, b, s in status if s == -1}
    tp_divisions = {gmap[int(g)] for g, correct in divisions.scores.items() if correct}
    fp_divisions = {lookup[pmap[int(p)]] for p in divisions.fp_forks}
    for key, actual in [('edge_tp', len(tp_edges)), ('edge_fp', len(fp_edges)),
                        ('division_tp', len(tp_divisions)), ('division_fp', len(fp_divisions))]:
        if score[key] != actual:
            raise ValueError('Official error-identity partition does not match its counts')
    stages = Counter(f'{f["category"]}:{f["stage"]}' for f in flags)
    return dict(tp_edges=sorted(tp_edges), fp_edges=sorted(fp_edges), tp_divisions=sorted(tp_divisions),
        fp_divisions=sorted(fp_divisions), stages=dict(stages), flags=flags,
        centers=dict(Counter(c['kind'] for c in centers.values())),
        matched_gt=sorted(matches.values()), unknown_unmatched_predictions=len(graph['nodes'])-len(matches),
        matches=sorted(matches.items()),
        graph_hash=graph_hash(graph['nodes'], graph['edges']))


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value


def transitions(before, after):
    result = {}
    for name in ['tp_edges', 'fp_edges', 'tp_divisions', 'fp_divisions', 'matched_gt']:
        old, new = set(map(tuples, before[name])), set(map(tuples, after[name]))
        if name.startswith('fp'):
            result[name+'_removed'], result[name+'_introduced'] = len(old-new), len(new-old)
        else:
            result[name+'_recovered'], result[name+'_lost'] = len(new-old), len(old-new)
    for stage in sorted(set(before['stages']) | set(after['stages'])):
        result[stage+'_before'] = before['stages'].get(stage, 0)
        result[stage+'_after'] = after['stages'].get(stage, 0)
    # Nonadditive partitions: a division repair can also change temporal edges.
    result['nonadditive_error_families'] = True
    return result


def one(task):
    row, arm, path = task
    cpu_budget(4)
    require_target_freeze(arm)
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    name = row['dataset']
    root = WORK / 'full_evaluation' / arm
    out = root / f'{name}.json'
    if out.exists():
        prior = read_json(out)
        if sha(path) != prior['prediction_sha256']:
            raise ValueError('An already evaluated prediction changed')
        return prior
    begin = time.monotonic()
    graph = load_graph(path)
    p0 = verified_graph(row)
    validity = validate(graph['nodes'], graph['edges'], row['image_shape'], allow_legacy_bounds=True)
    if arm.startswith(('D', 'A')) and not np.array_equal(graph['nodes'], p0['nodes']):
        raise ValueError('Fixed-observation module changed incumbent nodes')
    outside = np.any((graph['nodes'][:, 2:] < 0) | (graph['nodes'][:, 2:] >= np.array(row['image_shape'][1:])), axis=1)
    original = set(map(tuple, p0['nodes']))
    if any(tuple(n) not in original for n in graph['nodes'][outside]):
        raise ValueError('Candidate introduced an out-of-bounds observation')
    gn, ge = read_gt(DATA, name, row['physical_scale'])
    score, _, _ = evaluate_graph(name, graph['nodes'], graph['edges'], gn, ge, row['physical_scale'], row['estimated_total'])
    score.update(arm=arm, embryo=row['embryo'], status='measured', prediction_sha256=sha(path), validity=validity)
    raw = raw_centers(row, gn, ge)
    details = audit(row, graph, score, gn, ge, raw)
    write_json(root / 'audit' / f'{name}.json', details)
    if arm != 'P0':
        baseline = read_json(WORK / 'full_evaluation/P0/audit' / f'{name}.json')
        score['transitions'] = transitions(baseline, details)
    score['stages'], score['center_categories'] = details['stages'], details['centers']
    score['seconds'] = time.monotonic()-begin
    score['unchanged_nodes'] = np.array_equal(graph['nodes'], p0['nodes'])
    write_json(out, score, immutable=True)
    return score


def run(arm, prediction_root=None):
    # Early complete-inventory audits and the final queue share these exports.
    # One scorer at a time also keeps the total worker count at four.
    root = WORK / 'full_evaluation'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'scoring.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_locked(arm, prediction_root)


def run_locked(arm, prediction_root=None):
    require_target_freeze(arm)
    rows = inputs()
    if prediction_root is None:
        if arm in ['P0', 'C4_m6', 'C0']:
            tasks = [(r, arm, Path(r['baselines'][arm]['path'])) for r in rows]
        else:
            prediction_root = WORK / 'predictions' / arm
    if prediction_root is not None:
        tasks = [(r, arm, Path(prediction_root)/f'{r["dataset"]}.npz') for r in rows]
    missing = [r['dataset'] for r, _, p in tasks if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Complete {arm} inventory required; {len(missing)} clips missing')
    measured = []
    with Monitor(WORK / 'resources' / f'full-evaluation-{arm}.json') as monitor:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
            for item in pool.map(one, tasks):
                measured.append(item)
                monitor.check()
                if len(measured) % 20 == 0 or len(measured) == len(rows):
                    print(f'Full official {arm} score and changed-error audit: {len(measured)}/{len(rows)}', flush=True)
    if {r['dataset'] for r in measured} != {r['dataset'] for r in rows}:
        raise ValueError('Incomplete full evaluation')
    aggregate = [summary(measured, arm, e) for e in ['pooled', '44b6', '6bba']]
    write_json(WORK / 'full_evaluation' / arm / 'summary.json', dict(status='measured', arm=arm,
        groups=aggregate, independent_aggregation_parity=True, complete_expected_inventory=True))
    export()
    return aggregate


def export():
    baseline = read_json(RESULTS / 'baseline_validation.json')
    complete = {r['arm']: [r, *[e for e in baseline['embryos'] if e['arm'] == r['arm']]] for r in baseline['pooled']}
    for path in sorted((WORK/'full_evaluation').glob('*/summary.json')):
        result = read_json(path)
        complete[result['arm']] = result['groups']
    table = []
    for arm, groups in complete.items():
        for item in groups:
            group = dict(item)
            p0 = next(r for r in complete['P0'] if r['embryo'] == group['embryo'])
            c4 = next(r for r in complete['C4_m6'] if r['embryo'] == group['embryo'])
            group.update(delta_P0=group['score']-p0['score'], delta_C4_m6=group['score']-c4['score'],
                         node_adjustment=group['adj_edge_jaccard']-group['edge_jaccard'])
            table.append(group)
    write_csv(RESULTS/'scores.csv', [r for r in table if r['embryo'] == 'pooled'])
    write_csv(RESULTS/'per_embryo_scores.csv', [r for r in table if r['embryo'] != 'pooled'])
    changed, partitions, clips = [], {}, []
    for arm in complete:
        per = [read_json(p) for p in sorted((WORK/'full_evaluation'/arm).glob('*.json')) if p.name != 'summary.json']
        if len(per) != len(inputs()):
            if arm in ['P0', 'C4_m6', 'C0']:
                clips.extend(read_json(p) for p in sorted((WORK/'evaluation'/arm).glob('*.json')))
            continue
        clips.extend({k: v for k, v in r.items() if k not in ['validity', 'transitions', 'stages', 'center_categories']} for r in per)
        partitions[arm] = {}
        for embryo in ['pooled', '44b6', '6bba']:
            picked = [r for r in per if embryo == 'pooled' or r['embryo'] == embryo]
            counts, stages, centers = Counter(), Counter(), Counter()
            for r in picked:
                counts.update({k: v for k, v in r.get('transitions', {}).items() if k != 'nonadditive_error_families'})
                stages.update(r['stages']); centers.update(r['center_categories'])
            if counts:
                changed.append(dict(arm=arm, embryo=embryo, clips=len(picked), **counts))
            partitions[arm][embryo] = dict(stages=stages, centers=centers)
    if changed:
        write_csv(RESULTS/'error_transitions.csv', changed)
    if clips:
        write_csv(RESULTS/'per_clip_scores.csv', clips)
    write_json(RESULTS/'stage_attribution.json', dict(status='measured', complete_graph_partitions=partitions,
        sparse_unmatched_predictions='Unknown unless the official metric marks an evaluable FP edge/fork.',
        nonadditive_error_families=True, raw_center_attribution='Pinned original pre-ILP coordinates; no inferred detector oracle.'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    cpu_budget()
    run(args.arm, args.predictions)
