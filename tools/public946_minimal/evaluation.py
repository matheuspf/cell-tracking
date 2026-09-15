"""Fresh official rematching in a process that never performs inference."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .common import METRIC_REVISION, digest, load_arrays, read_json, sha, tree_hash, write_json


def gt_arrays(path):
    import numpy as np
    import zarr
    g = zarr.open_group(path, mode='r')
    nodes = np.column_stack([g['nodes/ids'][:], *[g[f'nodes/props/{a}/values'][:] for a in 'tzyx']]).astype(np.int64)
    edges = np.asarray(g['edges/ids'][:], np.int64).reshape(-1, 2)
    estimate = read_json(path / 'zarr.json')['attributes']['geff']['extra']['estimated_number_of_nodes']
    return nodes, edges, estimate


def one(row, prediction, destination, transform=None):
    import numpy as np
    from annotation_selection.metric_adapter import evaluate_graph
    start = time.perf_counter()
    ground_path = Path(row.get('original_path', row['path'])).with_suffix('.geff')
    n, e, estimate = gt_arrays(ground_path)
    gt_hashes = tree_hash(ground_path)
    if transform:
        axis = 4 if transform == 'reflect_x' else 3
        n[:, axis] = row['image_shape'][axis - 1] - 1 - n[:, axis]
    graph = load_arrays(prediction)
    fingerprint = digest(dict(prediction_sha256=sha(prediction), gt_sha256=gt_hashes['sha256'],
                              metric=METRIC_REVISION, scale=row['scale'], transform=transform,
                              original_serializer='v29_max_zero_natural_round_r1'))
    if destination.exists():
        previous = read_json(destination)
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Scoring receipt drift')
        return previous
    result, matches, tp = evaluate_graph(row['dataset'], graph['nodes'], graph['edges'], n, e, row['scale'], estimate)
    # The locked worker retains rounded, pre-clamp coordinates in original_nodes.
    # v29's actual CSV writer applies max(0, int(round(value))) at lines 3122-3124.
    # Reconstruct that writer here; never feed this diagnostic into inference.
    original_nodes = graph['original_nodes'].copy()
    lower_clamped = np.any(original_nodes[:, 2:] < 0, axis=1)
    original_nodes[:, 2:] = np.maximum(0, original_nodes[:, 2:])
    original, _, _ = evaluate_graph(row['dataset'], original_nodes, graph['edges'], n, e, row['scale'], estimate)
    gt_by_id = {int(r[0]): r for r in n}
    pred_by_id = {int(r[0]): r for r in graph['nodes']}
    matched_distances = [float(np.linalg.norm((pred_by_id[i][2:] - gt_by_id[j][2:]) * row['scale'])) for i, j in matches.items()]
    true_edges = sorted((matches[a], matches[b]) for a, b in tp)
    result.update(embryo=row['embryo'], contrast_q99_q10=row['contrast_q99_q10'],
                  original_export=original, tp_gt_edges=true_edges,
                  original_export_lower_clamped_nodes=int(lower_clamped.sum()),
                  original_export_outside_nodes=int(np.any(original_nodes[:, 2:] >= np.asarray(row['image_shape'][1:]), axis=1).sum()),
                  localization_sum_um=sum(matched_distances), localization_count=len(matched_distances),
                  localization_mean_um=float(np.mean(matched_distances)) if matched_distances else None,
                  fingerprint=fingerprint, prediction_sha256=sha(prediction), gt_sha256=gt_hashes['sha256'],
                  seconds=time.perf_counter() - start)
    write_json(destination, result)
    return result


def summarize(records, names):
    from annotation_selection.metric_adapter import aggregate
    pooled = aggregate(records, names)
    by_embryo = {}
    original_by_embryo = {}
    for embryo in sorted({r['embryo'] for r in records}):
        sub = [r for r in records if r['embryo'] == embryo]
        by_embryo[embryo] = aggregate(sub, [r['dataset'] for r in sub])
        original_by_embryo[embryo] = aggregate([r['original_export'] for r in sub], [r['dataset'] for r in sub])
    original = aggregate([r['original_export'] for r in records], names)
    return dict(pooled=pooled, per_embryo=by_embryo, original_export=original,
                original_export_per_embryo=original_by_embryo)


def run(args):
    if os.environ.get('PUBLIC946_INFERENCE') == '1':
        raise RuntimeError('Evaluator cannot run in an inference worker')
    job = read_json(args.job)
    rows = job['rows']
    root = Path(job['score_root'])
    records = []
    for row in rows:
        p = Path(job['prediction_root']) / row['dataset'] / 'final.npz'
        if not p.exists():
            raise FileNotFoundError(f'Missing full clip: {p}')
        records.append(one(row, p, root / (row['dataset'] + '.json'), job.get('transform')))
        if len(records) % 20 == 0:
            print(f'Fresh official matching {len(records)}/{len(rows)}', flush=True)
    result = summarize(records, [r['dataset'] for r in rows])
    result.update(completed=[r['dataset'] for r in rows], expected=[r['dataset'] for r in rows],
                  failed=[], metric_revision=METRIC_REVISION, job_sha256=sha(args.job))
    if job.get('baseline_score_root'):
        base = Path(job['baseline_score_root'])
        survival, lost, recovered = 0, 0, 0
        for r in records:
            original = read_json(base / (r['dataset'] + '.json'))
            before = set(map(tuple, original['tp_gt_edges']))
            after = set(map(tuple, r['tp_gt_edges']))
            survival += len(before & after)
            lost += len(before - after)
            recovered += len(after - before)
        result.update(original_tp_survived=survival, original_tp_lost=lost, recovered_tp=recovered)
    write_json(root / 'summary.json', result)
    print(f'Official {job["arm"]}: {result["pooled"]["score"]:.12f} ({len(rows)} full clips)', flush=True)
