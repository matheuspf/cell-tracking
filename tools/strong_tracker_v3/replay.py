"""Isolated, fingerprinted replays of the actual Harmonic repair call graph.

Only declarations/configuration are extracted; the original notebook is never
executed as a workflow, and no v1/v2 module global is imported or changed.
"""
from __future__ import annotations

import ast
import contextlib
import os
import time
from pathlib import Path

import numpy as np

from .common import (digest, graph_hash, load_graph, now, read_json, run_pool,
                     save_arrays, sha, validate, write_json)


VARIANTS = {
    'no_motion_no_safe_divisions': {'OUTPUT_SAFE_DIVISIONS': False},
    'no_motion_no_smoothing': {'OUTPUT_LINEFIT_SMOOTH': False},
    'no_motion_no_safe_divisions_no_smoothing': {
        'OUTPUT_SAFE_DIVISIONS': False, 'OUTPUT_LINEFIT_SMOOTH': False},
    'no_motion_no_pruning': {
        'OUTPUT_PRUNE_ISOLATED': False, 'OUTPUT_FILTER_SHORT_TRACKS': False},
    'no_motion_no_gaps': {'OUTPUT_GAP_CLOSE': False, 'OUTPUT_GAP2_RECOVERY': False},
}
CALL_ORDER = ['edge validity', 'motion disabled', 'single-parent guard',
              'close_single_frame_gaps', 'recover_strict_gap2',
              'add_safe_divisions_postlink', 'isolated-node pruning',
              'filter_short_track_components', 'linefit_smooth_output_graph',
              'integer serialization and bounds clipping']
_DETECTOR = None


def notebook_environment(source):
    """Read literal notebook environment assignments in execution order."""
    settings = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == 'os' and target.value.attr == 'environ'):
            try:
                key, value = ast.literal_eval(target.slice), ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(key, str) and isinstance(value, str):
                settings[key] = value
    return settings


def repair_namespace(ctx, *, settings=None, image_dir=None, fresh_heatmaps=False):
    global _DETECTOR
    path = ctx.full / 'harmonic_isolated.py'
    source = path.read_text()
    tree = ast.parse(source)
    # Source boundaries are checked by function membership and source hash in
    # every receipt. They include DeepCenter classes, never notebook execution.
    nodes = [node for node in tree.body if 19 <= node.lineno <= 77
             or 133 <= node.lineno <= 280 or 1522 <= node.lineno <= 3077]
    assert any(isinstance(n, ast.FunctionDef) and n.name == 'filter_output_graph' for n in nodes)
    assert not any(isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'DEEPCENTER_VETO_DETECTOR'
        for t in n.targets) for n in nodes)
    namespace = {'__name__': 'strong_tracker_v3_isolated_repair'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    namespace.update(TEST_DIR=Path(image_dir) if image_dir else ctx.data / 'train',
                     WORKING_DIR=ctx.out / 'replay', REPO_DIR=ctx.full / 'tracking_repo',
                     OUTPUT_MOTION_RELINK=False)
    namespace.update(settings or {})
    original_heatmap = namespace['deepcenter_heatmap_for_frame']
    if _DETECTOR is None:
        _DETECTOR = namespace['load_deepcenter_veto_detector']()
    state = {'bundle': _DETECTOR, 'cache_reads': 0, 'fresh_heatmaps': 0, 'input_heatmap_hashes': {}}
    heatmap_destination = ctx.out / ('fresh/heatmaps' if fresh_heatmaps else 'heatmaps')

    def heatmap(dataset, t, detector_bundle, frame_cache, heatmap_cache):
        for root in ([] if fresh_heatmaps else [ctx.v2 / 'heatmaps', heatmap_destination]):
            p = root / dataset / f'{t}.npy'
            if p.exists():
                state['cache_reads'] += 1
                if str(p) not in state['input_heatmap_hashes']:
                    state['input_heatmap_hashes'][str(p)] = sha(p)
                return np.load(p, allow_pickle=False)
        if state['bundle'] is None:
            state['bundle'] = namespace['load_deepcenter_veto_detector']()
        result = original_heatmap(dataset, t, state['bundle'], frame_cache, heatmap_cache)
        if result is not None:
            p = heatmap_destination / dataset / f'{t}.npy'
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, result, allow_pickle=False)
            state['fresh_heatmaps'] += 1
        return result

    namespace['deepcenter_heatmap_for_frame'] = heatmap
    namespace['_heatmap_state'] = state
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    return namespace


def as_arrays(nodes, edges):
    n = np.array([[int(i), int(nodes[i]['t']), *[float(nodes[i][a]) for a in 'zyx']]
                  for i in sorted(nodes)], float).reshape(-1, 5)
    e = np.array([[int(x['source_id']), int(x['target_id'])] for x in edges], np.int64).reshape(-1, 2)
    p = np.array([np.nan if x.get('edge_prob') is None else float(x['edge_prob']) for x in edges])
    return dict(nodes=n, edges=e, edge_prob=p)


def apply_repair(namespace, raw, name, shape):
    n = {int(r[0]): dict(node_id=int(r[0]), t=int(r[1]),
                         **dict(zip('zyx', map(float, r[2:])))) for r in raw['nodes']}
    probabilities = raw.get('edge_prob', np.full(len(raw['edges']), np.nan))
    e = [dict(source_id=int(a), target_id=int(b), edge_prob=float(p))
         for (a, b), p in zip(raw['edges'], probabilities)]
    nn, ee, stats = namespace['filter_output_graph'](
        n, e, dataset=name, deepcenter_bundle=namespace['_heatmap_state']['bundle'])
    result = as_arrays(nn, ee)
    exported = result['nodes'].copy()
    exported[:, 2:] = np.clip(np.rint(exported[:, 2:]), 0, np.asarray(shape[1:]) - 1)
    result['nodes'] = exported.astype(np.int64)
    validate(result['nodes'], result['edges'], shape)
    return result, stats


def one(task):
    ctx, row, variants = task
    from .inference import deny_annotations
    deny_annotations(allowed_geff_root=ctx.out / 'fresh')
    name = row['dataset']
    raw_path = ctx.v2 / 'raw' / f'{name}.npz'
    raw = load_graph(raw_path)
    output = []
    for variant in variants:
        settings = VARIANTS[variant]
        root = ctx.out / 'replay' / variant / name
        path = ctx.out / 'candidate_graphs' / variant / f'{name}.npz'
        fingerprint = digest(dict(raw=sha(raw_path), notebook=sha(ctx.full / 'harmonic_isolated.py'),
                                  replay=sha(Path(__file__)), settings=settings,
                                  shape=row['image_shape'], motion=False))
        if (root / 'complete.json').exists():
            receipt = read_json(root / 'complete.json')
            assert receipt['fingerprint'] == fingerprint, f'Stale replay {name}/{variant}'
            assert receipt['output_sha256'] == sha(path)
            assert all(sha(p) == h for p, h in receipt['input_heatmap_hashes'].items())
            assert sha(receipt['detector_path']) == receipt['detector_sha256']
            output.append(receipt)
            continue
        start = time.perf_counter()
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'run.log').open('w') as log, contextlib.redirect_stdout(log):
            namespace = repair_namespace(ctx, settings=settings)
            graph, stats = apply_repair(namespace, raw, name, row['image_shape'])
        save_arrays(path, nodes=graph['nodes'], edges=graph['edges'])
        receipt = dict(dataset=name, variant=variant, fingerprint=fingerprint,
                       output_sha256=sha(path), graph_hash=graph_hash(graph['nodes'], graph['edges']),
                       seconds=time.perf_counter()-start, stats=stats, settings=settings,
                       motion=False, annotation_access_blocked=True, created=now(),
                       heatmap_cache_reads=namespace['_heatmap_state']['cache_reads'],
                       fresh_heatmaps=namespace['_heatmap_state']['fresh_heatmaps'],
                       input_heatmap_hashes=namespace['_heatmap_state']['input_heatmap_hashes'],
                       detector_path=str(namespace['_heatmap_state']['bundle']['path']),
                       detector_sha256=sha(namespace['_heatmap_state']['bundle']['path']))
        write_json(root / 'complete.json', receipt)
        output.append(receipt)
    return dict(dataset=name, variants=len(output), seconds=sum(r['seconds'] for r in output))


def run(ctx, args):
    variants = [args.variant] if getattr(args, 'variant', None) else list(VARIANTS)
    rows = ctx.samples()
    if getattr(args, 'limit', None):
        rows = rows[:args.limit]
    lock_path = ctx.out / 'conditional_ablation_lock.json'
    lock = dict(variants=VARIANTS, motion=False, notebook_sha256=sha(ctx.full / 'harmonic_isolated.py'),
                code_sha256=sha(Path(__file__)), call_order=CALL_ORDER, both_directions_frozen=True,
                comparative_scores_exposed=False)
    if lock_path.exists():
        assert read_json(lock_path) == lock
    else:
        write_json(lock_path, lock)
    list(run_pool(one, [(ctx, row, variants) for row in rows], getattr(args, 'workers', 4)))
    receipts = [read_json(ctx.out / 'replay' / v / r['dataset'] / 'complete.json')
                for v in variants for r in rows]
    write_json(ctx.out / 'conditional_ablation_receipt.json', dict(
        samples=len(rows), variants=variants, annotation_access_blocked=True,
        settings=VARIANTS, call_order=CALL_ORDER,
        notebook_sha256=sha(ctx.full / 'harmonic_isolated.py'),
        raw_algorithm_bodies_unchanged=True, output_count=len(receipts),
        seconds=sum(r['seconds'] for r in receipts)))


if __name__ == '__main__':
    import argparse
    from .context import RunContext
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    run(RunContext.default(), parser.parse_args())
