"""Image/model-only worker. Evaluation runs in a different process."""
from __future__ import annotations

import contextlib
import copy
import os
import resource
import sys
import time
from pathlib import Path

from .common import (array_hash, check_resources, digest, load_arrays, now, read_json,
                     save_arrays, sha, write_json)


def arrays(nodes, edges):
    import numpy as np
    return dict(nodes=np.asarray([[i, nodes[i]['t'], *[nodes[i][a] for a in 'zyx']]
                                 for i in sorted(nodes)], dtype=np.float64).reshape(-1, 5),
                edges=np.asarray([[e['source_id'], e['target_id']] for e in edges], dtype=np.int64).reshape(-1, 2),
                edge_prob=np.asarray([e.get('edge_prob', float('nan')) for e in edges], dtype=np.float64))


def validate(nodes, edges, shape, allow_bounds=False):
    import numpy as np
    from collections import Counter
    if nodes.ndim != 2 or nodes.shape[1] != 5 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('Invalid graph array shape')
    if not np.isfinite(nodes).all() or not np.equal(nodes, np.rint(nodes)).all():
        raise ValueError('Nonfinite/noninteger nodes')
    if not np.isfinite(edges).all() or not np.equal(edges, np.rint(edges)).all():
        raise ValueError('Noninteger edges')
    ids = {int(r[0]): r for r in nodes}
    if len(ids) != len(nodes) or len(set(map(tuple, edges))) != len(edges):
        raise ValueError('Duplicate IDs/edges')
    if any(int(a) not in ids or int(b) not in ids for a, b in edges):
        raise ValueError('Missing endpoint')
    if any(ids[int(b)][1] != ids[int(a)][1] + 1 for a, b in edges):
        raise ValueError('Nonconsecutive edge')
    if max(Counter(edges[:, 1]).values(), default=0) > 1 or max(Counter(edges[:, 0]).values(), default=0) > 2:
        raise ValueError('Forbidden merge/degree')
    if np.any(nodes[:, 1] < 0) or np.any(nodes[:, 1] >= shape[0]):
        raise ValueError('Time outside clip')
    outside = np.any((nodes[:, 2:] < 0) | (nodes[:, 2:] >= np.asarray(shape[1:])), axis=1)
    if outside.any() and not allow_bounds:
        raise ValueError('Spatial coordinates outside clip')
    return dict(nodes=len(nodes), edges=len(edges), outside=int(outside.sum()),
                forks=sum(n == 2 for n in Counter(edges[:, 0]).values()))


def image_inventory_check(out, row):
    record = read_json(out / 'input_hashes' / (row.get('original_dataset', row['dataset']) + '.json'))
    for relative, expected in record['files'].items():
        if sha(Path(row['path']) / relative) != expected:
            raise ValueError(f'Input hash mismatch: {row["dataset"]}/{relative}')


def neural(job, row, directory):
    import numpy as np
    import torch
    import tracksdata as td
    from .neural import Evidence, import_predictor
    source_root = Path(job['source_root'])
    sys.path[:0] = [str(source_root / 'src'), str(source_root / 'scripts')]
    evidence = Evidence(job['modules'], job.get('transform'))
    predictor, adapter = import_predictor(source_root / 'scripts/predict_unet_transformer.py', job['modules'], evidence)
    (directory / 'predictor.py').write_text(adapter)
    device = torch.device('cuda')
    model, window, downsample = predictor.load_model(Path(job['primary_weights']), device)
    secondary, secondary_window, secondary_downsample = predictor.load_model(Path(job['secondary_weights']), device)
    if (window, downsample) != (secondary_window, secondary_downsample):
        raise ValueError('Model grid/context mismatch')
    evidence.attach(model)
    evidence.attach(secondary)
    cfg = predictor.PredictConfig(det_threshold=.965, threshold=.48, use_ilp=True,
        ilp_edge_weight=-1., ilp_appearance_weight=0., ilp_disappearance_weight=2., ilp_division_weight=1.2)
    # The original CLI uses its default pool_kernel_um=3.0 (checkpoint 5.0 is not read for cfg).
    start = time.perf_counter()
    coords, candidates = predictor.predict_video(model, Path(row['path']), device, cfg,
        window_size=window, unet_batch_size=4, downsample=downsample, secondary_model=secondary,
        secondary_edge_weight=.15, secondary_detection_weight=.80,
        secondary_link_mode='low_margin_consensus', secondary_mix_temperature=1., secondary_low_margin_max=.35)
    torch.cuda.synchronize()
    graph = predictor.build_graph(coords, candidates)
    if graph.num_edges():
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr('edge_prob'), appearance_weight=0.,
                                     disappearance_weight=2., division_weight=1.2)
        with predictor.suppress_output():
            graph = solver.solve(graph)
    n = {int(r['node_id']): dict(node_id=int(r['node_id']), t=int(r['t']),
             **{a: float(r[a]) for a in 'zyx'}) for r in graph.node_attrs().iter_rows(named=True)}
    e = [dict(source_id=int(r['source_id']), target_id=int(r['target_id']), edge_prob=float(r['edge_prob']))
         for r in graph.edge_attrs().iter_rows(named=True)]
    raw = arrays(n, e)
    save_arrays(directory / 'raw.npz', **raw)
    save_arrays(directory / 'evidence.npz', coords=coords, candidates=np.asarray(candidates, float).reshape(-1, 4),
                offsets=np.asarray(evidence.offsets).reshape(-1, 3),
                node_probabilities=np.asarray(evidence.node_probabilities), **evidence.probabilities)
    write_json(directory / 'neural.json', dict(seconds=time.perf_counter() - start,
        raw_sha256=sha(directory / 'raw.npz'), evidence_sha256=sha(directory / 'evidence.npz'),
        adapter_sha256=sha(directory / 'predictor.py'), actual_window=window, downsample=downsample,
        configuration=vars(cfg), summaries=evidence.summaries, graph_hash=array_hash(raw['nodes'], raw['edges']),
        provenance=job['neural_fingerprint'], fresh=True))
    del evidence, model, secondary, graph
    torch.cuda.empty_cache()
    return raw, directory / 'evidence.npz'


def repair(job, row, raw, evidence_path, directory):
    import numpy as np
    from .modules import gated_relink, localize_originals, fork_smooth
    from .source import repair_namespace
    mods = set(job['modules'])
    image_dir = Path(row['path']).parent
    # A symlink-free declaration namespace from the original archive; motion default explicit per arm.
    ns = repair_namespace(job['archive_source'], image_dir, directory,
                          {'OUTPUT_MOTION_RELINK': 'no_motion' not in mods})
    if job.get('transform'):
        original_frame = ns['read_test_frame']
        def read_frame(dataset, t, frame_cache):
            if t in frame_cache:
                return frame_cache[t]
            value = original_frame(dataset, t, {})
            if job['transform'] == 'shift_x1':
                value = np.take(value, np.maximum(np.arange(value.shape[-1]) - 1, 0), axis=-1)
            else:
                axis = -1 if job['transform'] == 'reflect_x' else -2
                value = np.flip(value, axis=axis).copy()
            frame_cache[t] = value
            return value
        ns['read_test_frame'] = read_frame
    bundle = ns['load_deepcenter_veto_detector']()
    original_heatmap = ns['deepcenter_heatmap_for_frame']
    cache_id = digest(dict(image=row['image_sha256'], transform=job.get('transform'),
                           source=job['original_source_sha256'], model=job['model_hashes'][job['deepcenter_weights']]))
    cache_root = Path(job['out']) / 'heatmaps' / cache_id
    heatmap_stats = dict(reads=0, generated=0, hashes={})

    def heatmap(dataset, t, detector_bundle, frame_cache, heatmap_cache):
        p = cache_root / f'{t}.npy'
        receipt = p.with_suffix('.json')
        if p.exists() and receipt.exists():
            record = read_json(receipt)
            if record['fingerprint'] != cache_id or record['sha256'] != sha(p):
                raise ValueError('Heatmap cache drift')
            heatmap_stats['reads'] += 1
            result = np.load(p, allow_pickle=False)
        else:
            result = original_heatmap(dataset, t, detector_bundle, frame_cache, heatmap_cache)
            if result is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, result, allow_pickle=False)
                write_json(receipt, dict(fingerprint=cache_id, sha256=sha(p)))
                heatmap_stats['generated'] += 1
        if result is not None:
            heatmap_stats['hashes'][str(t)] = sha(p)
        return result
    ns['deepcenter_heatmap_for_frame'] = heatmap
    nodes = {int(r[0]): dict(node_id=int(r[0]), t=int(r[1]), **dict(zip('zyx', map(float, r[2:])))) for r in raw['nodes']}
    edges = [dict(source_id=int(a), target_id=int(b), edge_prob=float(p)) for (a, b), p in zip(raw['edges'], raw['edge_prob'])]
    original_nodes = copy.deepcopy(nodes)
    evidence = load_arrays(evidence_path) if mods & {'E01', 'E02'} else None
    stage_records = []
    mechanism_records = {}

    def capture(stage, n, e):
        a = arrays(n, e)
        save_arrays(directory / f'{stage}.npz', **a)
        stage_records.append(dict(stage=stage, hash=array_hash(a['nodes'], a['edges']), nodes=len(n), edges=len(e)))

    capture('raw', nodes, edges)
    if 'E01' in mods:
        # Build lookups only for queried transition edge components, with complete source-column evidence.
        probability = {}
        for key in evidence:
            if not key.startswith('p_'):
                continue
            t = key[2:]
            ps, ss, tt = evidence[key], evidence['s_' + t], evidence['t_' + t]
            probability[int(t)] = (ps, {int(v): i for i, v in enumerate(ss)}, {int(v): i for i, v in enumerate(tt)})
        class Lookup:
            def get(self, edge, default=None):
                a, b = edge
                record = probability.get(nodes[a]['t']) if a in nodes else None
                if record is None or a not in record[1] or b not in record[2]:
                    return default
                return float(record[0][record[1][a], record[2][b]])
            def __getitem__(self, edge):
                return self.get(edge, float('nan'))
        # Original valid-edge list is captured immediately before it is handed to the relinker.
        original_filter = ns['filter_output_graph']
        import inspect
        from .source import replace_once
        source = inspect.getsource(original_filter)
        source = replace_once(source, '        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)',
            '        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n'
            '        motion_edges = _p946_gate(nodes_by_id, edges, motion_edges, stats)')
        def gate(n, before, proposed, stats):
            if not proposed:
                mechanism_records['E01'] = []
                return before
            chosen, records = gated_relink(n, before, proposed, Lookup())
            mechanism_records['E01'] = records
            return chosen
        ns['_p946_gate'] = gate
        exec(compile(source, '<E01-atomic-gate>', 'exec'), ns)
    for function, phase in [('close_single_frame_gaps', 'motion'), ('recover_strict_gap2', 'gap1'),
                            ('add_safe_divisions_postlink', 'gap2'), ('filter_short_track_components', 'divisions')]:
        original = ns[function]
        def wrapper(n, e, *a, _original=original, _phase=phase, **kw):
            capture(_phase, n, e)
            return _original(n, e, *a, **kw)
        ns[function] = wrapper
    original_smooth = ns['linefit_smooth_output_graph']

    def smooth(n, e, stats):
        capture('presmooth', n, e)
        if 'E02' in mods:
            offsets = {i: d for i, d in enumerate(evidence['offsets'])}
            mechanism_records['E02'] = localize_originals(n, original_nodes, offsets, [1, 4, 4])
            capture('localized', n, e)
        if 'E07' in mods:
            return fork_smooth(n, e, stats, ns['OUTPUT_LINEFIT_WEIGHT'], ns['OUTPUT_LINEFIT_WINDOW'])
        return original_smooth(n, e, stats)
    ns['linefit_smooth_output_graph'] = smooth
    nodes, edges, stats = ns['filter_output_graph'](nodes, edges, dataset=row['dataset'], deepcenter_bundle=bundle)
    result = arrays(nodes, edges)
    floats = result['nodes'].copy()
    original_export = floats.copy()
    original_export[:, 2:] = np.rint(original_export[:, 2:])
    original_export = original_export.astype(np.int64)
    original_validation = validate(original_export, result['edges'], row['image_shape'], allow_bounds=True)
    exported = original_export.copy()
    exported[:, 2:] = np.clip(exported[:, 2:], 0, np.asarray(row['image_shape'][1:]) - 1)
    validated = validate(exported, result['edges'], row['image_shape'])
    save_arrays(directory / 'final.npz', nodes=exported, edges=result['edges'],
                original_nodes=original_export, float_nodes=floats, edge_prob=result['edge_prob'])
    return dict(stats=stats, mechanisms=mechanism_records, stages=stage_records,
                validation=validated, original_validation=original_validation, heatmaps=heatmap_stats,
                final_sha256=sha(directory / 'final.npz'), graph_hash=array_hash(exported, result['edges']))


def run(args):
    import torch
    from .isolation import probe
    job = read_json(args.job)
    if os.environ.get('PUBLIC946_INFERENCE') != '1':
        raise RuntimeError('Worker must start with inherited annotation guard')
    out = Path(job['out'])
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(job['limits']['gpu_allocated_gib_max'] * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    for key in list(os.environ):
        if key.startswith('BIOHUB_'):
            del os.environ[key]
    os.environ.update(job['settings'])
    os.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0' if 'no_motion' in job['modules'] else '1'
    os.environ['BIOHUB_ALLOW_PIP_INSTALL'] = '0'
    for p, h in job['model_hashes'].items():
        if sha(p) != h:
            raise ValueError('Model fingerprint drift')
    row = job['row']
    assert probe(str(Path(row['path']).with_suffix('.geff')))
    image_inventory_check(out, row)
    check_resources(out, job['limits'])
    start = time.perf_counter()
    if job.get('reuse_neural'):
        reused = Path(job['reuse_neural'])
        receipt = read_json(reused / 'neural.json')
        if receipt['provenance'] != job['neural_fingerprint']:
            raise ValueError('Neural cache fingerprint mismatch')
        if sha(reused / 'raw.npz') != receipt['raw_sha256'] or sha(reused / 'evidence.npz') != receipt['evidence_sha256']:
            raise ValueError('Neural cache content mismatch')
        raw, evidence_path = load_arrays(reused / 'raw.npz'), reused / 'evidence.npz'
        neural_fresh = False
    else:
        raw, evidence_path = neural(job, row, directory)
        neural_fresh = True
    result = repair(job, row, raw, evidence_path, directory)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result.update(dataset=row['dataset'], arm=job['arm'], modules=job['modules'],
                  seconds=seconds, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                  rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  neural_fresh=neural_fresh, neural_evidence=str(evidence_path),
                  annotation_access_denied=True, job_sha256=sha(args.job), fingerprint=job['fingerprint'],
                  completed_at=now())
    write_json(directory / 'complete.json', result)
    print(f"Completed {job['arm']} {row['dataset']} in {seconds:.1f}s", flush=True)
