"""Fresh neural pilots through an isolated copy of the actual notebook CLI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def trace_raw_decoder(ctx):
    """Explain zero raw forks from actual candidates and exact ILP objectives."""
    import inspect
    import contextlib
    import io
    import numpy as np
    import polars as pl
    import tracksdata as td
    from .common import load_graph, sha, write_json
    predictor = ctx.full / 'tracking_repo/scripts/predict_unet_transformer.py'
    solver_source = Path(inspect.getfile(td.solvers.ILPSolver))
    rows = []
    for row in ctx.samples():
        name = row['dataset']
        prepath = ctx.full / 'inputs' / f'pre_ilp_{name}.npz'
        pre = load_graph(prepath)
        rawpath = ctx.v2 / 'raw' / f'{name}.npz'
        raw = load_graph(rawpath)
        scores = pre['edge_scores']
        _, outdegree = np.unique(scores[:, 0].astype(np.int64), return_counts=True)
        _, indegree = np.unique(scores[:, 1].astype(np.int64), return_counts=True)
        _, raw_degree = np.unique(raw['edges'][:, 0], return_counts=True)
        rows.append(dict(dataset=name, pre_ilp_edges=len(scores),
                         candidate_sources_with_two_or_more_children=int((outdegree >= 2).sum()),
                         candidate_targets_with_two_or_more_parents=int((indegree >= 2).sum()),
                         raw_forks=int((raw_degree == 2).sum()),
                         probability_min=float(scores[:, 2].min()), probability_max=float(scores[:, 2].max()),
                         pre_ilp_sha256=sha(prepath), raw_sha256=sha(rawpath)))
    fixtures = []
    for division_weight in [1.2, 0.1]:
        graph = td.graph.InMemoryGraph()
        graph.add_edge_attr_key('edge_prob', pl.Float64, 0.)
        nodes = [dict(t=0), dict(t=1)]
        edges = [(0, 1)]
        for branch in range(2):
            prev = 1
            for t in range(2, 10):
                i = len(nodes)
                nodes.append(dict(t=t))
                edges.append((prev, i))
                prev = i
        ids = graph.bulk_add_nodes(nodes)
        graph.bulk_add_edges([dict(source_id=ids[a], target_id=ids[b], edge_prob=.99) for a, b in edges])
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr('edge_prob'),
            appearance_weight=0., disappearance_weight=2., division_weight=division_weight)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = solver.solve(graph)
        _, counts = np.unique(result.edge_attrs()['source_id'].to_numpy(), return_counts=True)
        fixtures.append(dict(division_weight=division_weight, candidate_forks=1,
                             selected_forks=int((counts == 2).sum()), selected_edges=result.num_edges()))
    assert fixtures[0]['selected_forks'] == 0 and fixtures[1]['selected_forks'] == 1
    receipt = dict(samples=len(rows),
        predictor_sha256=sha(predictor), solver_sha256=sha(solver_source),
        predictor_lines=dict(softmax=786, candidate_threshold=789, greedy_constraints=804,
                             ilp_config=89, ilp_call=992),
        solver_lines=dict(incoming_flow=294, outgoing_flow=309, binary_variables=344),
        score_normalization='softmax(raw, dim=0): each target column normalizes over competing source parents; documentation row-normalized wording is inaccurate',
        candidate_generation='all source/next-frame target pairs with normalized probability >0.48; greedy degree limits None under use_ilp=True',
        final_constraints='binary flow: indegree <=1 (merge disabled), outdegree <=2 (one binary division variable)',
        objective=dict(edge_weight='-edge_probability', appearance_weight=0., disappearance_weight=2., division_weight=1.2),
        exact_dominance_reason='For any fork remove either daughter edge with p<=1, set parent division=0 and detached daughter appearance=1. All nodes and other edges stay selected; both flow constraints remain satisfied. Objective change = p + 0 - 1.2 <= -0.2. Thus every fork is strictly dominated by a free daughter birth, independent of candidate coverage and disappearance weight.',
        synthetic_actual_solver=fixtures, raw_forks=sum(r['raw_forks'] for r in rows),
        candidate_sources_with_two_or_more_children=sum(r['candidate_sources_with_two_or_more_children'] for r in rows),
        rows=rows)
    write_json(ctx.out / 'raw_decoder_trace.json', receipt)
    return receipt


def select_pilots(ctx):
    from .common import read_json, sha, write_json, now
    lock_path = ctx.out / 'fresh_pilot_lock.json'
    records = []
    for row in ctx.samples():
        path = ctx.data / 'train' / (row['dataset'] + '.zarr') / 'zarr.json'
        quantiles = read_json(path)['attributes']['image_statistics']['quantiles']
        records.append(dict(dataset=row['dataset'], embryo=row['embryo'],
                            contrast=float(quantiles['0.99']) - float(quantiles['0.1']),
                            metadata_sha256=sha(path), image_shape=row['image_shape']))
    selected = []
    for embryo in sorted(set(r['embryo'] for r in records)):
        group = sorted((r for r in records if r['embryo'] == embryo), key=lambda r: (r['contrast'], r['dataset']))
        selected.append(group[len(group)//2])
    lock = dict(selection_rule='median image metadata q99-q10 contrast per embryo; tie by name',
                selected=selected, candidates=len(records), labels_read=False, scores_used=False)
    if lock_path.exists():
        assert read_json(lock_path) == lock
    else:
        write_json(lock_path, lock)
    return selected


def checkpoint_raw_graph(graph, name):
    """Atomic prediction-only checkpoint called after the actual CLI save."""
    import numpy as np
    from .common import save_arrays, sha, write_json
    root = Path(os.environ['V3_RAW_CHECKPOINTS'])
    path = root / f'{name}.npz'
    attrs = graph.edge_attrs()
    save_arrays(path, nodes=graph.node_attrs().select('node_id', 't', 'z', 'y', 'x').to_numpy(),
        edges=attrs.select('source_id', 'target_id').to_numpy().astype(np.int64),
        edge_prob=attrs['edge_prob'].to_numpy() if 'edge_prob' in attrs.columns else np.empty(0))
    write_json(root / f'{name}.json', dict(sha256=sha(path),
        pipeline_fingerprint=os.environ['V3_PIPELINE_FINGERPRINT']))


def fresh_can_resume(name):
    from .common import read_json, sha
    root = Path(os.environ['V3_RAW_CHECKPOINTS'])
    path, receipt = root / f'{name}.npz', root / f'{name}.json'
    if not receipt.exists():
        return False
    record = read_json(receipt)
    assert record['pipeline_fingerprint'] == os.environ['V3_PIPELINE_FINGERPRINT'], 'Stale fresh neural checkpoint'
    assert record['sha256'] == sha(path), 'Fresh neural checkpoint changed'
    return True


def prepare(ctx, pilots, *, resumable=False):
    import shutil
    from .common import digest, sha, write_json
    from .replay import notebook_environment
    destination = ctx.out / 'fresh' / 'tracking_repo'
    original = ctx.full / 'tracking_repo'
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for folder in ['src', 'scripts']:
        for source in sorted((original / folder).rglob('*.py')):
            relative = source.relative_to(original)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_hashes[str(relative)] = sha(source)
    weights = destination / 'weights'
    if not weights.exists():
        weights.symlink_to((original / 'weights').resolve(), target_is_directory=True)
    image_dir = ctx.out / 'fresh' / 'inputs' / 'test'
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in pilots:
        target = image_dir / (row['dataset'] + '.zarr')
        if not target.exists():
            target.symlink_to(ctx.data / 'train' / target.name, target_is_directory=True)
    write_json(destination / 'fresh_splits.json', [dict(split=0, train=[], test=[p['dataset'] for p in pilots])])
    script = destination / 'scripts' / 'predict_unet_transformer.py'
    source = script.read_text()
    old_root = '/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full'
    assert source.count(old_root) == 2
    source = source.replace(old_root, str(ctx.out / 'fresh'))
    # Timing-only instrumentation around the same predict/build/ILP/save calls.
    source = source.replace('        coords, edges = predict_video(',
                            '        import time as _pilot_time\n        torch.cuda.synchronize()\n'
                            '        _pilot_start = _pilot_time.perf_counter()\n        coords, edges = predict_video(', 1)
    source = source.replace('        graph = build_graph(coords, edges)',
                            '        torch.cuda.synchronize()\n        _pilot_neural_end = _pilot_time.perf_counter()\n'
                            '        graph = build_graph(coords, edges)', 1)
    anchor = '        save_graph(graph, output_dir / f"{name}.geff")'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '\n'
        '        _pilot_record = dict(dataset=name, neural_io_seconds=_pilot_neural_end-_pilot_start, '
        'graph_ilp_export_seconds=_pilot_time.perf_counter()-_pilot_neural_end, '
        'cuda_peak_bytes=torch.cuda.max_memory_allocated(), '
        'nodes=graph.num_nodes(), edges=graph.num_edges())\n'
        '        with Path(os.environ["V3_FRESH_TIMINGS"]).open("a") as _pilot_handle:\n'
        '            _pilot_handle.write(json.dumps(_pilot_record)+"\\n")\n', 1)
    if resumable:
        loop = '    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):'
        assert source.count(loop) == 1
        source = source.replace(loop, loop + '\n'
            '        from strong_tracker_v3.fresh import fresh_can_resume, checkpoint_raw_graph\n'
            '        if fresh_can_resume(name):\n'
            '            print("Fresh neural checkpoint verified:", name, flush=True)\n'
            '            continue\n', 1)
        source = source.replace(anchor, anchor + '\n        checkpoint_raw_graph(graph, name)', 1)
    compile(source, str(script), 'exec')
    script.write_text(source)
    settings = notebook_environment((ctx.full / 'harmonic_isolated.py').read_text())
    settings['BIOHUB_SECONDARY_WEIGHTS'] = str(ctx.full / 'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth')
    settings['BIOHUB_DIAGNOSTIC_ARM'] = 'strong_tracker_v3_fresh'
    settings['BIOHUB_DATA_DIR'] = str(image_dir)
    settings['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'
    settings['V3_FRESH_TIMINGS'] = str(ctx.out / 'fresh' / 'neural_timings.jsonl')
    config = dict(source_hashes=source_hashes, copied_predictor_sha256=sha(script),
                  notebook_sha256=sha(ctx.full / 'harmonic_isolated.py'), settings=settings,
                  image_dir=str(image_dir), source_repo=str(destination),
                  changes=['relocate two diagnostic outputs', 'add synchronized stage timers'],
                  algorithm_bodies_unchanged=True, dataset_ids=[p['dataset'] for p in pilots],
                  resumable=resumable, model_hashes={str(p): sha(p) for p in [
                      original / 'weights/unet_transformer/split_0/edge_predictor_best.pth',
                      ctx.full / 'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth']})
    if resumable:
        settings['V3_RAW_CHECKPOINTS'] = str(ctx.out / 'fresh' / 'raw_checkpoints')
        settings['V3_PIPELINE_FINGERPRINT'] = digest(config)
    write_json(ctx.out / 'fresh' / 'pipeline_lock.json', config)
    return config


def worker(ctx):
    # This function is reached only after the __main__ startup audit hook.
    import runpy
    import time
    import resource
    from .common import (graph_hash, load_graph, now, read_json, save_arrays, sha,
                         validate, write_json)
    from .replay import apply_repair, repair_namespace
    import numpy as np
    import torch
    import tracksdata as td
    config = read_json(ctx.out / 'fresh' / 'pipeline_lock.json')
    start = time.perf_counter()
    assert torch.cuda.is_available(), 'Fresh inference requires the supplied CUDA runtime'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    os.environ.update(config['settings'])
    repo = Path(config['source_repo'])
    for name, expected in config['model_hashes'].items():
        assert sha(name) == expected
    predictor = repo / 'scripts' / 'predict_unet_transformer.py'
    assert sha(predictor) == config['copied_predictor_sha256']
    sys.path[:0] = [str(repo / 'src'), str(repo / 'scripts')]
    sys.argv = [str(predictor), '--data-dir', config['image_dir'], '--splits', str(repo / 'fresh_splits.json'),
                '--split', '0', '--weights', str(repo / 'weights/unet_transformer/split_0/edge_predictor_best.pth'),
                '--unet-batch-size', '4', '--det-threshold', config['settings']['BIOHUB_DET_THRESHOLD'],
                '--ilp-edge-weight', '-1.0', '--ilp-appearance-weight', config['settings']['BIOHUB_ILP_APPEARANCE_WEIGHT'],
                '--ilp-disappearance-weight', config['settings']['BIOHUB_ILP_DISAPPEARANCE_WEIGHT'],
                '--ilp-division-weight', config['settings']['BIOHUB_ILP_DIVISION_WEIGHT'], '--use-ilp']
    runpy.run_path(str(predictor), run_name='__main__')
    neural_end = time.perf_counter()
    records = []
    rows = {r['dataset']: r for r in ctx.samples()}
    for name in config['dataset_ids']:
        if config.get('resumable'):
            assert fresh_can_resume(name)
            raw = load_graph(ctx.out / 'fresh' / 'raw_checkpoints' / f'{name}.npz')
        else:
            path, = repo.glob(f'predictions/*/unet_transformer/split_0/{name}.geff')
            graph = td.graph.IndexedRXGraph.from_geff(path)
            if isinstance(graph, tuple):
                graph = graph[0]
            raw = dict(nodes=graph.node_attrs().select('node_id', 't', 'z', 'y', 'x').to_numpy(),
                       edges=graph.edge_attrs().select('source_id', 'target_id').to_numpy().astype(np.int64),
                       edge_prob=graph.edge_attrs()['edge_prob'].to_numpy())
        save_arrays(ctx.out / 'fresh' / 'raw' / f'{name}.npz', **raw)
        final_path = ctx.out / 'fresh' / 'incumbent' / f'{name}.npz'
        final_receipt = final_path.with_suffix('.json')
        if config.get('resumable') and final_receipt.exists():
            previous = read_json(final_receipt)
            assert previous['pipeline_lock_sha256'] == sha(ctx.out / 'fresh' / 'pipeline_lock.json')
            assert previous['output_sha256'] == sha(final_path)
            records.append(previous)
            continue
        repair_start = time.perf_counter()
        namespace = repair_namespace(ctx, image_dir=config['image_dir'], fresh_heatmaps=True)
        # Fresh mode routes heatmaps into its own directory to avoid reading any
        # pre-existing V3 heatmaps even when other lanes have run concurrently.
        result, stats = apply_repair(namespace, raw, name, rows[name]['image_shape'])
        save_arrays(final_path, nodes=result['nodes'], edges=result['edges'])
        record = dict(dataset=name, repair_seconds=time.perf_counter()-repair_start,
                            graph_hash=graph_hash(result['nodes'], result['edges']),
                            stats=stats, validity=validate(result['nodes'], result['edges'], rows[name]['image_shape']),
                            output_sha256=sha(final_path),
                            pipeline_lock_sha256=sha(ctx.out / 'fresh' / 'pipeline_lock.json'),
                            fresh_heatmap_queries=namespace['_heatmap_state']['fresh_heatmaps'])
        write_json(final_receipt, record)
        records.append(record)
        print('Fresh incumbent complete:', name, flush=True)
    write_json(ctx.out / 'fresh' / 'worker_receipt.json', dict(
        created=now(), completed=True, annotation_reads=0, startup_guard=True,
        fresh_image_neural_inference=True, graph_cache_reads_for_prediction=0,
        neural_stage_seconds=neural_end-start, total_seconds=time.perf_counter()-start,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pilot_records=records, pipeline_lock_sha256=sha(ctx.out / 'fresh' / 'pipeline_lock.json')))


def run(ctx, args):
    import subprocess
    import time
    from .common import graph_hash, load_graph, now, read_json, sha, write_json
    pilots = select_pilots(ctx)
    ctx.out.joinpath('fresh').mkdir(parents=True, exist_ok=True)
    done = ctx.out / 'fresh' / 'worker_receipt.json'
    config = prepare(ctx, pilots)
    start = time.perf_counter()
    if not done.exists():
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ctx.repo / 'tools'))
        # The mirrored runtime's CUDA wheels must precede the image toolkit.
        env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                                         if p and p != '/usr/local/cuda/lib64')
        with (ctx.out / 'fresh' / 'run.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'strong_tracker_v3.fresh', '--worker'],
                                    cwd=ctx.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failure = dict(status='fresh_image_failed', returncode=result.returncode,
                           seconds=time.perf_counter()-start, pilots=pilots,
                           log=str(ctx.out / 'fresh' / 'run.log'), cache_only=True)
            write_json(ctx.out / 'fresh_receipt.json', failure)
            return failure
    worker_receipt = read_json(done)
    assert worker_receipt['pipeline_lock_sha256'] == sha(ctx.out / 'fresh' / 'pipeline_lock.json')
    comparison = []
    for row in pilots:
        name = row['dataset']
        fresh = load_graph(ctx.out / 'fresh' / 'incumbent' / f'{name}.npz')
        cache = load_graph(ctx.incumbent(name))
        fresh_raw = load_graph(ctx.out / 'fresh' / 'raw' / f'{name}.npz')
        cache_raw = load_graph(ctx.v2 / 'raw' / f'{name}.npz')
        comparison.append(dict(dataset=name,
            raw_graph_equal=graph_hash(fresh_raw['nodes'], fresh_raw['edges']) == graph_hash(cache_raw['nodes'], cache_raw['edges']),
            final_graph_equal=graph_hash(fresh['nodes'], fresh['edges']) == graph_hash(cache['nodes'], cache['edges']),
            fresh_nodes=len(fresh['nodes']), cached_nodes=len(cache['nodes']),
            fresh_edges=len(fresh['edges']), cached_edges=len(cache['edges']),
            fresh_hash=graph_hash(fresh['nodes'], fresh['edges']), cache_hash=graph_hash(cache['nodes'], cache['edges'])))
    receipt = dict(status='fresh_image_pilots_complete', pilots=pilots,
                   comparison=comparison, **worker_receipt,
                   full_set_fresh_run=False, full_set_cached_run=True,
                   full_set_seconds_linear_projection=worker_receipt['total_seconds'] * 199 / len(pilots),
                   projection_assumptions='same 100x64x256x256 volumes; linear two-pilot throughput, repeated startup overestimates; density-dependent CPU decoding varies',
                   kaggle_runtime_proven=False)
    write_json(ctx.out / 'fresh_receipt.json', receipt)
    return receipt


def run_full(ctx, args):
    """Two isolated shards, each with immutable per-clip neural checkpoints."""
    import subprocess
    import time
    from dataclasses import replace
    from .common import digest, graph_hash, load_graph, read_json, sha, write_json
    rows = ctx.samples()
    count = getattr(args, 'workers', 2)
    assert 1 <= count <= 2, 'Full fresh initial concurrency is capped at two'
    start = time.perf_counter()
    processes, contexts, logs = [], [], []
    lock_path = ctx.out / 'fresh_full_lock.json'
    lock = dict(datasets=[r['dataset'] for r in rows], shards=count,
                code_sha256=sha(Path(__file__)), repair_sha256=sha(ctx.repo / 'tools/strong_tracker_v3/replay.py'),
                input_sha256=sha(ctx.out / 'inputs.json'), annotation_unavailable=True,
                reason='Two-pilot measured serial projection fits authorized compute and disk budget')
    write_json(lock_path, lock, immutable=True)
    for shard in range(count):
        part = replace(ctx, out=ctx.out / 'fresh_full' / f'shard{shard}')
        subset = rows[shard::count]
        write_json(part.out / 'inputs.json', subset, immutable=True)
        config = prepare(part, subset, resumable=True)
        contexts.append(part)
        done = part.out / 'fresh' / 'worker_receipt.json'
        if done.exists():
            assert read_json(done)['pipeline_lock_sha256'] == sha(part.out / 'fresh' / 'pipeline_lock.json')
            continue
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ctx.repo / 'tools'),
                   STRONG_TRACKER_V3_OUT=str(part.out))
        env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                                         if p and p != '/usr/local/cuda/lib64')
        log = (part.out / 'fresh' / 'run.log').open('a')
        logs.append(log)
        processes.append(subprocess.Popen([sys.executable, '-m', 'strong_tracker_v3.fresh',
                                           '--worker', '--out', str(part.out)],
                                          cwd=ctx.repo, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for log in logs:
        log.close()
    if any(codes):
        write_json(ctx.out / 'fresh_full_failure.json', dict(returncodes=codes, checkpoint_resume=True))
        raise RuntimeError(f'Fresh shard failed: {codes}')
    comparisons = []
    receipts = []
    for part in contexts:
        receipts.append(read_json(part.out / 'fresh' / 'worker_receipt.json'))
        for row in part.samples():
            name = row['dataset']
            fresh = load_graph(part.out / 'fresh/incumbent' / f'{name}.npz')
            old = load_graph(ctx.incumbent(name))
            raw = load_graph(part.out / 'fresh/raw' / f'{name}.npz')
            raw_old = load_graph(ctx.v2 / 'raw' / f'{name}.npz')
            comparisons.append(dict(dataset=name,
                final_graph_equal=graph_hash(fresh['nodes'], fresh['edges']) == graph_hash(old['nodes'], old['edges']),
                raw_graph_equal=graph_hash(raw['nodes'], raw['edges']) == graph_hash(raw_old['nodes'], raw_old['edges']),
                fresh_graph_path=str(part.out / 'fresh/incumbent' / f'{name}.npz'),
                fresh_graph_sha256=sha(part.out / 'fresh/incumbent' / f'{name}.npz')))
    assert len(comparisons) == len(rows) == 199
    result = dict(status='complete', samples=len(comparisons), wall_seconds=time.perf_counter()-start,
                  shard_receipts=receipts, comparison=comparisons, annotations_unavailable_from_start=True,
                  fresh_all_199=True, cached_graph_reads_for_prediction=0,
                  exact_incumbent_parity=sum(r['final_graph_equal'] for r in comparisons),
                  exact_raw_parity=sum(r['raw_graph_equal'] for r in comparisons))
    write_json(ctx.out / 'fresh_full_receipt.json', result)
    return result


if __name__ == '__main__':
    from .inference import deny_annotations
    # Install before NumPy, Torch, model imports, RunContext, images or graphs.
    root = Path(os.environ.get('STRONG_TRACKER_V3_OUT', '/kaggle/working/cell-tracking/strong-tracker-v3'))
    guard = deny_annotations(allowed_geff_root=root / 'fresh')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out')
    args = parser.parse_args()
    from .context import RunContext
    ctx = RunContext.default(out=args.out)
    if args.worker:
        worker(ctx)
    elif args.full:
        run_full(ctx, args)
    else:
        run(ctx, args)
