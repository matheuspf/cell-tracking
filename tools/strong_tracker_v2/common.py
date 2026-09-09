from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from annotation_selection.common import (DATA, METRIC_REV, REPO, clean, digest,
    graph_hash, load_graph, now, read_json, save_graph, sha, write_json)

V1 = Path('/kaggle/working/cell-tracking/annotation-selection-v1')
OUT = Path('/kaggle/working/cell-tracking/strong-tracker-v2')
WORK = REPO / 'work/strong-tracker-v2'
FULL = V1 / 'public_harmonic_full'
SEED = 20260909
SCALE = np.array([1.625, .40625, .40625])


def inventory():
    return read_json(V1 / 'inventory.json')


def stage(name, state, **kw):
    p = OUT / 'status.json'
    d = read_json(p) if p.exists() else {'study_id': 'strong-tracker-v2', 'stages': {}}
    d['stages'][name] = dict(status=state, updated=now(), **clean(kw))
    d['updated'] = now()
    write_json(p, d)


def stamp_sources():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / 'tools/strong_tracker_v2').glob('*.py'))}


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def raw_graph(name):
    cache = OUT / 'raw' / f'{name}.npz'
    if cache.exists():
        return load_graph(cache)
    import tracksdata as td
    p, = (FULL / 'tracking_repo/predictions').glob(f'*/unet_transformer/split_0/{name}.geff')
    g = td.graph.IndexedRXGraph.from_geff(p)
    if isinstance(g, tuple):
        g = g[0]
    a = g.node_attrs()
    n = a.select('node_id', 't', 'z', 'y', 'x').to_numpy()
    e = g.edge_attrs().select('source_id', 'target_id').to_numpy().astype(np.int64)
    attrs = g.edge_attrs()
    prob = attrs['edge_prob'].to_numpy() if 'edge_prob' in attrs.columns else np.full(len(e), np.nan)
    # Raw positions are saved as floats; round only at the notebook export boundary.
    save_arrays(cache, nodes=n, edges=e, edge_prob=prob)
    return dict(nodes=n, edges=e, edge_prob=prob)


def export_nodes(n):
    result = np.asarray(n).copy()
    result[:, 2:] = np.maximum(0, np.rint(result[:, 2:]))
    return result.astype(np.int64)


def adjacency(nodes, edges):
    ix = {int(n): i for i, n in enumerate(nodes[:, 0])}
    pred = [[] for _ in nodes]
    succ = [[] for _ in nodes]
    for a, b in edges:
        i, j = ix[int(a)], ix[int(b)]
        succ[i].append(j)
        pred[j].append(i)
    return ix, pred, succ


def validate(nodes, edges, shape, reference=None, allow_legacy_bounds=False):
    n, e = np.asarray(nodes), np.asarray(edges)
    assert n.ndim == 2 and n.shape[1] == 5 and e.ndim == 2 and e.shape[1] == 2
    assert np.isfinite(n).all() and np.equal(n, np.rint(n)).all()
    assert len(np.unique(n[:, 0])) == len(n), 'duplicate node IDs'
    assert len(set(map(tuple, e))) == len(e), 'duplicate edges'
    ix, pred, succ = adjacency(n, e)
    assert all(len(x) <= 1 for x in pred), 'forbidden merge'
    assert all(len(x) <= 2 for x in succ), 'more than two daughters'
    assert all(n[ix[int(b)], 1] == n[ix[int(a)], 1] + 1 for a, b in e), 'nonconsecutive edge'
    assert ((n[:, 1] >= 0) & (n[:, 1] < shape[0])).all()
    outside = np.any((n[:, 2:] < 0) | (n[:, 2:] >= np.array(shape[1:])), axis=1)
    if outside.any():
        assert allow_legacy_bounds, 'out of bounds'
        if reference is not None:
            old = {tuple(r) for r in reference}
            assert all(tuple(r) in old for r in n[outside]), 'new out-of-bounds point'
    return dict(nodes=len(n), edges=len(e), out_of_bounds=int(outside.sum()),
                forks=sum(len(x) == 2 for x in succ), valid=True)


def run_pool(fn, tasks, workers=16):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        fs = {pool.submit(fn, x): x for x in tasks}
        for i, f in enumerate(as_completed(fs), 1):
            try:
                result = f.result()
            except Exception as exc:
                print(f'FAILED {fn.__name__} {fs[f]}: {type(exc).__name__}: {exc}', flush=True)
                raise
            print(f'{fn.__name__} {i}/{len(fs)} {str(result)[:220]}', flush=True)
            yield result


def preservation_snapshot():
    p = OUT / 'preservation_before.json'
    if p.exists():
        return
    files = {str(x.relative_to(V1)): [x.stat().st_size, x.stat().st_mtime_ns]
             for x in V1.rglob('*') if x.is_file()}
    keys = ['artifact_manifest.json', 'public_prediction_lock.json', 'prediction_lock.json',
            'model_lock.json', 'public_score_rows.parquet', 'public_harmonic_full/harmonic_isolated.py']
    write_json(p, dict(created=now(), v1_files=files,
        critical_sha256={k: sha(V1/k) for k in keys}, source=stamp_sources()), immutable=True)


def preservation_check():
    before = read_json(OUT / 'preservation_before.json')
    current = {str(x.relative_to(V1)): [x.stat().st_size, x.stat().st_mtime_ns]
               for x in V1.rglob('*') if x.is_file()}
    assert before['v1_files'] == current, 'Sealed v1 store changed'
    assert all(sha(V1/k) == v for k, v in before['critical_sha256'].items())
    return dict(v1_files=len(current), unchanged=True)
