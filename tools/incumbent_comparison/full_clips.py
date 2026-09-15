"""Frozen complete-clip detector comparison with a common native linker.

Stages run in separate processes so inference cannot inherit evaluator label reads.
The public association checkpoint is training-exposed: this is a pipeline diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .common import REPO, WORK, OUT, DATA, GPU_LOCK, read, write, sha, now

PANEL = REPO / 'work/cellpose-ultrack-20260914/panel.json'
CP = PANEL.parent / 'predictions/cellpose_cpdino_vitb'
INC = Path('/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/inputs')
OLD = REPO / 'work/cellpose-refine-v1'
NATIVE = Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/repo')
PRIMARY = NATIVE.parent / 'weights/unet_transformer/split_0/edge_predictor_best.pth'
ROOT = WORK / 'full-clips'
SPACING = np.array([1.625, .40625, .40625])
SEEDS = (20260914, 314159)
METHODS = ('incumbent', 'cellpose', 'constant', 'source-20260914', 'source-314159',
           'both-20260914', 'both-314159', 'incumbent+cellpose', 'incumbent+source-20260914')


def npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def progress(stage, completed, total, **extra):
    value = dict(updated_utc=now(), stage=stage, completed=completed, total=total, **extra)
    write(ROOT / 'progress.json', value)
    print(json.dumps(value), flush=True)


def prepare():
    from tools.cellpose_refine.infer import all_heads_ready
    from tools.cellpose_refine.common import CHECKPOINT, CHECKPOINT_SHA
    from tools.annotation_selection.common import METRIC_REV
    cfg = read(REPO / 'configs/cellpose-refine-v1.json')
    sources = all_heads_ready(cfg)
    both = read(OUT / 'refit-models-lock.json')
    heads = [{**r, 'scope': 'source'} for r in sources]
    heads += [dict(scope='both', seed=r['seed'], path=str(WORK / 'models/refiner-all' / f"seed-{r['seed']}.pt"),
                   sha256=r['sha256']) for r in both['models']]
    for r in heads:
        assert sha(r['path']) == r['sha256']
    panel = read(PANEL)
    assert len(panel['clips']) == 6 and len(panel['frames']) == 600
    old = {r['key']: r for r in read(OLD / 'inputs/inference.json')['rows']}
    inputs = []
    for row in panel['frames']:
        pred = CP / (row['key'] + '.npz')
        receipt = read(pred.with_suffix('.json'))
        image_hash = sha(row['image_path'])
        assert receipt['input_sha256'] == image_hash
        entry = dict(row, prediction_path=str(pred), prediction_sha256=sha(pred), image_sha256=image_hash)
        if row['key'] in old:
            cache = Path(old[row['key']]['cache_path'])
            info = read(cache.with_suffix('.json'))
            assert info['annotation_read'] == 'none' and sha(cache) == info['cache_sha256']
            entry.update(feature_path=str(cache), feature_sha256=sha(cache))
        inputs.append(entry)
    offsets = {}
    incumbents = {}
    for source in ('44b6', '6bba'):
        path = OLD / 'offset-control' / f'fit-{source}.json'
        offsets[source] = dict(path=str(path), sha256=sha(path), offset=read(path)['offset_zyx_um'])
    for clip in panel['clips']:
        path = INC / f"pre_ilp_{clip['dataset']}.npz"
        meta = DATA / f"{clip['dataset']}.zarr/zarr.json"
        incumbents[clip['dataset']] = dict(path=str(path), sha256=sha(path), metadata_sha256=sha(meta))
    implementation = [Path(__file__), Path(__file__).with_name('common.py')]
    implementation += [REPO / 'tools/cellpose_refine' / f'{n}.py' for n in ('common', 'features', 'model', 'infer')]
    implementation += [REPO / 'tools/annotation_selection/metric_adapter.py', REPO / 'tools/cellpose_ultrack/track.py']
    plan = dict(created_utc=now(), panel_sha256=sha(PANEL), frames=inputs, clips=panel['clips'], methods=METHODS,
                head_models=heads, offsets=offsets, incumbent_inputs=incumbents,
                native_checkpoint=dict(path=str(PRIMARY), sha256=sha(PRIMARY)),
                cellpose_checkpoint=dict(path=str(CHECKPOINT), sha256=CHECKPOINT_SHA),
                native_source={str(p): sha(p) for p in sorted(NATIVE.rglob('*.py'))},
                implementation={str(p): sha(p) for p in implementation}, metric_revision=METRIC_REV,
                refiner_config_sha256=sha(REPO / 'configs/cellpose-refine-v1.json'),
                linking=dict(activation='softmax over sources (dim0)', threshold=.5, max_parents=1, max_children=2,
                             tie_break='descending (probability, source index, target index)', motion_gate_um=None,
                             node_filter=None, union_suppression_um=2.),
                feature_rule='Fresh native image and position features at each method integer centers, no nearest-proposal transfer. Original movie quantiles and (1,4,4) image stride.',
                selection='Both refiner seeds reported. No tuning or checkpoint selection on this panel. All graphs frozen before any new metric read.',
                scope='600 frames in six complete, previously reused training clips. Released native association weights are training-exposed. This controlled native greedy pipeline is not the full production Harmonic pipeline and cannot establish unseen-embryo generalization.')
    path = OUT / 'full-clips-plan.json'
    if path.exists():
        prior = read(path)
        assert {k: v for k, v in prior.items() if k != 'created_utc'} == {k: v for k, v in plan.items() if k != 'created_utc'}
    else:
        write(path, plan)
    progress('prepared', 0, 600, reused_feature_frames=sum('feature_path' in r for r in inputs))


def plan():
    p = read(OUT / 'full-clips-plan.json')
    for path, digest in p['implementation'].items():
        assert sha(path) == digest, path
    assert sha(PANEL) == p['panel_sha256']
    return p


def union(base, addition):
    centers = list(base['centers_zyx'])
    scores = list(base['scores'])
    for i in np.argsort(-addition['scores'], kind='stable'):
        point = addition['centers_zyx'][i]
        if centers and np.min(np.linalg.norm((np.asarray(centers) - point) * SPACING, axis=1)) <= 2.:
            continue
        centers.append(point)
        scores.append(addition['scores'][i])
    return dict(centers_zyx=np.asarray(centers, np.int64).reshape(-1, 3), scores=np.asarray(scores, np.float32))


def banks():
    import torch
    from tools.cellpose_refine.features import FrozenFeatures
    from tools.cellpose_refine.model import Refiner
    from tools.cellpose_refine.infer import refine_cached, install_inference_guard
    from tools.cellpose_refine.common import bounded_integer, configure
    from tools.detector_screen.cellpose_adapter import gpu_lock
    configure()
    install_inference_guard()
    p = plan()
    cfg = read(REPO / 'configs/cellpose-refine-v1.json')
    assert sha(REPO / 'configs/cellpose-refine-v1.json') == p['refiner_config_sha256']
    extractor = FrozenFeatures('cpu')
    heads = []
    for record in p['head_models']:
        assert sha(record['path']) == record['sha256']
        state = torch.load(record['path'], map_location='cpu', weights_only=True)
        head = Refiner(state['config'])
        head.load_state_dict(state['model'], strict=True)
        heads.append((record, head.eval().requires_grad_(False)))
    incumbent = {}
    for name, record in p['incumbent_inputs'].items():
        assert sha(record['path']) == record['sha256']
        with np.load(record['path'], allow_pickle=False) as f:
            incumbent[name] = dict(coords=f['coords'], scores=f['node_probabilities'])
    receipts = []
    for begin in range(0, len(p['frames']), 8):
        rows = p['frames'][begin:begin + 8]
        with gpu_lock(GPU_LOCK):
            extractor.net.to('cuda')
            extractor.device = torch.device('cuda')
            for _, head in heads:
                head.to('cuda')
            try:
                for row in rows:
                    key = row['key']
                    done_path = ROOT / 'banks-receipts' / (key + '.json')
                    if done_path.exists():
                        old = read(done_path)
                        assert old['plan_sha256'] == sha(OUT / 'full-clips-plan.json')
                        for r in old['banks']:
                            assert sha(ROOT / 'banks' / r['method'] / (key + '.npz')) == r['sha256']
                        receipts.extend(old['banks'])
                        continue
                    assert sha(row['prediction_path']) == row['prediction_sha256']
                    with np.load(row['prediction_path'], allow_pickle=False) as f:
                        queries = np.asarray(f['centers_zyx'], np.float64)
                        scores = f['scores'].copy()
                    if 'feature_path' in row:
                        assert sha(row['feature_path']) == row['feature_sha256']
                        with np.load(row['feature_path'], allow_pickle=False) as f:
                            np.testing.assert_array_equal(queries, f['queries'])
                            feature, patch, geometry = (f[k].copy() for k in ('feature', 'patch', 'geometry'))
                    else:
                        assert sha(row['image_path']) == row['image_sha256']
                        volume = np.load(row['image_path'], allow_pickle=False)
                        feature, patch, geometry = extractor.extract(volume, queries)
                    # Imported Cellpose enables TF32; the original FP32 heads disabled it.
                    torch.backends.cuda.matmul.allow_tf32 = False
                    shape = row['shape']
                    base = np.clip(np.rint(queries), 0, np.asarray(shape) - 1).astype(np.int64)
                    values = dict(cellpose=dict(centers_zyx=base, scores=scores))
                    source = {'44b6': '6bba', '6bba': '44b6'}[row['embryo']]
                    offset = np.array(p['offsets'][source]['offset'])
                    points, _ = bounded_integer(queries, queries + offset / SPACING, shape, 3.)
                    values['constant'] = dict(centers_zyx=points, scores=scores)
                    for record, head in heads:
                        if record['scope'] == 'source' and record['target'] != row['embryo']:
                            continue
                        refined = refine_cached(head, queries, feature, patch, geometry, shape, cfg)
                        points = refined['centers_zyx'].astype(np.int64)
                        assert len(points) == len(base) and np.all(np.linalg.norm((points - base) * SPACING, axis=1) <= 3.000001)
                        values[f"{record['scope']}-{record['seed']}"] = dict(centers_zyx=points, scores=scores)
                    inc = incumbent[row['dataset']]
                    selected = inc['coords'][:, 0] == row['time']
                    values['incumbent'] = dict(centers_zyx=np.clip(np.rint(inc['coords'][selected, 1:]), 0, np.asarray(shape) - 1).astype(np.int64), scores=inc['scores'][selected])
                    values['incumbent+cellpose'] = union(values['incumbent'], values['cellpose'])
                    values['incumbent+source-20260914'] = union(values['incumbent'], values['source-20260914'])
                    records = []
                    for method in METHODS:
                        dest = ROOT / 'banks' / method / (key + '.npz')
                        npz(dest, **values[method])
                        records.append(dict(method=method, key=key, sha256=sha(dest), count=len(values[method]['centers_zyx'])))
                    write(done_path, dict(plan_sha256=sha(OUT / 'full-clips-plan.json'), banks=records,
                                          annotation_access=False, created_utc=now()))
                    receipts.extend(records)
            finally:
                extractor.tokens = None
                extractor.net.to('cpu')
                for _, head in heads:
                    head.to('cpu')
                torch.cuda.empty_cache()
        progress('candidate banks', min(begin + 8, len(p['frames'])), len(p['frames']))
    write(OUT / 'full-clips-banks-lock.json', dict(created_utc=now(), plan_sha256=sha(OUT / 'full-clips-plan.json'), banks=receipts))


def greedy(probs):
    candidates = sorted([(float(probs[i, j]), int(i), int(j)) for i, j in zip(*np.where(probs > .5))], reverse=True)
    children, parents, edges = Counter(), Counter(), []
    for _, i, j in candidates:
        if children[i] < 2 and parents[j] < 1:
            edges.append((i, j))
            children[i] += 1
            parents[j] += 1
    return np.asarray(edges, np.int64).reshape(-1, 2)


def link():
    import torch
    from tools.detector_screen.cellpose_adapter import gpu_lock
    from tools.cellpose_refine.infer import install_inference_guard
    from tools.cellpose_ultrack.track import validate_graph
    install_inference_guard()
    p = plan()
    lock = read(OUT / 'full-clips-banks-lock.json')
    assert lock['plan_sha256'] == sha(OUT / 'full-clips-plan.json')
    for r in lock['banks']:
        assert sha(ROOT / 'banks' / r['method'] / (r['key'] + '.npz')) == r['sha256']
    for path, digest in p['native_source'].items():
        assert sha(path) == digest
    for path in (NATIVE / 'src', NATIVE / 'scripts'):
        sys.path.insert(0, str(path))
    import predict_unet_transformer as prediction
    import train_unet_transformer as training
    torch.set_num_threads(2)
    assert sha(PRIMARY) == p['native_checkpoint']['sha256']
    model, window, ds = prediction.load_model(PRIMARY, torch.device('cpu'))
    model.eval().requires_grad_(False)
    assert window == 2 and tuple(ds) == (1, 4, 4)
    receipts, completed = [], 0
    for clip in p['clips']:
        name = clip['dataset']
        rows = sorted([r for r in p['frames'] if r['dataset'] == name], key=lambda r: r['time'])
        assert [r['time'] for r in rows] == list(range(clip['shape'][0]))
        meta_path = DATA / f'{name}.zarr/zarr.json'
        assert sha(meta_path) == p['incumbent_inputs'][name]['metadata_sha256']
        quantiles = read(meta_path)['attributes']['image_statistics']['quantiles']
        low, high = float(quantiles['0.001']), float(quantiles['0.999'])
        images, points, offsets, nodes, edge_parts = [], {}, {}, {}, {}
        for row in rows:
            assert sha(row['image_path']) == row['image_sha256']
            image = np.load(row['image_path'], allow_pickle=False)[::1, ::4, ::4].astype(np.float32)
            images.append(torch.from_numpy(np.maximum((image - low) / (high - low + 1e-6), 0)))
        for method in METHODS:
            points[method], offsets[method], parts, count = [], [], [], 0
            for row in rows:
                with np.load(ROOT / 'banks' / method / (row['key'] + '.npz'), allow_pickle=False) as f:
                    xyz = f['centers_zyx'].astype(np.int64)
                points[method].append(xyz)
                offsets[method].append(count)
                parts.append(np.column_stack([np.arange(count, count + len(xyz)), np.full(len(xyz), row['time']), xyz]))
                count += len(xyz)
            nodes[method] = np.concatenate(parts).astype(np.int64)
            edge_parts[method] = []
        for begin in range(0, len(rows) - 1, 8):
            with gpu_lock(GPU_LOCK):
                model.to('cuda')
                try:
                    with torch.inference_mode():
                        for t in range(begin, min(begin + 8, len(rows) - 1)):
                            features, _ = model.encode(torch.stack(images[t:t + 2])[None].to('cuda'))
                            for method in METHODS:
                                coords, pos, masks, embeddings = [], [], [], []
                                for j in range(2):
                                    xyz = points[method][t + j].astype(np.float32)
                                    grid = xyz / np.array(ds, np.float32)
                                    c = torch.tensor(grid, device='cuda')[None]
                                    m = torch.ones((1, len(grid)), dtype=torch.bool, device='cuda')
                                    fpos = training.extract_pos_features(np.column_stack([np.full(len(grid), j), grid]), (2, 64, 64, 64))
                                    coords.append(c * torch.tensor(ds, device='cuda'))
                                    pos.append(torch.tensor(fpos, device='cuda')[None])
                                    masks.append(m)
                                    embeddings.append(model._index_features(features[:, j], c, m))
                                if all(e.shape[1] for e in embeddings):
                                    logits = model.predict_edges(*embeddings, *coords, *pos, *masks)[0]
                                    assert torch.isfinite(logits).all()
                                    edges = greedy(torch.softmax(logits, dim=0).cpu().numpy())
                                else:
                                    edges = np.empty((0, 2), np.int64)
                                edge_parts[method].append(edges + np.array([offsets[method][t], offsets[method][t + 1]]))
                            del features
                            completed += 1
                finally:
                    model.to('cpu')
                    torch.cuda.empty_cache()
            progress('native linking', completed, sum(c['shape'][0] - 1 for c in p['clips']), dataset=name)
        for method in METHODS:
            edges = np.concatenate(edge_parts[method]).astype(np.int64)
            validate_graph(nodes[method], edges, clip['shape'])
            dest = ROOT / 'graphs' / method / (name + '.npz')
            npz(dest, nodes=nodes[method], edges=edges)
            receipts.append(dict(method=method, dataset=name, sha256=sha(dest), nodes=len(nodes[method]), edges=len(edges), frames=len(rows)))
    write(OUT / 'full-clips-graphs-lock.json', dict(created_utc=now(), plan_sha256=sha(OUT / 'full-clips-plan.json'), banks_lock_sha256=sha(OUT / 'full-clips-banks-lock.json'), graphs=receipts))


def evaluate():
    import logging
    import subprocess
    import warnings
    import zarr
    from scipy.spatial.distance import cdist
    from tools.annotation_selection.common import OFFICIAL, METRIC_REV
    from tools.annotation_selection.metric_adapter import evaluate_graph, aggregate
    from tools.detector_screen.evaluate import matches
    from tools.cellpose_ultrack.evaluate import assert_matching
    from tools.cellpose_ultrack.track import validate_graph
    logging.disable(logging.WARNING)
    warnings.filterwarnings('ignore')
    p = plan()
    assert subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip() == METRIC_REV == p['metric_revision']
    assert not subprocess.check_output(['git', '-C', str(OFFICIAL), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    lock = read(OUT / 'full-clips-graphs-lock.json')
    assert lock['plan_sha256'] == sha(OUT / 'full-clips-plan.json') and len(lock['graphs']) == 6 * len(METHODS)
    for r in lock['graphs']:
        assert sha(ROOT / 'graphs' / r['method'] / (r['dataset'] + '.npz')) == r['sha256']
    opened, results = now(), []
    for clip in p['clips']:
        name = clip['dataset']
        gt = zarr.open_group(DATA / f'{name}.geff', mode='r')
        truth = np.column_stack([gt['nodes/ids'][:], *[gt[f'nodes/props/{a}/values'][:] for a in 'tzyx']]).astype(np.int64)
        gedges = np.asarray(gt['edges/ids'][:], np.int64).reshape(-1, 2)
        meta = dict(gt.attrs)['geff']
        scale = [next(a['scale'] for a in meta['axes'] if a['name'] == k) for k in 'zyx']
        np.testing.assert_allclose(scale, SPACING)
        for method in METHODS:
            path = ROOT / 'graphs' / method / (name + '.npz')
            with np.load(path, allow_pickle=False) as f:
                nodes, edges = f['nodes'], f['edges']
            validate_graph(nodes, edges, clip['shape'])
            result, mapping, true_edges = evaluate_graph(name, nodes, edges, truth, gedges, scale, meta['extra']['estimated_number_of_nodes'])
            assert_matching(mapping, nodes, truth, scale)
            counts = {str(r): 0 for r in range(1, 8)}
            crowd = {str(r): dict(pairs=0, both_at3=0, both_at7=0) for r in (7, 14)}
            for t in range(clip['shape'][0]):
                pred, target = nodes[nodes[:, 1] == t, 2:], truth[truth[:, 1] == t]
                matched = {r: matches(pred, target, t, float(r)) for r in range(1, 8)}
                for r, m in matched.items():
                    counts[str(r)] += len(m)
                distances = cdist(target[:, 2:] * SPACING, target[:, 2:] * SPACING)
                for r in (7, 14):
                    pairs = list(zip(*np.where(np.triu(distances <= r, k=1))))
                    crowd[str(r)]['pairs'] += len(pairs)
                    for delta in (3, 7):
                        recovered = set(matched[delta].values())
                        crowd[str(r)][f'both_at{delta}'] += sum(target[i, 0] in recovered and target[j, 0] in recovered for i, j in pairs)
            matched_gt = set(mapping.values())
            available = sum(int(a) in matched_gt and int(b) in matched_gt for a, b in gedges)
            result.update(method=method, embryo=clip['embryo'], gt_nodes=len(truth), gt_edges=len(gedges),
                          proposals=len(nodes), selected_edges=len(edges), detector_matches=counts, close=crowd,
                          available_gt_edges=available, linked_gt_edges=len(true_edges),
                          detector_recall={r: n / len(truth) for r, n in counts.items()}, graph_sha256=sha(path))
            results.append(result)
            progress('official scoring', len(results), 6 * len(METHODS), method=method, dataset=name)
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for embryo in ('pooled', '44b6', '6bba'):
            subset = [r for r in results if r['method'] == method and (embryo == 'pooled' or r['embryo'] == embryo)]
            s = aggregate(subset, [r['dataset'] for r in subset])
            total = sum(r['gt_nodes'] for r in subset)
            s.update(gt_nodes=total, proposals=sum(r['proposals'] for r in subset),
                     detector_recall={k: sum(r['detector_matches'][k] for r in subset) / total for k in map(str, range(1, 8))},
                     available_gt_edges=sum(r['available_gt_edges'] for r in subset), linked_gt_edges=sum(r['linked_gt_edges'] for r in subset),
                     close={str(k): {field: sum(r['close'][str(k)][field] for r in subset) for field in ('pairs', 'both_at3', 'both_at7')} for k in (7, 14)})
            summaries[method][embryo] = s
    write(OUT / 'full-clips-evaluation.json', dict(created_utc=now(), labels_opened_utc=opened, graphs_lock_sha256=sha(OUT / 'full-clips-graphs-lock.json'),
                                                 rows=results, summaries=summaries, scope=p['scope']))
    progress('complete', 54, 54)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'banks', 'link', 'evaluate'))
    args = parser.parse_args()
    {'prepare': prepare, 'banks': banks, 'link': link, 'evaluate': evaluate}[args.action]()
