"""Source-only three-state calibration using supported real raw/P0 hypotheses."""
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_graph, write_json
from .infer import embeddings
from .observations import ObservationBank, raw_graph
from .resources import Lease, Monitor
from .scoring import load_model


def fit(logits, labels):
    logits, labels = np.asarray(logits, float), np.asarray(labels, int)
    def objective(p):
        value = logits/np.exp(p[0])+np.array([0., p[1], p[2]])
        return float((logsumexp(value, axis=1)-value[np.arange(len(value)), labels]).mean()+.01*np.square(p).sum())
    result = minimize(objective, [0., 0., 0.], method='L-BFGS-B', bounds=[(-2., 2.), (-12., 12.), (-12., 12.)])
    if not result.success:
        raise RuntimeError('Observation source calibration failed to converge')
    return dict(temperature=float(np.exp(result.x[0])), state_intercepts=[0., float(result.x[1]), float(result.x[2])],
        regularization=.01, before_loss=objective([0., 0., 0.]), after_loss=objective(result.x),
        state_counts=np.bincount(labels, minlength=3).tolist(), rows=len(labels), converged=True,
        threshold='State logit must exceed keep; one fixed shared calibration, no margin grid')


def collect(model, spec, row, root):
    name = row['dataset']
    path = root / f'{name}.npz'
    if path.exists():
        receipt = read_json(path.with_suffix('.json'))
        if sha(path) != receipt['sha256'] or receipt['model_sha256'] != spec['weights_sha256']:
            raise ValueError('Observation calibration cache drift')
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    graph, raw = verified_graph(row), raw_graph(row)
    bank = ObservationBank(row, graph, raw)
    with np.load(WORK / 'observation_source' / spec['recipe']['source'] / f'{name}.npz', allow_pickle=False) as data:
        known = (data['labels'] >= 0) & (data['group'] >= 0)
        pairs, labels, groups = data['pairs'][known], data['labels'][known], data['group'][known]
    if not len(pairs):
        result = dict(logits=np.empty((0, 3), np.float32), labels=labels, group=np.asarray([], dtype=str))
    else:
        old_indices = set(map(int, pairs[:, 0]))
        old_indices.update(j for i in pairs[:, 0] for j in [*bank.pred[i], *bank.succ[i]])
        raw_indices = set(map(int, pairs[:, 1]))
        from types import SimpleNamespace
        with Lease(required_gib=8.):
            model.cuda()
            z, _ = embeddings(model, row, graph, bank, root/'embeddings/P0'/f'{name}.npz', spec['weights_sha256'], indices=old_indices)
            rz, _ = embeddings(model, row, raw, SimpleNamespace(pred=bank.raw_pred, succ=bank.raw_succ),
                root/'embeddings/raw'/f'{name}.npz', spec['weights_sha256'], indices=raw_indices)
            out = []
            with torch.inference_mode():
                for start in range(0, len(pairs), 2048):
                    pp = pairs[start:start+2048]
                    context = np.stack([z[sorted({int(i), *bank.pred[i], *bank.succ[i]})].mean(0) for i, _ in pp])
                    evidence = np.stack([bank.evidence(int(i), int(j)) for i, j in pp])
                    value = model.selection_scores(torch.as_tensor(z[pp[:, 0]], device='cuda'),
                        torch.as_tensor(rz[pp[:, 1]], device='cuda'), torch.as_tensor(context, device='cuda'),
                        torch.as_tensor(evidence, device='cuda'))
                    out.append(value.cpu().numpy())
            model.cpu()
            torch.cuda.empty_cache()
        result = dict(logits=np.concatenate(out), labels=labels, group=np.array([f'{name}:{g}' for g in groups]))
    save_arrays(path, **result)
    write_json(path.with_suffix('.json'), dict(sha256=sha(path), model_sha256=spec['weights_sha256'],
        source=spec['recipe']['source'], dataset=name, real_supported_rows=len(labels), synthetic_calibration=False))
    return result


def run(source, seed=20260915):
    from .guard import install
    install(source=source)
    torch.set_num_threads(2)
    package = WORK / 'training/O10_swap' / source / str(seed)
    if (package/'frozen_package.json').exists():
        from .source_screen import run as source_screen
        for arm in ['O10_swap', 'O10_restore']:
            source_screen(source, arm, seed)
        return read_json(package/'calibration.json')
    model, spec = load_model(package)
    split = read_json(RESULTS / 'split_manifest.json')['directions'][source]
    rows = [r for r in inputs() if r['embryo'] == source and split[r['dataset']]['partition'] == 'calibration']
    root = WORK / 'observation_calibration' / source / str(seed)
    tables = []
    begin = time.monotonic()
    with Monitor(root / 'resources.json') as monitor:
        for row in rows:
            tables.append(collect(model, spec, row, root))
            monitor.check()
            print(f'Observation calibration {source}: {len(tables)}/{len(rows)} source partitions', flush=True)
    x, y = np.concatenate([t['logits'] for t in tables]), np.concatenate([t['labels'] for t in tables])
    calibration = fit(x, y)
    group = np.concatenate([t['group'] for t in tables])
    losses = logsumexp(x, axis=1)-x[np.arange(len(x)), y]
    grouped_loss = np.mean([losses[group == g].mean() for g in np.unique(group)])
    receipt = dict(status='measured', source=source, seed=seed, calibration=calibration,
        source_grouped_selection_loss=float(grouped_loss), source_groups=len(np.unique(group)),
        resubstitution=False, independently_certified=False, seconds=time.monotonic()-begin,
        split_manifest_sha256=sha(RESULTS / 'split_manifest.json'), target_labels_read=False,
        synthetic_calibration=False, actual_unbalanced_supported_distribution=True)
    write_json(package / 'calibration.json', receipt, immutable=True)
    frozen = dict(spec, calibration=calibration, calibration_status='source-only frozen',
                  calibration_sha256=sha(package / 'calibration.json'))
    write_json(package / 'frozen_package.json', frozen, immutable=True)
    from .source_screen import run as source_screen
    for arm in ['O10_swap', 'O10_restore']:
        source_screen(source, arm, seed)
    return receipt
