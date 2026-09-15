"""Regularized source-only calibration on the unbalanced supported candidate bank."""
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_evidence, verified_graph, write_json
from .feasibility import from_record, records
from .resources import Lease, Monitor


def targets(rows):
    event_anchors = {r['anchor'] for r in rows if r['labels']['biological'] == 1}
    known, good = [], []
    for r in rows:
        label, kind = r['labels'], r['decision']['kind']
        positive = label['biological'] == 1
        negative = label['identity'] == 0 or label['metric_fork_target'] == 0
        if r['anchor'] in event_anchors:
            negative |= not positive and kind in ['keep', 'continuation_a', 'continuation_b', 'two_parents']
        else:
            positive |= kind != 'division' and label['identity'] == 1
        good.append(positive)
        known.append(positive or negative)
    return np.asarray(good, dtype=bool), np.asarray(known, dtype=bool)


def fit_binary(value, label, weight, editable=None):
    """One fixed regularized fit, without a threshold or margin grid."""
    value, label, weight = map(lambda a: np.asarray(a, float), [value, label, weight])
    editable = np.ones_like(value) if editable is None else np.asarray(editable, float)
    if not len(value) or not np.isfinite(value).all() or not np.isfinite(weight).all():
        raise ValueError('Calibration requires finite supported source rows')
    weight = weight/weight.sum()
    def loss(parameters):
        lt, bias = parameters
        logits = value/np.exp(lt)+bias*editable
        return float(np.sum(weight*(np.logaddexp(0, logits)-label*logits))+.01*(lt*lt+bias*bias))
    fitted = minimize(loss, np.array([0., 0.]), method='L-BFGS-B', bounds=[(-2., 2.), (-12., 12.)])
    if not fitted.success:
        raise RuntimeError(f'Source calibration optimizer did not converge: {fitted.message}')
    lt, bias = fitted.x
    return dict(temperature=float(np.exp(lt)), intercept=float(bias), regularization=.01,
                before_loss=loss([0., 0.]), after_loss=loss(fitted.x), rows=len(value),
                positives=int(label.sum()), effective_unbalanced_positive_fraction=float(weight@label),
                optimizer='L-BFGS-B', converged=True, threshold=0.)


def group_loss(values, good, known, group):
    group = np.asarray(group)
    out = []
    for g in np.unique(group):
        use = (group == g) & known
        pos = use & good
        if pos.any() and (use & ~good).any():
            out.append(float(logsumexp(values[use])-logsumexp(values[pos])))
    return dict(loss=float(np.mean(out)) if out else None, groups=len(out))


def collect(row, spec, model, folder):
    from .bank import EventBank
    from .infer import embeddings
    from .scoring import score
    name = row['dataset']
    destination = folder / f'{name}.npz'
    if destination.exists():
        receipt = read_json(destination.with_suffix('.json'))
        if sha(destination) != receipt['sha256'] or receipt['model_sha256'] != spec['weights_sha256']:
            raise ValueError('Calibration cache drift')
        with np.load(destination, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    graph, native = verified_graph(row), verified_evidence(row)
    rows = list(records(spec['recipe']['source'], name))
    # All alternatives and time anchors remain together; no row balancing/cap.
    decisions = [from_record(r) for r in rows]
    path = WORK / 'source' / spec['recipe']['source'] / 'pairs' / f'{name}.npz'
    with np.load(path, allow_pickle=False) as data:
        ii, pair_y, pair_groups = data['index'], data['labels'], data['group']
    required = {n for d in decisions for n in d.event[:3]}
    required.update(n for d in decisions for e in d.remove | d.add for n in e)
    required.update(map(int, native['pairs'][ii].flatten()))
    bank = EventBank(graph['nodes'], graph['edges'], native, row['physical_scale'])
    with Lease(required_gib=8.):
        model.cuda()
        z, valid = embeddings(model, row, graph, bank, folder / 'embeddings' / f'{name}.npz', spec['weights_sha256'], indices=required)
        pair_score = []
        with torch.inference_mode():
            zz = torch.as_tensor(z, device='cuda')
            for start in range(0, len(ii), 4096):
                ids = ii[start:start+4096]
                pp = torch.as_tensor(native['pairs'][ids], device='cuda')
                x = native['edge_features'][ids]
                normalized = np.clip((x-spec['mean'])/spec['scale'], -10, 10).astype(np.float32)
                value = model.link_scores(zz, pp, torch.as_tensor(normalized, device='cuda')).cpu().numpy()+x[:, 23]
                pair_score.extend(value)
        values, risks = [], []
        for start in range(0, len(rows), 1024):
            part = rows[start:start+1024]
            vv, rr = score(model, spec, graph['nodes'], native, z, valid, decisions[start:start+1024],
                           [r['event_features'] for r in part], row['physical_scale'])
            values.extend(vv)
            risks.extend(rr)
        model.cpu()
        torch.cuda.empty_cache()
    good, known = targets(rows)
    result = dict(value=np.asarray(values), risk=np.asarray(risks), good=good, known=known,
        group=np.asarray([r['group'] for r in rows]), editable=np.asarray([d.kind != 'keep' for d in decisions]),
        inverse_anchor_inclusion=np.asarray([1/r['sampling_probability'] for r in rows]),
        pair_score=np.asarray(pair_score), pair_y=pair_y, pair_group=pair_groups)
    save_arrays(destination, **result)
    write_json(destination.with_suffix('.json'), dict(sha256=sha(destination), model_sha256=spec['weights_sha256'],
        source=spec['recipe']['source'], dataset=name, decisions=len(rows), supported_pairs=len(ii),
        unknown_rows_masked=int((~known).sum()), target_labels_read=False))
    return result


def run(source, arm, seed=20260915):
    from .guard import install
    install(source=source)
    from .scoring import load_model
    torch.set_num_threads(2)
    package = WORK / 'training' / arm / source / str(seed)
    model, spec = load_model(package)
    if spec['recipe']['source'] != source:
        raise PermissionError('Calibration source/model mismatch')
    destination = package / 'calibration.json'
    if destination.exists():
        from .source_screen import run as source_replay
        source_replay(source, arm, seed)
        return read_json(destination)
    split = read_json(RESULTS / 'split_manifest.json')['directions'][source]
    rows = [r for r in inputs() if r['embryo'] == source and split[r['dataset']]['partition'] == 'calibration']
    source_events = sum(read_json(WORK / 'source' / source / 'receipts' / f'{r["dataset"]}.json')['covered_division_groups'] for r in rows)
    resubstitution = source_events < 2 and arm != 'A10'
    if resubstitution:
        rows = [r for r in inputs() if r['embryo'] == source]
    folder = WORK / 'calibration' / arm / source / str(seed)
    tables = []
    begin = time.monotonic()
    with Monitor(folder / 'resources.json') as monitor:
        for k, row in enumerate(rows, 1):
            tables.append(collect(row, spec, model, folder))
            monitor.check()
            print(f'Source calibration {arm}/{source}: {k}/{len(rows)} complete candidate partitions', flush=True)
    combine = lambda key: np.concatenate([t[key] for t in tables])
    if arm == 'A10':
        calibration = fit_binary(combine('pair_score'), combine('pair_y'), np.ones(len(combine('pair_y'))))
        diagnostic = dict(supported_pairs=len(combine('pair_y')))
    else:
        value, good, known = combine('value'), combine('good'), combine('known')
        calibration = fit_binary(value[known], good[known], combine('inverse_anchor_inclusion')[known], combine('editable')[known])
        calibrated = value/calibration['temperature']+calibration['intercept']*combine('editable')
        diagnostic = dict(before=group_loss(value, good, known, combine('group')),
                          after=group_loss(calibrated, good, known, combine('group')))
    receipt = dict(status='measured', source=source, arm=arm, seed=seed, calibration=calibration,
        source_grouped_decision=diagnostic, resubstitution=resubstitution,
        independence_certified=False, source_event_groups=source_events,
        split_manifest_sha256=sha(RESULTS / 'split_manifest.json'), clips=[r['dataset'] for r in rows],
        weights_sha256=spec['weights_sha256'], seconds=time.monotonic()-begin,
        fitting_distribution='Unbalanced supported candidates; deterministic ordinary-anchor inclusion weighted by nine. Unknown decisions are censored. No annotation-density feature.',
        checkpoint_selection='Fixed final checkpoint; this calibration does not select weights.', target_labels_read=False)
    write_json(destination, receipt, immutable=True)
    frozen = dict(spec, calibration=calibration, calibration_status='source-only frozen', calibration_sha256=sha(destination))
    if spec['recipe']['family'] == 'organoid':
        from .organoid_adapter import ORG_MODEL, ORG_SHA
        frozen['architecture_dependency'] = dict(path=str(ORG_MODEL / 'model.keras'), sha256=ORG_SHA)
    write_json(package / 'frozen_package.json', frozen, immutable=True)
    from .source_screen import run as source_replay
    source_replay(source, arm, seed)
    if arm == 'D10_frozen' and source == '6bba' and seed == 20260915:
        # Both control directions now exist. Use this serial queue boundary for
        # the mandatory fresh-image contract check; it reads no target metrics.
        import subprocess
        import sys
        log_path = WORK/'fresh_validation/controller.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.fresh_validation'],
                                   stdout=log, stderr=subprocess.STDOUT)
        write_json(WORK/'fresh_validation/controller.json', dict(returncode=child.returncode,
            log_sha256=sha(log_path), independent_training_lanes_continue=True))
    return receipt
