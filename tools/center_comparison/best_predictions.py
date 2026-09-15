"""Add retained P0 and highest-pooled C4_m6 graphs with official error locations."""
from __future__ import annotations

import gzip
import json
import shutil
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .export import lineage_roots, official_matches, point_rows
from .pipeline import RADII, WORK, read_geometry, read_gt, sha256, write_json

DEFAULT_PREDICTIONS = WORK / 'segmentation-tracking-v6-local/prior-v6/ram-outputs/predictions/P0'
DEFAULT_EVALUATION = WORK / 'ultrack-integration-v7/evaluation/P0'


def diagnose(nodes, edges, gt_nodes, gt_edges, spacing, estimate):
    """Read edge validity and division decisions from the pinned official scorer.

    Work on the complete clip before selecting display frames. In particular,
    annotation-window boundaries and unlabelled cells must not become false errors.
    Persisted IDs need explicit maps: graph-library IDs are unrelated row indices.
    """
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from annotation_selection.metric_adapter import make_graph, official
    from tracking_cellmot.division_metrics import score_divisions

    pred, pmap = make_graph(nodes, edges)
    gt, gmap = make_graph(gt_nodes, gt_edges)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        counts = official.evaluate(pred, gt, scale=tuple(spacing), max_distance=7.)
        if len(nodes) and len(gt_nodes) and not len(edges):
            pred.match(gt, matching=DistanceMatching(max_distance=7., scale=tuple(spacing), optimal=True))
        divisions = score_divisions(pred, gt, scale=tuple(spacing), max_distance=7.)
    keys = td.DEFAULT_ATTR_KEYS
    matches = {pmap[int(a)]: gmap[int(b)] for a, b in pred.node_attrs(
        attr_keys=[keys.NODE_ID, keys.MATCHED_NODE_ID]).iter_rows() if b is not None and b != -1} if len(nodes) and len(gt_nodes) else {}
    if len(set(matches.values())) != len(matches):
        raise ValueError('Official matching is not one-to-one')
    reverse = {g: p for p, g in matches.items()}
    pn = {int(n[0]): list(map(int, n)) for n in nodes}
    gn = {int(n[0]): list(map(int, n)) for n in gt_nodes}
    correct, incorrect = set(), set()
    if len(edges):
        attrs = official._evaluate_matched_graph(pred, gt)
        for a, b, matched, valid in attrs.select(
                keys.EDGE_SOURCE, keys.EDGE_TARGET, keys.MATCHED_EDGE_MASK, 'pred_valid').iter_rows():
            if matched:
                correct.add((pmap[int(a)], pmap[int(b)]))
            elif valid:
                incorrect.add((pmap[int(a)], pmap[int(b)]))
    recovered = {(matches[a], matches[b]) for a, b in correct}
    missed = set(map(tuple, gt_edges)) - recovered
    if (len(correct), len(incorrect), len(missed)) != (counts.edge_tp, counts.edge_fp, counts.edge_fn):
        raise ValueError('Official edge locations disagree with score counts')
    if (sum(divisions.scores.values()), len(divisions.fp_forks), len(divisions.scores) - sum(divisions.scores.values())) != (counts.division_tp, counts.division_fp, counts.division_fn):
        raise ValueError('Official division locations disagree with score counts')

    events = []
    def segment(method, a, b, status):
        lookup = gn if method == 'gt' else pn
        return dict(method=method, source=lookup[int(a)], target=lookup[int(b)], status=status)

    def context(gids, pids):
        # Local expected and actual links make a wrong destination or broken
        # continuation inspectable without guessing identities from coordinates.
        out = [segment('gt', a, b, 'expected') for a, b in gt_edges if a in gids or b in gids]
        out += [segment('best', a, b, 'prediction') for a, b in edges if a in pids or b in pids]
        return out

    for a, b in sorted(missed):
        a, b = int(a), int(b)
        present = [g for g in (a, b) if g in reverse]
        reason = ('Both annotated endpoints have P0 matches, but their assigned predictions are not linked.'
                  if len(present) == 2 else 'At least one annotated endpoint has no P0 assignment within 7 µm.')
        events.append(dict(id=f'edge_fn:{a}:{b}', kind='edge_fn', t=gn[b][1], method='gt', node=b,
            label=f'Missed link {a} → {b}', reason=reason,
            segments=[segment('gt', a, b, 'fn'), *context({a, b}, {reverse[g] for g in present})]))
    for a, b in sorted(incorrect):
        gids = {matches[p] for p in (a, b) if p in matches}
        events.append(dict(id=f'edge_fp:{a}:{b}', kind='edge_fp', t=pn[b][1], method='best', node=b,
            label=f'Incorrect link {a} → {b}',
            reason='The official metric counts this predicted link as incorrect against the evaluable sparse annotation graph.',
            segments=[segment('best', a, b, 'fp'), *context(gids, {a, b})]))
    for i, recovered in sorted(divisions.scores.items()):
        if not recovered:
            g = gmap[i]
            events.append(dict(id=f'division_fn:{g}', kind='division_fn', t=gn[g][1], method='gt', node=g,
                label=f'Missed division at annotation {g}',
                reason='The official division check did not recover the parent and both daughter lineages. Division matching is evaluated separately from the center assignments.',
                segments=context({g}, {reverse[g]} if g in reverse else set())))
    for i in sorted(divisions.fp_forks):
        p = pmap[i]
        events.append(dict(id=f'division_fp:{p}', kind='division_fp', t=pn[p][1], method='best', node=p,
            label=f'Incorrect division at prediction {p}',
            reason='The official division check counts this predicted fork as a false positive. This is a scored fork, not merely an unmatched cell.',
            segments=context({matches[p]} if p in matches else set(), {p})))
    recall = len(matches) / len(gt_nodes) if len(gt_nodes) else 0.
    metrics = official.per_sample_metrics(counts, estimate, recall)
    metrics['matched_nodes'] = len(matches)
    return dict(metrics=metrics, matches=matches, events=events,
                edge_status={f'{a}:{b}': 'tp' if (a, b) in correct else 'fp' if (a, b) in incorrect else 'unknown' for a, b in edges})


def center_events(frame, radius, method='best', name='P0'):
    matches = {g: (p, d) for p, g, d in frame['matches'][method][str(radius)]}
    nearest = {g: (p, d) for g, p, d in frame['model_nearest'][method]}
    events = []
    for g, *_ in frame['points']['gt']:
        pair = matches.get(g)
        if pair is not None and pair[1] < 3.:
            continue
        kind = 'center_miss' if pair is None else 'offset'
        if pair is None:
            close = nearest.get(g)
            reason = f'No one-to-one {name} assignment within {radius} µm.'
            if close:
                reason += f' Nearest prediction {close[0]} is {close[1]:.2f} µm away; nearest does not imply an assignment.'
        else:
            reason = f'Assigned prediction {pair[0]} is {pair[1]:.2f} µm from the annotation. The ≥3 µm offset flag is a localization diagnostic.'
        events.append(dict(id=f'{kind}:{g}', kind=kind, t=frame['t'], method='gt', node=g,
            label=f'{"Missed annotation" if pair is None else "Center offset"} {g}',
            distance_um=None if pair is None else pair[1], reason=reason, segments=[]))
    return events


def add_model(args, method, name, predictions, evaluation, description):
    from scipy.spatial.distance import cdist
    from strong_tracker_v3.common import graph_hash, validate
    from annotation_selection.common import METRIC_REV
    from .error_cases import CATEGORIES, center_case, tracking_cases

    site = args.output / 'site'
    index = json.loads((site / 'index.json').read_text())
    frames = {key: json.loads((site / 'frames' / f'{key}.json').read_text()) for key in index['frame_keys']}
    summaries, errors = {}, {str(int(r)): [] for r in RADII}
    cases = {str(int(r)): [] for r in RADII}
    for dataset in dict.fromkeys(f['dataset'] for f in frames.values()):
        by_time = {f['t']: f for f in frames.values() if f['dataset'] == dataset}
        source = predictions / f'{dataset}.npz'
        receipt_path = evaluation / f'{dataset}.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['metric_revision'] != METRIC_REV or receipt['prediction_sha256'] != sha256(source):
            raise ValueError(f'{name} receipt/source mismatch: {dataset}')
        with np.load(source, allow_pickle=False) as graph:
            nodes, edges = graph['nodes'], graph['edges']
        _, shape, spacing = read_geometry(args.data_root, dataset)
        validate(nodes, edges, shape)
        gt_nodes, gt_edges = read_gt(args.data_root, dataset, spacing)
        if graph_hash(nodes, edges) != receipt['graph_hash']:
            raise ValueError(f'{name} graph hash mismatch: {dataset}')
        diagnosis = diagnose(nodes, edges, gt_nodes, gt_edges, spacing, receipt['estimated_total'])
        explained = tracking_cases(diagnosis, nodes, edges, gt_nodes, gt_edges, spacing)
        for case in explained:
            case['reason'] = case['reason'].replace('P0', name)
            if case['method'] == 'best':
                case['method'] = method
        for event in diagnosis['events']:
            event['reason'] = event['reason'].replace('P0', name)
            if event['method'] == 'best':
                event['method'] = method
            for segment in event['segments']:
                if segment['method'] == 'best':
                    segment['method'] = method
        for k in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes', 'matched_nodes', 'adj_edge_jaccard']:
            if not np.isclose(diagnosis['metrics'][k], receipt[k], rtol=0, atol=1e-12):
                raise ValueError(f'Fresh {name} evaluation changed: {dataset} {k}')
        roots = lineage_roots(nodes, edges)
        times = {int(n[0]): int(n[1]) for n in nodes}
        for t, f in by_time.items():
            if f['shape'] != shape[1:] or not np.allclose(f['spacing'], spacing):
                raise ValueError(f'Viewer/{name} geometry mismatch: {f["key"]}')
            gt = point_rows(gt_nodes[gt_nodes[:, 1] == t][:, [0, 2, 3, 4]], shape[1:])
            if gt != f['points']['gt']:
                raise ValueError(f'Viewer annotations differ from canonical labels: {f["key"]}')
            rows = point_rows(nodes[nodes[:, 1] == t][:, [0, 2, 3, 4]], shape[1:])
            f['points'][method] = rows
            f['matches'][method] = {str(int(r)): official_matches(rows, gt, np.asarray(spacing), r) for r in RADII}
            # Error coloring must use exactly the same official assignment as
            # the displayed 7 µm centers, including tied alternatives.
            expected = {p: g for p, g in diagnosis['matches'].items() if times[p] == t}
            if expected != {p: g for p, g, _ in f['matches'][method]['7']}:
                raise ValueError(f'Per-frame/full-clip matching drift: {f["key"]}')
            f['tracks'][method] = {p[0]: roots[p[0]] for p in rows}
            f['links'][method] = [[times[int(a)], int(a), int(b)] for a, b in edges
                                   if times[int(b)] == t and times[int(a)] in by_time]
            f.setdefault('model_edge_status', {})[method] = {f'{a}:{b}': diagnosis['edge_status'][f'{a}:{b}'] for _, a, b in f['links'][method]}
            f.setdefault('model_nearest', {})[method] = []
            if gt and rows:
                distances = cdist(np.asarray(gt)[:, 1:4] * spacing, np.asarray(rows)[:, 1:4] * spacing)
                nearest = distances.argmin(axis=1)
                f.setdefault('model_nearest', {})[method] = [[g[0], rows[i][0], float(distances[j, i])] for j, (g, i) in enumerate(zip(gt, nearest))]
            with gzip.open(site / 'frames' / f'{f["key"]}.labels.gz', 'rb') as stream:
                labels = np.frombuffer(stream.read(), '<u4').reshape(f['shape'])
            for member in f['members'].values():
                member[method] = []
            for p in rows:
                label = int(labels[tuple(p[1:4])])
                if label:
                    f['members'][str(label)][method].append(p[0])
            tracking_events = [e for e in diagnosis['events'] if e['t'] == t]
            f.setdefault('model_errors', {})[method] = tracking_events
            for r in errors:
                centers = center_events(f, r, method, name)
                for event in [*tracking_events, *centers]:
                    errors[r].append(dict(event, id=f'{dataset}:{event["id"]}', dataset=dataset, key=f['key']))
                for case in [*(c for c in explained if c['t'] == t), *(center_case(e, f, method, r) for e in centers)]:
                    cases[r].append(dict(case, id=f'{dataset}:{case["id"]}', dataset=dataset, key=f['key'],
                                         related_flags=[f'{dataset}:{flag}' for flag in case['related_flags']]))
        summaries[dataset] = dict(diagnosis['metrics'], source=str(source), source_sha256=sha256(source),
            graph_hash=receipt['graph_hash'], evaluation_source=str(receipt_path), evaluation_sha256=sha256(receipt_path),
            frames_total=shape[0], exported_times=sorted(by_time),
            exported_tracking_errors=dict(Counter(e['kind'] for e in diagnosis['events'] if e['t'] in by_time)))
        print(f'{name}_VERIFIED {dataset}: {len(nodes)} nodes; official edge/division errors agree', flush=True)
    # Update only after every source/score/coordinate check passes. No images,
    # model predictions, FOCUS outputs or previous methods are regenerated.
    for key, f in frames.items():
        write_json(site / 'frames' / f'{key}.json', f)
    index['methods'][method] = f'{name} · {description}'
    review = dict(name=name, method=method, description=description, created=datetime.now(timezone.utc).isoformat(),
        metric_revision=METRIC_REV, datasets=summaries, errors=errors, cases=cases, categories=CATEGORIES,
        scope=f'{len(frames)} exported frames from {len(summaries)} clips; tracking evaluated on each complete clip before selecting display frames.',
        interpretation='Center flags use the selected radius. Tracking errors always use the official 7 µm metric. Unmatched predictions and unevaluable links are unknown, not false positives.')
    index.setdefault('prediction_reviews', {})[method] = review
    index.pop('best_prediction', None)
    write_json(site / 'index.json', index)
    for asset in ['index.html', 'viewer.js', 'error_view.js', 'style.css', 'detection_view.js', 'detection_style.css']:
        shutil.copyfile(Path(__file__).with_name(asset), site / asset)
    write_json(args.output / f'{method}_predictions_receipt.json', dict(
        created=review['created'], frames=len(frames), datasets=summaries,
        error_counts_at_7um=dict(Counter(e['kind'] for e in errors['7'])),
        ui_source_sha256={name: sha256(site / name) for name in ['index.html', 'viewer.js', 'error_view.js', 'style.css', 'detection_view.js', 'detection_style.css']}))
    print(f'{name} added to {len(frames)} frames; {len(errors["7"])} inspectable error flags at 7 µm', flush=True)


def add_best(args):
    add_model(args, 'best', 'P0', getattr(args, 'best_predictions', DEFAULT_PREDICTIONS),
              getattr(args, 'best_evaluation', DEFAULT_EVALUATION), 'retained complete pipeline')
    add_model(args, 'pooled', 'C4_m6', WORK / 'multidata-training-v4/predictions/C4_m6',
              WORK / 'multidata-training-v4/evaluation/C4_m6', 'highest pooled score')
    # Aggregate the identical 199-clip population; keep selected status distinct
    # from the maximum observed score. Do not compare against oracle experiments.
    from annotation_selection.metric_adapter import aggregate
    from annotation_selection.common import METRIC_REV
    site = args.output / 'site'
    index = json.loads((site / 'index.json').read_text())
    expected = {p.stem for p in DEFAULT_EVALUATION.glob('*.json')}
    if len(expected) != 199:
        raise ValueError('Expected the frozen complete 199-clip evaluation')
    for method, folder, pred_dir in [
        ('best', getattr(args, 'best_evaluation', DEFAULT_EVALUATION), getattr(args, 'best_predictions', DEFAULT_PREDICTIONS)),
        ('pooled', WORK / 'multidata-training-v4/evaluation/C4_m6', WORK / 'multidata-training-v4/predictions/C4_m6')]:
        rows = []
        for dataset in sorted(expected):
            row = json.loads((folder / f'{dataset}.json').read_text())
            if (row['dataset'] != dataset or row['metric_revision'] != METRIC_REV
                    or row['prediction_sha256'] != sha256(pred_dir / f'{dataset}.npz')):
                raise ValueError(f'Full-population prediction hash drift: {method} {dataset}')
            rows.append(row)
        review = index['prediction_reviews'][method]
        review['full_evaluation'] = aggregate(rows, sorted(expected))
        review['embryos'] = {e: aggregate([r for r in rows if r['embryo'] == e],
                           sorted(r['dataset'] for r in rows if r['embryo'] == e)) for e in ['44b6', '6bba']}
    index['prediction_reviews']['pooled']['selection_note'] = 'Highest recorded pooled score among complete non-oracle pipelines. Not adopted: 6bba regressed by 0.000034 versus C0 and the required replication did not qualify.'
    index['prediction_reviews']['best']['selection_note'] = 'Retained P0 pipeline; both embryo scores improved over C0. C0 is still available as the selected v3 comparison.'
    index['methods']['selected'] = 'C0 · selected v3 full pipeline'
    write_json(site / 'index.json', index)
