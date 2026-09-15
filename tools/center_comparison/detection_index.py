"""Exhaustive annotated-node and association audit, with pageable unknown predictions.

Microscopy remains in canonical Zarr stores. Small immutable graph snapshots and
an atomic SQLite catalog let the viewer inspect every timepoint without exporting
the entire image dataset or calling sparse unmatched cells false positives.
"""
from __future__ import annotations

import json
import sqlite3
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .pipeline import WORK, read_geometry, read_gt, sha256, write_json

RADIUS = 7.
OFFSET = 3.
KINDS = {
    'missing': ['No nearby proposal or final center', 'Detection candidate'],
    'proposal_only': ['Raw proposal present; no final center nearby', 'Selection / position ambiguity'],
    'conflict': ['Nearby final center assigned elsewhere', 'Assignment ambiguity'],
    'recovered': ['Raw proposal gap; final center present', 'Raw detection candidate'],
    'offset': ['Assigned center offset ≥3 µm', 'Localization candidate'],
    'crowded': ['Competing centers or annotations', 'Identity ambiguity'],
    'matched': ['Assigned center within 3 µm', 'Matched control'],
    'association': ['Both centers assigned; expected link absent', 'Association candidate'],
    'association_ambiguous': ['Missing link with uncertain center identity', 'Association ambiguity'],
    'wrong_link': ['Scored incorrect predicted connection', 'Association candidate'],
    'wrong_link_ambiguous': ['Incorrect connection with uncertain center identity', 'Association ambiguity'],
    'unlabeled': ['Prediction without an annotation assignment', 'Unknown, not a confirmed error'],
}


def classify_center(matched, raw_matched, raw_nearest, final_nearest, distance, crowded):
    """Evidence labels, never a claim to have diagnosed biological identity."""
    if not matched:
        if final_nearest is not None and final_nearest <= RADIUS:
            return 'conflict'
        if raw_nearest is not None and raw_nearest <= RADIUS:
            return 'proposal_only'
        return 'missing'
    if not raw_matched and (raw_nearest is None or raw_nearest > RADIUS):
        return 'recovered'
    if distance >= OFFSET:
        return 'offset'
    if crowded or not raw_matched:
        return 'crowded'
    return 'matched'


def spatial_evidence(gt, predicted, matches, spacing):
    """For every GT ID: nearest, candidate count, assignment and competitors."""
    evidence = {}
    reverse = {int(g): int(p) for p, g in matches.items()}
    for t in np.unique(gt[:, 1]):
        g = gt[gt[:, 1] == t]
        p = predicted[predicted[:, 1] == t]
        distances = cdist(g[:, 2:] * spacing, p[:, 2:] * spacing)
        near = distances <= RADIUS
        pindex = {int(row[0]): i for i, row in enumerate(p)}
        for i, row in enumerate(g):
            gid = int(row[0]); assigned = reverse.get(gid)
            j = pindex[assigned] if assigned is not None else None
            nearest = int(distances[i].argmin()) if len(p) else None
            evidence[gid] = dict(
                nearest_id=None if nearest is None else int(p[nearest, 0]),
                nearest=None if nearest is None else float(distances[i, nearest]),
                count=int(near[i].sum()), assigned=assigned,
                distance=None if j is None else float(distances[i, j]),
                competitors=0 if j is None else int(near[:, j].sum()) - 1,
                assigned_not_nearest=j is not None and j != nearest)
    return evidence


def audit_graph(nodes, edges, gt, gt_edges, spacing, receipt):
    import tracksdata as td
    from annotation_selection.metric_adapter import make_graph, official
    from tracksdata.metrics import DistanceMatching
    pred, pmap = make_graph(nodes, edges)
    truth, gmap = make_graph(gt, gt_edges)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        counts = official.evaluate(pred, truth, scale=tuple(spacing), max_distance=RADIUS)
        if not len(edges) and len(nodes) and len(gt):
            pred.match(truth, matching=DistanceMatching(max_distance=RADIUS, scale=tuple(spacing), optimal=True))
    keys = td.DEFAULT_ATTR_KEYS
    matches = {pmap[int(p)]: gmap[int(g)] for p, g in pred.node_attrs(
        attr_keys=[keys.NODE_ID, keys.MATCHED_NODE_ID]).iter_rows() if g is not None and g != -1} if len(nodes) and len(gt) else {}
    statuses = []
    if len(edges):
        attrs = official._evaluate_matched_graph(pred, truth)
        for a, b, correct, valid in attrs.select(keys.EDGE_SOURCE, keys.EDGE_TARGET, keys.MATCHED_EDGE_MASK, 'pred_valid').iter_rows():
            statuses.append([pmap[int(a)], pmap[int(b)], 1 if correct else -1 if valid else 0])
    for field in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn']:
        if getattr(counts, field) != receipt[field]:
            raise ValueError(f'Fresh official audit drift: {receipt["dataset"]} {field}')
    if len(matches) != receipt['matched_nodes'] or len(set(matches.values())) != len(matches):
        raise ValueError('Annotation matching count or uniqueness changed')
    if len(nodes) != receipt['num_pred_nodes']:
        raise ValueError('Predicted node count drift')
    if sum(s == -1 for _, _, s in statuses) != counts.edge_fp:
        raise ValueError('FP edge locations disagree with official counts')
    return matches, np.asarray(statuses, np.int64).reshape(-1, 3)


def match_array(matches, nodes, gt, spacing):
    pn = {int(n[0]): n[2:] for n in nodes}
    gn = {int(n[0]): n[2:] for n in gt}
    return np.asarray([[p, g, np.linalg.norm((pn[p] - gn[g]) * spacing)] for p, g in matches.items()], float).reshape(-1, 3)


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE clips (dataset TEXT PRIMARY KEY, info TEXT NOT NULL);
CREATE TABLE models (dataset TEXT, model TEXT, info TEXT NOT NULL, PRIMARY KEY(dataset,model));
CREATE TABLE cases (
 key TEXT PRIMARY KEY, dataset TEXT NOT NULL, model TEXT NOT NULL,
 entity TEXT NOT NULL, kind TEXT NOT NULL, ambiguous INTEGER NOT NULL,
 t INTEGER NOT NULL, node INTEGER NOT NULL, other INTEGER,
 distance REAL, raw_distance REAL, final_nearest REAL,
 raw_count INTEGER, final_count INTEGER, competitors INTEGER,
 raw_matched INTEGER, matched INTEGER, priority INTEGER NOT NULL
);
"""
INSERT = 'INSERT INTO cases VALUES (' + ','.join('?' for _ in range(18)) + ')'


def build(args):
    from annotation_selection.metric_adapter import match_nodes
    from annotation_selection.common import METRIC_REV
    from strong_tracker_v3.common import graph_hash, validate
    from .best_predictions import DEFAULT_PREDICTIONS, DEFAULT_EVALUATION
    root = args.output / 'detection-review'
    snapshots = root / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    building = root / 'review.build.sqlite'
    if building.exists():
        building.unlink()
    db = sqlite3.connect(building)
    db.executescript(SCHEMA)
    definitions = {
        'pooled': ('C4_m6', WORK / 'multidata-training-v4/predictions/C4_m6', WORK / 'multidata-training-v4/evaluation/C4_m6'),
        'best': ('P0', getattr(args, 'best_predictions', DEFAULT_PREDICTIONS), getattr(args, 'best_evaluation', DEFAULT_EVALUATION)),
    }
    datasets = sorted(p.stem for p in (args.data_root / 'train').glob('*.geff'))
    images = {p.stem for p in (args.data_root / 'train').glob('*.zarr')}
    if not datasets or set(datasets) != images:
        raise ValueError('Every training image must have paired ground truth')
    if getattr(args, 'audit_limit', None):
        datasets = datasets[:args.audit_limit]
    total_gt = total_frames = 0
    start = time.monotonic()
    for dataset in datasets:
        folder, shape, spacing_list = read_geometry(args.data_root, dataset)
        spacing = np.array(spacing_list)
        for t in range(shape[0]):
            chunk = folder / f'0/c/{t}/0/0/0'
            if not chunk.is_file() or not chunk.stat().st_size:
                raise FileNotFoundError(chunk)
        gt, gt_edges = read_gt(args.data_root, dataset, spacing)
        raw_path = WORK / f'annotation-selection-v1/public_harmonic_full/inputs/pre_ilp_{dataset}.npz'
        with np.load(raw_path, allow_pickle=False) as src:
            coords = src['coords']
            raw = np.column_stack([np.arange(len(coords)), coords]).astype(np.int64)
            confidence = src['node_probabilities']
        validate(raw, np.empty((0, 2), np.int64), shape)
        raw_matches = match_nodes(raw, np.empty((0, 2), np.int64), gt, gt_edges, spacing)
        raw_evidence = spatial_evidence(gt, raw, raw_matches, spacing)
        snapshot_base = f'{dataset}-{sha256(raw_path)[:12]}-{graph_hash(gt, gt_edges)[:12]}'
        common = snapshots / f'{snapshot_base}-common.npz'
        np.savez_compressed(common, gt=gt, gt_edges=gt_edges, raw=raw, confidence=confidence,
                            raw_matches=match_array(raw_matches, raw, gt, spacing))
        meta = json.loads((folder / 'zarr.json').read_text())['attributes']
        quantiles = meta.get('image_statistics', {}).get('quantiles', {})
        display_limits = [float(quantiles.get('0.01', 0)), float(quantiles.get('0.999', 65535))]
        clip = dict(dataset=dataset, embryo=dataset.split('_')[0], shape=shape, spacing=spacing_list,
                    annotations=len(gt), gt_edges=len(gt_edges), raw_predictions=len(raw),
                    raw_unmatched=len(raw) - len(raw_matches), raw_matched=len(raw_matches),
                    snapshot=common.name, image_root=str(folder), raw_source=str(raw_path),
                    raw_sha256=sha256(raw_path), gt_graph_hash=graph_hash(gt, gt_edges),
                    display_limits=display_limits, image_metadata_sha256=sha256(folder / 'zarr.json'))
        db.execute('INSERT INTO clips VALUES (?,?)', (dataset, json.dumps(clip)))
        total_gt += len(gt); total_frames += shape[0]
        for model, (name, pred_root, eval_root) in definitions.items():
            source = pred_root / f'{dataset}.npz'
            receipt_path = eval_root / f'{dataset}.json'
            receipt = json.loads(receipt_path.read_text())
            digest = sha256(source)
            if receipt['dataset'] != dataset or receipt['metric_revision'] != METRIC_REV or receipt['prediction_sha256'] != digest:
                raise ValueError(f'Prediction/evaluation identity mismatch: {model} {dataset}')
            with np.load(source, allow_pickle=False) as pred:
                nodes, edges = pred['nodes'], pred['edges']
            validate(nodes, edges, shape)
            if graph_hash(nodes, edges) != receipt['graph_hash']:
                raise ValueError('Graph hash mismatch')
            matches, edge_status = audit_graph(nodes, edges, gt, gt_edges, spacing, receipt)
            evidence = spatial_evidence(gt, nodes, matches, spacing)
            frozen = snapshots / f'{snapshot_base}-{model}-{digest[:12]}.npz'
            np.savez_compressed(frozen, nodes=nodes, edges=edges, matches=match_array(matches, nodes, gt, spacing), edge_status=edge_status)
            records = []
            ambiguous = {}
            gt_lookup = {int(n[0]): n for n in gt}
            for g in gt:
                gid = int(g[0]); r, p = raw_evidence[gid], evidence[gid]
                crowd = p['count'] > 1 or p['competitors'] > 0 or p['assigned_not_nearest'] or r['count'] > 1 or r['competitors'] > 0
                kind = classify_center(p['assigned'] is not None, r['assigned'] is not None, r['nearest'], p['nearest'], p['distance'], crowd)
                ambiguous[gid] = crowd or kind in ['conflict', 'proposal_only', 'crowded']
                priority = {'missing': 0, 'proposal_only': 1, 'conflict': 2, 'recovered': 3, 'offset': 4, 'crowded': 5, 'matched': 9}[kind]
                records.append((f'{model}:{dataset}:g:{gid}', dataset, model, 'gt', kind, int(ambiguous[gid]),
                    int(g[1]), gid, None, p['distance'], r['nearest'], p['nearest'], r['count'], p['count'],
                    p['competitors'], int(r['assigned'] is not None), int(p['assigned'] is not None), priority))
            reverse = {g: p for p, g in matches.items()}
            recovered = {(matches[int(a)], matches[int(b)]) for a, b, status in edge_status if status == 1}
            missed = {tuple(map(int, e)) for e in gt_edges} - recovered
            if len(missed) != receipt['edge_fn']:
                raise ValueError('Every official missed edge must be accounted for')
            represented_by_center = 0
            for a, b in sorted(missed):
                if a not in reverse or b not in reverse:
                    represented_by_center += 1
                    continue  # Both endpoint states remain in the annotated-node catalog.
                uncertain = ambiguous[a] or ambiguous[b] or evidence[a]['distance'] >= OFFSET or evidence[b]['distance'] >= OFFSET
                kind = 'association_ambiguous' if uncertain else 'association'
                records.append((f'{model}:{dataset}:e:{a}:{b}', dataset, model, 'edge', kind, int(uncertain),
                    int(gt_lookup[b][1]), a, b, None, None, None, None, None, None, 1, 1, 6))
            node_lookup = {int(n[0]): n for n in nodes}
            for a, b, status in edge_status:
                if status != -1:
                    continue
                a, b = int(a), int(b)
                gids = [matches.get(a), matches.get(b)]
                uncertain = any(g is None or ambiguous[g] or evidence[g]['distance'] >= OFFSET for g in gids)
                kind = 'wrong_link_ambiguous' if uncertain else 'wrong_link'
                records.append((f'{model}:{dataset}:f:{a}:{b}', dataset, model, 'pred_edge', kind, int(uncertain),
                    int(node_lookup[b][1]), a, b, None, None, None, None, None, None, 1, int(all(g is not None for g in gids)), 7))
            db.executemany(INSERT, records)
            counts = Counter(row[4] for row in records)
            info = dict(model=model, name=name, source=str(source), source_sha256=digest, snapshot=frozen.name,
                evaluation_source=str(receipt_path), evaluation_sha256=sha256(receipt_path),
                predictions=len(nodes), matched=len(matches), unmatched=len(nodes) - len(matches),
                kind_counts=dict(counts), ambiguous_labeled=sum(row[5] for row in records),
                edge_fn=receipt['edge_fn'], edge_fp=receipt['edge_fp'],
                missed_edges_represented_by_center_cases=represented_by_center,
                association_fn_cases=counts['association'] + counts['association_ambiguous'])
            db.execute('INSERT INTO models VALUES (?,?,?)', (dataset, model, json.dumps(info)))
        db.commit()
        print(f'DETECTION_INDEX {dataset} {len(gt)} annotations · {time.monotonic()-start:.1f}s', flush=True)
    db.executescript('''
      CREATE INDEX case_list ON cases(model,priority,dataset,t,key);
      CREATE INDEX case_clip ON cases(model,dataset,priority,t,key);
      CREATE INDEX case_filter ON cases(model,entity,kind);
      CREATE INDEX case_ambiguity ON cases(model,ambiguous);
    ''')
    info = dict(version=1, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), datasets=len(datasets),
                frames=total_frames, annotations=total_gt, radius_um=RADIUS, offset_flag_um=OFFSET,
                models={k: v[0] for k, v in definitions.items()}, metric_revision=METRIC_REV,
                kinds=KINDS, complete_training_population=set(datasets) == images,
                scope='All annotated nodes, all scored incorrect edges, and all missed edges via center or association cases. Every unmatched raw/final prediction remains pageable as unknown.',
                interpretation='Evidence groups are review candidates, not automatic biological diagnoses. Spatial assignment and temporal association are different decisions. Sparse annotations cannot establish whole-field false-positive detection counts.')
    db.execute('INSERT INTO metadata VALUES (?,?)', ('catalog', json.dumps(info)))
    db.commit(); db.close()
    building.replace(root / 'review.sqlite')
    info['database_sha256'] = sha256(root / 'review.sqlite')
    write_json(root / 'index_receipt.json', info)
    print(f'COMPLETE: {len(datasets)} clips, {total_frames} frames, {total_gt} annotations', flush=True)
