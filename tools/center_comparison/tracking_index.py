"""Reproducible full-population tracking audit over immutable prediction graphs.

Every catalog row is one official FN/FP flag, not an independent biological
event. Division decisions use the official local-window matching; endpoint
attribution for temporal edges uses the official whole-clip matching.
"""
from __future__ import annotations

import json
import sqlite3
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .detection_server import connect, snapshot
from .pipeline import REPO, read_geometry, read_gt, sha256, write_json

CATEGORIES = {
    'edge_endpoint': ['Link blocked by an unmatched center', 'Detection / observation selection'],
    'edge_link': ['Matched cells are not linked', 'Temporal association / decoding'],
    'edge_fp': ['Incorrect predicted connection', 'Observation identity / association'],
    'division_fn': ['Missed division', 'Division evidence / event decoding'],
    'division_fp': ['Incorrect predicted division', 'Division identity / event decoding'],
}
STAGES = {
    'proposal_gap': 'Missing endpoint has no detector candidate within 7 µm',
    'selection': 'Detector candidates nearby; final endpoints unavailable',
    'assignment': 'Nearby final centers lose the one-to-one assignment',
    'association': 'Both endpoints matched; their connection is absent',
    'competing': 'Wrong link uses an unmatched center near the expected cell',
    'unmatched': 'Wrong link touches an unmatched prediction',
    'wrong_identity': 'Wrong link joins two matched cells',
    'division_observation': 'Local division window lacks parent or daughter matches',
    'division_no_fork': 'Local matches available; no parent-side predicted fork',
    'division_topology': 'Parent-side fork exists; branches fail the division check',
    'division_pairing': 'Locally valid fork lost one-to-one division pairing',
    'division_cross': 'Daughter branches contradict distinct annotated lineages',
    'division_merge': 'Predicted daughter branches locally merge',
    'division_extra': 'Evaluable fork does not recover an annotated division',
}
COUNT_FIELDS = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
SCHEMA = '''
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE cases (
 key TEXT PRIMARY KEY, model TEXT NOT NULL, dataset TEXT NOT NULL, embryo TEXT NOT NULL,
 category TEXT NOT NULL, stage TEXT NOT NULL, t INTEGER NOT NULL, node INTEGER NOT NULL,
 other INTEGER, evidence TEXT NOT NULL
);
CREATE INDEX case_filter ON cases(model,category,embryo,dataset,t,key);
CREATE INDEX case_stage ON cases(model,stage,dataset,t,key);
'''


def adjacency(edges):
    before, after = defaultdict(list), defaultdict(list)
    for a, b in edges:
        before[int(b)].append(int(a))
        after[int(a)].append(int(b))
    return before, after


def edge_flags(common, final, centers, spacing):
    """Partition every official edge error exactly once, using matched endpoints."""
    gt = {int(n[0]): n for n in common['gt']}
    pred = {int(n[0]): n for n in final['nodes']}
    matches = {int(p): int(g) for p, g, _ in final['matches']}
    reverse = {g: p for p, g in matches.items()}
    before, after = adjacency(common['gt_edges'])
    recovered = {(matches[int(a)], matches[int(b)]) for a, b, s in final['edge_status'] if s == 1}
    flags = []
    for a, b in sorted(set(map(tuple, common['gt_edges'].astype(int))) - recovered):
        a, b = int(a), int(b)
        missing = [g for g in [a, b] if g not in reverse]
        if not missing:
            category, stage = 'edge_link', 'association'
        else:
            category = 'edge_endpoint'
            stage = ('proposal_gap' if any(centers[g]['raw_count'] == 0 for g in missing) else
                     'assignment' if any(centers[g]['final_count'] > 0 for g in missing) else 'selection')
        flags.append(dict(category=category, stage=stage, t=int(gt[b][1]), node=a, other=b,
            evidence=dict(missing_gt=missing, endpoint_kinds=[centers[g]['kind'] for g in [a, b]],
                          division_edge=len(after[a]) >= 2)))
    for a, b, s in final['edge_status']:
        if s != -1:
            continue
        a, b = int(a), int(b)
        unmatched = [p for p in [a, b] if p not in matches]
        near = []
        for p, expected in [(a, before.get(matches.get(b), [])), (b, after.get(matches.get(a), []))]:
            if p in matches:
                continue
            for g in expected:
                distance = float(np.linalg.norm((pred[p][2:] - gt[g][2:]) * spacing))
                if pred[p][1] == gt[g][1] and g in reverse and distance <= 7.:
                    near.append(dict(prediction=p, expected_gt=g, assigned_prediction=reverse[g], distance_um=distance))
        flags.append(dict(category='edge_fp', stage='competing' if near else 'unmatched' if unmatched else 'wrong_identity',
            t=int(pred[b][1]), node=a, other=b, evidence=dict(unmatched_predictions=unmatched, competing=near)))
    return flags


def division_flags(common, final, spacing, receipt):
    from annotation_selection.metric_adapter import make_graph
    from tracking_cellmot import division_metrics as dm

    pred, pmap = make_graph(final['nodes'], final['edges'])
    truth, gmap = make_graph(common['gt'], common['gt_edges'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = dm.score_divisions(pred, truth, scale=tuple(spacing), max_distance=7.)
        windows = dm.match_divisions(pred, truth, scale=tuple(spacing), max_distance=7.)
        _, cross, merged = dm._pred_division_fork_sets(pred, truth, tuple(spacing), 7.)
    observed = (sum(result.scores.values()), len(result.fp_forks), len(result.scores)-sum(result.scores.values()))
    if observed != tuple(receipt[k] for k in ['division_tp', 'division_fp', 'division_fn']):
        raise ValueError('Fresh official division locations disagree with evaluation')
    gt_windows = dm.extract_divisions(truth)
    gt = {int(n[0]): n for n in common['gt']}
    pn = {int(n[0]): n for n in final['nodes']}
    flags = []
    for g, recovered in sorted(result.scores.items()):
        if recovered:
            continue
        window = windows[g]
        attrs = dm._matched_node_attrs(window)
        local_matches = {pmap[int(p)]: gmap[int(q)] for p, q in attrs.iter_rows()}
        grouped = dm._matched_division_nodes(attrs, gt_windows[g], g)
        forks, valid = set(), set()
        if grouped is None:
            stage = 'division_observation'
        else:
            parents, daughters = grouped
            forks = {p for p in parents | {c for p in parents for c in window.successors(p)} if window.out_degree(p) >= 2}
            valid = {p for p in forks - cross - merged if dm._is_strongly_connected_division(window, p, parents, daughters)}
            stage = 'division_pairing' if valid else 'division_topology' if forks else 'division_no_fork'
        gid = gmap[g]
        flags.append(dict(category='division_fn', stage=stage, t=int(gt[gid][1]), node=gid, other=None,
            evidence=dict(local_matches=list(local_matches.items()),
                local_gt=[gmap[int(q)] for q in gt_windows[g].node_ids()],
                candidate_forks=[pmap[p] for p in sorted(forks)], valid_forks=[pmap[p] for p in sorted(valid)])))
    for p in sorted(result.fp_forks):
        pid = pmap[p]
        flags.append(dict(category='division_fp', stage='division_merge' if p in merged else 'division_cross' if p in cross else 'division_extra',
            t=int(pn[pid][1]), node=pid, other=None, evidence=dict(cross_component=p in cross, merged=p in merged)))
    return flags


def aggregate_counts(rows):
    """Official weighting, including clip-specific node-count multipliers."""
    from annotation_selection.metric_adapter import aggregate
    return aggregate(rows, [r['dataset'] for r in rows])


def scenarios(rows, base):
    """Accounting scenarios, not repaired graphs or additive achievable gains."""
    output = []
    for key, title in [('division_fn', 'Recover all missed divisions'), ('division_fp', 'Remove all false divisions'),
                       ('division_all', 'Perfect division term'), ('edge_endpoint', 'Recover endpoint-blocked links'),
                       ('edge_link', 'Recover links with matched endpoints'), ('edge_fp', 'Remove all incorrect links'),
                       ('edge_all', 'Perfect edge matching at the same node counts')]:
        changed = []
        for row in rows:
            r = dict(row)
            if key.startswith('division'):
                if key in ['division_fn', 'division_all']:
                    r['division_tp'] += r['division_fn']; r['division_fn'] = 0
                if key in ['division_fp', 'division_all']:
                    r['division_fp'] = 0
            elif key == 'edge_fp':
                r['edge_fp'] = 0
            else:
                n = r['edge_fn'] if key == 'edge_all' else r['categories'].get(key, 0)
                r['edge_tp'] += n; r['edge_fn'] -= n
                if key == 'edge_all':
                    r['edge_fp'] = 0
            changed.append(r)
        value = aggregate_counts(changed)['score']
        output.append(dict(key=key, title=title, score=value, delta=value-base['score']))
    return output


def summarize(rows):
    summary = aggregate_counts(rows)
    categories, stages, centers = Counter(), Counter(), Counter()
    for r in rows:
        categories.update(r['categories']); stages.update(r['stages']); centers.update(r['centers'])
    summary.update(clips=len(rows), annotations=sum(r['annotations'] for r in rows),
        matched_nodes=sum(r['matched_nodes'] for r in rows), categories=dict(categories), stages=dict(stages),
        centers=dict(centers), center_recall_micro=sum(r['matched_nodes'] for r in rows)/sum(r['annotations'] for r in rows),
        node_adjustment=summary['adj_edge_jaccard']-summary['edge_jaccard'])
    summary['scenarios'] = scenarios(rows, summary)
    return summary


def build(args):
    from annotation_selection.common import METRIC_REV
    from strong_tracker_v3.common import graph_hash
    from .detection_index import audit_graph, match_array

    root = args.output / 'tracking-review'
    root.mkdir(parents=True, exist_ok=True)
    source_db = connect(args.output)
    detection_sha = sha256(args.output / 'detection-review/review.sqlite')
    catalog = json.loads(source_db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
    if not catalog['complete_training_population']:
        raise ValueError('Build the complete detection index first')
    building = root / 'review.build.sqlite'
    building.unlink(missing_ok=True)
    db = sqlite3.connect(building)
    db.executescript(SCHEMA)
    rows = defaultdict(list)
    provenance = []
    started = time.monotonic()
    clips = source_db.execute('SELECT info FROM clips ORDER BY dataset').fetchall()
    for index, (clip_json,) in enumerate(clips):
        clip = json.loads(clip_json); dataset = clip['dataset']
        common = snapshot(str(args.output / 'detection-review/snapshots' / clip['snapshot']))
        _, shape, spacing = read_geometry(args.data_root, dataset)
        gt, gt_edges = read_gt(args.data_root, dataset, spacing)
        if shape != clip['shape'] or spacing != clip['spacing'] or graph_hash(gt, gt_edges) != clip['gt_graph_hash']:
            raise ValueError(f'Canonical geometry/annotation drift: {dataset}')
        if not np.array_equal(gt, common['gt']) or not np.array_equal(gt_edges, common['gt_edges']):
            raise ValueError('Annotation snapshot drift')
        if sha256(Path(clip['raw_source'])) != clip['raw_sha256']:
            raise ValueError('Raw detection source drift')
        for model in catalog['models']:
            mi = json.loads(source_db.execute('SELECT info FROM models WHERE model=? AND dataset=?', (model, dataset)).fetchone()[0])
            receipt = json.loads(Path(mi['evaluation_source']).read_text())
            if sha256(Path(mi['source'])) != mi['source_sha256'] or mi['source_sha256'] != receipt['prediction_sha256'] or receipt['metric_revision'] != METRIC_REV:
                raise ValueError('Prediction/receipt identity drift')
            if sha256(Path(mi['evaluation_source'])) != mi['evaluation_sha256']:
                raise ValueError('Evaluation receipt drift')
            final = snapshot(str(args.output / 'detection-review/snapshots' / mi['snapshot']))
            if graph_hash(final['nodes'], final['edges']) != receipt['graph_hash']:
                raise ValueError('Frozen prediction graph drift')
            matches, statuses = audit_graph(final['nodes'], final['edges'], gt, gt_edges, spacing, receipt)
            fresh_matches = match_array(matches, final['nodes'], gt, np.array(spacing))
            # Graph attribute tables need not preserve iteration order across runtimes.
            if (len(fresh_matches) != len(final['matches']) or
                set(map(tuple, fresh_matches)) != set(map(tuple, final['matches'])) or
                len(statuses) != len(final['edge_status']) or
                set(map(tuple, statuses)) != set(map(tuple, final['edge_status']))):
                raise ValueError('Fresh official matching/edge status changed')
            centers = {r['node']: dict(r) for r in source_db.execute("SELECT * FROM cases WHERE model=? AND dataset=? AND entity='gt'", (model, dataset))}
            flags = edge_flags(common, final, centers, np.array(spacing)) + division_flags(common, final, spacing, receipt)
            categories = Counter(f['category'] for f in flags)
            if (categories['edge_endpoint']+categories['edge_link'], categories['edge_fp'], categories['division_fn'], categories['division_fp']) != tuple(receipt[k] for k in ['edge_fn','edge_fp','division_fn','division_fp']):
                raise ValueError('Error partition does not reproduce official counts')
            for f in flags:
                key = f"{model}:{dataset}:{f['category']}:{f['node']}" + (f":{f['other']}" if f['other'] is not None else '')
                db.execute('INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)', (key,model,dataset,clip['embryo'],f['category'],f['stage'],f['t'],f['node'],f['other'],json.dumps(f['evidence'])))
            row = dict(receipt, embryo=clip['embryo'], annotations=len(gt), categories=dict(categories),
                stages=dict(Counter(f['stage'] for f in flags)), centers=dict(Counter(c['kind'] for c in centers.values())))
            rows[model].append(row)
            provenance.append(dict(model=model,dataset=dataset,prediction_sha256=mi['source_sha256'],
                evaluation_sha256=mi['evaluation_sha256'],gt_graph_hash=clip['gt_graph_hash']))
        db.commit()
        print(f'TRACKING_AUDIT {index+1}/{len(clips)} {dataset} {time.monotonic()-started:.1f}s', flush=True)
    report = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(), metric_revision=METRIC_REV,
        detection_database_sha256=detection_sha, datasets=len(clips), frames=catalog['frames'],
        categories=CATEGORIES, stages=STAGES, models={}, provenance=provenance,
        source='Frozen graphs, canonical GEFF/Zarr, and fresh pinned official full-clip edge and local-window division scoring.',
        scope='All 199 public training clips, two reused embryos. The dominant incumbent detection checkpoint has documented training exposure to all 199 clips. These are local diagnostics, not held-out or leaderboard scores.',
        counting='One row per official error flag. A wrong connection can cause FP and FN; division flags can overlap edge errors. Category totals are not counts of independent biological mistakes.',
        scenario_note='Accounting scenarios change only the named TP/FP/FN counts and reaggregate with official weights and fixed node counts. They are not rescored graph repairs, achievable forecasts, or additive gains.')
    for model, model_rows in rows.items():
        summaries = {embryo:summarize([r for r in model_rows if r['embryo']==embryo]) for embryo in sorted({r['embryo'] for r in model_rows})}
        summaries['all'] = summarize(model_rows)
        report['models'][model] = dict(name=catalog['models'][model],
            status='Highest recorded complete pooled score; not adopted' if model=='pooled' else 'Retained pipeline after adoption checks',
            summaries=summaries, clips=[{k:r[k] for k in ['dataset','embryo',*COUNT_FIELDS,'adj_edge_jaccard','estimated_total','matched_nodes','annotations','categories','stages','centers']} for r in model_rows])
    db.execute('INSERT INTO metadata VALUES (?,?)', ('report',json.dumps(report,allow_nan=False)))
    db.commit(); db.close(); source_db.close()
    building.replace(root / 'review.sqlite')
    write_json(root / 'report.json', report)
    # A concise, portable result excludes raw coordinates and graphs.
    write_json(REPO / 'results/pipeline-errors-20260915/summary.json', report)
    print(f'COMPLETE {len(provenance)} fresh audits; report and tracking index ready', flush=True)
