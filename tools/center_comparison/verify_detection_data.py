"""Independently verify full-population coverage and physical detection evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .detection_server import catalog, connect, list_cases, snapshot
from .pipeline import DEFAULT_OUTPUT, sha256, write_json


def evidence(nodes, gt, matches, spacing):
    """Use spatial trees independently of the exporter's pairwise distance matrix."""
    lookup = {int(n[0]): n for n in nodes}
    truth = {int(n[0]): n for n in gt}
    assert len(lookup) == len(nodes) and len(truth) == len(gt)
    mapping = {int(p): int(g) for p, g, _ in matches}
    assert len(mapping) == len(matches) == len(set(mapping.values()))
    reverse = {g: p for p, g in mapping.items()}
    for p, g, d in matches:
        a, b = lookup[int(p)], truth[int(g)]
        assert a[1] == b[1]
        assert np.isclose(np.linalg.norm((a[2:] - b[2:])*spacing), d, atol=1e-10)
        assert d <= 7
    result = {}
    for t in np.unique(gt[:, 1]):
        p = nodes[nodes[:, 1] == t]
        g = gt[gt[:, 1] == t]
        tree = cKDTree(p[:, 2:]*spacing)
        distances, _ = tree.query(g[:, 2:]*spacing)
        counts = tree.query_ball_point(g[:, 2:]*spacing, 7, return_length=True)
        truth_tree = cKDTree(g[:, 2:]*spacing)
        for row, nearest, count in zip(g, distances, counts, strict=True):
            assigned = reverse.get(int(row[0]))
            distance = None if assigned is None else np.linalg.norm((lookup[assigned][2:]-row[2:])*spacing)
            competitors = 0 if assigned is None else len(truth_tree.query_ball_point(lookup[assigned][2:]*spacing, 7))-1
            result[int(row[0])] = (None if not len(p) else nearest, count, distance, assigned, competitors)
    return result, mapping


def verify(root):
    db = connect(root)
    receipt = json.loads((root / 'detection-review/index_receipt.json').read_text())
    assert receipt['database_sha256'] == sha256(root / 'detection-review/review.sqlite')
    totals = Counter()
    for dataset, encoded in db.execute('SELECT dataset,info FROM clips ORDER BY dataset'):
        clip = json.loads(encoded)
        common = snapshot(str(root / 'detection-review/snapshots' / clip['snapshot']))
        gt = common['gt']; spacing = np.array(clip['spacing'])
        raw_evidence, _ = evidence(common['raw'], gt, common['raw_matches'], spacing)
        assert clip['annotations'] == len(gt)
        assert clip['raw_unmatched'] == len(common['raw']) - len(common['raw_matches'])
        totals.update(clips=1, frames=clip['shape'][0], annotations=len(gt))
        for model, encoded in db.execute('SELECT model,info FROM models WHERE dataset=?', (dataset,)):
            info = json.loads(encoded)
            pred = snapshot(str(root / 'detection-review/snapshots' / info['snapshot']))
            final_evidence, mapping = evidence(pred['nodes'], gt, pred['matches'], spacing)
            rows = [dict(row) for row in db.execute('SELECT * FROM cases WHERE dataset=? AND model=?', (dataset, model))]
            centers = {r['node']: r for r in rows if r['entity'] == 'gt'}
            assert set(centers) == set(gt[:, 0]) and len(centers) == len(gt)
            for gid, row in centers.items():
                raw, final = raw_evidence[gid], final_evidence[gid]
                for field, value in [('raw_distance', raw[0]), ('final_nearest', final[0]), ('distance', final[2])]:
                    assert row[field] is None if value is None else np.isclose(row[field], value, atol=1e-10)
                assert (row['raw_count'], row['final_count'], row['competitors']) == (raw[1], final[1], final[4])
                assert bool(row['matched']) == (final[3] is not None)
                assert bool(row['raw_matched']) == (raw[3] is not None)
                if row['kind'] == 'matched':
                    assert final[2] < 3 and final[1] == 1 and final[4] == 0
                if not row['matched']:
                    assert row['kind'] in ['missing', 'proposal_only', 'conflict']
            matched_gt = set(mapping.values())
            correct = {(mapping[int(a)], mapping[int(b)]) for a, b, status in pred['edge_status'] if status == 1}
            gt_edges = {tuple(map(int, e)) for e in common['gt_edges']}
            missing = gt_edges - correct
            assert len(missing) == info['edge_fn'] and correct <= gt_edges
            association = {e for e in missing if set(e) <= matched_gt}
            assert association == {(r['node'], r['other']) for r in rows if r['entity'] == 'edge'}
            assert len(missing - association) == info['missed_edges_represented_by_center_cases']
            incorrect = {(int(a), int(b)) for a, b, status in pred['edge_status'] if status == -1}
            assert len(incorrect) == info['edge_fp']
            assert incorrect == {(r['node'], r['other']) for r in rows if r['entity'] == 'pred_edge'}
            assert info['unmatched'] == len(pred['nodes']) - len(mapping)
            totals.update(model_clips=1, gt_case_rows=len(centers), scored_incorrect_edges=len(incorrect),
                          missing_edges_accounted_for=len(missing), matches_checked=len(mapping))
    overview = catalog(db)
    assert totals['clips'] == overview['datasets'] and totals['annotations'] == overview['annotations']
    for model in overview['models']:
        raw_gaps = db.execute("SELECT count(*) FROM cases WHERE model=? AND entity='gt' AND raw_count=0", (model,)).fetchone()[0]
        assert overview['counts'][model]['missing'] == raw_gaps
        for pool, count in [('raw', overview['raw_unmatched']), ('final', overview['counts'][model]['unlabeled'])]:
            last = list_cases(root, db, dict(model=model, filter='unlabeled', pool=pool, offset=str(count-1)))
            assert last['total'] == count and len(last['rows']) == 1
            assert last['rows'][0]['ambiguous'] == 1 and last['rows'][0]['kind'] == 'unlabeled'
    db.close()
    result = dict(passed=True, **totals, all_annotated_nodes_present=True,
                  physical_distances_and_one_to_one_assignments=True, candidate_counts_independently_verified=True,
                  all_official_edge_errors_accounted_for=True, unmatched_pool_last_pages_verified=True,
                  database_sha256=receipt['database_sha256'], counts=overview['counts'],
                  raw_unmatched=overview['raw_unmatched'])
    write_json(root / 'detection-review/data-validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output)
