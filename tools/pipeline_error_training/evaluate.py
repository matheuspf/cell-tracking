"""Separate annotation-owning evaluator. Training/inference never import this."""
import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .common import (
    DATA, RECEIPTS, RESULTS, STUDY, WORK, inputs, read_json, save_arrays,
    verified_graph, write_json,
)
from .resources import Monitor, cpu_budget


def baseline_one(row):
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    cpu_budget(4)
    gt_nodes, gt_edges = read_gt(DATA, row['dataset'], row['physical_scale'])
    output = []
    for arm in ['P0', 'C4_m6', 'C0']:
        start = time.monotonic()
        graph = verified_graph(row, arm)
        score, matches, tp = evaluate_graph(row['dataset'], graph['nodes'], graph['edges'],
            gt_nodes, gt_edges, row['physical_scale'], row['estimated_total'])
        score.update(embryo=row['embryo'], arm=arm, seconds=time.monotonic()-start,
                     status='measured', prediction_sha256=row['baselines'][arm]['sha256'])
        if arm in RECEIPTS:
            expected = read_json(RECEIPTS[arm] / f'{row["dataset"]}.json')
            for key in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn',
                        'matched_nodes', 'num_pred_nodes', 'adj_edge_jaccard']:
                if not np.isclose(score[key], expected[key], rtol=0, atol=1e-12):
                    raise ValueError(f'Fresh control score mismatch {arm}/{row["dataset"]}/{key}')
        write_json(WORK / 'evaluation' / arm / f'{row["dataset"]}.json', score)
        save_arrays(WORK / 'evaluation' / arm / f'{row["dataset"]}.npz',
                    matches=np.asarray(sorted(matches.items()), np.int64).reshape(-1, 2),
                    tp_edges=np.asarray(sorted(tp), np.int64).reshape(-1, 2))
        output.append(score)
    return output


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, arm, embryo='pooled'):
    from annotation_selection.metric_adapter import aggregate
    chosen = [r for r in rows if r['arm'] == arm and (embryo == 'pooled' or r['embryo'] == embryo)]
    result = aggregate(chosen, sorted(r['dataset'] for r in chosen))
    result.update(result.pop('counts'))
    result.update(arm=arm, embryo=embryo, status='measured', clips=len(chosen),
                  matched_nodes=sum(r['matched_nodes'] for r in chosen))
    return result


def baselines():
    cpu_budget()
    started = time.monotonic()
    rows = []
    with Monitor(WORK / 'resources/baselines.json') as monitor:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
            for i, scores in enumerate(pool.map(baseline_one, inputs()), 1):
                rows.extend(scores)
                monitor.check()
                if i % 10 == 0 or i == 199:
                    print(f'Fresh control scores: {i}/199 clips, {i*3}/597 graphs', flush=True)
    pooled = [summary(rows, arm) for arm in ['P0', 'C4_m6', 'C0']]
    embryos = [summary(rows, arm, e) for arm in ['P0', 'C4_m6', 'C0'] for e in ['44b6', '6bba']]
    expected = read_json(STUDY)['baselines']
    for row in pooled:
        ref = expected[row['arm']]
        if not np.isclose(row['score'], ref['score'], rtol=0, atol=1e-12):
            raise ValueError('Pooled baseline does not reproduce the registered reference')
        for kind in ['edge', 'division']:
            if [row[f'{kind}_{s}'] for s in ['tp', 'fp', 'fn']] != ref[f'{kind}_counts']:
                raise ValueError('Baseline pooled count mismatch')
    write_csv(RESULTS / 'scores.csv', pooled)
    write_csv(RESULTS / 'per_embryo_scores.csv', embryos)
    write_csv(RESULTS / 'per_clip_scores.csv', rows)
    write_json(RESULTS / 'baseline_validation.json', dict(status='measured', clips=199, graphs=597,
        exact_nodes=True, hashes_verified=True, fresh_metric_counts_match=True,
        independent_aggregation_parity=True, seconds=time.monotonic()-started, pooled=pooled, embryos=embryos))
    print('All baseline pooled counts and scores reproduced', flush=True)
