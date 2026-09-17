"""Hash all baseline graphs; independently aggregate and freshly replay four clips."""
import time
import numpy as np

from .common import (RESULTS, PRIOR_RESULTS, PRIOR_WORK, OFFICIAL, DATA, STUDY,
                     read_json, write_json, write_csv, sha, verified_graph, inputs)


def run():
    from annotation_selection.metric_adapter import evaluate_graph, aggregate
    from center_comparison.pipeline import read_gt
    start = time.monotonic()
    manifest = read_json(PRIOR_RESULTS / 'input_manifest.json')
    for relative, expected in manifest['metric_sources'].items():
        if sha(OFFICIAL / relative) != expected:
            raise ValueError('Pinned official metric source changed')
    write_json(RESULTS / 'input_manifest.json', manifest, immutable=True)
    split = read_json(PRIOR_RESULTS / 'split_manifest.json')
    split.update(reused_from=str(PRIOR_RESULTS / 'split_manifest.json'),
                 temporal_operator='two ordered attention layers', minimum_purge_frames=9)
    write_json(RESULTS / 'split_manifest.json', split, immutable=True)
    write_json(RESULTS / 'exposure_manifest.json', dict(
        scope='Source-only direct fitting on exposed P0; exploratory inner split; not clean OOF',
        prior_model_manifest_sha256=sha(PRIOR_RESULTS / 'model_input_manifest.json'),
        inherited=read_json(PRIOR_RESULTS / 'model_input_manifest.json'),
        image_encoder='Random initialization; upstream P0 observations remain exposed',
        source_groups_independent=False, global_acquisition_offsets_available=False,
        no_new_external_data=True), immutable=True)
    receipts = []
    graph_receipts = []
    for row in inputs():
        for arm in ('P0', 'C4_m6'):
            graph = verified_graph(row, arm)
            receipt_path = PRIOR_WORK / 'evaluation' / arm / (row['dataset']+'.json')
            receipt = read_json(receipt_path)
            if receipt['graph_hash'] != row['baselines'][arm]['graph_hash']:
                raise ValueError('Baseline metric receipt graph mismatch')
            receipts.append(dict(receipt, arm=arm, embryo=row['embryo']))
            graph_receipts.append(dict(dataset=row['dataset'], arm=arm,
                nodes=len(graph['nodes']), graph_sha256=sha(row['baselines'][arm]['path']),
                receipt_sha256=sha(receipt_path)))
        if len(graph_receipts) % 40 == 0:
            print(f'Baseline hashes: {len(graph_receipts)//2}/199', flush=True)
    pooled, embryos = [], []
    for arm in ('P0', 'C4_m6'):
        for embryo in ('pooled', '44b6', '6bba'):
            rs = [r for r in receipts if r['arm'] == arm and
                  (embryo == 'pooled' or r['embryo'] == embryo)]
            a = aggregate(rs, sorted(r['dataset'] for r in rs))
            a.update(a.pop('counts'))
            a.update(arm=arm, embryo=embryo, clips=len(rs), status='verified',
                     matched_nodes=sum(r['matched_nodes'] for r in rs))
            (pooled if embryo == 'pooled' else embryos).append(a)
    expected = read_json(STUDY)['baselines']
    for r in pooled:
        np.testing.assert_allclose(r['score'], expected[r['arm']], rtol=0, atol=1e-12)
    replay = []
    for embryo in ('44b6', '6bba'):
        ordered = sorted(inputs(embryo), key=lambda r: (r['baselines']['P0']['nodes'], r['dataset']))
        for row in (ordered[len(ordered)//2], ordered[-1]):
            gn, ge = read_gt(DATA, row['dataset'], row['physical_scale'])
            for arm in ('P0', 'C4_m6'):
                graph = verified_graph(row, arm)
                score, _, _ = evaluate_graph(row['dataset'], graph['nodes'], graph['edges'],
                                            gn, ge, row['physical_scale'], row['estimated_total'])
                old = next(r for r in receipts if r['arm'] == arm and r['dataset'] == row['dataset'])
                for key in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn',
                            'num_pred_nodes','matched_nodes','adj_edge_jaccard'):
                    np.testing.assert_allclose(score[key], old[key], rtol=0, atol=1e-12)
                replay.append(dict(dataset=row['dataset'], arm=arm, exact=True))
            print(f'Fresh baseline replay: {row["dataset"]}', flush=True)
    write_csv(RESULTS / 'scores.csv', pooled)
    write_csv(RESULTS / 'per_embryo_scores.csv', embryos)
    write_csv(RESULTS / 'baseline_per_clip_scores.csv', receipts)
    result = dict(status='measured', graphs=graph_receipts, pooled=pooled, embryos=embryos,
                  fresh_replays=replay, seconds=time.monotonic()-start,
                  independent_aggregation_parity=True, unchanged_nodes=4108943)
    write_json(RESULTS / 'baseline_validation.json', result)
    return result
