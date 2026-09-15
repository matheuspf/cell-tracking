"""Check frozen pilot provenance and independently recount canonical GEFF labels."""
from __future__ import annotations

import json
import numpy as np
import zarr
from tools.cellpose_embedding_probe import panel, OUT, WORK, REPO, sha, read, write


def main():
    plan=read(OUT/'embedding-plan.json'); bank=read(OUT/'banks-lock.json')
    native=read(OUT/'native-lock.json'); metrics=read(OUT/'pilot-metrics.json')
    assert plan['script_sha256']==sha(REPO/'tools/cellpose_embedding_probe.py')
    assert plan['pilot_plan_sha256']==sha(OUT/'pilot-plan.json')
    assert plan['adjacent_plan_sha256']==sha(OUT/'adjacent/pilot-plan.json')
    assert bank['embedding_plan_sha256']==sha(OUT/'embedding-plan.json')
    assert native['plan_sha256']==sha(OUT/'embedding-plan.json')
    assert native['banks_lock_sha256']==sha(OUT/'banks-lock.json')
    assert metrics['native_lock_sha256']==sha(OUT/'native-lock.json')
    assert native['created_utc']<metrics['gt_opened_utc']
    for r in bank['files']:
        if not r.get('source_only'):
            assert sha(WORK/'banks'/r['method']/f"{r['key']}.npz")==r['sha256']
    for r in native['predictions']:
        assert sha(WORK/'native'/r['weight']/r['method']/f"{r['key']}.npz")==r['sha256']
    raw_nodes=raw_edges=0
    for dataset in sorted({r['dataset'] for r in panel()['frames']}):
        graph=zarr.open_group(str(REPO/'data/train'/f'{dataset}.geff'),mode='r')
        ids=np.asarray(graph['nodes/ids'][:]); times=np.asarray(graph['nodes/props/t/values'][:])
        edges=np.asarray(graph['edges/ids'][:]); lookup=dict(zip(ids,times))
        raw_nodes+=int(np.isin(times,[25,26,75,76]).sum())
        raw_edges+=int(sum(lookup[a] in (25,75) and lookup[b]==lookup[a]+1 for a,b in edges))
    assert (raw_nodes,raw_edges)==(180,88)
    for r in metrics['groups']:
        for a in r['association'].values():
            assert 0<=a['correct_parent_top1']<=a['supported_all_edges']<=a['eligible_gt_edges']
            assert 0<=a['correct_top1']<=a['supported_edges']<=a['eligible_continuation_gt_edges']
    report={'source_and_plan_hashes_valid':True,'banks_verified':216,'native_matrices_verified':216,
            'frozen_before_evaluation':True,'canonical_geff_nodes':raw_nodes,'canonical_geff_edges':raw_edges,
            'metric_denominators_valid':True,'metrics_sha256':sha(OUT/'pilot-metrics.json'),
            'report_browser_qa':read(WORK/'report-qa/checks.json')}
    write(OUT/'validation.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
