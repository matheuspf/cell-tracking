from pipeline_error_training.accounting import transitions


def test_changed_errors_record_simultaneous_recoveries_and_regressions():
    old = dict(tp_edges=[[1, 2], [3, 4]], fp_edges=[[[7, 0, 1, 2, 3], [8, 1, 1, 2, 3]]],
               tp_divisions=[1], fp_divisions=[[9, 2, 1, 2, 3]], matched_gt=[1, 2, 3],
               stages={'edge_endpoint:selection': 2, 'edge_fp:competing': 1})
    new = dict(tp_edges=[[1, 2], [4, 5]], fp_edges=[[[7, 0, 1, 2, 3], [18, 1, 1, 2, 4]]],
               tp_divisions=[11], fp_divisions=[], matched_gt=[1, 2, 5],
               stages={'edge_endpoint:selection': 1, 'edge_link:association': 1})
    out = transitions(old, new)
    for key in ['tp_edges_recovered', 'tp_edges_lost', 'tp_divisions_recovered', 'tp_divisions_lost',
                'fp_edges_removed', 'fp_edges_introduced', 'fp_divisions_removed', 'matched_gt_recovered', 'matched_gt_lost']:
        assert out[key] == 1
    assert out['fp_divisions_introduced'] == 0
    assert out['edge_fp:competing_after'] == 0
    assert out['nonadditive_error_families']
