"""Contracts that can silently invalidate an apparently plausible microscopy view."""
import numpy as np
import pytest

from center_comparison.export import extract_centers, lineage_roots, point_rows, preview_links


def test_native_mask_ids_and_empty_masks():
    labels = np.zeros((5, 8, 9), np.uint32)
    labels[1:3, 2:4, 3:6] = 17
    labels[4, 7, 8] = 902
    raw = np.ones(labels.shape, np.uint16)
    modes, sizes = extract_centers(labels, raw)
    assert sizes == {17: 12, 902: 1}
    assert modes['focus_centroid'].tolist() == [[17, 1.5, 2.5, 4], [902, 4, 7, 8]]
    # A uniform object must fall back to its geometric center when weights vanish.
    assert np.allclose(modes['focus_weighted'], modes['focus_centroid'])
    empty, sizes = extract_centers(np.zeros_like(labels), raw)
    assert not sizes and all(a.shape == (0, 4) for a in empty.values())


def test_exact_centers_and_submission_rounding_remain_distinct():
    rows = point_rows(np.array([[91, 2.5, 3.5, 8.9]]), [5, 8, 9])
    assert rows == [[91, 2, 4, 8, 2.5, 3.5, 8.9]]
    with pytest.raises(ValueError, match='Nonfinite'):
        point_rows([[0, np.nan, 0, 0]], [5, 8, 9])


def test_preview_assignment_preserves_cardinality_and_respects_physical_depth():
    # Greedy nearest-neighbor association can consume the only feasible partner
    # of the second object. Optimal association must retain both links.
    previous = point_rows([[10, 0, 0, 4], [20, 0, 0, 0]], [8, 8, 20])
    current = point_rows([[30, 0, 0, 0], [40, 0, 0, 10]], [8, 8, 20])
    links = preview_links(previous, current, np.ones(3), radius=7)
    assert {(a, b) for a, b, _ in links} == {(10, 40), (20, 30)}
    previous = point_rows([[10, 0, 0, 0]], [8, 8, 20])
    current = point_rows([[30, 1, 0, 0], [40, 0, 0, 2]], [8, 8, 20])
    assert preview_links(previous, current, np.array([4, 1, 1]), radius=3) == [[10, 40, 2.]]


def test_lineage_identity_survives_a_division_without_inventing_edges():
    nodes = np.array([[80, 0], [4, 1], [12, 1], [70, 2], [99, 0]])
    edges = np.array([[80, 4], [80, 12], [12, 70]])
    assert lineage_roots(nodes, edges) == {80: 4, 4: 4, 12: 4, 70: 4, 99: 99}


def test_official_errors_distinguish_unknown_cells_from_evaluable_wrong_links():
    pytest.importorskip('tracksdata')
    from center_comparison.best_predictions import diagnose
    # IDs are deliberately unrelated to row indices. The unlabelled 50→51
    # track is unknown, while 20→31 contradicts an annotated continuation.
    gt = np.array([[101, 0, 0, 0, 0], [102, 1, 0, 0, 0],
                   [201, 0, 0, 0, 20], [202, 1, 0, 0, 20],
                   [301, 1, 0, 0, 50], [401, 2, 0, 0, 90]])
    ge = np.array([[101, 102], [201, 202], [202, 401]])
    pred = np.array([[10, 0, 0, 0, 0], [11, 1, 0, 0, 0],
                     [20, 0, 0, 0, 20], [21, 1, 0, 0, 20],
                     [31, 1, 0, 0, 50], [41, 2, 0, 0, 110],
                     [50, 0, 0, 0, 70], [51, 1, 0, 0, 80]])
    pe = np.array([[10, 11], [20, 31], [21, 41], [50, 51]])
    result = diagnose(pred, pe, gt, ge, np.ones(3), len(pred))
    assert result['edge_status'] == {'10:11': 'tp', '20:31': 'fp', '21:41': 'fp', '50:51': 'unknown'}
    missed = {e['node']: e for e in result['events'] if e['kind'] == 'edge_fn'}
    assert set(missed) == {202, 401}
    assert 'Both annotated endpoints' in missed[202]['reason']
    assert 'no P0 assignment' in missed[401]['reason']
    assert missed[202]['segments'][0]['source'] == gt[2].tolist()
    assert result['metrics']['edge_fn'] == 2
    from center_comparison.error_cases import tracking_cases
    cases = tracking_cases(result, pred, pe, gt, ge, np.ones(3))
    assert {e['category'] for e in cases} == {'link_wrong', 'link_detection'}
    wrong = next(e for e in cases if e['category'] == 'link_wrong')
    assert set(wrong['related_flags']) == {'edge_fn:201:202', 'edge_fp:20:31'}
    assert wrong['scene']['times'] == [0, 1]
    assert len(wrong['scene']['expected']) == len(wrong['scene']['actual']) == 1
    assert wrong['scene']['actual'][0]['target'] == 'p:31'
    assert all(c['prediction'] is not None for c in wrong['scene']['cells'])
    # Removing the wrong outgoing edge changes the diagnosis to a pure link
    # omission; the two correctly detected endpoints are still present.
    disconnected = pe[[0, 2, 3]]
    result = diagnose(pred, disconnected, gt, ge, np.ones(3), len(pred))
    cases = tracking_cases(result, pred, disconnected, gt, ge, np.ones(3))
    missing = next(e for e in cases if e['category'] == 'link_missing')
    assert missing['scene']['actual'] == []
    assert all(c['prediction'] is not None for c in missing['scene']['cells'])


def test_official_missed_division_survives_display_window_cropping():
    pytest.importorskip('tracksdata')
    from center_comparison.best_predictions import diagnose
    gt = np.array([[901, 0, 0, 0, 10], [19, 1, 0, 0, 9], [24, 1, 0, 0, 12],
                   [87, 2, 0, 0, 8], [93, 2, 0, 0, 13]])
    ge = np.array([[901, 19], [901, 24], [19, 87], [24, 93]])
    pred = gt.copy()
    pred[:, 0] += 1000
    pe = np.array([[1901, 1019], [1019, 1087], [1024, 1093]])
    result = diagnose(pred, pe, gt, ge, np.ones(3), len(pred))
    errors = [e for e in result['events'] if e['kind'] == 'division_fn']
    assert result['metrics']['division_fn'] == 1
    assert len(errors) == 1 and errors[0]['node'] == 901 and errors[0]['t'] == 0
    assert {e['target'][0] for e in errors[0]['segments'] if e['method'] == 'gt'} == {19, 24}
    from center_comparison.error_cases import tracking_cases
    cases = tracking_cases(result, pred, pe, gt, ge, np.ones(3))
    daughter = next(e for e in cases if e['category'] == 'link_missing')
    assert daughter['title'] == 'One daughter connection is missing'
    assert [e['recovered'] for e in daughter['scene']['expected']] == [True, False]
    assert daughter['scene']['actual'][0]['status'] == 'tp'


def test_center_miss_is_an_assignment_failure_even_when_nearest_is_close():
    from center_comparison.best_predictions import center_events
    frame = dict(t=20, points={'gt': [[9], [10], [11]]},
                 matches={'pooled': {'7': [[55, 10, 2.], [56, 11, 3.5]]}},
                 model_nearest={'pooled': [[9, 55, 1.], [10, 55, 2.], [11, 56, 3.5]]})
    events = center_events(frame, '7', 'pooled', 'C4_m6')
    assert [e['kind'] for e in events] == ['center_miss', 'offset']
    assert 'Nearest prediction 55 is 1.00' in events[0]['reason']
    assert 'C4_m6 assignment' in events[0]['reason']


def test_center_cases_keep_competition_for_assignment_separate_from_absence():
    from center_comparison.best_predictions import center_events
    from center_comparison.error_cases import center_case
    frame = dict(t=7, spacing=[1, 1, 1], points={'gt': [[10, 0, 0, 1], [11, 0, 0, 2]],
                 'best': [[91, 0, 0, 1]]}, matches={'best': {'1': [[91, 11, 1.]]}},
                 model_nearest={'best': [[10, 91, 0.], [11, 91, 1.]]})
    event = center_events(frame, '1')[0]
    conflict = center_case(event, frame, 'best', '1')
    assert conflict['category'] == 'center_conflict'
    assert conflict['scene']['times'] == [7]
    assert conflict['scene']['cells'][0]['prediction'] is None
    assert conflict['scene']['cells'][1]['prediction'] == 'p:91'
    assert conflict['scene']['actual'] == []  # Same-frame assignment is not a tracking edge.
    assert conflict['scene']['center_probe']['assigned'] is False
    assert conflict['scene']['center_probe']['delta_um'] == [0., 0., 0.]
    frame['model_nearest']['best'][0][2] = 5.
    frame['points']['gt'][0][3] = 6
    missing = center_case(event, frame, 'best', '1')
    assert missing['category'] == 'center_missing'
    assert missing['scene']['center_probe']['delta_um'] == [0., 0., -5.]
    assert missing['scene']['center_probe']['distance_um'] == 5.
