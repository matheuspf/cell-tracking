"""Evidence boundaries for detection review, rather than inferred biological truth."""
import gzip
import json
import sqlite3

import numpy as np
import pytest

from center_comparison.detection_index import INSERT, SCHEMA, classify_center, spatial_evidence
from center_comparison.detection_server import get_case, gray_frame, list_cases


@pytest.mark.parametrize('inputs,expected', [
    ((False, False, 9., 10., None, False), 'missing'),
    ((False, True, 1., 10., None, False), 'proposal_only'),
    ((False, False, 9., 1., None, False), 'conflict'),
    ((True, False, 9., 1., 1., False), 'recovered'),
    ((True, True, 1., 4., 4., False), 'offset'),
    ((True, True, 1., 1., 1., True), 'crowded'),
    ((True, True, 1., 1., 1., False), 'matched'),
    ((False, False, None, None, None, False), 'missing'),
])
def test_no_proposal_is_distinct_from_unassigned_and_downstream_gap(inputs, expected):
    assert classify_center(*inputs) == expected


def test_crowding_does_not_turn_a_nearest_point_into_an_assignment():
    gt = np.array([[100, 0, 0, 0, 0], [200, 0, 0, 0, 2], [300, 1, 0, 0, 0]])
    pred = np.array([[17, 0, 0, 0, 1], [91, 0, 2, 0, 0]])
    evidence = spatial_evidence(gt, pred, {17: 100}, np.array([4., 1., 1.]))
    assert evidence[100]['assigned'] == 17
    assert evidence[100]['competitors'] == 1
    assert evidence[200]['assigned'] is None
    assert evidence[200]['nearest_id'] == 17
    assert evidence[200]['count'] == 1  # Z offset 2 is 8 µm, outside the gate.
    assert evidence[300]['nearest'] is None  # A neighboring timepoint cannot match.


@pytest.fixture
def unknown_catalog(tmp_path):
    root = tmp_path / 'detection-review'
    snapshots = root / 'snapshots'; snapshots.mkdir(parents=True)
    db = sqlite3.connect(':memory:'); db.row_factory = sqlite3.Row; db.executescript(SCHEMA)
    for name, ids in [('one', [10, 20, 30]), ('two', [40, 50, 60])]:
        nodes = np.array([[ids[0], 0, 1, 2, 3], [ids[1], 0, 1, 2, 5], [ids[2], 1, 1, 2, 5]])
        gt = np.array([[100, 0, 1, 2, 3], [200, 1, 1, 2, 5]])
        matches = np.array([[ids[0], 100, 0.]])
        common = name + '-common.npz'; model = name + '-model.npz'
        np.savez_compressed(snapshots / common, raw=nodes, gt=gt, gt_edges=np.empty((0, 2), int),
                            raw_matches=matches, confidence=np.ones(3))
        np.savez_compressed(snapshots / model, nodes=nodes, matches=matches, edges=np.empty((0, 2), int), edge_status=np.empty((0, 3), int))
        clip = dict(dataset=name, shape=[2, 4, 8, 9], spacing=[4, 1, 1], display_limits=[0, 100],
                    snapshot=common, raw_unmatched=2, raw_sha256='fixture-raw', gt_graph_hash='fixture-gt')
        info = dict(snapshot=model, unmatched=2, name='fixture', source_sha256='fixture-pred')
        db.execute('INSERT INTO clips VALUES (?,?)', (name, json.dumps(clip)))
        db.execute('INSERT INTO models VALUES (?,?,?)', (name, 'pooled', json.dumps(info)))
    yield tmp_path, db
    db.close()


def test_unknown_pool_pagination_crosses_clips_without_dropping_last_cases(unknown_catalog):
    root, db = unknown_catalog
    rows = list_cases(root, db, dict(filter='unlabeled', offset='1', limit='2'))
    assert rows['total'] == 4
    assert [r['key'] for r in rows['rows']] == ['pooled:one:p:30', 'pooled:two:p:50']
    last = list_cases(root, db, dict(filter='unlabeled', offset='3', limit='40'))
    assert [r['node'] for r in last['rows']] == [60]
    raw = list_cases(root, db, dict(filter='unlabeled', pool='raw', t='1'))
    assert raw['total'] == 2
    assert [r['key'] for r in raw['rows']] == ['pooled:one:r:30', 'pooled:two:r:60']


def test_default_catalog_includes_all_flagged_centers_and_edges_across_clips(unknown_catalog):
    root, db = unknown_catalog
    examples = [('one', 'gt', 'missing', 0), ('two', 'gt', 'crowded', 5),
                ('one', 'edge', 'association', 6), ('two', 'pred_edge', 'wrong_link_ambiguous', 7),
                ('one', 'gt', 'matched', 9)]
    for i, (dataset, entity, kind, priority) in enumerate(examples):
        db.execute(INSERT, (f'case{i}', dataset, 'pooled', entity, kind, 1, 0, i, None,
                            None, None, None, None, None, None, None, None, priority))
    first = list_cases(root, db, dict(limit='2'))
    second = list_cases(root, db, dict(limit='2', offset='2'))
    assert first['total'] == second['total'] == 4
    assert [r['key'] for r in [*first['rows'], *second['rows']]] == ['case0', 'case1', 'case2', 'case3']
    linked = list_cases(root, db, dict(key='case3', limit='2'))
    assert linked['offset'] == 2 and linked['rows'][-1]['key'] == 'case3'
    stale_page = list_cases(root, db, dict(offset='9999', limit='2'))
    assert stale_page == second
    assert list_cases(root, db, dict(filter='detection'))['total'] == 2
    assert list_cases(root, db, dict(filter='association'))['total'] == 2


def test_unmatched_prediction_context_keeps_its_frame_and_unknown_identity(unknown_catalog):
    root, db = unknown_catalog
    context = get_case(root, db, 'pooled:one:p:30')
    assert context['case']['t'] == 1
    assert context['case']['kind'] == 'unlabeled'
    assert context['focus']['gt'] == []
    assert context['focus']['final'] == [30]
    assert context['anchor'] == [1, 2, 5]
    assert all(n[1] == f['t'] for f in context['frames'] for rows in f['points'].values() for n in rows)
    with pytest.raises(ValueError, match='not an unmatched'):
        get_case(root, db, 'pooled:one:p:10')
    with pytest.raises(ValueError, match='Unknown clip'):
        get_case(root, db, 'pooled:../../etc:r:1')


def test_native_frames_and_missing_chunks_cannot_be_substituted(tmp_path):
    import zarr
    group = zarr.open_group(str(tmp_path / 'image.zarr'), mode='w')
    raw = np.arange(1, 1 + 3*2*4*5, dtype=np.uint16).reshape(3, 2, 4, 5)
    group.create_array('0', data=raw, chunks=(1, 2, 4, 5))
    args = (str(tmp_path / 'image.zarr'), 1, raw.shape, (0, 200), 'fixture')
    compressed = gray_frame(*args)
    expected = np.clip(raw[1].astype(np.float32)*255/200, 0, 255).astype(np.uint8)
    assert np.array_equal(np.frombuffer(gzip.decompress(compressed), np.uint8).reshape(2, 4, 5), expected)
    (tmp_path / 'image.zarr/0/c/2/0/0/0').unlink()
    with pytest.raises(FileNotFoundError):
        gray_frame(str(tmp_path / 'image.zarr'), 2, raw.shape, (0, 200), 'fixture')
