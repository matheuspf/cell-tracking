"""Validate exported artifacts against masks, graph edges, and the prior pilot."""
from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import tifffile

from .export import official_matches
from .pipeline import DEFAULT_OUTPUT, PILOT, write_json


def verify(output=DEFAULT_OUTPUT, pilot=PILOT):
    manifest = json.loads((output / 'manifest.json').read_text())
    index = json.loads((output / 'site/index.json').read_text())
    reference = json.loads((pilot / 'results.json').read_text())
    mapping = {'detector': 'repo_detector', 'selected': 'repo_final',
               'watershed': 'repo_seeded_watershed', 'focus_centroid': 'focus_centroid',
               'focus_weighted': 'focus_weighted', 'focus_peak': 'focus_peak'}
    prior = {(r['dataset'], r['time'], r['method']): r for r in reference['per_frame']}
    tested = 0
    graph_sets = {}
    gt_lookup = {}
    spacing = {}
    for seq in index['sequences']:
        spacing[seq['dataset']] = np.asarray(seq['spacing'])
        if seq['continuous']:
            assert np.array_equal(np.diff(seq['times']), np.ones(len(seq['times']) - 1))
        with np.load(output / 'graphs' / f"{seq['dataset']}.npz") as graph:
            gt_lookup[seq['dataset']] = {int(n[0]): n.tolist() for n in graph['gt_nodes']}
            graph_sets[seq['dataset']] = {kind: {tuple(map(int, e)) for e in graph[f'{kind}_edges']}
                                          for kind in ['gt', 'selected']}
    model_nodes = {}
    for method, review in index['prediction_reviews'].items():
        assert review['full_evaluation']['n'] == 199
        for dataset, receipt in review['datasets'].items():
            with np.load(receipt['source'], allow_pickle=False) as graph:
                graph_sets[dataset][method] = set(map(tuple, graph['edges']))
                model_nodes[dataset, method] = {int(n[0]): n.tolist() for n in graph['nodes']}
        for radius, events in review['errors'].items():
            assert len({e['id'] for e in events}) == len(events)
            for event in events:
                assert event['key'] in index['frame_keys']
                if event['kind'] == 'offset':
                    assert 3 <= event['distance_um'] <= float(radius)
                for edge in event['segments']:
                    assert tuple([edge['source'][0], edge['target'][0]]) in graph_sets[event['dataset']][edge['method']]
            cases = review['cases'][radius]
            assert len({c['id'] for c in cases}) == len(cases)
            # Grouping must neither hide a scoring flag nor invent a new one.
            assert {flag for c in cases for flag in c['related_flags']} == {e['id'] for e in events}
            for case in cases:
                scene = case['scene']
                lookup = {p['key']: p for p in scene['points']}
                cells = {c['annotation']: c for c in scene['cells']}
                assert len(lookup) == len(scene['points'])
                assert case['category'] in review['categories']
                for point in scene['points']:
                    original = (gt_lookup[case['dataset']] if point['kind'] == 'annotation'
                                else model_nodes[case['dataset'], method])[point['id']]
                    assert [point['id'], point['t'], *point['zyx']] == original
                    assert point['t'] in scene['times']
                for edge in scene['actual']:
                    assert (lookup[edge['source']]['id'], lookup[edge['target']]['id']) in graph_sets[case['dataset']][method]
                for edge in scene['expected']:
                    assert (lookup[edge['source']]['id'], lookup[edge['target']]['id']) in graph_sets[case['dataset']]['gt']
                    a, b = cells[edge['source']]['prediction'], cells[edge['target']]['prediction']
                    present = a is not None and b is not None and (lookup[a]['id'], lookup[b]['id']) in graph_sets[case['dataset']][method]
                    assert edge['recovered'] == present
                if case['category'] in ['link_missing', 'link_detection']:
                    focus = scene['focus_connection']
                    found = all(cells[focus[k]]['prediction'] is not None for k in ['source', 'target'])
                    assert found == (case['category'] == 'link_missing')
                if case['category'] == 'link_wrong':
                    assert any(e['status'] == 'fp' for e in scene['actual'])
                if case['category'].startswith('center_'):
                    assert scene['times'] == [case['t']] and not scene['actual']
                    probe = scene.get('center_probe')
                    if probe:
                        a, b = lookup[probe['annotation']], lookup[probe['prediction']]
                        delta = (np.asarray(b['zyx']) - a['zyx']) * spacing[case['dataset']]
                        assert np.allclose(probe['delta_um'], delta)
                        assert np.isclose(probe['distance_um'], np.linalg.norm(delta))
                        assert probe['assigned'] == (cells[a['key']]['prediction'] == b['key'])
    for f in manifest['frames']:
        key = f['key']
        exported = json.loads((output / 'site/frames' / f'{key}.json').read_text())
        native_labels = tifffile.imread(output / 'frames' / key / 'masks.tif')
        with gzip.open(output / 'site/frames' / f'{key}.labels.gz', 'rb') as stream:
            labels = np.frombuffer(stream.read(), '<u4').reshape(f['shape'])
        assert np.array_equal(labels, native_labels), key
        assert set(map(int, exported['sizes'])) == set(np.unique(labels)) - {0}
        with gzip.open(output / 'site/frames' / f'{key}.gray.gz', 'rb') as stream:
            gray = np.frombuffer(stream.read(), np.uint8).reshape(f['shape'])
        raw = tifffile.imread(output / 'frames' / key / 'image.tif')
        low, high = exported['display_limits']
        assert np.array_equal(gray, np.clip((raw.astype(float) - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8))
        for method, rows in exported['points'].items():
            assert len(rows) == len({r[0] for r in rows})
            assert all(all(0 <= c < dim for c, dim in zip(r[1:4], f['shape'])) for r in rows)
            if method == 'gt':
                continue
            ref = prior.get((f['dataset'], f['t'], mapping.get(method)))
            if method in index['prediction_reviews']:
                lookup = model_nodes[f['dataset'], method]
                expected = [[n[0], *n[2:]] for n in lookup.values() if n[1] == f['t']]
                assert [r[:4] for r in rows] == expected
                for label, members in exported['members'].items():
                    assert set(members[method]) == {r[0] for r in rows if labels[tuple(r[1:4])] == int(label)}
            for radius, pairs in exported['matches'][method].items():
                assert len(pairs) == len({p for p, _, _ in pairs}) == len({g for _, g, _ in pairs})
                assert all(d <= float(radius) + 1e-10 for _, _, d in pairs)
                pmap = {r[0]: np.array(r[1:4]) for r in rows}
                gmap = {r[0]: np.array(r[1:4]) for r in exported['points']['gt']}
                assert all(np.isclose(np.linalg.norm((pmap[p] - gmap[g]) * f['spacing']), d) for p, g, d in pairs)
                if ref:
                    assert len(rows) == ref['predicted']
                    assert len(pairs) == ref['matches_at_radius_um'][str(float(radius))], (key, method, radius)
                    tested += 1
                    if radius == '7':
                        assert np.allclose(sorted(d for _, _, d in pairs), sorted(ref['distances_um_at_7']))
        for method in ['gt', 'selected', *index['prediction_reviews']]:
            assert all((a, b) in graph_sets[f['dataset']][method] for _, a, b in exported['links'][method])
            if method in index['prediction_reviews']:
                lookup = model_nodes[f['dataset'], method]
                for event in exported['model_errors'][method]:
                    assert event['t'] == f['t']
                    if event['method'] == method:
                        assert lookup[event['node']][1] == event['t']
                assert set(exported['model_edge_status'][method].values()) <= {'tp', 'fp', 'unknown'}
        for method in ['focus_centroid', 'focus_weighted', 'focus_peak']:
            edges = exported['links'][method]
            assert all(t == f['t'] - 1 for t, _, _ in edges)
            assert len(edges) == len({a for _, a, _ in edges}) == len({b for _, _, b in edges})
    # Non-greedy and anisotropic matching probes are run in the actual metric runtime.
    result = official_matches([[9, 0, 0, 4], [8, 0, 0, 0]], [[30, 0, 0, 0], [40, 0, 0, 10]], np.ones(3), 7.)
    assert {(a, b) for a, b, _ in result} == {(9, 40), (8, 30)}
    result = official_matches([[9, 1, 0, 0], [8, 0, 0, 2]], [[30, 0, 0, 0]], np.array([4., 1., 1.]), 3.)
    assert result == [[8, 30, 2.]]
    receipt = dict(passed=True, frames=len(manifest['frames']), pilot_method_radius_comparisons=tested,
                   native_mask_bytes_exact=True, display_voxels_exact=True,
                   assignment_distances_verified=True, saved_graph_edges_verified=True,
                   complete_pipeline_models=list(index['prediction_reviews']),
                   source_prediction_points_exact=True, error_context_edges_verified=True,
                   explained_case_nodes_exact=True, explained_cases_cover_all_scoring_flags=True,
                   center_displacement_axes_verified=True,
                   center_failures_distinct_from_link_failures=True, correct_daughter_links_preserved=True,
                   focus_preview_one_to_one_consecutive_only=True)
    write_json(output / 'data_validation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--pilot', type=Path, default=PILOT)
    a = p.parse_args()
    verify(a.output, a.pilot)
