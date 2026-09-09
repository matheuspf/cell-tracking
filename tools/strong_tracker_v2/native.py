"""Image/prediction-only candidate construction. No annotation reads or imports."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import FULL, OUT, SCALE, V1, adjacency, load_graph, save_arrays

EDGE_FEATURES = ['native_probability', 'native_missing', 'existing_edge', 'distance_um',
    'forward_residual_um', 'backward_residual_um', 'mutual_motion_residual_um',
    'forward_distance_rank', 'reverse_distance_rank', 'source_response', 'target_response',
    'intensity_difference', 'intensity_ratio', 'source_density', 'target_density',
    'source_native_confidence', 'target_native_confidence', 'source_origin_inserted',
    'target_origin_inserted', 'source_history_missing', 'target_future_missing',
    'source_track_remaining', 'target_track_age']

NODE_EXTRA = ['native_confidence', 'native_confidence_missing', 'origin_inserted',
    'source_displacement_um', 'best_outgoing_probability', 'second_outgoing_probability',
    'outgoing_probability_margin', 'best_incoming_probability', 'native_incoming_missing',
    'native_outgoing_missing']


def provenance(name, nodes, raw):
    p = np.load(FULL / 'inputs' / f'pre_ilp_{name}.npz')
    coords, probs = p['coords'], p['node_probabilities']
    orig = {int(r[0]): r for r in raw['nodes']}
    mapping = np.full(len(nodes), -1, np.int64)
    displacement = np.zeros(len(nodes), np.float32)
    for i, r in enumerate(nodes):
        j = int(r[0])
        if j in orig:
            assert 0 <= j < len(coords) and np.array_equal(orig[j][1:], coords[j]), 'raw ID provenance drift'
            mapping[i] = j
            displacement[i] = np.linalg.norm((r[2:] - orig[j][2:]) * SCALE)
    assert len(np.unique(mapping[mapping >= 0])) == np.sum(mapping >= 0)
    confidence = np.zeros(len(nodes), np.float32)
    confidence[mapping >= 0] = probs[mapping[mapping >= 0]]
    return mapping, confidence, displacement, p['edge_scores']


def build(name, base=None, raw=None):
    # Cached raw export is label-free and prepared by V200, not a GEFF inference read.
    b = base if base is not None else load_graph(V1/'baseline/public'/f'{name}.npz')
    r = raw if raw is not None else load_graph(OUT/'raw'/f'{name}.npz')
    n, e = b['nodes'], b['edges']
    ix, pred, succ = adjacency(n, e)
    pos = n[:, 2:] * SCALE
    mapping, conf, displacement, es = provenance(name, n, r)
    native_to_i = {int(j): i for i, j in enumerate(mapping) if j >= 0}
    native_prob = {}
    for a, c, prob, *_ in es:
        if int(a) in native_to_i and int(c) in native_to_i:
            i, j = native_to_i[int(a)], native_to_i[int(c)]
            if n[j, 1] == n[i, 1]+1:
                native_prob[i, j] = max(native_prob.get((i, j), 0.), float(prob))
    pairs = set(native_prob)
    existing = {(ix[int(a)], ix[int(c)]) for a, c in e}
    pairs.update(existing)
    # Cap image-only physical neighbors, in both directions, without GT gating.
    for t in np.unique(n[:, 1])[:-1]:
        a, c = np.flatnonzero(n[:, 1] == t), np.flatnonzero(n[:, 1] == t+1)
        if not len(a) or not len(c):
            continue
        d, j = cKDTree(pos[c]).query(pos[a], k=min(4, len(c)))
        for ii, dd, jj in zip(a, np.asarray(d).reshape(len(a), -1), np.asarray(j).reshape(len(a), -1)):
            pairs.update((int(ii), int(c[k])) for v, k in zip(dd, jj) if v <= 14.)
        d, j = cKDTree(pos[a]).query(pos[c], k=min(2, len(a)))
        for jj, dd, ii in zip(c, np.asarray(d).reshape(len(c), -1), np.asarray(j).reshape(len(c), -1)):
            pairs.update((int(a[k]), int(jj)) for v, k in zip(dd, ii) if v <= 14.)
    pairs = np.asarray(sorted(pairs), np.int64).reshape(-1, 2)
    a, c = pairs.T
    vp = np.zeros_like(pos)
    vn = np.zeros_like(pos)
    hm, fm = np.ones(len(n), bool), np.ones(len(n), bool)
    for i in range(len(n)):
        if len(pred[i]) == 1:
            vp[i] = pos[i] - pos[pred[i][0]]
            hm[i] = False
        if len(succ[i]) == 1:
            vn[i] = pos[succ[i][0]] - pos[i]
            fm[i] = False
    delta = pos[c] - pos[a]
    distance = np.linalg.norm(delta, axis=1)
    pp = np.array([native_prob.get(tuple(p), 0.) for p in pairs], np.float32)
    has = np.array([tuple(p) in native_prob for p in pairs])
    ex = np.array([tuple(p) in existing for p in pairs])
    fr, rr = np.zeros(len(pairs)), np.zeros(len(pairs))
    for dest, ids in [(fr, a), (rr, c)]:
        order = np.lexsort((np.arange(len(ids)), distance, ids))
        split = np.r_[0, np.flatnonzero(np.diff(ids[order]))+1]
        dest[order] = np.arange(len(order)) - np.repeat(split, np.diff(np.r_[split, len(order)]))
    f = b['features']
    x = np.column_stack([pp, ~has, ex, distance,
        np.linalg.norm(delta - .5*vp[a], axis=1),
        np.linalg.norm(delta - .5*vn[c], axis=1),
        np.linalg.norm(delta - .25*(vp[a]+vn[c]), axis=1), fr, rr,
        f[a, 5], f[c, 5], np.abs(f[a, 6]-f[c, 6]),
        np.minimum(f[a, 6], f[c, 6])/(np.maximum(f[a, 6], f[c, 6])+.01),
        f[a, 8], f[c, 8], conf[a], conf[c], mapping[a]<0, mapping[c]<0,
        hm[a], fm[c], f[a, 12], f[c, 11]]).astype(np.float32)
    node = np.zeros((len(n), len(NODE_EXTRA)), np.float32)
    node[:, :4] = np.column_stack([conf, mapping<0, mapping<0, displacement])
    node[:, 8:] = 1.
    for j, (i, k) in enumerate(pairs):
        if has[j]:
            if pp[j] > node[i, 4]:
                node[i, 5] = node[i, 4]
                node[i, 4] = pp[j]
            elif pp[j] > node[i, 5]:
                node[i, 5] = pp[j]
            node[k, 7] = max(node[k, 7], pp[j])
            node[k, 8], node[i, 9] = 0., 0.
    node[:, 6] = node[:, 4]-node[:, 5]
    return dict(pairs=pairs, edge_features=x, node_extra=node, native_index=mapping)


def cached(name):
    return load_graph(OUT/'native'/f'{name}.npz')
