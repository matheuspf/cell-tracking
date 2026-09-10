"""NumPy-only reference contracts; not a trainer or an official graph evaluator."""
from __future__ import annotations

import hashlib
import numpy as np

NATIVE_SPACING = np.array([1.625, .40625, .40625], dtype=np.float64)


def points_on_image(points_native, *, sequence: bool, pooled_static: bool = False):
    """Sequence images are ALREADY XY/4. Do not resample them again."""
    p = np.asarray(points_native, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 ZYX points')
    stride = np.array([1, 4, 4] if sequence or pooled_static else [1, 1, 1])
    return p / stride, NATIVE_SPACING * stride


def normalize_image(image, *, representation: str, low_native: float, high_native: float):
    """Equivalent normalization for explicit uint16-scale or [0,1] inputs."""
    x = np.asarray(image, dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite([low_native, high_native]).all():
        raise ValueError('Nonfinite image/normalizer')
    if high_native <= low_native:
        raise ValueError('Normalizer must have a positive range')
    if representation == 'unit':
        if ((x < 0) | (x > 1)).any():
            raise ValueError('Unit representation outside [0,1]')
        x = x * 65535.
    elif representation == 'uint16_scale':
        if ((x < 0) | (x > 65535)).any():
            raise ValueError('Native representation outside uint16 range')
    else:
        raise ValueError('Intensity representation must be explicit')
    return np.clip((x - low_native) / (high_native - low_native), 0, 1)


def _integers(values, name):
    a = np.asarray(values)
    if a.dtype.kind not in 'iu' or a.dtype.kind == 'b':
        raise ValueError(f'{name} must have integer dtype')
    return a.astype(np.int64, copy=False)


def validate_graph(node_ids, t, points, edges, *, shape=None):
    """Edges index rows, NOT node IDs. Returns counts, never a quality score."""
    ids, time, edge = (_integers(node_ids, 'node_ids'), _integers(t, 't'),
                       _integers(edges, 'edges'))
    pos = np.asarray(points, dtype=np.float64)
    n = len(ids)
    if ids.ndim != 1 or time.shape != (n,) or pos.shape != (n, 3):
        raise ValueError('Invalid node array shapes')
    if edge.ndim != 2 or edge.shape[1] != 2:
        raise ValueError('Expected Ex2 edge rows')
    if len(np.unique(ids)) != n or (time < 0).any() or not np.isfinite(pos).all():
        raise ValueError('Duplicate IDs, invalid times or nonfinite positions')
    if edge.size and ((edge < 0).any() or (edge >= n).any()):
        raise ValueError('Edge row index outside node table')
    if len(np.unique(edge, axis=0)) != len(edge):
        raise ValueError('Duplicate edges')
    if edge.size and not np.all(time[edge[:, 1]] == time[edge[:, 0]] + 1):
        raise ValueError('Edges must connect consecutive frames')
    indegree = np.bincount(edge[:, 1], minlength=n)
    outdegree = np.bincount(edge[:, 0], minlength=n)
    if (indegree > 1).any() or (outdegree > 2).any():
        raise ValueError('Merge or more than two children')
    if shape is not None:
        sh = np.asarray(shape)
        if sh.shape != (4,) or sh.dtype.kind not in 'iu' or (sh <= 0).any():
            raise ValueError('Expected positive integer TZYX shape')
        if (time >= sh[0]).any() or (pos < 0).any() or (pos >= sh[1:]).any():
            raise ValueError('Observation outside image')
    return {'nodes': n, 'edges': len(edge), 'division_parents': int((outdegree == 2).sum())}


def tracklet_ids(t, edges):
    """Reconstruct nonbranching segments. Never use synthetic clone IDs as tracks."""
    time, edge = _integers(t, 't'), _integers(edges, 'edges')
    n = len(time)
    validate_graph(np.arange(n), time, np.zeros((n, 3)), edge)
    pred = [[] for _ in range(n)]
    succ = [[] for _ in range(n)]
    for a, b in edge:
        pred[b].append(a); succ[a].append(b)
    units = np.full(n, -1, dtype=np.int64)
    next_unit = 0
    for i in np.argsort(time, kind='stable'):
        if len(pred[i]) == 1 and len(succ[pred[i][0]]) == 1:
            units[i] = units[pred[i][0]]
        else:
            units[i] = next_unit; next_unit += 1
    return units


def dense_fork_targets(t, edges, n_frames: int, *, complete_future=None):
    """Dense-source labels only; visibility/crop censoring must be supplied.

    Do not call this for sparse Biohub or non-exhaustive Zoo annotations.
    A static image and the final sequence frame provide NO nondivision labels.
    """
    time, edge = _integers(t, 't'), _integers(edges, 'edges')
    if type(n_frames) is not int or n_frames < 1 or (time >= n_frames).any():
        raise ValueError('Invalid observed frame range')
    validate_graph(np.arange(len(time)), time, np.zeros((len(time), 3)), edge)
    target = np.bincount(edge[:, 0], minlength=len(time)) == 2
    mask = time < n_frames - 1
    if complete_future is not None:
        f = np.asarray(complete_future)
        if f.dtype != bool or f.shape != time.shape:
            raise ValueError('complete_future must be a node-aligned Boolean mask')
        mask &= f
    return target, mask


def padded_time_window(image, *, anchor: int, before: int, after: int):
    """Preserve missing-time masks; six real frames never become nine observed frames."""
    x = np.asarray(image)
    if x.ndim != 4 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite TZYX image')
    if any(type(v) is not int for v in (anchor, before, after)):
        raise ValueError('Window indices must be integers')
    if not 0 <= anchor < len(x) or before < 0 or after < 0:
        raise ValueError('Invalid temporal window')
    indices = np.arange(anchor - before, anchor + after + 1)
    valid = (indices >= 0) & (indices < len(x))
    out = np.zeros((len(indices), *x.shape[1:]), dtype=x.dtype)
    out[valid] = x[indices[valid]]
    return out, valid


def edge_supervision(gt_ids, candidate_edges, truth_edges, *, dense: bool):
    """Candidate-to-GT mapping uses -1 for unmatched. Unknown labels are -1.

    Sparse negatives need a contradicted known outgoing/incoming relationship.
    Dense synthetic unmatched/distractor candidates can be true negatives.
    """
    ids = _integers(gt_ids, 'gt_ids')
    candidates = _integers(candidate_edges, 'candidate_edges')
    truth = _integers(truth_edges, 'truth_edges')
    if ids.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != 2 or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError('Malformed supervision tables')
    if candidates.size and ((candidates < 0).any() or (candidates >= len(ids)).any()):
        raise ValueError('Candidate row index out of bounds')
    if (truth < 0).any():
        raise ValueError('Truth IDs cannot be unmatched sentinels')
    gs = set(map(tuple, truth.tolist()))
    outgoing = set(truth[:, 0].tolist()); incoming = set(truth[:, 1].tolist())
    labels = np.full(len(candidates), -1, dtype=np.int8)
    for k, (a, b) in enumerate(candidates):
        s, d = int(ids[a]), int(ids[b])
        if (s, d) in gs:
            labels[k] = 1
        elif dense or s in outgoing or d in incoming:
            labels[k] = 0
    return labels


def assert_partition(records):
    """All aliases/windows of a provenance group must share a partition."""
    groups, representations = {}, {}
    for r in records:
        for key in ('provenance_group', 'representation_group', 'partition'):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError(f'Missing {key}')
        if r['partition'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown partition')
        for mapping, key in ((groups, 'provenance_group'), (representations, 'representation_group')):
            if r[key] in mapping and mapping[r[key]] != r['partition']:
                raise ValueError('A provenance group or duplicate representation crosses partitions')
            mapping[r[key]] = r['partition']
    return True


def stable_partition(group: str, seed: int = 20260909):
    """Generator sanity split only; hashing does not establish biological independence."""
    if not isinstance(group, str) or not group:
        raise ValueError('Nonempty canonical group required')
    u = int.from_bytes(hashlib.sha256(f'{seed}:{group}'.encode()).digest()[:8], 'big') / 2**64
    return 'train' if u < .8 else 'validation' if u < .9 else 'test'


def require_task(source_kind: str, task: str):
    allowed = {
        'synthetic_static': {'centers'},
        'synthetic_sequence': {'centers', 'edges', 'events', 'images'},
        'zoo_graph': {'weak_edges', 'weak_events', 'geometry'},
        'zoo_mouse': {'weak_edges', 'geometry'},
        'riken_points': {'spatial_audit'},
    }
    if task not in allowed.get(source_kind, set()):
        raise ValueError(f'{source_kind} does not supply {task} supervision')
    return True


def raw_fork_advantage(second_edge_probability, *, division_cost=1.2, birth_cost=0., learned_gain=0.):
    """Toy fixed-node comparison; not a replacement for the actual ILP test."""
    values = np.asarray([second_edge_probability, division_cost, birth_cost, learned_gain], dtype=float)
    if not np.isfinite(values).all() or not 0 <= second_edge_probability <= 1:
        raise ValueError('Invalid objective parameters')
    return float(second_edge_probability + birth_cost + learned_gain - division_cost)
