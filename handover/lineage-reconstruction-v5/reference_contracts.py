"""Small v5 reference contracts, NOT a tracker or official graph evaluator.

Requires NumPy. Candidate graphs may branch to many alternatives; only selected
output graphs have binary-lineage degree limits. Unknown sparse labels stay -1.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable, Mapping

import numpy as np


def integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} cannot be boolean')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def grid_to_native(points, stride_zyx=(1., 1., 1.), origin_zyx=(0., 0., 0.)):
    """Fractional network-grid points -> native voxels; no hidden rounding."""
    p = np.asarray(points, dtype=np.float64)
    s = np.asarray(stride_zyx, dtype=np.float64)
    o = np.asarray(origin_zyx, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or s.shape != (3,) or o.shape != (3,):
        raise ValueError('Expected N x 3 points and three-axis transforms')
    if not all(np.isfinite(a).all() for a in (p, s, o)) or (s <= 0).any():
        raise ValueError('Transforms must be finite with positive stride')
    return p * s + o


def serialize_positions(points, shape_zyx):
    """Round valid native points and record clipping caused only by rounding."""
    p = np.asarray(points, dtype=np.float64)
    shape = np.array([integer(v, 'shape', 1) for v in shape_zyx], dtype=np.int64)
    if shape.shape != (3,) or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected N x 3 points and three-axis shape')
    if not np.isfinite(p).all() or (p < 0).any() or (p >= shape).any():
        raise ValueError('Continuous native centers must already be inside the image')
    rounded = np.rint(p).astype(np.int64)
    clipped = np.clip(rounded, 0, shape - 1)
    return clipped, int(np.any(clipped != rounded, axis=1).sum())


def checked_times(node_times: Mapping[int, int]) -> dict[int, int]:
    return {integer(k, 'node ID'): integer(v, 'frame') for k, v in node_times.items()}


def checked_edges(node_times: Mapping[int, int], edges: Iterable[tuple[int, int]]):
    times = checked_times(node_times)
    result, seen = [], set()
    for item in edges:
        if len(item) != 2:
            raise ValueError('Each edge needs two endpoints')
        a, b = (integer(v, 'edge endpoint') for v in item)
        if a not in times or b not in times:
            raise ValueError('Dangling edge')
        if times[b] != times[a] + 1:
            raise ValueError('Final candidate links must be consecutive and forward')
        if (a, b) in seen:
            raise ValueError('Duplicate edge')
        seen.add((a, b)); result.append((a, b))
    return times, result


def validate_selected_graph(node_times, edges):
    times, edges = checked_edges(node_times, edges)
    incoming = {i: 0 for i in times}; outgoing = {i: 0 for i in times}
    for a, b in edges:
        incoming[b] += 1; outgoing[a] += 1
    if max(incoming.values(), default=0) > 1:
        raise ValueError('Selected graph contains a merge')
    if max(outgoing.values(), default=0) > 2:
        raise ValueError('Selected graph contains more than two children')
    return {'nodes': len(times), 'edges': len(edges),
            'forks': sum(v == 2 for v in outgoing.values())}


def validate_region_mapping(node_times, assignments, require_complete=True):
    """assignments: (frame, positive mask label, stable candidate ID)."""
    times = checked_times(node_times); keys = set(); candidates = set()
    for frame, mask_label, candidate in assignments:
        frame = integer(frame, 'frame')
        mask_label = integer(mask_label, 'mask label', 1)
        candidate = integer(candidate, 'candidate ID')
        if candidate not in times or times[candidate] != frame:
            raise ValueError('Mask/candidate frame mismatch')
        if (frame, mask_label) in keys or candidate in candidates:
            raise ValueError('Mask/candidate mapping is not one-to-one')
        keys.add((frame, mask_label)); candidates.add(candidate)
    if require_complete and candidates != set(times):
        raise ValueError('Fixed-node region mapping omitted candidates')
    return {'represented': len(candidates), 'unrepresented': len(times) - len(candidates)}


def sparse_edge_targets(node_times, matched_gt_ids, candidate_edges, gt_edges,
                        ambiguous_nodes=()):
    """Conservative reference targets: positive=1, supported negative=0, unknown=-1.

    A known predecessor constrains incoming alternatives. A single annotated child
    does not constrain all outgoing alternatives. Two established children exhaust
    the binary-source capacity. This consumes source TRAINING annotations only;
    matching itself must be done by the official adapter, not this helper.
    """
    times, pairs = checked_edges(node_times, candidate_edges)
    if set(matched_gt_ids) != set(times):
        raise ValueError('Need a match/unknown entry for every candidate')
    matches = {i: integer(matched_gt_ids[i], 'GT match', -1) for i in times}
    known = [i for i in matches.values() if i >= 0]
    if len(known) != len(set(known)):
        raise ValueError('Reference node matching must be one-to-one')
    ambiguous = set(ambiguous_nodes)
    if not ambiguous <= set(times):
        raise ValueError('Unknown ambiguous node')
    truth, incoming, outgoing = set(), {}, {}
    for item in gt_edges:
        if len(item) != 2:
            raise ValueError('Invalid truth edge')
        a, b = (integer(v, 'GT endpoint') for v in item)
        if a == b or (a, b) in truth:
            raise ValueError('Self/duplicate truth edge')
        if b in incoming and incoming[b] != a:
            raise ValueError('GT merge violates this binary-lineage contract')
        truth.add((a, b)); incoming[b] = a
        outgoing.setdefault(a, set()).add(b)
        if len(outgoing[a]) > 2:
            raise ValueError('GT source has more than two children')
    result = np.full(len(pairs), -1, dtype=np.int8)
    for k, (a, b) in enumerate(pairs):
        if a in ambiguous or b in ambiguous:
            continue
        ga, gb = matches[a], matches[b]
        if (ga, gb) in truth:
            result[k] = 1
        elif gb in incoming or len(outgoing.get(ga, ())) == 2:
            result[k] = 0
    return result


def fuse_window_logits(observations):
    """Normalized evidence, not multiplication across overlapping windows.

    Each observation is (source_id, target_id, window_id, logit, weight).
    Repeated identical (edge,window) records are counted once; conflicts fail.
    Distinct windows with the same logit do not inflate confidence.
    """
    unique = {}
    for a, b, window, logit, weight in observations:
        a = integer(a, 'source'); b = integer(b, 'target')
        if a == b or not isinstance(window, str) or not window:
            raise ValueError('Invalid edge/window identity')
        l = number(logit, 'logit'); w = number(weight, 'weight')
        if w <= 0:
            raise ValueError('Evidence weight must be positive')
        key = (a, b, window)
        if key in unique and unique[key] != (l, w):
            raise ValueError('Conflicting duplicate window evidence')
        unique[key] = (l, w)
    numerator, denominator = {}, {}
    for (a, b, _), (l, w) in sorted(unique.items()):
        key = (a, b)
        numerator[key] = numerator.get(key, 0.) + w * l
        denominator[key] = denominator.get(key, 0.) + w
    return {key: numerator[key] / denominator[key] for key in sorted(numerator)}


def fork_cost_difference(second_edge_reward, division_cost, birth_cost=0., event_gain=0.):
    """Tiny regression fixture: fork cost minus continuation+daughter-birth cost.

    Negative favors a fork. Not the proposed full likelihood objective.
    The old bounded edge reward cannot overcome cost 1.2 with free birth.
    """
    p = number(second_edge_reward, 'reward')
    d = number(division_cost, 'division cost')
    b = number(birth_cost, 'birth cost')
    g = number(event_gain, 'event gain')
    return d - p - b - g
