"""The eight fixed recipes; no learned parameters and no evaluator imports."""
from __future__ import annotations

import collections
import math
import numpy as np


def gated_relink(nodes, before, proposal, probability):
    """Atomic connected components of the edge symmetric difference (E01)."""
    key = lambda e: (int(e['source_id']), int(e['target_id']))
    old, new = {key(e): e for e in before}, {key(e): e for e in proposal}
    removed, added = set(old) - set(new), set(new) - set(old)
    adjacency = collections.defaultdict(set)
    for a, b in removed | added:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(adjacency)
    while unseen:
        stack = [min(unseen)]
        component = set()
        while stack:
            n = stack.pop()
            if n in component:
                continue
            component.add(n)
            stack.extend(adjacency[n] - component)
        unseen -= component
        components.append(component)
    indeg_old, outdeg_old = collections.Counter(b for a, b in old), collections.Counter(a for a, b in old)
    indeg_new, outdeg_new = collections.Counter(b for a, b in new), collections.Counter(a for a, b in new)
    chosen = dict(old)
    receipts = []
    for comp in components:
        rm = sorted(e for e in removed if e[0] in comp)
        add = sorted(e for e in added if e[0] in comp)
        same_degree = all(indeg_old[n] == indeg_new[n] and outdeg_old[n] == outdeg_new[n] for n in comp)
        no_fork = all(outdeg_old[n] <= 1 and outdeg_new[n] <= 1 for n in comp)
        present = all(n in nodes for n in comp)
        values = [probability.get(e, float('nan')) for e in rm + add]
        finite = all(math.isfinite(v) for v in values)
        eligible = same_degree and no_fork and present and finite
        gain = math.fsum(probability[e] for e in add) - math.fsum(probability[e] for e in rm) if finite else None
        accept = eligible and gain > 0
        if accept:
            for e in rm:
                del chosen[e]
            chosen.update({e: new[e] for e in add})
        receipts.append(dict(removed=rm, added=add, eligible=eligible, accepted=accept,
                             degree_preserved=same_degree, no_fork=no_fork,
                             complete_evidence=finite, probability_gain=gain))
    # Keep original edge ordering; append accepted proposals in original proposal order.
    result = [e for e in before if key(e) in chosen]
    result += [e for e in proposal if key(e) in chosen and key(e) not in old]
    return result, receipts


def parabola_offset(minus, zero, plus):
    """Three-point concave local-maximum fit, independently per axis (E02)."""
    a, b, c = np.broadcast_arrays(np.asarray(minus, float), np.asarray(zero, float), np.asarray(plus, float))
    denominator = a - 2 * b + c
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = .5 * (a - c) / denominator
    valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (denominator < 0)
             & (b >= a) & (b >= c) & np.isfinite(delta) & (np.abs(delta) <= .5))
    return np.where(valid, delta, 0.)


def peak_offsets(response, coords):
    shape = np.asarray(response.shape)
    coords = np.asarray(coords, int).reshape(-1, 3)
    offsets = np.zeros_like(coords, dtype=np.float64)
    supported = np.zeros_like(coords, dtype=bool)
    for axis in range(3):
        good = (coords[:, axis] > 0) & (coords[:, axis] < shape[axis] - 1)
        c = coords[good]
        low, high = c.copy(), c.copy()
        low[:, axis] -= 1
        high[:, axis] += 1
        offsets[good, axis] = parabola_offset(response[tuple(low.T)], response[tuple(c.T)], response[tuple(high.T)])
        supported[good, axis] = True
    return offsets, supported


def localize_originals(nodes, original_nodes, offsets, stride):
    shifted = 0
    eligible = 0
    displacement = []
    for i, node in nodes.items():
        if i not in original_nodes or i not in offsets:
            continue
        before = np.asarray([original_nodes[i][a] for a in 'zyx'], float)
        current = np.asarray([node[a] for a in 'zyx'], float)
        if not np.array_equal(before, current):
            continue
        eligible += 1
        delta = np.asarray(offsets[i], float) * np.asarray(stride)
        for a, v in zip('zyx', current + delta):
            node[a] = float(v)
        shifted += int(np.any(delta != 0))
        displacement.append(delta.tolist())
    return dict(eligible_originals=eligible, shifted_originals=shifted,
                displacement_voxels=displacement)


def trilinear_features(features, coords, mask):
    """E03 reference interpolation, including exact integer gather fast path."""
    import torch
    import torch.nn.functional as F
    b, c, z, y, x = features.shape
    shape = coords.new_tensor([z, y, x])
    valid_coords = coords[mask]
    if valid_coords.numel() and ((valid_coords < 0).any() or (valid_coords > shape - 1).any()):
        raise ValueError('Feature coordinates outside audited grid')
    # Avoid numerical coordinate conversion noise where interpolation is exactly gather.
    if torch.equal(valid_coords, valid_coords.trunc()):
        result = features.new_zeros((b, coords.shape[1], c))
        for i in range(b):
            q = coords[i, mask[i]].long()
            result[i, mask[i]] = features[i, :, q[:, 0], q[:, 1], q[:, 2]].T
        return result
    grid = (2 * (coords + .5) / shape - 1).flip(-1).reshape(b, -1, 1, 1, 3)
    sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled[:, :, :, 0, 0].transpose(1, 2) * mask.unsqueeze(-1)


def context_windows(length, window, transition):
    """Fully observed forward windows containing t and t+1 (E06)."""
    t = transition
    return [s for s in range(max(0, t + 2 - window), min(t, length - window) + 1)]


def median_views(views, masks=None):
    import torch
    stack = torch.stack(views)
    if masks is None:
        ordered = stack.sort(dim=0).values
        n = len(views)
        return (ordered[(n - 1) // 2] + ordered[n // 2]) * .5
    mask = torch.broadcast_to(torch.stack(masks), stack.shape)
    count = mask.sum(0)
    if (count == 0).any():
        raise ValueError('No supported detection view')
    ordered = torch.where(mask, stack, torch.full_like(stack, float('inf'))).sort(dim=0).values
    a = ordered.gather(0, ((count - 1) // 2).unsqueeze(0))[0]
    b = ordered.gather(0, (count // 2).unsqueeze(0))[0]
    return (a + b) * .5


def inverse_phase(response, phase_grid):
    """Pull shifted response at q+phase; never wrap unsupported boundaries (E04)."""
    import torch
    import torch.nn.functional as F
    z, y, x = response.shape[-3:]
    axes = [torch.arange(n, device=response.device, dtype=response.dtype) + d
            for n, d in zip((z, y, x), phase_grid)]
    zz, yy, xx = torch.meshgrid(*axes, indexing='ij')
    valid = (zz >= 0) & (zz <= z - 1) & (yy >= 0) & (yy <= y - 1) & (xx >= 0) & (xx <= x - 1)
    grid = torch.stack((2 * (xx + .5) / x - 1, 2 * (yy + .5) / y - 1,
                        2 * (zz + .5) / z - 1), -1)[None]
    result = F.grid_sample(response.reshape(-1, 1, z, y, x), grid.expand(response.numel() // (z*y*x), -1, -1, -1, -1),
                           mode='bilinear', padding_mode='border', align_corners=False).reshape(response.shape)
    return result, valid


def fork_anchors(nodes, edges):
    pred, succ = collections.defaultdict(list), collections.defaultdict(list)
    for edge in edges:
        a, b = int(edge['source_id']), int(edge['target_id'])
        if a in nodes and b in nodes and nodes[b]['t'] == nodes[a]['t'] + 1:
            pred[b].append(a)
            succ[a].append(b)
    forks = {n for n in nodes if len(succ[n]) > 1}
    anchors = set(forks)
    for n in forks:
        anchors.update(pred[n])
        anchors.update(succ[n])
    return forks, anchors, pred, succ


def fork_smooth(nodes, edges, stats, weight=.8, window=2):
    """Same public polyfit arithmetic, with E07 anchors and segment restriction."""
    forks, anchors, pred, succ = fork_anchors(nodes, edges)
    original = {i: np.asarray([nodes[i][a] for a in 'zyx'], np.float64) for i in nodes}
    updates = {}
    for i in sorted(nodes):
        if i in anchors:
            continue
        neighborhood = [(0, i)]
        current = i
        for step in range(1, window + 1):
            previous = pred[current]
            if len(previous) != 1 or previous[0] in forks:
                break
            current = previous[0]
            neighborhood.append((-step, current))
        current = i
        for step in range(1, window + 1):
            following = succ[current]
            if len(following) != 1 or following[0] in forks:
                break
            current = following[0]
            neighborhood.append((step, current))
        if len(neighborhood) < 3:
            stats['linefit_skipped_nodes'] += 1
            continue
        dts = np.asarray([dt for dt, _ in neighborhood], np.float64)
        coords = np.stack([original[n] for _, n in neighborhood])
        fitted = np.asarray([np.polyval(np.polyfit(dts, coords[:, a], 1), 0.) for a in range(3)], np.float64)
        if not np.isfinite(fitted).all():
            stats['linefit_skipped_nodes'] += 1
            continue
        updates[i] = (1 - weight) * original[i] + weight * fitted
    for i, position in updates.items():
        for axis, value in zip('zyx', position):
            nodes[i][axis] = float(value)
    stats['linefit_smoothed_nodes'] = len(updates)
    stats['E07_forks'] = len(forks)
    stats['E07_anchors'] = len(anchors)
    return nodes
