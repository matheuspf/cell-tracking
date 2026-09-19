"""Explicit native data, sparse supervision, and zero-origin coarse transforms."""
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
import zarr

SPACING = np.array([1.625, .40625, .40625], dtype=np.float32)
STRIDE = np.array([1., 4., 4.], dtype=np.float32)


def normalize(frame):
    lo, hi = np.percentile(frame, [1, 99])
    image = np.clip((frame.astype(np.float32)-lo)/max(hi-lo, 1.), 0, 1).astype(np.float32)
    return image[:, ::4, ::4].copy(), (float(lo), float(hi))


def background(frame, coarse):
    mean = uniform_filter(coarse, size=3, mode='constant')
    contrast = np.sqrt(np.maximum(0, uniform_filter(coarse**2, size=3, mode='constant')-mean**2))
    # Quantiles use only valid local neighborhoods; boundary padding is unknown.
    interior = np.zeros(coarse.shape, bool)
    interior[1:-1, 1:-1, 1:-1] = True
    intensity = frame[:, ::4, ::4]
    low = intensity < np.percentile(frame, 10)
    quiet = contrast < np.percentile(contrast[interior], 20)
    peaks = (coarse == maximum_filter(coarse, size=3)) & (coarse > 0)
    saturated = coarse >= 1
    return low & quiet & interior & ~peaks & ~saturated


def centers_target(shape, centers):
    target = np.zeros(shape, np.float32)
    support = np.zeros(shape, bool)
    lattice_um = SPACING*STRIDE
    for point in centers:
        c = point*SPACING
        lower = np.maximum(0, np.ceil((c-3.25)/lattice_um).astype(int))
        upper = np.minimum(shape, np.floor((c+3.25)/lattice_um).astype(int)+1)
        if np.any(lower >= upper):
            continue
        grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lower, upper)], indexing='ij'), -1)
        d2 = ((grid*lattice_um-c)**2).sum(-1)
        known = d2 <= 3.25**2
        sl = tuple(slice(a, b) for a, b in zip(lower, upper))
        target[sl] = np.maximum(target[sl], np.where(known, np.exp(-d2/(2*1.625**2)), 0))
        support[sl] |= known
    return target, support


def labels(path):
    g = zarr.open_group(str(path), mode='r')
    nodes = np.column_stack([g['nodes/ids'][:], *[g[f'nodes/props/{c}/values'][:] for c in 'tzyx']])
    return nodes.astype(np.int64), np.asarray(g['edges/ids'][:], np.int64)


def pair(data, name, t):
    image = zarr.open_array(str(data/'train'/f'{name}.zarr'/'0'), mode='r')
    nodes, edges = labels(data/'train'/f'{name}.geff')
    images, targets, positives, backgrounds, query = [], [], [], [], []
    for frame in (t, t+1):
        raw = np.asarray(image[frame])
        x, _ = normalize(raw)
        points = nodes[nodes[:, 1] == frame]
        y, known = centers_target(x.shape, points[:, 2:].astype(np.float32))
        bg = background(raw, x) & ~known
        images.append(x[None]); targets.append(y); positives.append(known); backgrounds.append(bg); query.append(points)
    return dict(images=np.stack(images), targets=np.stack(targets), positives=np.stack(positives),
                backgrounds=np.stack(backgrounds), query=query, edges=edges, dataset=name, time=t)


def supported_incoming(a_ids, b_ids, edges):
    edge_set = set(map(tuple, edges.tolist()))
    annotated_children = set(edges[:, 1].tolist()) if len(edges) else set()
    return np.array([[1 if (int(a), int(b)) in edge_set else 0 if b in annotated_children else -1
                      for b in b_ids] for a in a_ids], dtype=np.int8).reshape(len(a_ids), len(b_ids))
