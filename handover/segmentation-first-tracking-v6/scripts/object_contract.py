"""Small NumPy reference contracts, NOT a detector or production tracker.

Coordinates use voxel centers, ZYX, with half-open boxes. Motion for mask_iou
is an explicit integer RAW-grid displacement, not micrometers or optical flow.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def ints(value, size, name, nonnegative=True):
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in 'iu':
        raise ValueError(f'{name}: expected {size} integer coordinates')
    if a.dtype.kind == 'u' and np.any(a > np.iinfo(np.int64).max):
        raise ValueError(f'{name}: integer overflow')
    a = a.astype(np.int64)
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name}: negative coordinate')
    return a


@dataclass(frozen=True)
class Object3D:
    bbox: tuple[int, ...]
    mask: np.ndarray  # Tight, owned, read-only binary crop.
    centroid_vox: tuple[float, ...]
    centroid_um: tuple[float, ...]
    volume_vox: int
    volume_um3: float
    covariance_um2: np.ndarray
    spacing_zyx: tuple[float, ...]


def extract(mask, origin=(0, 0, 0), spacing=(1., 1., 1.)):
    a = np.asarray(mask)
    o = ints(origin, 3, 'origin')
    s = np.asarray(spacing, dtype=float)
    if a.ndim != 3 or a.dtype != np.bool_ or not a.any():
        raise ValueError('mask must be a nonempty 3D boolean object')
    if s.shape != (3,) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('spacing must be finite positive ZYX')
    points = np.argwhere(a)
    lo, hi = points.min(0), points.max(0) + 1
    if np.any(o > np.iinfo(np.int64).max - hi):
        raise ValueError('origin plus mask extent overflows')
    crop = a[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))].copy()
    crop.setflags(write=False)
    raw = points.astype(float) + o
    mean = raw.mean(0)
    centered = (raw - mean) * s
    cov = centered.T @ centered / len(raw)
    cov.setflags(write=False)
    return Object3D(tuple(map(int, np.r_[lo+o, hi+o])), crop,
                    tuple(mean), tuple(mean*s), len(raw), float(len(raw)*np.prod(s)),
                    cov, tuple(s))


def bbox_iou(a, b):
    a, b = ints(a, 6, 'bbox A', False), ints(b, 6, 'bbox B', False)
    if np.any(a[3:] <= a[:3]) or np.any(b[3:] <= b[:3]):
        raise ValueError('boxes must have positive half-open extents')
    # Python integers avoid overflow in volume products.
    volume = lambda widths: math.prod(int(v) for v in widths)
    inter = volume(np.maximum(0, np.minimum(a[3:], b[3:]) - np.maximum(a[:3], b[:3])))
    return inter / (volume(a[3:]-a[:3]) + volume(b[3:]-b[:3]) - inter)


def mask_iou(source, target, shift_vox=(0, 0, 0)):
    if source.spacing_zyx != target.spacing_zyx:
        raise ValueError('resample into a common physical grid before overlap')
    shift = ints(shift_vox, 3, 'shift', False)
    a, b = np.array(source.bbox), np.array(target.bbox)
    a = a + np.r_[shift, shift]
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    if np.any(hi <= lo):
        return 0.
    sl = tuple(slice(int(l-v), int(h-v)) for l,h,v in zip(lo,hi,a[:3]))
    tl = tuple(slice(int(l-v), int(h-v)) for l,h,v in zip(lo,hi,b[:3]))
    inter = int(np.count_nonzero(source.mask[sl] & target.mask[tl]))
    return inter / (source.volume_vox + target.volume_vox - inter)


def validate_selection(selected, candidates, conflicts):
    chosen, known = list(selected), set(candidates)
    if len(set(chosen)) != len(chosen) or not set(chosen) <= known:
        raise ValueError('duplicate or unknown selected hypothesis')
    for a, b in conflicts:
        if a == b or a not in known or b not in known:
            raise ValueError('invalid conflict edge')
        if a in chosen and b in chosen:
            raise ValueError('incompatible simultaneous segmentation hypotheses')
    return True


def validate_graph(nodes, edges, shape_tzyx):
    n, e = np.asarray(nodes), np.asarray(edges)
    shape = ints(shape_tzyx, 4, 'shape')
    if np.any(shape <= 0):
        raise ValueError('positive TZYX shape required')
    if n.ndim != 2 or n.shape[1:] != (5,) or n.dtype.kind not in 'iu':
        raise ValueError('nodes must be integer [id,t,z,y,x]')
    if e.ndim != 2 or e.shape[1:] != (2,) or e.dtype.kind not in 'iu':
        raise ValueError('edges must be integer [source_id,target_id]')
    if np.any(n < 0) or np.any(n[:,1:] >= shape):
        raise ValueError('node coordinates/IDs outside contract')
    ids = [int(v) for v in n[:,0]]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate node ID')
    when = dict(zip(ids, map(int, n[:,1])))
    pairs = [tuple(map(int, row)) for row in e]
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate edge')
    incoming, outgoing = {}, {}
    for a,b in pairs:
        if a not in when or b not in when or when[b] != when[a]+1:
            raise ValueError('missing endpoint or non-adjacent time edge')
        incoming[b] = incoming.get(b,0)+1
        outgoing[a] = outgoing.get(a,0)+1
    if max(incoming.values(),default=0)>1 or max(outgoing.values(),default=0)>2:
        raise ValueError('merge or more than two daughters')
    return True


def dense_label_bytes(shape_tzyx=(100,64,256,256), clips=199, detectors=1, bytes_per_voxel=4):
    shape = ints(shape_tzyx,4,'shape')
    values = [clips,detectors,bytes_per_voxel]
    if np.any(shape <= 0) or any(type(v) is not int or v < 1 for v in values):
        raise ValueError('positive sizes required')
    return math.prod(map(int,shape))*clips*detectors*bytes_per_voxel
