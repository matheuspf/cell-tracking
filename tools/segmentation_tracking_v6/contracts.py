"""Small mask-contract reference; not a segmenter, tracker, or official scorer.

A bbox is [z0,y0,x0,z1,y1,x1), in one documented voxel grid. Coordinates
refer to voxel centers at integer indices. Mask comparisons require the SAME
lattice, origin convention and spacing. Production adapters must also verify
transforms and grid IDs; they may not silently resample masks here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import numpy as np


def _integers(values, n, name):
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain {n} integers")
    return tuple(int(x) for x in a)


@dataclass(frozen=True)
class Region:
    dataset: str
    time: int
    provider: str
    instance: int
    grid_id: str
    bbox: tuple[int, int, int, int, int, int]
    mask: np.ndarray
    spacing: tuple[float, float, float]

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.dataset, self.provider, self.grid_id)):
            raise ValueError("dataset, provider and grid_id must be nonempty")
        if type(self.time) is not int or self.time < 0 or type(self.instance) is not int or self.instance <= 0:
            raise ValueError("invalid time or positive instance ID")
        b = _integers(self.bbox, 6, "bbox")
        shape = tuple(b[k + 3] - b[k] for k in range(3))
        m = np.asarray(self.mask)
        if any(s <= 0 for s in shape) or m.dtype != bool or m.shape != shape or not m.any():
            raise ValueError("mask must be nonempty boolean data matching its half-open bbox")
        s = np.asarray(self.spacing, dtype=float)
        if s.shape != (3,) or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError("spacing must be three finite positive numbers")
        m = m.copy(); m.setflags(write=False)
        object.__setattr__(self, "bbox", b)
        object.__setattr__(self, "mask", m)
        object.__setattr__(self, "spacing", tuple(float(x) for x in s))

    @property
    def key(self):
        return (self.dataset, self.time, self.provider, self.instance)

    @property
    def voxel_count(self):
        return int(self.mask.sum())

    @property
    def volume_um3(self):
        return self.voxel_count * float(np.prod(self.spacing))

    def points(self):
        return np.argwhere(self.mask) + np.asarray(self.bbox[:3])

    def center(self, inside=False):
        p = self.points(); mean = p.mean(axis=0)
        if inside:
            # Deterministic occupied-voxel representative, not a per-GT correction.
            d = ((p - mean) * np.asarray(self.spacing)) ** 2
            return p[np.argmin(d.sum(axis=1))].copy()
        return mean

    def covariance_um2(self):
        p = self.points() * np.asarray(self.spacing)
        x = p - p.mean(axis=0)
        return x.T @ x / len(p)

    def mask_sha256(self):
        return hashlib.sha256(np.packbits(self.mask.ravel()).tobytes()).hexdigest()


def from_labels(labels, label, *, dataset="sample", time=0, provider="test",
                grid_id="native", spacing=(1., 1., 1.)):
    a = np.asarray(labels)
    if a.ndim != 3 or a.dtype.kind not in "iu" or (a < 0).any():
        raise ValueError("labels must be a nonnegative integer ZYX array")
    if type(label) is not int or label <= 0:
        raise ValueError("label must be positive")
    p = np.argwhere(a == label)
    if not len(p):
        raise ValueError("requested label is absent; do not fabricate a mask")
    lo, hi = p.min(axis=0), p.max(axis=0) + 1
    section = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return Region(dataset, time, provider, label, grid_id,
                  tuple(int(x) for x in np.r_[lo, hi]), a[section] == label, spacing)


def _compatible(a, b):
    if a.dataset != b.dataset or a.grid_id != b.grid_id or a.spacing != b.spacing:
        raise ValueError("mask comparison requires identical dataset/grid/spacing")


def intersection_voxels(a, b):
    _compatible(a, b)
    lo = np.maximum(a.bbox[:3], b.bbox[:3]); hi = np.minimum(a.bbox[3:], b.bbox[3:])
    if np.any(hi <= lo):
        return 0
    def crop(r):
        origin = np.asarray(r.bbox[:3])
        return r.mask[tuple(slice(int(l), int(h)) for l, h in zip(lo-origin, hi-origin))]
    return int(np.count_nonzero(crop(a) & crop(b)))


def mask_iou(a, b):
    inter = intersection_voxels(a, b)
    return inter / (a.voxel_count + b.voxel_count - inter)


def bbox_iou(a, b):
    _compatible(a, b)
    size = np.maximum(0, np.minimum(a.bbox[3:], b.bbox[3:]) - np.maximum(a.bbox[:3], b.bbox[:3]))
    inter = int(np.prod(size))
    va = int(np.prod(np.asarray(a.bbox[3:]) - a.bbox[:3]))
    vb = int(np.prod(np.asarray(b.bbox[3:]) - b.bbox[:3]))
    return inter / (va + vb - inter)


def translate(a, shift_zyx, *, target_time=None):
    """Integer translation reference: no clipping, wrapping, or pair-fitted flow.

    A production image-derived warp needs its own validity/occlusion mask.
    Ultrack uses target-to-source shifts; callers must test that convention.
    """
    s = _integers(shift_zyx, 3, "shift")
    b = tuple(a.bbox[k] + s[k % 3] for k in range(6))
    return replace(a, bbox=b, time=a.time if target_time is None else target_time)


def daughter_union_features(parent_in_daughter_frame, a, b):
    p = parent_in_daughter_frame
    _compatible(p, a); _compatible(p, b)
    if a.time != b.time or p.time != a.time or a.key == b.key:
        raise ValueError("require distinct daughters and aligned masks at one time")
    ab = intersection_voxels(a, b)
    # Triple intersection from local sparse coordinates, only for reference fixtures.
    ap = {tuple(x) for x in a.points()}; bp = {tuple(x) for x in b.points()}
    triple = len({tuple(x) for x in p.points()} & ap & bp)
    pi = intersection_voxels(p, a) + intersection_voxels(p, b) - triple
    union = a.voxel_count + b.voxel_count - ab
    return dict(union_iou=pi / (p.voxel_count + union - pi),
                daughter_volume_ratio=union / p.voxel_count,
                daughter_overlap_fraction=ab / min(a.voxel_count, b.voxel_count))


def validate_edges(nodes, edges):
    """nodes maps unique observation IDs to integer frame indices; no track IDs."""
    if not nodes or any(type(k) is not int or type(t) is not int or t < 0 for k,t in nodes.items()):
        raise ValueError("invalid observation/frame map")
    edges = [tuple(e) for e in edges]
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate edge")
    incoming = {}; outgoing = {}
    for edge in edges:
        if len(edge) != 2 or any(type(k) is not int or k not in nodes for k in edge):
            raise ValueError("missing/noninteger endpoint")
        a,b = edge
        if nodes[b] != nodes[a] + 1:
            raise ValueError("only consecutive temporal edges are exportable")
        incoming[b] = incoming.get(b,0)+1; outgoing[a] = outgoing.get(a,0)+1
    if max(incoming.values(), default=0)>1 or max(outgoing.values(), default=0)>2:
        raise ValueError("forbidden merge or excessive children")
    return True
