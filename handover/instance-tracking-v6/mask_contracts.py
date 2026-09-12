"""NumPy reference contracts for v6. Not segmenter inference or official scoring.

Masks retain actual cropped support on an explicitly identified registered grid.
Bounding boxes are half open. A translation moves SOURCE support toward TARGET;
Ultrack's target-backshift convention needs its own tested adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np


def _int3(value: Iterable[int], name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != (3,) or a.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be three integers')
    return a.astype(np.int64)


def _spacing(value: Iterable[float]) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('spacing must be finite positive ZYX')
    return a


@dataclass(frozen=True)
class Key:
    dataset: str
    frame: int
    producer: str
    label_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError('dataset required')
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError('producer required')
        if type(self.frame) is not int or self.frame < 0:
            raise ValueError('frame must be a nonnegative integer')
        if type(self.label_id) is not int or self.label_id <= 0:
            raise ValueError('background/invalid label is not an instance')


@dataclass(frozen=True)
class Instance:
    key: Key
    lo: tuple[int, int, int]
    mask: np.ndarray
    grid_id: str = 'native'

    def __post_init__(self) -> None:
        if not isinstance(self.key, Key):
            raise ValueError('a frame/model-scoped Key is required')
        lo = _int3(self.lo, 'bbox lo')
        m = np.asarray(self.mask)
        if (lo < 0).any() or not isinstance(self.grid_id, str) or not self.grid_id:
            raise ValueError('nonnegative origin and grid identity required')
        if m.dtype != bool or m.ndim != 3 or min(m.shape) == 0 or not m.any():
            raise ValueError('nonempty 3D boolean support required; missing is not empty')
        # No array alias may mutate the already measured instance.
        m = m.copy()
        m.flags.writeable = False
        object.__setattr__(self, 'mask', m)
        object.__setattr__(self, 'lo', tuple(int(v) for v in lo))

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.lo) + self.mask.shape

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    def coordinates(self) -> np.ndarray:
        return np.argwhere(self.mask) + self.lo


def _same_grid(a: Instance, b: Instance) -> None:
    if a.grid_id != b.grid_id or a.key.dataset != b.key.dataset:
        raise ValueError('instances must share dataset and registered grid')


def extract_instances(labels: np.ndarray, dataset: str, frame: int,
                      producer: str, grid_id: str = 'native') -> list[Instance]:
    """Small-array reference; production should avoid scanning once per label."""
    x = np.asarray(labels)
    if x.ndim != 3 or x.dtype.kind not in 'iu' or (x < 0).any():
        raise ValueError('expected nonnegative integer ZYX labels')
    result = []
    for label in np.unique(x):
        if label == 0:
            continue
        coords = np.argwhere(x == label)
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        result.append(Instance(Key(dataset, frame, producer, int(label)),
                               tuple(lo), x[slices] == label, grid_id))
    return result


def describe(a: Instance, spacing=(1., 1., 1.)) -> dict:
    s = _spacing(spacing)
    points = a.coordinates().astype(float)
    center = points.mean(axis=0)
    q = (points - center) * s
    cov = (q.T @ q) / len(points)
    # All voxels carry equal support mass; not an intensity-weighted cell center.
    interior = points[np.argmin(np.square(q).sum(axis=1))]
    return {
        'voxel_count': a.count,
        'volume_um3': float(a.count * np.prod(s)),
        'centroid_grid_zyx': center.tolist(),
        'centroid_um_from_grid_origin': (center * s).tolist(),
        'interior_nearest_centroid_grid_zyx': interior.tolist(),
        'bbox_lo_zyx': list(a.lo), 'bbox_hi_exclusive_zyx': a.hi.tolist(),
        'bbox_extent_um_zyx': ((a.hi - a.lo) * s).tolist(),
        'covariance_um2': cov.tolist(),
        'occupancy': float(a.count / np.prod(a.mask.shape)),
    }


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    """Stable under arbitrary positive ID renaming; 0 remains background."""
    x = np.asarray(labels)
    if x.dtype.kind not in 'iu' or (x < 0).any():
        raise ValueError('integer nonnegative labels required')
    vals, first, inverse = np.unique(x, return_index=True, return_inverse=True)
    rank = np.zeros(len(vals), np.int64)
    positive = np.flatnonzero(vals != 0)
    order = positive[np.argsort(first[positive], kind='stable')]
    rank[order] = np.arange(1, len(order) + 1)
    return rank[inverse].reshape(x.shape)


def bbox_iou(a: Instance, b: Instance, source_shift=(0, 0, 0)) -> float:
    _same_grid(a, b)
    shift = _int3(source_shift, 'source shift')
    lo = np.maximum(np.asarray(a.lo) + shift, b.lo)
    hi = np.minimum(a.hi + shift, b.hi)
    intersection = int(np.prod(np.maximum(hi - lo, 0)))
    va, vb = int(np.prod(a.mask.shape)), int(np.prod(b.mask.shape))
    return intersection / (va + vb - intersection)


def overlap(a: Instance, b: Instance, source_shift=(0, 0, 0)) -> dict:
    """Actual support overlap with SOURCE integer translation; no np.roll."""
    _same_grid(a, b)
    shift = _int3(source_shift, 'source shift')
    alo, blo = np.asarray(a.lo) + shift, np.asarray(b.lo)
    lo, hi = np.maximum(alo, blo), np.minimum(a.hi + shift, b.hi)
    intersection = 0
    if (hi > lo).all():
        sa = tuple(slice(int(x), int(y)) for x, y in zip(lo - alo, hi - alo))
        sb = tuple(slice(int(x), int(y)) for x, y in zip(lo - blo, hi - blo))
        intersection = int(np.logical_and(a.mask[sa], b.mask[sb]).sum())
    union = a.count + b.count - intersection
    return {'intersection_voxels': intersection, 'union_voxels': union,
            'iou': intersection / union,
            'source_covered': intersection / a.count,
            'target_covered': intersection / b.count}


def union_instance(a: Instance, b: Instance, *, max_bbox_voxels=4_000_000) -> Instance:
    _same_grid(a, b)
    if a.key.frame != b.key.frame or a.key == b.key:
        raise ValueError('union requires distinct hypotheses in the same frame')
    lo, hi = np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi)
    size = math.prod(int(v) for v in hi - lo)
    if size > max_bbox_voxels:
        raise MemoryError('reference union bbox exceeds cap')
    out = np.zeros(tuple(hi - lo), bool)
    for part in (a, b):
        sl = tuple(slice(int(x), int(y)) for x, y in zip(np.asarray(part.lo) - lo, part.hi - lo))
        out[sl] |= part.mask
    return Instance(Key(a.key.dataset, a.key.frame,
                        f'union({a.key.producer}:{a.key.label_id},{b.key.producer}:{b.key.label_id})', 1),
                    tuple(lo), out, a.grid_id)


def division_features(parent: Instance, a: Instance, b: Instance,
                      spacing=(1., 1., 1.), source_shift=(0, 0, 0)) -> dict:
    """Soft image-support features; not a division classifier or hard mass rule."""
    _same_grid(parent, a)
    _same_grid(parent, b)
    if a.key.frame != parent.key.frame + 1 or b.key.frame != a.key.frame:
        raise ValueError('division observations must be consecutive frames')
    if a.key == b.key or overlap(a, b)['intersection_voxels']:
        raise ValueError('two selected daughter masks must be distinct and disjoint')
    u = union_instance(a, b)
    stats = overlap(parent, u, source_shift)
    s = _spacing(spacing)
    shift = _int3(source_shift, 'source shift')
    p, ca, cb = parent.coordinates().mean(0), a.coordinates().mean(0), b.coordinates().mean(0)
    bary = (a.count * ca + b.count * cb) / (a.count + b.count)
    stats.update(volume_ratio=(a.count + b.count) / parent.count,
                 sister_distance_um=float(np.linalg.norm((ca - cb) * s)),
                 parent_to_union_center_um=float(np.linalg.norm((bary - p - shift) * s)))
    return stats


def grid_to_native(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    p, a = np.asarray(points, dtype=float), np.asarray(affine, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or a.shape != (4, 4):
        raise ValueError('expected points[N,3] and affine[4,4]')
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError('finite coordinates required')
    if not np.allclose(a[3], [0, 0, 0, 1]) or abs(np.linalg.det(a[:3, :3])) < 1e-12:
        raise ValueError('invertible affine voxel-center mapping required')
    return p @ a[:3, :3].T + a[:3, 3]


def sparse_link_label(source_gt: int | None, target_gt: int | None,
                      gt_edges: Iterable[tuple[int, int]]) -> int:
    """Conservative TRAINING support: +1, 0, or -1 unknown. NOT scorer FP logic."""
    edges = set(gt_edges)
    if source_gt is not None and target_gt is not None and (source_gt, target_gt) in edges:
        return 1
    known_parents = {s for s, t in edges if target_gt is not None and t == target_gt}
    if known_parents and source_gt not in known_parents:
        return 0
    children = {t for s, t in edges if source_gt is not None and s == source_gt}
    if len(children) >= 2 and target_gt not in children:
        return 0
    return -1


def storage_gib(clips: int, frames: int, shape: Iterable[int], bytes_per_voxel=2) -> float:
    dims = _int3(shape, 'shape')
    if any(type(x) is not int or x <= 0 for x in (clips, frames, bytes_per_voxel)) or (dims <= 0).any():
        raise ValueError('positive integer counts required')
    return clips * frames * math.prod(int(x) for x in dims) * bytes_per_voxel / 2**30


def score_for_division_counts(adjusted_edge: float, tp: int, fp: int, fn: int) -> float:
    if not math.isfinite(adjusted_edge) or adjusted_edge < 0:
        raise ValueError('valid adjusted edge score required')
    if any(type(x) is not int or x < 0 for x in (tp, fp, fn)):
        raise ValueError('nonnegative integer counts required')
    den = tp + fp + fn
    return adjusted_edge + (0.1 * tp / den if den else 0.0)
