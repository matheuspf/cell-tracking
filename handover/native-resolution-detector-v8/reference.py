"""Reference contracts, not a detector or an official evaluator.

Grid indices locate voxel centers: native_position = origin + stride * index.
Masks identify supervised entries. Missing annotations do not imply zero targets.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import torch
import torch.nn.functional as F


def _triplet(value: Sequence[float], name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def _points(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("points must have finite shape (N,3) in ZYX order")
    return result


@dataclass(frozen=True)
class Grid:
    stride_native: tuple[float, float, float] = (1., 1., 1.)
    origin_native: tuple[float, float, float] = (0., 0., 0.)
    spacing_um: tuple[float, float, float] = (1.625, .40625, .40625)

    def __post_init__(self) -> None:
        for name, positive in (("stride_native", True), ("origin_native", False), ("spacing_um", True)):
            object.__setattr__(self, name, tuple(_triplet(getattr(self, name), name, positive)))

    def to_native(self, coordinates: np.ndarray) -> np.ndarray:
        return _points(coordinates) * self.stride_native + self.origin_native

    def from_native(self, coordinates: np.ndarray) -> np.ndarray:
        return (_points(coordinates) - self.origin_native) / self.stride_native

    def encode(self, points_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest anchor + residual in micrometers; never average collisions.

        This does not clamp to image/crop bounds. The production caller must
        validate target inclusion and carry a separate valid-core mask.
        """
        points = _points(points_native)
        indices = np.floor(self.from_native(points) + .5).astype(np.int64)
        residual_um = (points - self.to_native(indices)) * self.spacing_um
        return indices, residual_um

    def decode(self, indices: np.ndarray, residual_um: np.ndarray) -> np.ndarray:
        indices, residual_um = _points(indices), _points(residual_um)
        if indices.shape != residual_um.shape:
            raise ValueError("offset and anchor arrays must have identical shapes")
        return self.to_native(indices) + residual_um / self.spacing_um


def collision_groups(indices: np.ndarray) -> list[list[int]]:
    """Expose rows sharing one output bin; targets remain separate upstream."""
    indices = _points(indices)
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("bin indices must be integral")
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, point in enumerate(indices):
        groups.setdefault(tuple(int(x) for x in point), []).append(i)
    return [rows for rows in groups.values() if len(rows) > 1]


def support_precedence(human: np.ndarray, pseudo: np.ndarray, background: np.ndarray) -> dict[str, np.ndarray]:
    """Human > positive pseudo > background; this is NOT an object matcher.

    Real target generation must first deduplicate object hypotheses and exclude
    unsupported visible regions from background. Unknown receives no loss.
    """
    arrays = [np.asarray(x) for x in (human, pseudo, background)]
    if any(x.dtype != bool for x in arrays) or len({x.shape for x in arrays}) != 1:
        raise ValueError("support masks must be boolean with identical shapes")
    h, p, b = arrays
    p = p & ~h
    b = b & ~h & ~p
    return {"human": h.copy(), "pseudo": p, "background": b, "unknown": ~(h | p | b)}


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, support: torch.Tensor,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted mean on supported entries; unknown targets may be NaN.

    Normalize human/pseudo/background categories separately in production, then
    apply the registered category coefficients. Do NOT multiply a dense NaN loss
    by zero, and do not interpret the returned scalar as a calibrated PU risk.
    """
    if logits.shape != targets.shape or logits.shape != support.shape:
        raise ValueError("logits, targets and support must have equal shape")
    if support.dtype != torch.bool or not logits.is_floating_point() or not targets.is_floating_point():
        raise ValueError("floating predictions/targets and boolean support required")
    if weights is not None and weights.shape != logits.shape:
        raise ValueError("weights shape mismatch")
    if not bool(support.any()):
        return logits.reshape(-1)[:0].sum()
    selected = logits[support].float()
    truth = targets[support].float()
    if not bool(torch.isfinite(selected).all() and torch.isfinite(truth).all()):
        raise ValueError("supported logits/targets must be finite")
    if not bool(((truth >= 0) & (truth <= 1)).all()):
        raise ValueError("supported BCE targets must be in [0,1]")
    w = torch.ones_like(selected) if weights is None else weights[support].float()
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("supported weights must be finite and nonnegative")
    if float(w.sum()) == 0:
        return selected[:0].sum()
    return (F.binary_cross_entropy_with_logits(selected, truth, reduction="none") * w).sum() / w.sum()


def masked_offsets(prediction: torch.Tensor, targets: torch.Tensor, support: torch.Tensor,
                   beta_um: float = 1.) -> torch.Tensor:
    """Huber/Smooth-L1 on supported point vectors expressed in micrometers."""
    if prediction.shape != targets.shape or prediction.ndim < 2 or prediction.shape[-1] != 3:
        raise ValueError("offsets require equal shapes ending in ZYX=3")
    if support.shape != prediction.shape[:-1] or support.dtype != torch.bool:
        raise ValueError("one boolean support value per offset vector required")
    if not np.isfinite(beta_um) or beta_um <= 0:
        raise ValueError("beta_um must be finite and positive")
    if not bool(support.any()):
        return prediction.reshape(-1)[:0].sum()
    p, t = prediction[support].float(), targets[support].float()
    if not bool(torch.isfinite(p).all() and torch.isfinite(t).all()):
        raise ValueError("supported offsets must be finite")
    return F.smooth_l1_loss(p, t, beta=beta_um, reduction="mean")


def assert_source_isolated(root: str, assets: Mapping[str, Mapping], source: str, target: str) -> set[str]:
    """Fail closed on declared provenance; claims still require actual receipts.

    Every node records provenance, evidence, biohub_training_embryos,
    biohub_calibration_embryos and parents. 'Training' includes unsupervised data
    and pseudo-label/feature construction, not just explicitly supervised labels.
    Passing this metadata checker is not proof that a missing receipt exists.
    """
    if not source or not target or source == target:
        raise ValueError("distinct source and target embryos required")
    permitted = {"random_initialization", "verified_source_only", "verified_external_no_target_exposure"}
    visiting, visited = set(), set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"provenance cycle: {name}")
        if name in visited:
            return
        if name not in assets:
            raise ValueError(f"missing dependency: {name}")
        node = assets[name]
        required = {"provenance", "evidence", "biohub_training_embryos", "biohub_calibration_embryos", "parents"}
        if not required.issubset(node):
            raise ValueError(f"incomplete provenance: {name}")
        if node["provenance"] not in permitted or not isinstance(node["evidence"], str) or not node["evidence"]:
            raise ValueError(f"unverified dependency: {name}")
        for key in ("biohub_training_embryos", "biohub_calibration_embryos", "parents"):
            if not isinstance(node[key], (list, tuple)) or any(not isinstance(x, str) or not x for x in node[key]):
                raise ValueError(f"invalid provenance list: {name}/{key}")
        used = set(node["biohub_training_embryos"]) | set(node["biohub_calibration_embryos"])
        if not used.issubset({source}):
            raise ValueError(f"non-source embryo in training/calibration: {name}: {sorted(used)}")
        visiting.add(name)
        for parent in node["parents"]:
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    visit(root)
    return visited


def budget_indices(scores: np.ndarray, frames: np.ndarray, identifiers: np.ndarray, k: int) -> np.ndarray:
    """Diagnostic top-K genuine candidates per frame; no GT and no point filling."""
    scores, frames, identifiers = map(np.asarray, (scores, frames, identifiers))
    if scores.ndim != 1 or frames.shape != scores.shape or identifiers.shape != scores.shape:
        raise ValueError("scores, frames and IDs must be equal-length vectors")
    if not np.isfinite(scores).all() or frames.dtype.kind not in "iu" or identifiers.dtype.kind not in "iu":
        raise ValueError("finite scores and integer frame/ID vectors required")
    if (frames < 0).any() or type(k) is not int or k < 0:
        raise ValueError("nonnegative frames and integer K required")
    if len(set(zip(frames.tolist(), identifiers.tolist()))) != len(scores):
        raise ValueError("duplicate frame/candidate identity")
    selected = []
    for frame in np.unique(frames):
        rows = np.flatnonzero(frames == frame)
        order = np.lexsort((identifiers[rows], -scores[rows]))
        selected.extend(rows[order[:k]].tolist())
    return np.asarray(selected, dtype=np.int64)
