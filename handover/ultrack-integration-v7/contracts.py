"""Small observation-ancestry contract, NOT a competition scorer or exporter."""
from __future__ import annotations
import math
from numbers import Real, Integral
from typing import Iterable, Mapping, Any


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite integer")
    if not isinstance(value, Integral) and abs(value) >= 2**53:
        raise ValueError(f"{name} is too large for a floating-point identifier")
    out = int(value)
    if out != value:
        raise ValueError(f"{name} must be integral")
    return out


def validate_observations(rows: Iterable[Mapping[str, Any]], shape: tuple[int, ...],
                          *, integer_coordinates: bool = True) -> dict:
    """Use actual id/parent_id, ignoring track_id. Coordinates are native voxels.

    This checks topology/coordinates only. Real mask ownership, hierarchy
    exclusion and inside-mask support require the maintained region contracts.
    """
    if len(shape) != 4 or any(integer(x, "shape") <= 0 for x in shape):
        raise ValueError("Expected a positive TZYX shape")
    observations = {}
    for row in rows:
        missing = set(("id", "parent_id", "t", "z", "y", "x")) - row.keys()
        if missing:
            raise ValueError(f"Missing observation fields: {sorted(missing)}")
        node_id, parent = integer(row["id"], "id"), integer(row["parent_id"], "parent_id")
        if node_id < 0 or parent < -1 or node_id in observations:
            raise ValueError("Invalid/duplicate observation ID or parent sentinel")
        t = integer(row["t"], "t")
        if not 0 <= t < shape[0]:
            raise ValueError("Time outside clip")
        for name, bound in zip(("z", "y", "x"), shape[1:]):
            val = row[name]
            if isinstance(val, bool) or not isinstance(val, Real) or not math.isfinite(val) or not 0 <= val < bound:
                raise ValueError("Coordinate outside native volume or not finite")
            if integer_coordinates:
                integer(val, name)
        observations[node_id] = (parent, t)
    children = {node_id: [] for node_id in observations}
    edges = []
    for node_id, (parent, t) in observations.items():
        if parent == -1:
            continue
        if parent not in observations:
            raise ValueError("Selected child references absent/unselected parent")
        if observations[parent][1] + 1 != t:
            raise ValueError("Edges must go from t to t+1")
        children[parent].append(node_id)
        if len(children[parent]) > 2:
            raise ValueError("More than two daughters")
        edges.append((parent, node_id))
    return {"nodes": len(observations), "edges": sorted(edges),
            "divisions": sum(len(v) == 2 for v in children.values())}
