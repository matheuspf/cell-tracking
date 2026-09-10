"""Fixed-node atomic edit reference; not a learned policy or production decoder."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping
import math

Edge = tuple[int, int]


def integer(v: object) -> int:
    if type(v) is not int or v < 0:
        raise ValueError('nonnegative Python integer required')
    return v


def edges_checked(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    result = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError('edge needs two IDs')
        result.append((integer(edge[0]), integer(edge[1])))
    if len(result) != len(set(result)):
        raise ValueError('duplicate edges')
    return tuple(sorted(result))


@dataclass(frozen=True)
class Graph:
    """Only IDs, times and edges. Coordinates are unchanged outside this helper."""
    node_times: tuple[tuple[int, int], ...]
    edges: tuple[Edge, ...]

    @classmethod
    def build(cls, node_times: Mapping[int, int], edges: Iterable[Edge]) -> 'Graph':
        times = tuple(sorted((integer(n), integer(t)) for n,t in node_times.items()))
        pairs = edges_checked(edges)
        tmap = dict(times)
        incoming, outgoing = {}, {}
        for a,b in pairs:
            if a not in tmap or b not in tmap:
                raise ValueError('dangling endpoint')
            if tmap[b] != tmap[a] + 1:
                raise ValueError('edges must connect consecutive frames')
            incoming[b] = incoming.get(b,0) + 1
            outgoing[a] = outgoing.get(a,0) + 1
            if incoming[b] > 1 or outgoing[a] > 2:
                raise ValueError('merge or outdegree above two')
        return cls(times, pairs)


def apply_atomic(graph: Graph, remove: Iterable[Edge], add: Iterable[Edge],
                 editable_nodes: Iterable[int], frozen_edges: Iterable[Edge] = ()) -> Graph:
    # Validate even manually constructed Graph objects before accepting an edit.
    if len(dict(graph.node_times)) != len(graph.node_times):
        raise ValueError('duplicate node IDs')
    original = Graph.build(dict(graph.node_times), graph.edges)
    r, a = set(edges_checked(remove)), set(edges_checked(add))
    old, frozen = set(original.edges), set(edges_checked(frozen_edges))
    allowed = set(integer(n) for n in editable_nodes)
    if not allowed <= set(dict(original.node_times)):
        raise ValueError('unknown editable node')
    if not frozen <= old:
        raise ValueError('frozen edge absent from incumbent')
    if r & a or not r <= old or a & old:
        raise ValueError('overlapping, nonexistent or already-present edit')
    if r & frozen:
        raise ValueError('fixed boundary edge cannot be removed')
    if any(x not in allowed for e in r|a for x in e):
        raise ValueError('edit crosses declared boundary')
    return Graph.build(dict(original.node_times), (old-r)|a)


def protected_external_owners(parent: int, daughters: Iterable[int],
                              incoming: Mapping[int,int], evidence: Mapping[Edge,Mapping],
                              probability_floor: float=.98, distance_ceiling_um: float=2.) -> set[int]:
    """Per-owner protection. Missing evidence abstains; free daughters don't mask it.

    This is a conservative boundary guard, not a rule that donor reassignment can
    never occur. A joint decoder can propose a separately scored donor alternative.
    """
    if not 0 <= probability_floor <= 1 or not math.isfinite(distance_ceiling_um) or distance_ceiling_um < 0:
        raise ValueError('invalid protection thresholds')
    protected = set()
    for daughter in daughters:
        owner = incoming.get(daughter)
        if owner is None or owner == parent:
            continue
        item = evidence.get((owner,daughter))
        if item is None or item.get('probability') is None or item.get('distance_um') is None:
            protected.add(owner)
            continue
        p,d = float(item['probability']), float(item['distance_um'])
        if not math.isfinite(p) or not math.isfinite(d) or not 0 <= p <= 1 or d < 0:
            raise ValueError('invalid owner evidence')
        if p >= probability_floor and d <= distance_ceiling_um:
            protected.add(owner)
    return protected
