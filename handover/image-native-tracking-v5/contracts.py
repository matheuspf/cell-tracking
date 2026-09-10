"""Small reference contracts for v5; not a graph scorer, trainer, or full decoder.

Standard-library only. All numerical scenarios assume unchanged adjusted-edge
contribution. Match assignment and official graph scoring must be run locally.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative Python integer")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def supported_edge_labels(
    candidate_edges: Sequence[tuple[int, int]],
    matched_gt: Mapping[int, int],
    gt_edges: Iterable[tuple[int, int]],
) -> list[int]:
    """Conservative labels on an already matched, single-dataset candidate bank.

    1: a matched GT edge. 0: a competing incoming edge when the correct incoming
    edge for that target is also represented. -1: unknown/censored. A mother with
    one known child does not prove an unrelated target is not a second daughter.
    Missing matches are omitted from matched_gt. IDs are not array row offsets.
    Inputs must already have valid, one-to-one time/space matches.
    """
    edges = [tuple(e) for e in candidate_edges]
    truth = {tuple(e) for e in gt_edges}
    for e in [*edges, *truth]:
        if len(e) != 2 or e[0] == e[1]:
            raise ValueError("edges require distinct source/target IDs")
        _integer(e[0], "source ID"); _integer(e[1], "target ID")
    if len(edges) != len(set(edges)):
        raise ValueError("duplicate candidate edge")
    if len(matched_gt.values()) != len(set(matched_gt.values())):
        raise ValueError("GT matches must be one-to-one")
    for p, g in matched_gt.items():
        _integer(p, "predicted ID"); _integer(g, "GT ID")
    gt_parents: dict[int, int] = {}
    for a, b in truth:
        if b in gt_parents and gt_parents[b] != a:
            raise ValueError("GT merge is not supported")
        gt_parents[b] = a
    positives = {(a, b) for a, b in edges
                 if a in matched_gt and b in matched_gt
                 and (matched_gt[a], matched_gt[b]) in truth}
    supported_targets = {b for _, b in positives}
    return [1 if e in positives else 0 if e[1] in supported_targets else -1
            for e in edges]


def canonical_fork(parent: int, daughters: Sequence[int]) -> tuple[int, int, int]:
    """An unordered *hypothesis* identity; says nothing about its truth."""
    _integer(parent, "parent")
    if len(daughters) != 2:
        raise ValueError("exactly two daughters required")
    a, b = daughters
    _integer(a, "daughter"); _integer(b, "daughter")
    if a == b or parent in (a, b):
        raise ValueError("fork must involve three distinct nodes")
    return (parent, min(a, b), max(a, b))


def equivalent_support_score(rows: Iterable[tuple[str, float]]) -> float:
    """Deduplicate identical predicted supports, then compute log-mean-exp.

    Call once *within* an event equivalence class. Distinct possible biological
    events must not be mixed. IDs must be label-blind predicted support identities.
    Duplicate IDs with inconsistent scores are an integrity error, not an average.
    """
    supports: dict[str, float] = {}
    for identity, score in rows:
        if not isinstance(identity, str) or not identity:
            raise ValueError("nonempty support identity required")
        score = _finite(score, "support log score")
        if identity in supports and supports[identity] != score:
            raise ValueError("same support has different scores")
        supports[identity] = score
    if not supports:
        raise ValueError("no observed support")
    values = list(supports.values()); top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values) / len(values))


def raw_fork_advantage(second_edge_reward: float, division_cost: float,
                       birth_cost: float = 0.0) -> float:
    """Local toy energy: fork versus same continuation plus independent birth.

    Positive favors a fork. Other graph terms and constraints are held equal.
    This reproduces the old dominance counterexample, not the complete solver.
    """
    r = _finite(second_edge_reward, "edge reward")
    d = _finite(division_cost, "division cost")
    b = _finite(birth_cost, "birth cost")
    return r + b - d


def validate_lineage(
    node_times: Mapping[int, int], edges: Iterable[tuple[int, int]],
    exclusion_groups: Iterable[Sequence[int]] = (),
) -> None:
    """Validate an exported selected graph. Exclusion groups refer to candidates.

    Only selected IDs belong in node_times. Degree/time validity is not evidence
    of correct biology or correct image coordinates. Coordinate checks are separate.
    """
    for node, t in node_times.items():
        _integer(node, "node ID"); _integer(t, "time")
    seen: set[tuple[int, int]] = set(); incoming: dict[int, int] = {}; outgoing: dict[int, int] = {}
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("two endpoints required")
        a, b = edge
        if a not in node_times or b not in node_times:
            raise ValueError("dangling edge")
        if (a, b) in seen:
            raise ValueError("duplicate edge")
        if node_times[b] != node_times[a] + 1:
            raise ValueError("only consecutive forward-frame links may be exported")
        seen.add((a, b)); incoming[b] = incoming.get(b, 0) + 1; outgoing[a] = outgoing.get(a, 0) + 1
        if incoming[b] > 1 or outgoing[a] > 2:
            raise ValueError("merge or more than two children")
    for group in exclusion_groups:
        group = tuple(group)
        if len(group) != len(set(group)):
            raise ValueError("duplicate ID within exclusion group")
        for node in group:
            _integer(node, "exclusion candidate ID")
        if sum(n in node_times for n in group) > 1:
            raise ValueError("incompatible observations selected together")


def division_budget(incumbent_score: float, tp: int, fp: int, fn: int,
                    target: float = 0.95) -> dict:
    """Conditional score budget, not an optimization oracle or full run aggregator."""
    score = _finite(incumbent_score, "score"); target = _finite(target, "target")
    for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
        _integer(value, key)
    if tp + fp + fn == 0 or tp + fn == 0:
        raise ValueError("defined division denominator and observed GT forks required")
    division_j = tp / (tp + fp + fn)
    adjusted_edge = score - 0.1 * division_j
    required = (target - adjusted_edge) / 0.1
    return {"incumbent": score, "target": target, "delta_required": target - score,
            "gt_divisions": tp + fn, "division_jaccard": division_j,
            "adjusted_edge_contribution": adjusted_edge,
            "required_division_jaccard_if_edges_unchanged": required,
            "required_adjusted_edge_if_divisions_unchanged": target - 0.1 * division_j,
            "division_only_arithmetically_feasible": required <= 1.0}


def division_scenario(budget: Mapping, tp: int, fp: int) -> dict:
    tp = _integer(tp, "tp"); fp = _integer(fp, "fp")
    gt = _integer(budget["gt_divisions"], "GT divisions")
    if tp > gt or gt + fp == 0:
        raise ValueError("invalid counterfactual counts")
    score = _finite(budget["adjusted_edge_contribution"], "edge contribution") + 0.1 * tp / (gt + fp)
    return {"division_tp": tp, "division_fp": fp, "division_fn": gt - tp,
            "conditional_score": score, "delta": score - budget["incumbent"],
            "assumption": "adjusted edge contribution unchanged; not a graph experiment"}
