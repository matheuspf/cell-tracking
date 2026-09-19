"""Small v11 reference contracts, not a trainer, scorer or security sandbox."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Expected a finite value')
    return value


def _labels(values: Sequence[int]) -> None:
    if any(type(value) is not int or value not in (-1, 0, 1) for value in values):
        raise ValueError('Labels must be integer -1, 0 or 1')


def occurrence_target(risks: Sequence[int], *, complete: bool = True) -> int:
    """Supported metric-risk target, NOT a biological nondivision assertion."""
    _labels(risks)
    if type(complete) is not bool:
        raise ValueError('Explicit completeness must be boolean')
    if not complete:
        return -1
    if 1 in risks:
        return 1
    return 0 if risks and all(value == 0 for value in risks) else -1


def logsumexp(values: Sequence[float]) -> float:
    values = [_finite(value) for value in values]
    if not values:
        raise ValueError('No legal actions')
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def canonical_logits(rows: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Production must canonicalize complete edits before computing features."""
    result: dict[str, float] = {}
    for key, value in rows:
        if not isinstance(key, str) or not key:
            raise ValueError('An action needs a nonempty canonical ID')
        value = _finite(value)
        if key in result and result[key] != value:
            raise ValueError('Canonical duplicate has inconsistent logits')
        result[key] = value
    return dict(sorted(result.items()))


def fork_gain(occurrence_logit: float, actions: Mapping[str, float], chosen: str,
              structural: float, no_fork: Sequence[float], margin: float) -> float:
    """Keep utility zero; conditional normalization spans every legal fork."""
    if chosen not in actions:
        raise ValueError('Chosen action is absent from the deployment bank')
    margin = _finite(margin)
    if margin < 0:
        raise ValueError('Negative safety margin')
    return (_finite(occurrence_logit) + _finite(actions[chosen])
            - logsumexp(list(actions.values())) + _finite(structural)
            - max([0.0, *[_finite(value) for value in no_fork]]) - margin)


def masked_pair_loss(logits: Sequence[float], labels: Sequence[int]) -> float:
    if len(logits) != len(labels):
        raise ValueError('Logit/label length mismatch')
    _labels(labels)
    values = [_finite(value) for value in logits]
    positive = [v for v, label in zip(values, labels) if label == 1]
    known = [v for v, label in zip(values, labels) if label >= 0]
    if not positive:
        return 0.0  # Negative-only groups train occurrence, not a fake keep label.
    return logsumexp(known) - logsumexp(positive)


def learning_rate(step: int, updates: int, base: float = 3e-4,
                  floor: float = 1e-6, warmup_cap: int = 200) -> float:
    """One-based sequential warmup then cosine; the final step reaches floor."""
    if type(step) is not int or type(updates) is not int or not 1 <= step <= updates:
        raise ValueError('Invalid step')
    if updates < 2 or type(warmup_cap) is not int or warmup_cap < 1:
        raise ValueError('Invalid schedule')
    base, floor = _finite(base), _finite(floor)
    if not 0 <= floor <= base or base == 0:
        raise ValueError('Invalid rates')
    warmup = min(updates - 1, warmup_cap, max(1, math.floor(0.05 * updates)))
    if step <= warmup:
        return base * step / warmup
    position = (step - warmup) / (updates - warmup)
    return floor + (base - floor) * (1 + math.cos(math.pi * position)) / 2


def aggregate(rows: Sequence[Mapping[str, Any]], *, empty_edge: float,
              empty_division: float) -> float:
    """Caller must verify official empty conventions; never average clip scores."""
    weighted, weight = [], 0
    divisions = [0, 0, 0]
    for row in rows:
        counts = [row[key] for key in ('edge_tp', 'edge_fp', 'edge_fn',
                                      'division_tp', 'division_fp', 'division_fn')]
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('Counts must be nonnegative integers')
        current = sum(counts[:3])
        weight += current
        weighted.append(current * _finite(row['adjusted_edge_jaccard']))
        divisions = [a + b for a, b in zip(divisions, counts[3:])]
    edge = math.fsum(weighted) / weight if weight else _finite(empty_edge)
    div = divisions[0] / sum(divisions) if sum(divisions) else _finite(empty_division)
    return edge + 0.1 * div


def audit_model_ancestry(artifacts: Mapping[str, Mapping[str, Any]], root: str,
                         source: str) -> set[str]:
    """Structural model-parent check; does not prove truth of submitted receipts."""
    if source not in ('44b6', '6bba'):
        raise ValueError('Unknown source')
    seen: set[str] = set()
    active: set[str] = set()

    def visit(key: str) -> None:
        if key in active:
            raise ValueError('Cyclic ancestry')
        if key in seen:
            return
        if key not in artifacts:
            raise ValueError('Missing parent')
        item = artifacts[key]
        if item.get('exposure') not in ('source_only', 'weight_free_code'):
            raise ValueError('Unqualified exposure')
        reads = item.get('embryos_read')
        parents = item.get('parents')
        if not isinstance(reads, list) or not isinstance(parents, list):
            raise ValueError('Missing explicit ancestry/access record')
        if any(not isinstance(p, str) for p in parents):
            raise ValueError('Invalid parent identifier')
        if not set(reads) <= {source}:
            raise ValueError('Target-embryo dependency')
        if item['exposure'] == 'weight_free_code' and reads:
            raise ValueError('Code-only artifact declares data exposure')
        digest = item.get('sha256', '')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing artifact SHA256')
        active.add(key)
        for parent in parents:
            visit(parent)
        active.remove(key)
        seen.add(key)

    visit(root)
    return seen


def validate_registry(study: Mapping[str, Any]) -> None:
    """Fail closed on accidental expansion or weakening of this fixed plan."""
    expected = {
        'id': 'division-reliability-v11',
        'base_commit': '08061f0a224387594dd3e915462bafb5d5928fb2',
        'status': 'planned_not_executed',
        'seeds': [20260918, 314159],
        'arms': ['C00', 'C01', 'C11'],
        'primary_comparison': ['C11', 'C01'],
        'upstream_update_options': [24000, 16000, 8000],
        'event_update_options': [4000, 2000],
        'margin_options': [2, 4, 6, 8],
        'max_new_neural_fits': 8,
        'parent_unique_action_cap': 4096,
        'target_selection': False,
        'operational_neural_matrix': False,
        'automatic_production_adoption': False,
        'v10_results_observed_remotely': False,
        'inference_requires_explicit_package': True,
    }
    for key, value in expected.items():
        if study.get(key) != value or type(study.get(key)) is not type(value):
            raise ValueError(f'Registry mismatch: {key}')
    if study.get('directions') != [{'source': '44b6', 'target': '6bba'},
                                    {'source': '6bba', 'target': '44b6'}]:
        raise ValueError('Both source directions are required')
    budget = study['budget']
    if budget['gpu_lease_hours'] != 72 or budget['inference_reserve_hours'] < 16:
        raise ValueError('Budget or reserve drift')
    if sum(budget['initial_allocations_hours'].values()) > budget['gpu_lease_hours']:
        raise ValueError('Overallocated budget')
    if budget['projection_multiplier'] < 1.25:
        raise ValueError('Missing execution margin')
    if study['occurrence_negative_rule'] != 'all_legal_forks_supported_zero':
        raise ValueError('Unsafe sparse-label rule')
    if study['raw_data_or_weights_committed'] is not False:
        raise ValueError('Large/private artifact publication is prohibited')
