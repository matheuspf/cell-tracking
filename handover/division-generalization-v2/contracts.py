#!/usr/bin/env python3
"""Planning contracts, not a trainer or an independent experiment validator."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

P0 = 0.934864986413134
P0_DIV_TP, P0_DIV_FP, DIV_TRUTH = 29, 92, 151
P0_EDGE = P0 - 0.1 * P0_DIV_TP / (DIV_TRUTH + P0_DIV_FP)


def integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def lr_factor(step: int, total: int = 4096, warmup: int = 128,
              decay_start: int = 3277, floor: float = 0.1) -> float:
    """Zero-based optimizer step: warmup, plateau, then non-overlapping cosine."""
    for name, value in [('step', step), ('total', total), ('warmup', warmup),
                        ('decay_start', decay_start)]:
        integer(value, name)
    if not 0 < warmup <= decay_start < total - 1 or not 0 <= step < total:
        raise ValueError('Invalid schedule interval; do not shrink a production run')
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError('Invalid learning-rate floor')
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / (total - 1 - decay_start)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def required_divisions(target: float = 0.95, false_positives: int = 92,
                       edge_contribution: float = P0_EDGE) -> dict:
    """Count-only illustration: unchanged adjusted edge contribution, not a forecast."""
    integer(false_positives, 'false_positives')
    if not math.isfinite(target) or not math.isfinite(edge_contribution):
        raise ValueError('Scores must be finite')
    needed = max(0, math.ceil((target - edge_contribution)
                             * (DIV_TRUTH + false_positives) / 0.1 - 1e-12))
    return dict(required_tp=needed, additional_tp=needed-P0_DIV_TP,
                feasible_by_counts=needed <= DIV_TRUTH,
                score_at_required_tp=(edge_contribution + 0.1*needed
                                      / (DIV_TRUTH+false_positives)
                                      if needed <= DIV_TRUTH else None),
                scope='Count-only; edge contribution held fixed; not repaired graphs')


def training_receipt_errors(receipt: dict) -> list[str]:
    """Reject a pilot masquerading as a completed training test; verify evidence locally."""
    errors = []
    for key, minimum in [('joint_optimizer_updates', 4096),
                         ('positive_groups_seen', 1), ('negative_only_groups_seen', 1)]:
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f'{key}: require integer >= {minimum}')
    for key in ['full_source_screen', 'literal_zero_path_test',
                'counterfactual_label_parity', 'finite_gradient_checks',
                'random_background_stream', 'checkpoint_resume_parity']:
        if receipt.get(key) is not True:
            errors.append(f'{key}: measured evidence required')
    unknown = receipt.get('unknown_as_negative_count')
    if isinstance(unknown, bool) or not isinstance(unknown, int) or unknown != 0:
        errors.append('Unknown sparse hypotheses must not become negative labels')
    if receipt.get('new_target_used_for_training') is not False:
        errors.append('Target-outcome fitting is forbidden')
    return errors


def study_errors(study: dict) -> list[str]:
    errors = []
    if study.get('status') != 'planned_not_executed':
        errors.append('The delivered handover must not claim executed results')
    for key in ['base_commit', 'base_tree', 'metric_revision']:
        value = study.get(key, '')
        if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
            errors.append(f'Invalid pinned SHA: {key}')
    train = study.get('training', {})
    if train.get('joint_updates') != 4096 or train.get('automatic_budget_shrink') is not False:
        errors.append('Joint training floor or no-shrink contract changed')
    try:
        schedule = [lr_factor(i, train['joint_updates'], train['warmup_updates'],
                              train['decay_start']) for i in range(train['joint_updates'])]
        if max(schedule) != 1.0 or not all(math.isfinite(x) and x > 0 for x in schedule):
            errors.append('Learning-rate schedule failed')
    except (KeyError, ValueError, TypeError) as exc:
        errors.append(f'Invalid schedule: {exc}')
    resources = study.get('resources', {})
    if resources.get('gpu_total_gib', math.inf) > 20 or resources.get('rss_gib', math.inf) > 50:
        errors.append('Memory envelope exceeds supplied hardware allowance')
    if resources.get('total_cpu_threads', math.inf) > 16:
        errors.append('CPU thread envelope exceeds 16')
    if sum(resources.get('lease_hour_reservations', {}).values()) != resources.get('lease_hours_cap'):
        errors.append('Resource reservations do not reconcile')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path(__file__).with_name('study.json'))
    parser.add_argument('--training-receipt', type=Path)
    args = parser.parse_args()
    try:
        errors = study_errors(json.loads(args.study.read_text(encoding='utf-8')))
        if args.training_receipt:
            errors.extend(training_receipt_errors(json.loads(args.training_receipt.read_text(encoding='utf-8'))))
        result = dict(ok=not errors, errors=errors,
                      score_illustrations={str(fp): required_divisions(false_positives=fp)
                                           for fp in [92, 102, 112, 132]},
                      scope='Plan/receipt-schema checks only; no data, model or scorer execution')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = dict(ok=False, errors=[str(exc)])
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
