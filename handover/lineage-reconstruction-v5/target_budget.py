#!/usr/bin/env python3
"""Calculated division-only score requirements. Not a graph evaluator or forecast."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def scenarios(baseline, target=None):
    target = float(baseline['target_score'] if target is None else target)
    score = float(baseline['score'])
    if not math.isfinite(target) or not math.isfinite(score):
        raise ValueError('Finite target and baseline required')
    c = baseline['counts']
    for key in ('division_tp', 'division_fp', 'division_fn'):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError('Division counts must be nonnegative integers')
    gt = c['division_tp'] + c['division_fn']
    if not gt:
        raise ValueError('No annotated divisions; this sensitivity is undefined')
    observed = c['division_tp'] / (gt + c['division_fp'])
    adjusted = score - .1 * observed
    if not math.isclose(adjusted, baseline['adjusted_edge_contribution'], abs_tol=1e-12):
        raise ValueError('Baseline components disagree')
    required_d = max(0., (target - adjusted) / .1)
    required_tp = math.ceil(required_d * (gt + c['division_fp']) - 1e-12)
    cases = [(c['division_tp'], c['division_fp']), (66, 92), (60, 60), (70, 70)]
    rows = []
    for tp, fp in cases:
        if tp > gt:
            continue
        s = adjusted + .1 * tp / (gt + fp)
        rows.append({'division_tp': tp, 'division_fp': fp, 'division_fn': gt-tp,
                     'calculated_score': s, 'delta': s-score})
    return {'type': 'conditional_arithmetic_not_prediction', 'baseline_score': score,
            'target': target, 'gap': target-score, 'assumption': 'adjusted edge contribution unchanged',
            'adjusted_edge_contribution': adjusted, 'required_division_jaccard': required_d,
            'required_tp_at_unchanged_fp': required_tp,
            'feasible_at_unchanged_fp': required_tp <= gt,
            'additional_true_divisions_at_unchanged_fp': max(0, required_tp-c['division_tp']),
            'cases': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path(__file__).with_name('baseline.json'))
    p.add_argument('--target', type=float)
    args = p.parse_args()
    result = scenarios(json.loads(args.baseline.read_text()), args.target)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
