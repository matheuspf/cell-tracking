#!/usr/bin/env python3
"""Strict count arithmetic after official matching, never a graph evaluator."""
from __future__ import annotations
import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping

COUNTS = ('edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes')


def count(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name}: boolean is not a count')
    try:
        x = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f'{name}: invalid count') from exc
    if not x.is_finite() or x < 0 or x != x.to_integral_value():
        raise ValueError(f'{name}: expected nonnegative integer')
    return int(x)


def finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name}: boolean is not numeric')
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'{name}: nonfinite')
    return x


def summary(rows: Iterable[Mapping], expected: Iterable[str], alpha: float = .1) -> dict:
    rows, expected = list(rows), list(expected)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('expected sample list must be nonempty and unique')
    names = [r['dataset'] for r in rows]
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError('duplicate, missing or extra sample')
    variants = {r.get('variant') for r in rows}
    if len(variants) != 1:
        raise ValueError('mixed variants')
    alpha = finite(alpha, 'alpha')
    if alpha < 0:
        raise ValueError('alpha must be nonnegative')
    totals = dict.fromkeys(COUNTS, 0)
    weighted_adjusted, total_weight, n_adj = 0., 0, 0
    for r in rows:
        c = {k: count(r[k], k) for k in COUNTS}
        estimate = finite(r['estimated_total'], 'estimated_total')
        if estimate <= 0:
            raise ValueError('positive estimated_total required; never impute GT count')
        for k in COUNTS:
            totals[k] += c[k]
        w = c['edge_tp'] + c['edge_fp'] + c['edge_fn']
        if w:
            j = c['edge_tp'] / w
            adj = max(0., j * (1 + alpha - alpha * c['num_pred_nodes'] / estimate))
            weighted_adjusted += w * adj
            total_weight += w
            n_adj += 1
    if not total_weight:
        raise ValueError('no evaluable edge denominator; score undefined')
    dd = sum(totals[k] for k in ('division_tp', 'division_fp', 'division_fn'))
    d = totals['division_tp'] / dd if dd else None
    a = weighted_adjusted / total_weight
    return dict(n=len(rows), n_adj=n_adj, **totals, edge_jaccard=totals['edge_tp'] / total_weight,
                adj_edge_jaccard=a, division_jaccard=d, score=a + (.1*d if d is not None else 0.))


def compare(base: Iterable[Mapping], candidate: Iterable[Mapping], expected: Iterable[str],
            useful_gain: float = .002, stretch_gain: float = .02, tolerance: float = 1e-10) -> dict:
    base, candidate, expected = list(base), list(candidate), list(expected)
    s0, s1 = summary(base, expected), summary(candidate, expected)
    b = {r['dataset']: r for r in base}
    c = {r['dataset']: r for r in candidate}
    for name in expected:
        for prefix in ('edge', 'division'):
            gt0 = count(b[name][prefix+'_tp'], 'tp') + count(b[name][prefix+'_fn'], 'fn')
            gt1 = count(c[name][prefix+'_tp'], 'tp') + count(c[name][prefix+'_fn'], 'fn')
            if gt0 != gt1:
                raise ValueError('ground-truth event denominator changed')
        if finite(b[name]['estimated_total'], 'estimate') != finite(c[name]['estimated_total'], 'estimate'):
            raise ValueError('total estimate changed')
        if not b[name].get('embryo') or b[name]['embryo'] != c[name].get('embryo'):
            raise ValueError('missing or changed embryo assignment')
    deltas = {}
    for embryo in sorted({r['embryo'] for r in base}):
        ids = [n for n in expected if b[n]['embryo'] == embryo]
        a = summary([b[n] for n in ids], ids)
        z = summary([c[n] for n in ids], ids)
        deltas[embryo] = z['score'] - a['score']
    delta = s1['score'] - s0['score']
    eligible = delta > tolerance and all(d >= -tolerance for d in deltas.values())
    decision = ('stretch_local_gain' if delta >= stretch_gain else
                'useful_local_gain' if delta >= useful_gain else 'small_local_gain') if eligible else 'retain_incumbent'
    return dict(baseline=s0, candidate=s1, delta=delta, embryo_deltas=deltas,
                eligible_by_counts_only=eligible, decision=decision,
                caveat='Also require provenance, fresh-matching, full-inference and integrity checks; not a statistical guarantee.')


def scenarios() -> list[dict]:
    measured = json.loads(Path(__file__).with_name('measured_baseline.json').read_text())['incumbent']
    gt = measured['division_tp'] + measured['division_fn']
    result = []
    for tp, fp in [(29,92), (29,0), (60,60), (75,50), (80,40)]:
        d = tp / (gt + fp)
        s = measured['adj_edge_jaccard'] + .1*d
        result.append(dict(division_tp=tp, division_fp=fp, division_fn=gt-tp, division_jaccard=d,
                           score=s, delta=s-measured['score'], assumption='adjusted edge score held fixed; hypothetical'))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('scenarios')
    q = sub.add_parser('compare')
    q.add_argument('--baseline', type=Path, required=True)
    q.add_argument('--candidate', type=Path, required=True)
    q.add_argument('--expected', type=Path, required=True, help='JSON array from locked sample manifest')
    args = p.parse_args()
    if args.command == 'scenarios':
        result = scenarios()
    else:
        def read_csv(path):
            with path.open(newline='') as f:
                return list(csv.DictReader(f))
        result = compare(read_csv(args.baseline), read_csv(args.candidate), json.loads(args.expected.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
