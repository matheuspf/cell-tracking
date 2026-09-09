#!/usr/bin/env python3
"""Score arithmetic from recorded v1 aggregates; NOT a graph evaluator.

Division scenarios hold the adjusted edge term fixed. Filtering scenarios assume
uniform node retention and uniform TP retention in EVERY sample, unchanged FP
counts and divisions, fresh matching having no additional effects, and positive
unclipped count multipliers. Association scenarios report RAW edge Jaccard only.
These scenarios quantify requirements, not achievable results or leaderboard gain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must not be boolean')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def division_score(adjusted_edge: float, gt_divisions: int, tp: int, fp: int) -> dict:
    a = finite(adjusted_edge, 'adjusted_edge')
    g = nonnegative_int(gt_divisions, 'gt_divisions')
    t = nonnegative_int(tp, 'tp')
    f = nonnegative_int(fp, 'fp')
    if a < 0 or t > g:
        raise ValueError('negative edge contribution or TP greater than GT divisions')
    d = t / (g + f) if g + f else None
    return {'division_tp': t, 'division_fp': f, 'division_fn': g-t,
            'division_jaccard': d, 'score': a + (0.1*d if d is not None else 0.0),
            'assumption': 'adjusted edge contribution unchanged; counts are scenarios'}


def filtering_requirement(raw_edge: float, adjusted_edge: float,
                          keep: float, target_gain: float = 0.0) -> dict:
    j, a, r, gain = [finite(v, k) for v, k in
                    [(raw_edge, 'raw_edge'), (adjusted_edge, 'adjusted_edge'),
                     (keep, 'keep'), (target_gain, 'target_gain')]]
    if not 0 < j <= 1 or not 0 < a <= 1.1*j + 1e-12 or not 0 <= r <= 1 or gain < 0:
        raise ValueError('invalid Jaccard, positive unclipped adjustment, keep or gain')
    # From exact sample-weighted aggregation under the stated assumptions:
    # A(r) = 1.1*J - r*(1.1*J - A(1)). No global count-ratio substitution.
    k = max(0.0, 1.1*j-a)
    edge_at_fixed_counts = a + (1-r)*k
    u = (a+gain) / edge_at_fixed_counts
    return {'keep': r, 'target_score_gain': gain,
            'count_only_max_gain_under_assumptions': (1-r)*k,
            'required_uniform_tp_retention': u,
            'independent_endpoint_recall_approximation': math.sqrt(u),
            'feasible_by_retention_alone': u <= 1.0+1e-12,
            'assumption': 'same fractional TP/node retention in every sample; fixed FP/divisions; no clipping/rematching effects'}


def repaired_edge_jaccard(tp: int, fp: int, fn: int, repairs: int) -> dict:
    t, f, n, x = [nonnegative_int(v, k) for v, k in
                  [(tp,'tp'),(fp,'fp'),(fn,'fn'),(repairs,'repairs')]]
    if x > min(f,n):
        raise ValueError('each repair needs one original FP and one missing GT edge')
    denominator = t+f+n-x
    if not denominator:
        raise ValueError('undefined edge Jaccard')
    return {'corrected_wrong_links': x, 'edge_tp': t+x, 'edge_fp': f-x,
            'edge_fn': n-x, 'raw_edge_jaccard': (t+x)/denominator,
            'assumption': 'each repair replaces one FP by one missing GT edge; not an adjusted score'}


def generate(baseline: dict) -> dict:
    b = baseline['public_pooled']
    j = b['edge_tp']/(b['edge_tp']+b['edge_fp']+b['edge_fn'])
    d = b['division_tp']/(b['division_tp']+b['division_fp']+b['division_fn'])
    a = b['adjusted_edge_jaccard']
    if not math.isclose(j,b['edge_jaccard'],rel_tol=1e-12):
        raise ValueError('baseline edge counts disagree')
    s = a + .1*d
    if not math.isclose(s,b['score'],rel_tol=1e-12):
        raise ValueError('baseline score disagrees')
    g = b['division_tp']+b['division_fn']
    divisions = []
    for tp, fp in [(23,97),(23,0),(50,70),(60,60),(75,50),(80,40),(100,40)]:
        r = division_score(a,g,tp,fp)
        r['score_delta'] = r['score']-s
        divisions.append(r)
    filtering = [filtering_requirement(j,a,r,gain)
                 for r in [1.0,.99,.95,.9,.8,.7,.5,.3]
                 for gain in [0.0,.024]]
    links = [repaired_edge_jaccard(b['edge_tp'],b['edge_fp'],b['edge_fn'],x)
             for x in [0,500,1000,1500,2000]]
    for r in links:
        r['raw_edge_jaccard_delta'] = r['raw_edge_jaccard']-j
    return {'status':'calculated_scenarios_not_experiment_results',
            'source_commit':baseline['source_commit'], 'baseline_score':s,
            'division_scenarios':divisions, 'filtering_requirements':filtering,
            'association_scenarios':links}


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=here/'measured_baseline.json')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    result = generate(json.loads(args.baseline.read_text()))
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    else:
        print(text,end='')


if __name__=='__main__':
    main()
