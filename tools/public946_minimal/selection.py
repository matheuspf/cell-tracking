"""Frozen eligibility and ranking. Pilot scores cannot select recipes."""
from .common import ARMS


def eligible(candidate, baseline, noise, expected=199):
    reasons = []
    if len(candidate.get('completed', [])) != expected or candidate.get('failed'):
        reasons.append('incomplete_population')
    if set(candidate.get('completed', [])) != set(baseline.get('completed', [])):
        reasons.append('population_mismatch')
    if candidate.get('metric_revision') != baseline.get('metric_revision'):
        reasons.append('metric_mismatch')
    if candidate['pooled']['score'] - baseline['pooled']['score'] <= noise['score']:
        reasons.append('nonpositive_or_inconclusive_pooled_delta')
    for embryo, scores in baseline['per_embryo'].items():
        if candidate['per_embryo'][embryo]['score'] - scores['score'] <= noise['score']:
            reasons.append('nonpositive_or_inconclusive_embryo_' + embryo)
    if candidate['pooled']['division_jaccard'] - baseline['pooled']['division_jaccard'] < -noise['division_jaccard']:
        reasons.append('division_jaccard_regression')
    return dict(eligible=not reasons, reasons=reasons)


def ranking(arm, candidate, baseline, seconds):
    deltas = [candidate['per_embryo'][e]['score'] - baseline['per_embryo'][e]['score'] for e in baseline['per_embryo']]
    return (-min(deltas), -(candidate['pooled']['score'] - baseline['pooled']['score']),
            candidate.get('module_count', len(ARMS[arm]['modules'])), seconds, arm)
