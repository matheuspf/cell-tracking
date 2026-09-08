#!/usr/bin/env python3
"""Strict arithmetic helpers, NOT a replacement for official graph matching.

Python 3.12 / standard library only. Membership labels mean matched to sparse
GT, never 'real cell versus background'. Real scores require fresh official
matching and division evaluation for every graph variant first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

COUNT_KEYS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
              "division_fn", "num_pred_nodes")


def finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def count(value: object, name: str) -> int:
    result = finite(value, name)
    if result < 0 or result != math.floor(result):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(result)


def strict_summary(rows: Iterable[dict], expected_ids: Iterable[str]) -> dict:
    """Mirror positive-estimate aggregation; reject missing/duplicate samples.

    Input rows: dataset, estimated_total, and COUNT_KEYS. Zero edge-denominator
    rows carry zero edge weight. A run with no edge denominator is undefined.
    Empty division denominator drops the division term, as upstream does.
    The expected IDs must come from the locked evaluation split, not predictions.
    """
    rows = list(rows)
    expected = list(expected_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_ids must be nonempty and unique")
    ids = [r["dataset"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("scored datasets must equal expected_ids exactly")
    totals = {key: 0 for key in COUNT_KEYS}
    adjusted_numerator = 0.0
    total_weight = 0
    n_adj = 0
    details = []
    for raw in rows:
        r = {key: count(raw[key], key) for key in COUNT_KEYS}
        n_total = finite(raw["estimated_total"], "estimated_total")
        if n_total <= 0:
            raise ValueError("estimated_total must be positive; never impute GT count")
        for key in COUNT_KEYS:
            totals[key] += r[key]
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        c = r["num_pred_nodes"] / n_total
        multiplier = 1.1 - 0.1 * c
        edge = r["edge_tp"] / w if w else None
        adj = max(0.0, edge * multiplier) if edge is not None else None
        if w:
            adjusted_numerator += w * adj
            total_weight += w
            n_adj += 1
        details.append({"dataset": raw["dataset"], "count_ratio": c,
                        "multiplier": multiplier, "edge_weight": w,
                        "edge_jaccard": edge, "adj_edge_jaccard": adj})
    if total_weight == 0:
        raise ValueError("combined score undefined: no evaluable edge denominator")
    div_denom = sum(totals[k] for k in ("division_tp", "division_fp", "division_fn"))
    division = totals["division_tp"] / div_denom if div_denom else None
    adjusted = adjusted_numerator / total_weight
    return {"n": len(rows), "n_adj": n_adj, "counts": totals,
            "edge_jaccard": totals["edge_tp"] / total_weight,
            "adj_edge_jaccard": adjusted, "division_jaccard": division,
            "score": adjusted + (0.1 * division if division is not None else 0.0),
            "samples": details}


def membership_stats(labels: Iterable[int], keep: Iterable[bool]) -> dict:
    """Binary keep/annotation confusion table; unrelated to tracking-edge FP."""
    labels, keep = list(labels), list(keep)
    if not labels or len(labels) != len(keep):
        raise ValueError("nonempty equal-length labels and keep required")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("labels must be integer 0 or 1")
    if any(type(k) is not bool for k in keep):
        raise ValueError("keep must contain booleans")
    tp = sum(y == 1 and k for y, k in zip(labels, keep))
    fp = sum(y == 0 and k for y, k in zip(labels, keep))
    fn = sum(y == 1 and not k for y, k in zip(labels, keep))
    tn = len(labels) - tp - fp - fn
    ratio = lambda a, b: a / b if b else None
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    p = (tp + fn) / len(labels)
    deleted_rate = ratio(fn, fn + tn)
    return {"n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "annotation_prevalence": p, "retained_fraction": (tp + fp) / len(labels),
            "annotation_recall": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "accuracy": (tp + tn) / len(labels),
            "phi": (tp * tn - fp * fn) / math.sqrt(denom) if denom else None,
            "deleted_annotation_rate": deleted_rate,
            "deleted_rate_relative_to_population": deleted_rate / p
            if deleted_rate is not None and p else None}


def membership_curve(rows: Iterable[dict], fractions: Iterable[float], seed: int = 20260908) -> list[dict]:
    """Keep the top fraction WITHIN EACH DATASET, using label-blind tie breaks.

    Input rows: dataset, candidate_id, annotation_label (0/1), score.
    Scores must already be out-of-fold. This function cannot certify provenance.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no candidate rows")
    groups = defaultdict(list)
    seen = set()
    labels = []
    for i, r in enumerate(rows):
        key = (str(r["dataset"]), str(r["candidate_id"]))
        if key in seen:
            raise ValueError("duplicate dataset/candidate_id")
        seen.add(key)
        y = count(r["annotation_label"], "annotation_label")
        if y not in (0, 1):
            raise ValueError("annotation_label must be 0/1")
        labels.append(y)
        s = finite(r["score"], "score")
        tie = hashlib.sha256(json.dumps([seed, *key]).encode()).hexdigest()
        groups[key[0]].append((-s, tie, i))
    ranked = {key: sorted(values) for key, values in groups.items()}
    result = []
    for fraction in fractions:
        r = finite(fraction, "retained fraction")
        if not 0 <= r <= 1:
            raise ValueError("retained fraction must be in [0, 1]")
        mask = [False] * len(rows)
        for group in ranked.values():
            for _, _, i in group[:math.ceil(r * len(group))]:
                mask[i] = True
        stats = membership_stats(labels, mask)
        stats["requested_fraction"] = r
        stats["per_dataset"] = {
            name: membership_stats([labels[i] for _, _, i in group],
                                   [mask[i] for _, _, i in group])
            for name, group in ranked.items()}
        result.append(stats)
    return result


def required_tp_retention(j0: float, d1: float, c0: float, retained: float,
                          target: float, fp_to_gt: float = 0.0,
                          fp_retained: float = 1.0) -> dict:
    """Single-sample, fixed-matching sensitivity, NOT a multi-video score.

    u = TP1/TP0, b=FP0/GT_edges, v=FP1/FP0.
    J1/J0 = u*(1+b)/(1+v*b). Independent endpoint recall ~= sqrt(u).
    Coherent graph filtering need not follow the square approximation.
    d1 is assumed division Jaccard AFTER filtering.
    """
    values = {name: finite(val, name) for name, val in locals().copy().items()}
    j0, d1, c0, retained, target, fp_to_gt, fp_retained = (
        values[k] for k in ("j0", "d1", "c0", "retained", "target", "fp_to_gt", "fp_retained"))
    if not (0 < j0 <= 1 and 0 <= d1 <= 1 and c0 >= 0 and 0 <= retained <= 1
            and fp_to_gt >= 0 and 0 <= fp_retained <= 1):
        raise ValueError("invalid Jaccard, count ratio, retention, or FP parameter")
    multiplier = 1.1 - 0.1 * c0 * retained
    if multiplier <= 0:
        return {"required_tp_retention": None, "independent_node_recall": None,
                "reason": "nonpositive edge multiplier"}
    u = max(0.0, (target - 0.1 * d1) / (j0 * multiplier)
            * (1 + fp_retained * fp_to_gt) / (1 + fp_to_gt))
    return {"required_tp_retention": u, "independent_node_recall": math.sqrt(u),
            "feasible_by_tp_retention_only": u <= 1, "multiplier": multiplier,
            "assumption": "single sample, fixed matching; division after filtering supplied"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="Aggregate fresh official per-sample counts")
    summary.add_argument("--rows", type=Path, required=True)
    summary.add_argument("--expected", type=Path, required=True, help="JSON array of locked sample names")
    curve = sub.add_parser("membership", help="Evaluate OOF annotation-membership scores")
    curve.add_argument("--rows", type=Path, required=True)
    curve.add_argument("--fractions", default="1,.9,.8,.7,.5,.3,.1")
    for p in (summary, curve):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.rows.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.command == "summary":
        result = strict_summary(rows, json.loads(args.expected.read_text()))
    else:
        result = membership_curve(rows, [float(x) for x in args.fractions.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
