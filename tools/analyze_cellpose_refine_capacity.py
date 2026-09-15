"""Evaluation-only integer-feasible recall capacity of the fixed proposal bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist

from tools.cellpose_refine.common import (EMBRYOS, RESULTS, SCREEN, SPACING, WORK,
                                         config, now, sha, write_json)

RADII = (1, 2, 3, 7)
CAPS = (100, 200, 400)


def feasible_distances(base, targets, shape, movement=3., max_radius=7.):
    """Each entry is the closest integer center reachable by that one proposal."""
    base, targets = np.asarray(base), np.asarray(targets)
    assert np.array_equal(base, np.rint(base)) and np.array_equal(targets, np.rint(targets))
    maxima = np.floor(movement / SPACING).astype(int)
    offsets = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in maxima], indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = offsets[np.square(offsets * SPACING).sum(axis=1) <= movement**2 + 1e-9]
    distances = cdist(targets * SPACING, base * SPACING)
    result = np.full(distances.shape, np.inf)
    eligible = np.argwhere(distances <= movement + max_radius + 1e-9)
    for start in range(0, len(eligible), 256):
        g, p = eligible[start:start + 256].T
        options = base[p, None, :] + offsets[None, :, :]
        inside = ((options >= 0) & (options < np.asarray(shape))).all(axis=-1)
        squared = np.square((options - targets[g, None, :]) * SPACING).sum(axis=-1)
        squared[~inside] = np.inf
        result[g, p] = np.sqrt(squared.min(axis=1))
    return result


def capacity(distances, radius):
    if not all(distances.shape):
        return 0
    matched = maximum_bipartite_matching(csr_matrix(distances <= radius + 1e-9), perm_type="column")
    return int((matched >= 0).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    cfg = config()
    recipe = {"created_utc": now(), "script_sha256": sha(__file__), "movement_um": cfg["movement_limit_um"],
              "radii_um": RADII, "absolute_candidate_caps": CAPS,
              "purpose": "Diagnostic upper bound only; opens GT after the four-fit evaluation exists"}
    plan_path = RESULTS / "integer-capacity-plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        assert old["script_sha256"] == recipe["script_sha256"]
    else:
        write_json(plan_path, recipe)
    metrics_path = RESULTS / "metrics.json"
    while args.wait and not metrics_path.exists():
        time.sleep(15)
    metrics = json.loads(metrics_path.read_text())
    lock_path = WORK / "predictions-lock.json"
    lock = json.loads(lock_path.read_text())
    assert metrics["evaluation_opened"]["predictions_lock_sha256"] == sha(lock_path)
    assert len(lock["models"]) == 4
    panel_bytes = (SCREEN / "panel.json").read_bytes()
    gt_bytes = (SCREEN / "evaluation/ground_truth.json").read_bytes()
    assert hashlib.sha256(panel_bytes + gt_bytes).hexdigest() == metrics["data_fingerprint"]
    gt = json.loads(gt_bytes)
    rows = []
    for frame in json.loads(panel_bytes)["frames"]:
        if frame["role"] != "assessment":
            continue
        truth = np.asarray(gt["frames"][frame["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
        with np.load(SCREEN / "predictions/cellpose_cpdino_vitb" / (frame["key"] + ".npz"), allow_pickle=False) as p:
            base, scores = np.rint(p["centers_zyx"]).astype(np.float64), p["scores"]
        feasible = feasible_distances(base, truth[:, 2:], frame["shape"], cfg["movement_limit_um"])
        relaxed = cdist(truth[:, 2:] * SPACING, base * SPACING)
        order = np.argsort(-scores, kind="stable")
        rows.append({"key": frame["key"], "embryo": frame["embryo"], "gt": len(truth),
                     "integer_capacity": {str(r): capacity(feasible, r) for r in RADII},
                     "continuous_relaxation": {str(r): capacity(relaxed, r + cfg["movement_limit_um"]) for r in RADII},
                     "budgets": {str(k): {str(r): capacity(feasible[:, order[:k]], r) for r in (3, 7)} for k in CAPS}})
    groups = []
    for embryo in (*EMBRYOS, "pooled"):
        selected = [r for r in rows if embryo == "pooled" or r["embryo"] == embryo]
        count = sum(r["gt"] for r in selected)
        best = {str(r): sum(row["integer_capacity"][str(r)] for row in selected) for r in RADII}
        relaxed = {str(r): sum(row["continuous_relaxation"][str(r)] for row in selected) for r in RADII}
        budgets = {str(k): {str(r): sum(row["budgets"][str(k)][str(r)] for row in selected) for r in (3, 7)} for k in CAPS}
        gap = {}
        for seed, values in metrics["seeds"].items():
            actual = next(g for g in values["integer"] if g["embryo"] == embryo)
            gaps = {str(r): best[str(r)] - actual["matches"][f"{r}.0"] for r in RADII}
            assert min(gaps.values()) >= 0
            gap[seed] = gaps
        groups.append({"embryo": embryo, "gt": count, "integer_capacity": best,
                       "integer_capacity_recall": {r: n / count for r, n in best.items()},
                       "continuous_relaxation": relaxed, "integer_loss_from_relaxation": {r: relaxed[r] - best[r] for r in best},
                       "budgets": budgets, "remaining_count_gap_by_seed": gap})
    write_json(WORK / "evaluation/integer-capacity-frames.json", rows)
    write_json(RESULTS / "integer-capacity.json", {"created_utc": now(), "groups": groups,
        "plan_sha256": sha(plan_path), "metrics_sha256": sha(metrics_path), "data_fingerprint": metrics["data_fingerprint"],
        "interpretation": "Per-proposal feasible integer destinations plus maximum-cardinality matching give an optimistic recall bound at fixed candidate count/rank and 3um movement. It ignores biological proposal identity, learned decisions and the official distance-weight preference. It is neither a predicted gain nor a bound on the competition graph score."})
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
