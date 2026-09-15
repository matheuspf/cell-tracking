"""Exact adjacent-frame HOCT objective with fixed observations and optional divisions.

With upstream node cost -10, selecting an otherwise isolated node costs at most
-10 + .5 + .25 < 0. Thus all observations are selected. For adjacent-frame links,
birth/death/split terms separate by transition. Two source slots reduce this
remaining binary flow problem to minimum-cost bipartite assignment. A second
slot costs division+termination more than the first, so it cannot appear alone.
"""
from __future__ import annotations

import argparse
from itertools import product
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import CONFIG, ROOT, arrays, pilot, read, save, sha, write


def transition(source_ids, target_ids, pairs, similarity, target_orphan, *, birth=.5, death=.25,
               division=.25, bias=.5, allow_division=True):
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    ns, nt = len(source_ids), len(target_ids)
    if not ns or not nt or not len(pairs):
        return np.empty((0, 2), np.int64), 0.
    source = {int(n): i for i, n in enumerate(source_ids)}
    target = {int(n): i for i, n in enumerate(target_ids)}
    slots = 2 if allow_division else 1
    costs = np.full((slots * ns, nt + slots * ns), np.inf)
    costs[:, nt:] = 0.
    start_cost = birth * (1 - np.asarray(target_orphan))
    for (a, b), p in zip(pairs, similarity):
        i, j = source[int(a)], target[int(b)]
        costs[i, j] = bias - p - start_cost[j] - death
        if allow_division:
            costs[ns + i, j] = bias - p - start_cost[j] + division
    rows, cols = linear_sum_assignment(costs)
    chosen = [(int(source_ids[r % ns]), int(target_ids[c])) for r, c in zip(rows, cols) if c < nt]
    return np.asarray(sorted(chosen), np.int64).reshape(-1, 2), float(costs[rows, cols].sum())


def objective(edges, pairs, similarity, source_ids, target_ids, orphan, *, birth=.5, death=.25, division=.25, bias=.5):
    """Original objective relative to selecting all nodes with no temporal edges."""
    p = {tuple(map(int, edge)): float(value) for edge, value in zip(pairs, similarity)}
    q = dict(zip(map(int, target_ids), orphan))
    src_count = {}
    result = 0.
    for a, b in edges:
        a, b = int(a), int(b)
        result += bias - p[(a, b)] - birth * (1 - q[b])
        src_count[a] = src_count.get(a, 0) + 1
    result -= death * len(src_count)
    result += division * sum(n == 2 for n in src_count.values())
    return result


def run_one(name, model, allow_division, execution="cpu"):
    from tools.cellpose_ultrack.track import validate_graph, write_csv
    feature_path = ROOT / "cellpose/features" / f"{name}.npz"
    score_directory = "scores" if execution == "cpu" else "scores_cooperative_cuda"
    score_path = ROOT / "cellpose" / score_directory / model / f"{name}.npz"
    arm = f"cellpose_{model}" + ("" if allow_division else "_no_division")
    if execution != "cpu":
        arm += "_cuda"
    output = ROOT / "graphs" / arm / f"{name}.npz"
    inputs = dict(features=sha(feature_path), scores=sha(score_path), score_receipt=sha(score_path.with_suffix(".json")),
                  config=sha(CONFIG), code=sha(__file__))
    if output.with_suffix(".json").exists():
        receipt = read(output.with_suffix(".json"))
        if receipt["inputs"] != inputs or receipt["sha256"] != sha(output):
            raise ValueError("Stale decoded graph")
        return
    data, scores = arrays(feature_path), arrays(score_path)
    nodes, pairs = data["nodes"], data["pairs"]
    similarity, orphan = scores["similarity"], scores["orphan"]
    ni = {int(n[0]): i for i, n in enumerate(nodes)}
    src_times = np.array([nodes[ni[int(a)], 1] for a in pairs[:, 0]])
    selected = []
    costs = []
    started = time.monotonic()
    for t in range(int(nodes[:, 1].max())):
        ii = np.flatnonzero(nodes[:, 1] == t)
        jj = np.flatnonzero(nodes[:, 1] == t + 1)
        mask = src_times == t
        edges, cost = transition(nodes[ii, 0], nodes[jj, 0], pairs[mask], similarity[mask], orphan[jj],
                                 allow_division=allow_division)
        selected.append(edges)
        costs.append(cost)
    edges = np.concatenate(selected)
    clip = next(c for c in pilot() if c["dataset"] == name)
    validate_graph(nodes, edges, clip["shape"])
    save(output, nodes=nodes, edges=edges)
    csv_path = output.with_suffix(".csv")
    write_csv(csv_path, name, nodes, edges)
    write(output.with_suffix(".json"), dict(dataset=name, arm=arm, inputs=inputs, sha256=sha(output),
          csv_sha256=sha(csv_path), nodes=len(nodes), edges=len(edges), seconds=time.monotonic()-started,
          all_observations_preserved=True, baseline_edges_used=False, predictions_fitted_on_biohub=model == "source_residual",
          divisions=int(np.sum(np.unique(edges[:, 0], return_counts=True)[1] == 2)),
          objective_relative_to_no_edges=float(sum(costs)), exact_assignment=True,
          solver="Adjacent-frame fixed-observation reduction of the upstream HOCT flow objective"))
    print("HOCT GRAPH", arm, name, len(edges), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+", default=["general_v1", "ctc_v0"])
    parser.add_argument("--execution", choices=["cpu", "cooperative_cuda"], default="cpu")
    args = parser.parse_args()
    for model, name, divisions in product(args.models, args.datasets or read(CONFIG)["pilot_clips"], [True, False]):
        run_one(name, model, divisions, args.execution)


if __name__ == "__main__":
    main()
