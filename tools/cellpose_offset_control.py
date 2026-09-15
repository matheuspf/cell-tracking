"""Exploratory source-only constant-offset control after the frozen-head study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq

from tools.cellpose_refine.common import (EMBRYOS, REPO, RESULTS, SCREEN, SPACING, WORK,
    bounded_integer, config, json_hash, now, save_npz, sha, validate_ancestry, write_json)

OUT = WORK / "offset-control"


def fit(source):
    from tools.cellpose_refine.train import install_source_guard
    install_source_guard(source)
    path = WORK / "inputs" / f"train-{source}.json"
    lock = json.loads((WORK / "feature-lock.json").read_text())
    assert sha(path) == lock["source_summaries"][source]["manifest_sha256"]
    manifest = json.loads(path.read_text())
    validate_ancestry(source, manifest)
    target_groups, weight_groups = [], []
    for field in ("natural_rows", "jitter_rows"):
        targets, weights = [], []
        for row in manifest[field]:
            assert row["embryo"] == source
            p = Path(row["target_path"] if field == "natural_rows" else row["query_path"])
            assert sha(p) == row["target_sha256"] if field == "natural_rows" else sha(p) == row["query_sha256"]
            with np.load(p, allow_pickle=False) as data:
                valid = data["weight"] > 0
                targets.append(data["target_um"][valid].astype(np.float64))
                weights.append(data["weight"][valid].astype(np.float64))
        target = np.concatenate(targets)
        weight = np.concatenate(weights) / len(target)
        target_groups.append(target); weight_groups.append(weight)
    target, weight = np.concatenate(target_groups), np.concatenate(weight_groups)
    beta = config()["training"]["loss_beta_um"]
    delta = np.array([brentq(lambda x: np.sum(weight * np.clip((x - target[:, axis]) / beta, -1, 1)), -3., 3., xtol=1e-12)
                      for axis in range(3)])
    assert np.linalg.norm(delta) <= 3.
    def loss(x):
        error = np.abs(x - target)
        per_axis = np.where(error < beta, error**2 / (2 * beta), error - beta / 2)
        return float(np.sum(per_axis.mean(axis=1) * weight) / weight.sum())
    assert loss(delta) <= loss(np.zeros(3))
    result = {"created_utc": now(), "source": source, "target": next(e for e in EMBRYOS if e != source),
        "offset_zyx_um": delta.tolist(), "source_manifest_sha256": sha(path), "script_sha256": sha(__file__),
        "source_loss_before": loss(np.zeros(3)), "source_loss_after": loss(delta),
        "objective": "Constant minimizer of weighted SmoothL1, with equal natural/jitter sampling mass before their original per-query weights; population objective, not a replay of stochastic minibatch denominators",
        "fit_queries": [len(t) for t in target_groups], "target_scores_read_for_fit": False,
        "adaptation_exposure": {**manifest["adaptation_exposure"], "calibration": [source]},
        "inherited_exposure": manifest["inherited_exposure"]}
    write_json(OUT / f"fit-{source}.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-source", choices=EMBRYOS)
    args = parser.parse_args()
    if args.fit_source:
        fit(args.fit_source)
        return
    plan = {"created_utc": now(), "script_sha256": sha(__file__), "config_sha256": json_hash(config()),
        "exploratory_after_v1": True, "motivation": "Separate source-wide offset correction from image-dependent head effects",
        "fit_labels": "Opposite embryo only; existing fixed source targets and weights", "movement_limit_um": 3.,
        "selection": "One deterministic constant per source; no target-driven parameter search or seed selection"}
    write_json(RESULTS / "offset-control-plan.json", plan)
    for source in EMBRYOS:
        subprocess.run([sys.executable, "-m", "tools.cellpose_offset_control", "--fit-source", source], cwd=REPO, check=True)
    fits = [json.loads((OUT / f"fit-{source}.json").read_text()) for source in EMBRYOS]
    rows = json.loads((WORK / "inputs/inference.json").read_text())["rows"]
    hashes = {}
    for row in rows:
        fitted = next(f for f in fits if f["target"] == row["embryo"])
        assert fitted["source"] != row["embryo"]
        assert sha(row["prediction_path"]) == row["prediction_sha256"]
        with np.load(row["prediction_path"], allow_pickle=False) as p:
            base, scores = p["centers_zyx"].astype(np.float64), p["scores"].copy()
        continuous = np.clip(base + np.asarray(fitted["offset_zyx_um"]) / SPACING, 0, np.asarray(row["shape"]) - 1)
        integer, projected = bounded_integer(base, continuous, row["shape"])
        path = OUT / "predictions" / (row["key"] + ".npz")
        save_npz(path, centers_zyx=integer.astype(np.float64), float_centers_zyx=continuous, scores=scores,
                 integer_bound_projected=projected)
        hashes[row["key"]] = sha(path)
    prediction_lock = {"created_utc": now(), "fits": fits, "predictions": hashes,
                       "plan_sha256": sha(RESULTS / "offset-control-plan.json"), "target_scores_opened": False}
    lock_path = OUT / "predictions-lock.json"
    write_json(lock_path, prediction_lock)
    # Both fitted constants and all predictions exist before any control score or GT read.
    opened = now()
    panel_bytes = (SCREEN / "panel.json").read_bytes()
    gt_bytes = (SCREEN / "evaluation/ground_truth.json").read_bytes()
    gt = json.loads(gt_bytes)
    previous = json.loads((RESULTS / "metrics.json").read_text())
    assert hashlib.sha256(panel_bytes + gt_bytes).hexdigest() == previous["data_fingerprint"]
    from tools.detector_screen.analyze_headroom import frame_analysis, aggregate
    import logging
    logging.disable(logging.WARNING)
    groups = {}
    for encoding in ("integer", "float"):
        frame_results = []
        for row in rows:
            if row["role"] != "assessment":
                continue
            truth = np.asarray(gt["frames"][row["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
            path = OUT / "predictions" / (row["key"] + ".npz")
            assert sha(path) == hashes[row["key"]]
            with np.load(path, allow_pickle=False) as p:
                centers = p["centers_zyx" if encoding == "integer" else "float_centers_zyx"]
                frame_results.append(frame_analysis(row, truth, centers, p["scores"], encoding))
        groups[encoding] = [aggregate(frame_results, gt, e) for e in (*EMBRYOS, "pooled")]
        write_json(OUT / f"{encoding}-detail.json", frame_results)
    gates = []
    for embryo in EMBRYOS:
        b = next(g for g in previous["baseline"]["integer"] if g["embryo"] == embryo)
        r = next(g for g in groups["integer"] if g["embryo"] == embryo)
        bf = next(g for g in previous["baseline"]["float"] if g["embryo"] == embryo)
        rf = next(g for g in groups["float"] if g["embryo"] == embryo)
        q3 = float(np.mean([r["budgets"][str(k)]["recall"]["3.0"] - b["budgets"][str(k)]["recall"]["3.0"] for k in (100, 200, 400)]))
        r7 = [r["recall"]["7.0"] - b["recall"]["7.0"],
              *[r["budgets"][str(k)]["recall"]["7.0"] - b["budgets"][str(k)]["recall"]["7.0"] for k in (100, 200, 400)]]
        error = rf["mean_error_censored_at_7_um"] - bf["mean_error_censored_at_7_um"]
        gates.append({"target_embryo": embryo, "q3_delta": q3, "r7_deltas_full_100_200_400": r7,
                      "float_censored_error_delta_um": error, "passes_same_engineering_gate":
                      q3 > 0 and r["recall"]["3.0"] > b["recall"]["3.0"] and error < 0 and min(r7) >= -.002})
    write_json(RESULTS / "offset-control.json", {"created_utc": now(), "fits": fits, "groups": groups,
        "gates": gates, "all_groups_pass": all(g["passes_same_engineering_gate"] for g in gates),
        "data_fingerprint": previous["data_fingerprint"], "predictions_lock_sha256": sha(lock_path), "target_scores_opened_utc": opened,
        "interpretation": "Exploratory control motivated by v1 results on the same two reused embryos. No new independent test; inherited Cellpose exposure remains unresolved. Counts and ranks stay fixed."})
    print(json.dumps({"gates": gates, "all_groups_pass": all(g["passes_same_engineering_gate"] for g in gates)}), flush=True)


if __name__ == "__main__":
    main()
