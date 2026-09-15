"""Open target labels only after every registered fit and prediction is frozen."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import warnings
import numpy as np

from .common import (EMBRYOS, REPO, RESULTS, SCREEN, SPACING, WORK, config, json_hash, now, sha, write_json)


def main():
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore", message="No matching edges")
    cfg = config()
    lock_path = WORK / "predictions-lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["config_sha256"] == json_hash(cfg)
    training_lock_path = WORK / "training-lock.json"
    assert lock["training_lock_sha256"] == sha(training_lock_path)
    training_lock = json.loads(training_lock_path.read_text())
    for name, expected_sha in training_lock["source_sha256"].items():
        assert sha(REPO / name) == expected_sha, name
    assert len(lock["models"]) == 4
    for model in lock["models"]:
        assert model["source"] != model["target"]
        assert sha(model["path"]) == model["sha256"]
    panel_bytes = (SCREEN / "panel.json").read_bytes()
    panel = json.loads(panel_bytes)
    all_keys = {r["key"] for r in panel["frames"]}
    expected = {(seed, k) for seed in cfg["seeds"] for k in all_keys}
    assert {(r["seed"], r["key"]) for r in lock["predictions"]} == expected
    assert len(lock["predictions"]) == len(expected)
    for prediction in lock["predictions"]:
        path = WORK / "predictions" / f"seed-{prediction['seed']}" / (prediction["key"] + ".npz")
        assert prediction["training_source"] != prediction["embryo"]
        assert sha(path) == prediction["prediction_sha256"]
    # This timestamp is after the complete all-fit/all-prediction lock and before GT reads.
    opening = {"created_utc": now(), "predictions_lock_sha256": sha(lock_path),
               "all_registered_models_and_predictions_verified_before_gt_access": True}
    write_json(WORK / "evaluation-opened.json", opening)
    gt_bytes = (SCREEN / "evaluation/ground_truth.json").read_bytes()
    gt = json.loads(gt_bytes)
    from .diagnostics import DEFINITIONS, make_strata, signed_and_stratified
    from tools.detector_screen.analyze_headroom import frame_analysis, aggregate
    from tools.detector_screen.evaluate import matches
    baseline_path = RESULTS.parent / "detector-development-20260914/localization-headroom.json"
    baseline = json.loads(baseline_path.read_text())
    fingerprint = hashlib.sha256(panel_bytes + gt_bytes).hexdigest()
    assert fingerprint == baseline["data_fingerprint"]
    frames = [r for r in panel["frames"] if r["role"] == "assessment"]
    strata = make_strata(frames, gt)
    results = {"created_utc": now(), "data_fingerprint": fingerprint,
               "config_sha256": json_hash(cfg), "evaluation_opened": opening,
               "baseline": baseline["methods"]["cellpose_cpdino_vitb"], "seeds": {}, "stratum_definitions": DEFINITIONS,
               "scope": "Source-isolated cross-embryo adaptation on previously examined embryos; inherited cpDINO exposure unresolved. No temporal assignment or official graph score."}
    baseline_rows = {}
    for encoding in ("integer", "float"):
        path = WORK.parent / "detector-development-20260914" / f"cellpose_cpdino_vitb-{encoding}.json"
        baseline_rows[encoding] = {r["key"]: r for r in json.loads(path.read_text())}
        for frame in frames:
            truth = np.asarray(gt["frames"][frame["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
            with np.load(SCREEN / "predictions/cellpose_cpdino_vitb" / (frame["key"] + ".npz"), allow_pickle=False) as p:
                centers = p["centers_zyx"].astype(np.float64)
            if encoding == "integer":
                centers = np.rint(centers)
            old_row = baseline_rows[encoding][frame["key"]]
            old_row["_found_ids"] = {"3.0": list(matches(centers, truth, frame["time"], 3.).values()),
                                     "7.0": [int(k) for k in old_row["_records"]]}
        for group in results["baseline"][encoding]:
            group["additional_diagnostics"] = signed_and_stratified(list(baseline_rows[encoding].values()), strata, group["embryo"])
    for seed in cfg["seeds"]:
        seed_result = {}
        for encoding in ("integer", "float"):
            rows = []
            for i, row in enumerate(frames):
                truth = np.asarray(gt["frames"][row["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
                pred_path = WORK / "predictions" / f"seed-{seed}" / (row["key"] + ".npz")
                with np.load(pred_path, allow_pickle=False) as p, np.load(SCREEN / "predictions/cellpose_cpdino_vitb" / (row["key"] + ".npz"), allow_pickle=False) as b:
                    scores = p["scores"].copy()
                    np.testing.assert_array_equal(scores, b["scores"])
                    centers = p["centers_zyx" if encoding == "integer" else "float_centers_zyx"].copy()
                    assert len(centers) == len(b["centers_zyx"])
                    if encoding == "integer":
                        assert np.array_equal(centers, np.rint(centers))
                        assert (np.linalg.norm((centers - np.rint(b["centers_zyx"])) * SPACING, axis=1) <= 3 + 1e-9).all()
                    base_centers = np.rint(b["centers_zyx"]) if encoding == "integer" else b["centers_zyx"].astype(np.float64)
                item = frame_analysis(row, truth, centers, scores, encoding)
                item["exact_coordinate_collisions"] = {
                    "refined": len(centers) - len(np.unique(centers, axis=0)),
                    "baseline": len(base_centers) - len(np.unique(base_centers, axis=0))}
                base_item = baseline_rows[encoding][row["key"]]
                own, old = item["_records"], base_item["_records"]
                common = set(own) & set(old)
                item["paired_common_gt_at_7"] = {"n": len(common),
                    "new_error_sum_um": sum(own[k]["error_um"] for k in common),
                    "baseline_error_sum_um": sum(old[k]["error_um"] for k in common),
                    "gained": len(set(own) - set(old)), "lost": len(set(old) - set(own))}
                new3 = set(matches(centers, truth, row["time"], 3.).values())
                old3 = set(matches(base_centers, truth, row["time"], 3.).values())
                item["matched_gt_at_3_changes"] = {"gained": len(new3 - old3), "lost": len(old3 - new3)}
                item["_found_ids"] = {"3.0": list(new3), "7.0": [int(k) for k in own]}
                rows.append(item)
                if (i + 1) % 100 == 0:
                    print(json.dumps({"seed": seed, "encoding": encoding, "frames": i + 1}), flush=True)
            groups = [aggregate(rows, gt, e) for e in (*EMBRYOS, "pooled")]
            for group in groups:
                selected = [r for r in rows if group["embryo"] == "pooled" or r["embryo"] == group["embryo"]]
                group["paired_common_gt_at_7"] = {k: sum(r["paired_common_gt_at_7"][k] for r in selected)
                                                   for k in selected[0]["paired_common_gt_at_7"]}
                group["matched_gt_at_3_changes"] = {k: sum(r["matched_gt_at_3_changes"][k] for r in selected) for k in ("gained", "lost")}
                group["exact_coordinate_collisions"] = {
                    k: sum(r["exact_coordinate_collisions"][k] for r in selected) for k in ("refined", "baseline")}
                group["additional_diagnostics"] = signed_and_stratified(rows, strata, group["embryo"])
            seed_result[encoding] = groups
            write_json(WORK / "evaluation" / f"seed-{seed}-{encoding}-detail.json", rows)
            write_json(RESULTS / f"seed-{seed}-{encoding}-frames.json", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        results["seeds"][str(seed)] = seed_result
    promotion = []
    for seed, result in results["seeds"].items():
        for embryo in EMBRYOS:
            b = next(r for r in results["baseline"]["integer"] if r["embryo"] == embryo)
            r = next(r for r in result["integer"] if r["embryo"] == embryo)
            bf = next(r for r in results["baseline"]["float"] if r["embryo"] == embryo)
            rf = next(r for r in result["float"] if r["embryo"] == embryo)
            q3_delta = float(np.mean([r["budgets"][str(k)]["recall"]["3.0"] - b["budgets"][str(k)]["recall"]["3.0"] for k in cfg["evaluation"]["absolute_candidate_caps"]]))
            r7_delta = r["recall"]["7.0"] - b["recall"]["7.0"]
            cap_r7_deltas = {str(k): r["budgets"][str(k)]["recall"]["7.0"] - b["budgets"][str(k)]["recall"]["7.0"] for k in cfg["evaluation"]["absolute_candidate_caps"]}
            error_delta = rf["mean_error_censored_at_7_um"] - bf["mean_error_censored_at_7_um"]
            passed = q3_delta > 0 and r["recall"]["3.0"] > b["recall"]["3.0"] and error_delta < 0 and min([r7_delta, *cap_r7_deltas.values()]) >= -cfg["evaluation"]["max_r7_drop_per_embryo"]
            promotion.append({"seed": int(seed), "target_embryo": embryo, "q3_delta": q3_delta,
                              "full_r3_delta": r["recall"]["3.0"] - b["recall"]["3.0"], "full_r7_delta": r7_delta,
                              "cap_r7_deltas": cap_r7_deltas, "float_censored_mean_error_delta_um": error_delta,
                              "passes_declared_engineering_gate": bool(passed)})
    results["promotion_checks"] = promotion
    results["all_four_fit_target_groups_pass"] = all(r["passes_declared_engineering_gate"] for r in promotion)
    results["evaluation_source_sha256"] = {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("diagnostics.py")]}
    results["interpretation"] = "Engineering gate on two reused embryos, not statistical proof of biological generalization. Counts/ranking fixed. Any later target-informed recipe change is another exploratory iteration."
    write_json(RESULTS / "metrics.json", results)
    print(json.dumps({"promotion_checks": promotion, "all_groups_pass": results["all_four_fit_target_groups_pass"]}), flush=True)


if __name__ == "__main__":
    main()
