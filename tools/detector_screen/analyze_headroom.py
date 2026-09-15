"""Localization, crowding and detection headroom from fixed benchmark outputs.

This computes diagnostics only. It does not train models or infer temporal links.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist

from .evaluate import REPO, ROOT, SPACING, matches

OUT = REPO / "results/detector-development-20260914"
WORK = REPO / "work/detector-development-20260914"
METHODS = ["incumbent", "cellpose_cpdino_vitb", "organoid"]
RADII = [1., 2., 3., 4., 5., 6., 7.]
CAPS = [100, 200, 400]


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                max=float(a.max()))


def frame_analysis(row, truth, centers, scores, encoding):
    physical = centers * SPACING
    target = truth[:, 2:] * SPACING
    distances = cdist(target, physical)
    gt_distances = cdist(target, target)
    np.fill_diagonal(gt_distances, np.inf)
    nearest_gt = gt_distances.min(axis=1) if len(truth) else np.empty(0)
    nearest_pred = distances.min(axis=1) if len(centers) else np.full(len(truth), np.inf)
    by_id = {int(g[0]): i for i, g in enumerate(truth)}
    mappings = {str(radius): matches(centers, truth, row["time"], radius) for radius in RADII}
    m7 = mappings["7.0"]
    found = {radius: set(mapping.values()) for radius, mapping in mappings.items()}
    records = {}
    competing_margins = []
    safe = 0
    for pred, gt_id in m7.items():
        i = by_id[gt_id]
        error = float(distances[i, pred])
        others = np.delete(distances[:, pred], i)
        if len(others):
            competing_margins.append(float(others.min() - error))
        safe += error < nearest_gt[i] / 2
        records[str(gt_id)] = dict(predicted_um=physical[pred].tolist(), gt_um=target[i].tolist(), error_um=error)
    strata = {}
    for name, selected in {
        "neighbor_within_7um": nearest_gt <= 7,
        "neighbor_7_to_14um": (nearest_gt > 7) & (nearest_gt <= 14),
        "no_observed_neighbor_within_14um": nearest_gt > 14,
    }.items():
        ids = set(map(int, truth[selected, 0]))
        strata[name] = dict(gt=len(ids), matched={r: len(ids & values) for r, values in found.items()})
    pairs = {}
    for gate in (7, 14):
        a, b = np.where(np.triu(gt_distances <= gate, 1))
        pairs[str(gate)] = dict(
            pairs=len(a),
            both_matched={r: sum(int(truth[i, 0]) in ids and int(truth[j, 0]) in ids for i, j in zip(a, b))
                          for r, ids in found.items()})
    result = dict(key=row["key"], dataset=row["dataset"], embryo=row["embryo"], time=row["time"],
                  encoding=encoding, gt=len(truth), predicted=len(centers),
                  matches={r: len(m) for r, m in mappings.items()}, strata=strata, close_pairs=pairs,
                  gt_with_multiple_candidates_at_7=int(((distances <= 7).sum(axis=1) >= 2).sum()),
                  predictions_eligible_for_multiple_gt_at_7=int(((distances <= 7).sum(axis=0) >= 2).sum()),
                  matched_candidates_eligible_for_multiple_gt_at_7=int(sum((distances[:, p] <= 7).sum() >= 2 for p in m7)),
                  nearest_coverage_at_7=int((nearest_pred <= 7).sum()),
                  matched_inside_half_nearest_gt_spacing=int(safe),
                  matched_with_competitor_margin_at_most_1um=sum(x <= 1 for x in competing_margins),
                  _records=records, _competitor_margins=competing_margins,
                  _errors=[v["error_um"] for v in records.values()])
    misses = np.array([int(g[0]) not in found["7.0"] for g in truth], dtype=bool)
    d = nearest_pred[misses]
    result["miss_nearest_distance_bands"] = {
        "within_7_but_unmatched": int((d <= 7).sum()),
        "7_to_8": int(((d > 7) & (d <= 8)).sum()),
        "8_to_10": int(((d > 8) & (d <= 10)).sum()),
        "10_to_14": int(((d > 10) & (d <= 14)).sum()),
        "over_14_or_absent": int((d > 14).sum()),
    }
    result["refinement_capacity"] = {}
    for delta in (0, 1, 2, 3, 7):
        allowed = distances <= 7 + delta
        matching = maximum_bipartite_matching(csr_matrix(allowed), perm_type="column") if len(truth) and len(centers) else np.full(len(truth), -1)
        result["refinement_capacity"][str(delta)] = int((matching >= 0).sum())
    result["budgets"] = {}
    if encoding == "integer":
        order = np.argsort(-scores, kind="stable")
        for cap in CAPS:
            selected = centers[order[:cap]]
            result["budgets"][str(cap)] = dict(predicted=len(selected),
                matched={str(r): len(matches(selected, truth, row["time"], r)) for r in (3., 7.)})
    return result


def aggregate(rows, gt, embryo):
    rows = [r for r in rows if embryo == "pooled" or r["embryo"] == embryo]
    ng = sum(r["gt"] for r in rows)
    matched = {str(g): sum(r["matches"][str(g)] for r in rows) for g in RADII}
    errors = [x for r in rows for x in r["_errors"]]
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["dataset"], {})[r["time"]] = r
    eligible = available = one_missing = both_missing = 0
    displacement_errors = []
    for dataset, frames in by_clip.items():
        clip = gt["clips"][dataset]
        times = {int(n[0]): int(n[1]) for n in clip["nodes"]}
        for a, b in clip["edges"]:
            if times[a] not in frames or times[b] not in frames:
                continue
            eligible += 1
            ra = frames[times[a]]["_records"].get(str(a))
            rb = frames[times[b]]["_records"].get(str(b))
            if ra and rb:
                available += 1
                predicted = np.asarray(rb["predicted_um"]) - ra["predicted_um"]
                true = np.asarray(rb["gt_um"]) - ra["gt_um"]
                displacement_errors.append(float(np.linalg.norm(predicted - true)))
            elif ra or rb:
                one_missing += 1
            else:
                both_missing += 1
    strata = {}
    for name in rows[0]["strata"]:
        n = sum(r["strata"][name]["gt"] for r in rows)
        m = {str(g): sum(r["strata"][name]["matched"][str(g)] for r in rows) for g in RADII}
        strata[name] = dict(gt=n, matches=m, recall={g: v / n if n else None for g, v in m.items()})
    pair_summary = {}
    for gate in ("7", "14"):
        n = sum(r["close_pairs"][gate]["pairs"] for r in rows)
        m = {str(g): sum(r["close_pairs"][gate]["both_matched"][str(g)] for r in rows) for g in RADII}
        pair_summary[gate] = dict(pairs=n, both_matched=m, recall={g: v / n if n else None for g, v in m.items()})
    budgets = {}
    for cap in rows[0]["budgets"]:
        budgets[cap] = dict(predicted=sum(r["budgets"][cap]["predicted"] for r in rows),
            recall={str(g): sum(r["budgets"][cap]["matched"][str(g)] for r in rows) / ng for g in (3., 7.)})
    count_fields = ["gt_with_multiple_candidates_at_7", "predictions_eligible_for_multiple_gt_at_7",
                    "matched_candidates_eligible_for_multiple_gt_at_7", "nearest_coverage_at_7",
                    "matched_inside_half_nearest_gt_spacing", "matched_with_competitor_margin_at_most_1um"]
    return dict(embryo=embryo, frames=len(rows), clips=len(by_clip), gt=ng,
        predicted=sum(r["predicted"] for r in rows), matches=matched,
        recall={g: v / ng for g, v in matched.items()},
        matched_error_um=distribution(errors),
        mean_error_censored_at_7_um=(sum(errors) + 7 * (ng - len(errors))) / ng,
        annotated_competitor_margin_um=distribution([x for r in rows for x in r["_competitor_margins"]]),
        **{field: int(sum(r[field] for r in rows)) for field in count_fields},
        strata=strata, close_pairs=pair_summary, budgets=budgets,
        refinement_capacity={d: sum(r["refinement_capacity"][d] for r in rows) for d in ("0", "1", "2", "3", "7")},
        miss_nearest_distance_bands={name: sum(r["miss_nearest_distance_bands"][name] for r in rows)
                                    for name in rows[0]["miss_nearest_distance_bands"]},
        edge_endpoints=dict(eligible=eligible, available=available, one_missing=one_missing,
            both_missing=both_missing, availability=available / eligible,
            recoverable_fraction_to_perfect=1 - available / eligible),
        displacement_error_on_covered_gt_edges_um=distribution(displacement_errors))


def main():
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore", message="No matching edges")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    panel_bytes = (ROOT / "panel.json").read_bytes()
    gt_bytes = (ROOT / "evaluation/ground_truth.json").read_bytes()
    panel, gt = json.loads(panel_bytes), json.loads(gt_bytes)
    frames = [r for r in panel["frames"] if r["role"] == "assessment"]
    result = dict(data_fingerprint=hashlib.sha256(panel_bytes + gt_bytes).hexdigest(),
        panel_definition_sha256=panel["definition_sha256"], frames=400, radii_um=RADII,
        absolute_candidate_caps=CAPS, methods={},
        limitations=[
            "Fixed-output descriptive analysis on the previously examined panel, not a new independent test.",
            "Incumbent training provenance is uncertain with evidence of target exposure; it is a leakage-risk reference, not a clean performance ceiling.",
            "Cellpose pretraining overlap is audited separately and is not certified absent by these measurements.",
            "Crowding among annotated cells underestimates true crowding because GEFF is sparse; extra candidates are not proven extra biological cells.",
            "Floating predictions are compared against integer GT reference points; this cannot establish subvoxel biological ground truth.",
            "Refinement capacity is an optimistic maximum-cardinality bound within radius 7+delta, assuming each candidate may move delta micrometers. It ignores integer-grid feasibility, learned decisions and the official weight preference; it is not achieved recall.",
            "Both-endpoint availability is a detector diagnostic under fixed per-frame matches, not an official score or predicted-link accuracy.",
            "Displacement error uses known GT edge correspondences solely for diagnosis; no temporal links are inferred."])
    for method in METHODS:
        old = json.loads((ROOT / "evaluation" / (method + ".json")).read_text())
        assert old["data_fingerprint"] == result["data_fingerprint"]
        old_by_key = {r["key"]: r for r in old["per_frame"]}
        model = {}
        for encoding in ("integer", "float"):
            path = WORK / f"{method}-{encoding}.json"
            rows = []
            for i, row in enumerate(frames):
                truth = np.asarray(gt["frames"][row["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
                with np.load(ROOT / "predictions" / method / (row["key"] + ".npz"), allow_pickle=False) as p:
                    centers = np.asarray(p["centers_zyx"], dtype=np.float64)
                    scores = np.asarray(p["scores"], dtype=np.float64)
                assert np.isfinite(centers).all() and np.isfinite(scores).all()
                if encoding == "integer":
                    centers = np.rint(centers)
                centers = np.clip(centers, 0, np.asarray(row["shape"]) - 1)
                item = frame_analysis(row, truth, centers, scores, encoding)
                if encoding == "integer":
                    for radius in ("5.0", "6.0", "7.0"):
                        assert item["matches"][radius] == old_by_key[row["key"]]["matches"][radius], (method, row["key"], radius)
                rows.append(item)
                if (i + 1) % 100 == 0:
                    print(method, encoding, i + 1, "/", len(frames), flush=True)
            model[encoding] = [aggregate(rows, gt, embryo) for embryo in ("44b6", "6bba", "pooled")]
            path.write_text(json.dumps(rows, indent=2) + "\n")
            public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
            (OUT / f"{method}-{encoding}-frames.json").write_text(json.dumps(public_rows, indent=2) + "\n")
        result["methods"][method] = model
        (OUT / "localization-headroom.json").write_text(json.dumps(result, indent=2) + "\n")
        print(method, "COMPLETE", json.dumps(model["integer"][-1]["recall"]), flush=True)
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "localization-headroom.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
