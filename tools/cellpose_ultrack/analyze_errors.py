"""Rescore frozen baselines on the six pilot clips and audit tracking errors."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import zarr

from tools.annotation_selection.common import OFFICIAL, METRIC_REV, write_json
from tools.annotation_selection.metric_adapter import aggregate, evaluate_graph, match_nodes
from tools.cellpose_ultrack.evaluate import assert_matching
from tools.cellpose_ultrack.track import REPO, ROOT, validate_graph
from tools.detector_screen.cellpose_adapter import sha256

OUT = ROOT / "error-analysis"
BASELINE = Path("/kaggle/working/cell-tracking/strong-tracker-v3/selected_predictions")
PUBLIC = Path("/kaggle/working/cell-tracking/annotation-selection-v1/baseline/public")


def inspect_graph(name, path, clip, gt_nodes, gt_edges, estimate, arm, output=OUT):
    with np.load(path, allow_pickle=False) as data:
        nodes, edges = data["nodes"], data["edges"]
    validate_graph(nodes, edges, clip["shape"])
    row, matches, tp = evaluate_graph(name, nodes, edges, gt_nodes, gt_edges,
                                      clip["spacing_um"], estimate)
    assert_matching(matches, nodes, gt_nodes, clip["spacing_um"])
    reverse = {g: p for p, g in matches.items()}
    recovered = {(matches[int(a)], matches[int(b)]) for a, b in tp}
    truth = set(map(tuple, gt_edges))
    out_gt, in_gt = set(gt_edges[:, 0]), set(gt_edges[:, 1])
    degree = Counter(map(int, edges[:, 0]))
    pred_positions = {int(n[0]): n[2:] for n in nodes}
    gt_positions = {int(n[0]): n[2:] for n in gt_nodes}
    gt_children, gt_parents = {}, {}
    for a, b in gt_edges:
        gt_children.setdefault(int(a), []).append(int(b))
        gt_parents.setdefault(int(b), []).append(int(a))
    fp, fp_kinds, fp_endpoint_evidence = [], Counter(), Counter()
    for a, b in map(tuple, edges):
        ga, gb = matches.get(int(a)), matches.get(int(b))
        if (ga, gb) not in truth and (ga in out_gt or gb in in_gt):
            fp.append((int(a), int(b)))
            if ga is not None and gb is not None:
                fp_kinds["both_matched_wrong_identity"] += 1
            elif ga is not None:
                fp_kinds["matched_source_unmatched_target"] += 1
            else:
                fp_kinds["unmatched_source_matched_target"] += 1
            if (ga is None) != (gb is None):
                expected = gt_parents[gb] if ga is None else gt_children[ga]
                point = pred_positions[int(a if ga is None else b)]
                distance = min(float(np.linalg.norm((point - gt_positions[g]) * clip["spacing_um"])) for g in expected)
                bucket = ("within_7um_of_expected_gt" if distance <= 7.000001 else
                          "between_7_and_10um_of_expected_gt" if distance <= 10 else
                          "beyond_10um_of_expected_gt")
                fp_endpoint_evidence[bucket] += 1
    if len(fp) != row["edge_fp"] or len(recovered) != row["edge_tp"]:
        raise ValueError("Independent sparse-edge audit differs from official score")
    available = {(int(a), int(b)) for a, b in gt_edges if int(a) in reverse and int(b) in reverse}
    row.update(arm=arm, embryo=clip["embryo"], graph_file_sha256=sha256(path),
               gt_nodes=len(gt_nodes), gt_edges=len(gt_edges),
               available_gt_edges=len(available), fp_categories=dict(fp_kinds),
               unmatched_fp_endpoint_distance=dict(fp_endpoint_evidence),
               fp_edges_from_predicted_forks=sum(degree[a] == 2 for a, _ in fp),
               predicted_forks=sum(n == 2 for n in degree.values()),
               edge_precision=row["edge_tp"] / (row["edge_tp"] + row["edge_fp"]),
               edge_recall=row["edge_tp"] / len(gt_edges))
    row["summary"] = aggregate([row], [name])
    folder = output / arm
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}-matches.npz",
                        matched_ids=np.array(sorted(matches.items()), dtype=np.int64).reshape(-1, 2),
                        true_gt_edges=np.array(sorted(recovered), dtype=np.int64).reshape(-1, 2),
                        false_pred_edges=np.array(fp, dtype=np.int64).reshape(-1, 2))
    return row, nodes, edges, reverse, recovered, available


def cellpose_causes(clip, nodes, edges, reverse, recovered, available, gt_nodes, gt_edges,
                    source_root=ROOT, output=OUT, arm="cellpose"):
    name, scale = clip["dataset"], np.asarray(clip["spacing_um"])
    parts, offset = [], 0
    for t in range(clip["shape"][0]):
        with np.load(source_root / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.npz", allow_pickle=False) as f:
            centers = np.rint(f["centers_zyx"]).astype(np.int64)
        parts.append(np.column_stack([np.arange(offset, offset + len(centers)), np.full(len(centers), t), centers]))
        offset += len(centers)
    raw = np.concatenate(parts)
    raw_matches = match_nodes(raw, np.empty((0, 2), dtype=np.int64), gt_nodes, gt_edges, scale)
    assert_matching(raw_matches, raw, gt_nodes, scale)
    raw_gt = set(raw_matches.values())
    selected = {int(n[0]): n for n in nodes}
    gt_times = {int(n[0]): int(n[1]) for n in gt_nodes}
    with sqlite3.connect(f"file:{source_root / 'tracking' / name / 'data.db'}?mode=ro", uri=True) as db:
        links = {(int(a), int(b)): float(w) for a, b, w in db.execute("SELECT source_id,target_id,weight FROM links")}
        all_nodes = np.asarray(db.execute("SELECT id,t,z,y,x FROM nodes").fetchall())
        if db.execute("SELECT count(*) FROM nodes WHERE z_shift!=0 OR y_shift!=0 OR x_shift!=0").fetchone()[0]:
            raise ValueError("Unexpected motion shift in the stock-linker audit")
    source_positions = {t: all_nodes[all_nodes[:, 1] == t, 2:] * scale for t in range(100)}
    missing, reasons = [], Counter()
    edge_set = set(map(tuple, edges))
    for a, b in map(tuple, gt_edges):
        a, b = int(a), int(b)
        if (a, b) in recovered:
            continue
        record = dict(source_gt=a, target_gt=b, time=gt_times[a])
        if (a, b) not in available:
            reason = ("lost_during_ultrack_selection" if a in raw_gt and b in raw_gt
                      else "missing_from_raw_cellpose_and_selected_graph")
        else:
            pair = reverse[a], reverse[b]
            assert pair not in edge_set
            if pair in links:
                reason = "candidate_link_present_but_not_selected"
                record["candidate_iou"] = links[pair]
            else:
                delta = (selected[pair[0]][2:] - selected[pair[1]][2:]) * scale
                distance = float(np.linalg.norm(delta))
                record["distance_um"] = distance
                if distance >= 15.0:
                    reason = "candidate_link_outside_distance_gate"
                else:
                    ds = np.linalg.norm(source_positions[gt_times[a]] - selected[pair[1]][2:] * scale, axis=1)
                    closer = int(np.sum(ds < distance - 1e-7))
                    tied = int(np.sum(np.abs(ds - distance) <= 1e-7))
                    record.update(closer_source_hypotheses=closer, tied_source_hypotheses=tied)
                    if closer >= 10:
                        reason = "candidate_link_lost_to_ten_nearest_hypotheses"
                    elif closer + tied > 10:
                        reason = "candidate_link_nearest_cutoff_tie_or_iou_pruning"
                    else:
                        reason = "candidate_link_lost_to_top_five_iou_pruning"
        record["reason"] = reason
        reasons[reason] += 1
        missing.append(record)
    if len(missing) != len(gt_edges) - len(recovered):
        raise ValueError("Incomplete FN classification")
    # Fixed 20-frame core boundaries; this is descriptive, not a windowing ablation.
    boundary = {(int(a), int(b)) for a, b in gt_edges if gt_times[int(a)] % 20 == 19}
    result = dict(fn_reasons=dict(reasons), raw_matched_nodes=len(raw_matches),
                  raw_gt_edges_available=sum(int(a) in raw_gt and int(b) in raw_gt for a, b in gt_edges),
                  boundary_edges=len(boundary), boundary_missed=len(boundary - recovered),
                  interior_edges=len(gt_edges) - len(boundary),
                  interior_missed=len(set(map(tuple, gt_edges)) - boundary - recovered),
                  missing_gt_edges=missing)
    write_json(output / arm / f"{name}-causes.json", result)
    return result


def main():
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip() != METRIC_REV:
        raise ValueError("Official metric revision changed")
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Official metric checkout has changed")
    panel = json.loads((ROOT / "panel.json").read_text())
    old = json.loads((ROOT / "evaluation/summary.json").read_text())
    old_rows = {r["dataset"]: r for r in old["rows"]}
    attestation = json.loads((REPO / "results/strong-tracker-v3/export_scoring_manifest.json").read_text())
    if attestation["metric_revision"] != METRIC_REV or attestation["variant"] != "A_residual_m3.0":
        raise ValueError("Wrong baseline attestation")
    rows, diagnostics = [], []
    for clip in panel["clips"]:
        name = clip["dataset"]
        gt = zarr.open_group(Path(clip["image_path"]).with_suffix(".geff"), mode="r")
        gn = np.column_stack([gt["nodes/ids"][:], *[gt[f"nodes/props/{a}/values"][:] for a in "tzyx"]]).astype(np.int64)
        ge = np.asarray(gt["edges/ids"][:], dtype=np.int64).reshape(-1, 2)
        estimate = gt.attrs["geff"]["extra"]["estimated_number_of_nodes"]
        baseline_path = BASELINE / f"{name}.npz"
        if sha256(baseline_path) != attestation["graph_file_sha256"][name]:
            raise ValueError("Baseline graph changed from its scored export")
        inspected = {}
        for arm, path in [("cellpose", ROOT / "tracking" / name / "graph.npz"),
                          ("v3", baseline_path), ("public_harmonic", PUBLIC / f"{name}.npz")]:
            result = inspect_graph(name, path, clip, gn, ge, estimate, arm)
            row = result[0]
            if arm == "cellpose":
                if row["summary"]["counts"] != old_rows[name]["single_clip_summary"]["counts"]:
                    raise ValueError("Cellpose fresh scores differ from the original run")
                if not np.isclose(row["summary"]["score"], old_rows[name]["single_clip_summary"]["score"], atol=1e-12, rtol=0):
                    raise ValueError("Cellpose fresh aggregate drift")
            write_json(OUT / arm / f"{name}.json", row)
            rows.append(row)
            inspected[arm] = result
            print(arm, name, row["summary"]["score"], flush=True)
        cp = inspected["cellpose"]
        causes = cellpose_causes(clip, *cp[1:], gn, ge)
        lost = inspected["v3"][4] - cp[4]
        gained = cp[4] - inspected["v3"][4]
        cause_map = {(r["source_gt"], r["target_gt"]): r["reason"] for r in causes["missing_gt_edges"]}
        diagnostics.append(dict(dataset=name, **{k:v for k,v in causes.items() if k != "missing_gt_edges"},
                                baseline_only_correct_gt_edges=len(lost), cellpose_only_correct_gt_edges=len(gained),
                                baseline_only_edge_reasons=dict(Counter(cause_map[p] for p in lost))))
    summaries = {}
    for arm in ("cellpose", "v3", "public_harmonic"):
        subset = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {group: aggregate([r for r in subset if group == "pooled" or r["embryo"] == group],
                                          [r["dataset"] for r in subset if group == "pooled" or r["embryo"] == group])
                          for group in ("pooled", "44b6", "6bba")}
    cp, base = summaries["cellpose"]["pooled"], summaries["v3"]["pooled"]
    g = cp["counts"]["edge_tp"] + cp["counts"]["edge_fn"]
    cfp, bfp, cfn, bfn = [s["counts"][key] for s,key in [(cp,"edge_fp"),(base,"edge_fp"),(cp,"edge_fn"),(base,"edge_fn")]]
    fn_term = (cfn - bfn) * .5 * (1 / (g + cfp) + 1 / (g + bfp))
    fp_term = .5 * ((g - cfn) + (g - bfn)) * (1 / (g + bfp) - 1 / (g + cfp))
    count_term = (base["adj_edge_jaccard"] - base["edge_jaccard"]) - (cp["adj_edge_jaccard"] - cp["edge_jaccard"])
    division_term = .1 * (base["division_jaccard"] - cp["division_jaccard"])
    gap = base["score"] - cp["score"]
    if not np.isclose(fn_term + fp_term + count_term + division_term, gap, atol=1e-12, rtol=0):
        raise ValueError("Gap decomposition failed")
    attribution = dict(total_gap=gap, excess_edge_false_positives=fp_term, missing_true_edges=fn_term,
                       node_count_adjustment=count_term, division_bonus=division_term,
                       interpretation="Exact descriptive score decomposition. FP/FN contributions average both substitution orders (two-variable Shapley); this is not a causal detector/linker ablation.")
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete_six_clip_cohort=True,
                  metric_revision=METRIC_REV, baseline="v3 A_residual_m3.0", source_panel_sha256=sha256(ROOT / "panel.json"),
                  summaries=summaries, rows=rows, diagnostics=diagnostics, gap_attribution=attribution,
                  all_199_reference=dict(v3=.934802374260586, public_harmonic=.9117740142186423),
                  scope="Fresh official scoring of existing frozen graphs on the identical six full clips. No models, predictions, candidate links or tracker settings changed.")
    write_json(OUT / "summary.json", output)
    write_json(REPO / "results/cellpose-ultrack-20260914/error-analysis.json", output)
    print(json.dumps(dict(summaries=summaries, gap_attribution=attribution), indent=2), flush=True)


if __name__ == "__main__":
    main()
