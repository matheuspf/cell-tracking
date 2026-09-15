"""Audit the completed persistence arms against stock and v3, without fitting."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
import zarr

from tools.annotation_selection.common import METRIC_REV, OFFICIAL, now, write_json
from tools.annotation_selection.metric_adapter import aggregate, match_nodes
from tools.cellpose_ultrack.analyze_errors import BASELINE, inspect_graph, cellpose_causes
from tools.cellpose_ultrack.event_costs import bank_fingerprint
from tools.cellpose_ultrack.persistent_divisions import OUT, STUDY, paths
from tools.cellpose_ultrack.track import REPO, ROOT
from tools.detector_screen.cellpose_adapter import sha256

WORK = REPO / "work" / STUDY / "error-analysis"


def gap_decomposition(current, reference):
    """Exact descriptive decomposition of the full score difference."""
    g = current["counts"]["edge_tp"] + current["counts"]["edge_fn"]
    cf, rf = [s["counts"]["edge_fp"] for s in (current, reference)]
    cn, rn = [s["counts"]["edge_fn"] for s in (current, reference)]
    result = dict(
        excess_edge_false_positives=.5 * (2*g-cn-rn) * (1/(g+rf)-1/(g+cf)),
        missing_true_edges=(cn-rn) * .5 * (1/(g+cf)+1/(g+rf)),
        node_count_adjustment=(reference["adj_edge_jaccard"]-reference["edge_jaccard"]) -
                              (current["adj_edge_jaccard"]-current["edge_jaccard"]),
        division_bonus=.1*(reference["division_jaccard"]-current["division_jaccard"]),
    )
    gap = reference["score"]-current["score"]
    if not np.isclose(sum(result.values()), gap, atol=1e-12, rtol=0):
        raise ValueError("Gap accounting failed")
    return dict(total_gap=gap, **result,
                interpretation="Exact score accounting; FP/FN terms average both substitution orders. It does not causally distinguish detector and linker errors.")


def missed_node_proximity(nodes, gt_nodes, matched_gt, scale):
    """Distinguish absent nearby centers from competition in one-to-one matching."""
    result = Counter()
    records = []
    for row in gt_nodes:
        if int(row[0]) in matched_gt:
            continue
        candidates = nodes[nodes[:,1] == row[1],2:]
        distance = float(np.min(np.linalg.norm((candidates-row[2:])*scale, axis=1))) if len(candidates) else None
        kind = ("nearby_center_but_unmatched" if distance is not None and distance <= 7.000001 else
                "nearest_center_7_to_10um" if distance is not None and distance <= 10 else
                "no_center_within_10um")
        result[kind] += 1
        records.append(dict(gt_id=int(row[0]), time=int(row[1]), nearest_center_um=distance, kind=kind))
    result["unmatched_gt_nodes"] = len(records)
    return dict(counts=dict(result), nodes=records)


def link_context(nodes, edges, reverse, causes):
    children, parents = defaultdict(list), {}
    for p,c in edges:
        children[int(p)].append(int(c))
        parents[int(c)] = int(p)
    contexts = Counter()
    for row in causes["missing_gt_edges"]:
        if row["reason"] != "candidate_link_present_but_not_selected":
            continue
        p,c = reverse[row["source_gt"]], reverse[row["target_gt"]]
        state = ("source_continues_elsewhere" if children[p] else "source_ends") + "__" + (
            "target_has_other_parent" if c in parents else "target_starts_new_track")
        contexts[state] += 1
    return dict(contexts)


def main():
    if subprocess.check_output(["git","-C",str(OFFICIAL),"rev-parse","HEAD"],text=True).strip() != METRIC_REV:
        raise ValueError("Official scorer revision changed")
    if subprocess.check_output(["git","-C",str(OFFICIAL),"status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("Official scorer has modified files")
    panel = json.loads((ROOT / "panel.json").read_text())
    plan = json.loads((OUT / "plan.json").read_text())
    if sha256(ROOT / "panel.json") != plan["panel_sha256"]:
        raise ValueError("Cohort changed")
    attestation = json.loads((REPO / "results/strong-tracker-v3/export_scoring_manifest.json").read_text())
    if attestation["metric_revision"] != METRIC_REV or attestation["variant"] != "A_residual_m3.0":
        raise ValueError("Wrong baseline attestation")
    roots = {"current":paths("pair-image")[0], "control":paths("pair-control")[0], "stock":ROOT}
    saved = {a:json.loads((p / "evaluation/summary.json").read_text()) for a,p in roots.items()}
    previous = json.loads((REPO / "results/cellpose-ultrack-20260914/error-analysis.json").read_text())
    rows, diagnostics, changes, provenance = [], [], [], []
    for clip in panel["clips"]:
        name = clip["dataset"]
        gt_path = Path(clip["image_path"]).with_suffix(".geff")
        gt = zarr.open_group(gt_path,mode="r")
        gn = np.column_stack([gt["nodes/ids"][:],*[gt[f"nodes/props/{a}/values"][:] for a in "tzyx"]]).astype(np.int64)
        ge = np.asarray(gt["edges/ids"][:],dtype=np.int64).reshape(-1,2)
        meta = gt.attrs["geff"]
        scale = np.array([next(a["scale"] for a in meta["axes"] if a["name"]==d) for d in "zyx"])
        if not np.allclose(scale,clip["spacing_um"]):
            raise ValueError("Changed physical scale")
        estimate = meta["extra"]["estimated_number_of_nodes"]
        bank = bank_fingerprint(ROOT / "tracking" / name / "data.db")
        inspected = {}
        for arm in ("current","control","stock","v3"):
            path = BASELINE / f"{name}.npz" if arm=="v3" else roots[arm] / "tracking" / name / "graph.npz"
            if arm=="v3":
                if sha256(path) != attestation["graph_file_sha256"][name]:
                    raise ValueError("v3 export changed")
                expected = next(r for r in previous["rows"] if r["arm"]=="v3" and r["dataset"]==name)["summary"]
            else:
                expected = next(r for r in saved[arm]["rows"] if r["dataset"]==name)["single_clip_summary"]
                receipt = json.loads((path.parent / "result.json").read_text())
                if sha256(path) != receipt["graph_sha256"] or bank_fingerprint(path.parent / "data.db") != bank:
                    raise ValueError("Frozen graph or candidate bank changed")
            value = inspect_graph(name,path,clip,gn,ge,estimate,arm,output=WORK)
            if value[0]["summary"]["counts"] != expected["counts"] or not np.isclose(value[0]["summary"]["score"],expected["score"],atol=1e-12,rtol=0):
                raise ValueError("Fresh audit disagrees with saved official score")
            inspected[arm] = value
            rows.append(value[0])
            write_json(WORK / arm / f"{name}.json",value[0])
        raw_parts, offset = [], 0
        for t in range(clip["shape"][0]):
            with np.load(ROOT / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.npz",allow_pickle=False) as d:
                centers = np.rint(d["centers_zyx"]).astype(np.int64)
            raw_parts.append(np.column_stack([np.arange(offset,offset+len(centers)),np.full(len(centers),t),centers]))
            offset += len(centers)
        raw = np.concatenate(raw_parts)
        raw_matches = match_nodes(raw,np.empty((0,2),np.int64),gn,ge,scale)
        raw_proximity = missed_node_proximity(raw,gn,set(raw_matches.values()),scale)
        for arm in ("current","control"):
            value = inspected[arm]
            causes = cellpose_causes(clip,*value[1:],gn,ge,source_root=roots[arm],output=WORK,arm=arm)
            selected_proximity = missed_node_proximity(value[1],gn,set(value[3]),scale)
            write_json(WORK / arm / f"{name}-missed-nodes.json",dict(raw=raw_proximity,selected=selected_proximity))
            cause_map = {(r["source_gt"],r["target_gt"]):r["reason"] for r in causes["missing_gt_edges"]}
            lost = inspected["v3"][4] - value[4]
            diagnostics.append(dict(dataset=name,arm=arm,
                               **{k:v for k,v in causes.items() if k!="missing_gt_edges"},
                               v3_only_correct_gt_edges=len(lost),
                               current_only_correct_vs_v3=len(value[4]-inspected["v3"][4]),
                               v3_only_edge_reasons=dict(Counter(cause_map[p] for p in lost)),
                               unselected_candidate_context=link_context(value[1],value[2],value[3],causes),
                               missed_raw_node_proximity=raw_proximity["counts"],
                               missed_selected_node_proximity=selected_proximity["counts"]))
        current = inspected["current"]
        for reference in ("control","stock","v3"):
            other = inspected[reference]
            changes.append(dict(dataset=name,reference=reference,
                                correct_edges_lost=len(other[4]-current[4]),
                                correct_edges_gained=len(current[4]-other[4]),
                                false_edges_change=current[0]["edge_fp"]-other[0]["edge_fp"],
                                fn_change=current[0]["edge_fn"]-other[0]["edge_fn"]))
        provenance.append(dict(dataset=name,gt_metadata_sha256=sha256(gt_path / "zarr.json"),
                               candidate_bank=bank,
                               graphs={a:v[0]["graph_file_sha256"] for a,v in inspected.items()}))
        print(name,json.dumps({a:dict(score=v[0]["summary"]["score"],fp=v[0]["edge_fp"],fn=v[0]["edge_fn"]) for a,v in inspected.items()}),flush=True)
    summaries, totals = {}, {}
    for arm in ("current","control","stock","v3"):
        subset = [r for r in rows if r["arm"]==arm]
        summaries[arm] = {g:aggregate([r for r in subset if g=="pooled" or r["embryo"]==g],
                                    [r["dataset"] for r in subset if g=="pooled" or r["embryo"]==g])
                          for g in ("pooled","44b6","6bba")}
        fields = ("fp_categories","unmatched_fp_endpoint_distance")
        total = {f:dict(sum((Counter(r[f]) for r in subset),Counter())) for f in fields}
        total.update({f:sum(r[f] for r in subset) for f in ("gt_nodes","matched_nodes","available_gt_edges","fp_edges_from_predicted_forks","predicted_forks")})
        total["pooled_annotated_node_recall"] = total["matched_nodes"]/total["gt_nodes"]
        if arm in ("current","control"):
            ds = [d for d in diagnostics if d["arm"]==arm]
            for f in ("fn_reasons","unselected_candidate_context","missed_raw_node_proximity","missed_selected_node_proximity","v3_only_edge_reasons"):
                total[f] = dict(sum((Counter(d[f]) for d in ds),Counter()))
            for f in ("raw_matched_nodes","raw_gt_edges_available","boundary_edges","boundary_missed","interior_edges","interior_missed"):
                total[f] = sum(d[f] for d in ds)
        totals[arm] = total
    decompositions = {a:gap_decomposition(summaries["current"]["pooled"],summaries[a]["pooled"]) for a in ("control","stock","v3")}
    result = dict(created_utc=now(),study=STUDY,metric_revision=METRIC_REV,
                  panel_sha256=sha256(ROOT / "panel.json"),plan_sha256=sha256(OUT / "plan.json"),
                  summaries=summaries,totals=totals,rows=rows,diagnostics=diagnostics,edge_changes=changes,
                  gap_decompositions=decompositions,provenance=provenance,
                  scope="Fresh official matching and post-prediction diagnosis on the same six full clips. Graphs, candidate banks and scorer unchanged. No inference, fitting or parameter changes. Proximity diagnostics do not label every unmatched object as biologically false.")
    write_json(OUT / "error-analysis.json",result)
    print(json.dumps(dict(totals=totals,gap_decompositions=decompositions),indent=2),flush=True)


if __name__ == "__main__":
    main()
