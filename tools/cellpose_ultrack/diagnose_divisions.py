"""Inspect annotated division windows after tracking, without changing predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tracksdata as td
import zarr

from tools.annotation_selection.common import write_json
from tools.annotation_selection.metric_adapter import make_graph
from tools.cellpose_ultrack.track import REPO, ROOT
from tracking_cellmot import division_metrics as official


def window_evidence(matched, truth, divider):
    attrs = matched.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,
                                         td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    pairs = {int(p): int(g) for p, g in attrs.iter_rows() if g is not None and g != -1}
    sides = [{divider, *truth.predecessors(divider)},
             *[{child, *truth.successors(child)} for child in truth.successors(divider)]]
    evidence = [{p for p, g in pairs.items() if g in side} for side in sides]
    nearby = evidence[0] | {child for p in evidence[0] for child in matched.successors(p)}
    return dict(parent_side_matched=bool(evidence[0]),
                daughter_sides_matched=[bool(side) for side in evidence[1:]],
                all_three_sides_matched=all(bool(side) for side in evidence),
                predicted_forks_near_parent=sum(matched.out_degree(p) == 2 for p in nearby))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    root = parser.parse_args().root
    panel = json.loads((root / "panel.json").read_text())
    results = []
    for clip in panel["clips"]:
        name = clip["dataset"]
        score_path = root / "evaluation" / f"{name}.json"
        if not score_path.exists():
            continue
        score = json.loads(score_path.read_text())
        if not score["gt_divisions"]:
            continue
        path = root / "division-diagnostics" / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["graph_hash"] != score["graph_hash"]:
                raise ValueError("Previously diagnosed tracking graph changed")
            results.append(result)
            continue
        with np.load(root / "tracking" / name / "graph.npz", allow_pickle=False) as data:
            pred, _ = make_graph(data["nodes"], data["edges"])
        gt = zarr.open_group(Path(clip["image_path"]).with_suffix(".geff"), mode="r")
        truth_nodes = np.column_stack([gt["nodes/ids"][:],
                                      *[gt[f"nodes/props/{a}/values"][:] for a in "tzyx"]]).astype(np.int64)
        truth, gt_ids = make_graph(truth_nodes, np.asarray(gt["edges/ids"][:], dtype=np.int64))
        times = {int(row[0]): int(row[1]) for row in truth_nodes}
        raw_parts, count = [], 0
        for t in range(clip["shape"][0]):
            with np.load(root / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.npz", allow_pickle=False) as data:
                centers = np.rint(data["centers_zyx"]).astype(np.int64)
            raw_parts.append(np.column_stack([np.arange(count, count + len(centers)),
                                              np.full(len(centers), t), centers]))
            count += len(centers)
        raw, _ = make_graph(np.concatenate(raw_parts), np.empty((0, 2), dtype=np.int64))
        division_scores = official.score_divisions(pred, truth, tuple(clip["spacing_um"]), 7.0)
        if (sum(division_scores.scores.values()) != score["division_tp"]
                or len(division_scores.fp_forks) != score["division_fp"]):
            raise ValueError("Division diagnostics differ from the saved official score")
        selected_matches = official.match_divisions(pred, truth, tuple(clip["spacing_um"]), 7.0)
        raw_matches = official.match_divisions(raw, truth, tuple(clip["spacing_um"]), 7.0)
        windows = official.extract_divisions(truth)
        events = [dict(gt_dividing_node=gt_ids[divider], time=times[gt_ids[divider]],
                       recovered=bool(recovered),
                       raw_cellpose=window_evidence(raw_matches[divider], windows[divider], divider),
                       selected_ultrack=window_evidence(selected_matches[divider], windows[divider], divider))
                  for divider, recovered in division_scores.scores.items()]
        result = dict(dataset=name, graph_hash=score["graph_hash"], events=events,
                      description="Official local division-window matching; parent side includes its predecessor, daughter sides include their successors. Availability is a diagnostic, not an alternative division score.")
        write_json(path, result)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    if (root / "evaluation/summary.json").exists():
        summary = json.loads((root / "evaluation/summary.json").read_text())
        if sum(len(r["events"]) for r in results) != sum(r["gt_divisions"] for r in summary["rows"]):
            raise ValueError("Missing annotated division diagnostics")
        write_json(REPO / "results" / root.name / "division-diagnostics.json", results)


if __name__ == "__main__":
    main()
