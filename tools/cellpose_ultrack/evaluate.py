"""Fresh official scoring of complete Cellpose/ultrack node-and-edge graphs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import zarr

from tools.annotation_selection.common import OFFICIAL, METRIC_REV, graph_hash, write_json
from tools.annotation_selection.metric_adapter import aggregate, evaluate_graph, match_nodes
from tools.cellpose_ultrack.track import ROOT, REPO, validate_graph
from tools.detector_screen.cellpose_adapter import sha256


def assert_matching(matches, nodes, gt_nodes, spacing):
    pred = {int(r[0]): r for r in nodes}
    truth = {int(r[0]): r for r in gt_nodes}
    if len(matches) != len(set(matches.values())):
        raise ValueError("Nonbijective official matching")
    errors = []
    for a, b in matches.items():
        if pred[a][1] != truth[b][1]:
            raise ValueError("Official matcher crossed timepoints")
        distance = float(np.linalg.norm((pred[a][2:] - truth[b][2:]) * spacing))
        if distance > 7.000001:
            raise ValueError("Official matcher exceeded the 7 micrometer gate")
        errors.append(distance)
    return errors


def evaluate_one(root, clip, config):
    name = clip["dataset"]
    directory = root / "tracking" / name
    receipt = json.loads((directory / "result.json").read_text())
    if not receipt["full_clip"] or receipt["shape"] != clip["shape"]:
        raise ValueError("Incomplete movies cannot be used for the full competition metric")
    if receipt["config_sha256"] != sha256(config):
        raise ValueError("Scoring recipe differs from the actual tracker recipe")
    graph_path = directory / "graph.npz"
    if sha256(graph_path) != receipt["graph_sha256"]:
        raise ValueError("Tracking graph changed after completion")
    graph = np.load(graph_path, allow_pickle=False)
    nodes, edges = graph["nodes"], graph["edges"]
    validate_graph(nodes, edges, clip["shape"])
    ground_path = Path(clip["image_path"]).with_suffix(".geff")
    gt = zarr.open_group(ground_path, mode="r")
    gt_nodes = np.column_stack([gt["nodes/ids"][:], *[gt[f"nodes/props/{a}/values"][:] for a in "tzyx"]]).astype(np.int64)
    gt_edges = np.asarray(gt["edges/ids"][:], dtype=np.int64).reshape(-1, 2)
    metadata = dict(gt.attrs)["geff"]
    gt_scale = [next(a["scale"] for a in metadata["axes"] if a["name"] == axis) for axis in "zyx"]
    if not np.allclose(clip["spacing_um"], gt_scale):
        raise ValueError("Image and annotation scales disagree")
    estimate = metadata["extra"]["estimated_number_of_nodes"]
    output = root / "evaluation" / f"{name}.json"
    raw_paths = [root / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.npz"
                 for t in range(clip["shape"][0])]
    raw_hashes = [sha256(p) for p in raw_paths]
    stamp = dict(graph_sha256=sha256(graph_path), gt_graph_sha256=graph_hash(gt_nodes, gt_edges),
                 gt_metadata_sha256=sha256(ground_path / "zarr.json"), estimate=estimate,
                 tracking_receipt_sha256=sha256(directory / "result.json"),
                 config_sha256=sha256(config), metric_revision=METRIC_REV,
                 raw_cellpose_sha256s=raw_hashes)
    if output.exists():
        old = json.loads(output.read_text())
        if old["inputs"] != stamp:
            raise ValueError("Cached scoring inputs changed")
        return old
    started = time.perf_counter()
    row, matches, true_edges = evaluate_graph(name, nodes, edges, gt_nodes, gt_edges, clip["spacing_um"], estimate)
    errors = assert_matching(matches, nodes, gt_nodes, clip["spacing_um"])
    matched_gt = set(matches.values())
    available = sum(int(a) in matched_gt and int(b) in matched_gt for a, b in gt_edges)
    # Separately diagnose the unchanged detector's available endpoints before ultrack.
    # This matching cannot affect any saved prediction or tracking input.
    raw_parts = []
    count = 0
    for t in range(clip["shape"][0]):
        path = root / "predictions/cellpose_cpdino_vitb" / f"{name}-t{t:03}.npz"
        with np.load(path, allow_pickle=False) as data:
            centers = np.rint(data["centers_zyx"]).astype(np.int64)
        raw_parts.append(np.column_stack([np.arange(count, count + len(centers)),
                                          np.full(len(centers), t), centers]))
        count += len(centers)
    raw_nodes = np.concatenate(raw_parts).astype(np.int64)
    empty_edges = np.empty((0, 2), np.int64)
    validate_graph(raw_nodes, empty_edges, clip["shape"])
    raw_matches = match_nodes(raw_nodes, empty_edges, gt_nodes, gt_edges, clip["spacing_um"])
    assert_matching(raw_matches, raw_nodes, gt_nodes, clip["spacing_um"])
    raw_gt = set(raw_matches.values())
    raw_available = sum(int(a) in raw_gt and int(b) in raw_gt for a, b in gt_edges)
    row.update(embryo=clip["embryo"], inputs=stamp, gt_nodes=len(gt_nodes), gt_edges=len(gt_edges),
               gt_divisions=int(np.sum(np.unique(gt_edges[:, 0], return_counts=True)[1] == 2)),
               selected_edges=len(edges), predicted_divisions=receipt["predicted_divisions"],
               count_ratio=len(nodes) / estimate, adjustment_multiplier=1.1 - 0.1 * len(nodes) / estimate,
               matched_mean_error_um=float(np.mean(errors)) if errors else None,
               matching_gates_passed=True, gt_edges_with_matched_endpoints=available,
               linked_fraction_of_available_gt_edges=len(true_edges) / available if available else None,
               raw_cellpose=dict(nodes=len(raw_nodes), matched_nodes=len(raw_matches),
                                 gt_node_recall=len(raw_matches) / len(gt_nodes),
                                 gt_edges_with_matched_endpoints=raw_available,
                                 matched_gt_lost_after_ultrack=len(raw_gt - matched_gt),
                                 matched_gt_gained_after_ultrack=len(matched_gt - raw_gt)),
               seconds=time.perf_counter() - started)
    row["single_clip_summary"] = aggregate([row], [name])
    write_json(output, row)
    print(f"SCORED {name}: score={row['single_clip_summary']['score']:.8f}; "
          f"edge={row['edge_jaccard']:.8f}; nodes={len(nodes)}; "
          f"division TP/FP/FN={row['division_tp']}/{row['division_fp']}/{row['division_fn']}", flush=True)
    return row


def run(args):
    recipe = json.loads(args.config.read_text())
    revision = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    if revision != METRIC_REV or revision != recipe["metric_revision"]:
        raise ValueError("Official metric revision changed")
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Official metric checkout has modified tracked files")
    panel = json.loads((args.root / "panel.json").read_text())
    clips = [c for c in panel["clips"] if not args.datasets or c["dataset"] in args.datasets]
    if not clips or (args.datasets and len(clips) != len(set(args.datasets))):
        raise ValueError("Unknown dataset selection")
    rows = [evaluate_one(args.root, clip, args.config) for clip in clips]
    names = [r["dataset"] for r in rows]
    summaries = {"pooled": aggregate(rows, names)}
    for embryo in sorted({r["embryo"] for r in rows}):
        subset = [r for r in rows if r["embryo"] == embryo]
        summaries[embryo] = aggregate(subset, [r["dataset"] for r in subset])
    result = dict(schema=1, model=recipe.get("model_description", "Cellpose cpDINO-ViT-B + ultrack stock IoU/CBC"), complete_panel=not bool(args.datasets),
                  expected_datasets=[c["dataset"] for c in panel["clips"]], scored_datasets=names,
                  frames=sum(c["shape"][0] for c in clips), summaries=summaries, rows=rows,
                  recipe=recipe, config_sha256=sha256(args.config), panel_sha256=sha256(args.root / "panel.json"),
                  metric_revision=METRIC_REV,
                  metric_versions={p: importlib.metadata.version(p) for p in ("tracksdata", "polars", "rustworkx", "numpy")},
                  scope="Complete local training clips; no hidden-test or leaderboard evaluation. No fitted parameters. Reused embryos; inherited pretraining exposure unresolved.")
    if not args.datasets:
        write_json(args.root / "evaluation/summary.json", result)
        destination = REPO / "results" / args.root.name
        compact = json.loads(json.dumps(result))
        for row in compact["rows"]:
            row["inputs"].pop("raw_cellpose_sha256s", None)
        write_json(destination / "summary.json", compact)
        write_json(destination / "panel.json", {k: v for k, v in panel.items() if k != "frames"})
        write_json(destination / "tracking-receipts.json", [json.loads((args.root / "tracking" / n / "result.json").read_text()) for n in names])
    print(json.dumps({"scored_clips": len(names), "complete_panel": not bool(args.datasets), "summaries": summaries}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=REPO / "configs/cellpose-ultrack-windowed-v1.json")
    parser.add_argument("--datasets", nargs="+")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
