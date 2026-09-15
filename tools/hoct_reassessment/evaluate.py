"""Fresh full competition metric, after freezing every graph in a comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import time

import numpy as np
import zarr

from .common import CONFIG, CP, OLD, RESULTS, ROOT, arrays, pilot, read, save, sha, write


def graph_for(name, arm):
    if arm in ("v3", "previous_H_general_J"):
        from tools.image_native_tracking_v5.common import graph
        return graph(name, "C0" if arm == "v3" else "H_general_J")
    if arm == "ultrack_no_division":
        return arrays(CP.parent / "cellpose-ultrack-event-costs-20260914-no-division-control/tracking" / name / "graph.npz")
    path = ROOT / "graphs" / arm / f"{name}.npz"
    receipt = read(path.with_suffix(".json"))
    if sha(path) != receipt["sha256"]:
        raise ValueError("Graph changed after inference")
    return arrays(path)


def one(task):
    arm, clip, expected = task
    from tools.annotation_selection.common import graph_hash
    from tools.annotation_selection.metric_adapter import evaluate_graph
    from tools.cellpose_ultrack.evaluate import assert_matching
    from tools.cellpose_ultrack.track import validate_graph

    name = clip["dataset"]
    graph = graph_for(name, arm)
    nodes, edges = graph["nodes"], graph["edges"]
    if graph_hash(nodes, edges) != expected:
        raise ValueError("Graph changed after comparison lock")
    validate_graph(nodes, edges, clip["shape"])
    truth_path = Path(clip["image_path"]).with_suffix(".geff")
    truth = zarr.open_group(truth_path, mode="r")
    gt_nodes = np.column_stack([truth["nodes/ids"][:], *[truth[f"nodes/props/{k}/values"][:] for k in "tzyx"]]).astype(np.int64)
    gt_edges = np.asarray(truth["edges/ids"][:], np.int64).reshape(-1, 2)
    metadata = truth.attrs["geff"]
    scale = [next(a["scale"] for a in metadata["axes"] if a["name"] == axis) for axis in "zyx"]
    np.testing.assert_array_equal(scale, clip["spacing_um"])
    estimate = metadata["extra"]["estimated_number_of_nodes"]
    inputs = dict(graph=expected, gt=graph_hash(gt_nodes, gt_edges), metadata=sha(truth_path / "zarr.json"),
                  metric=read(CONFIG)["metric_revision"], evaluator=sha(__file__))
    dest = ROOT / "evaluation" / arm / f"{name}.json"
    if dest.exists():
        existing = read(dest)
        if existing["inputs"] != inputs:
            raise ValueError("Stale evaluation")
        return existing
    started = time.monotonic()
    row, matches, tp = evaluate_graph(name, nodes, edges, gt_nodes, gt_edges, scale, estimate)
    errors = assert_matching(matches, nodes, gt_nodes, np.asarray(scale))
    available = set(matches.values())
    row.update(arm=arm, embryo=clip["embryo"], inputs=inputs, seconds=time.monotonic()-started,
               predicted_edges=len(edges), predicted_divisions=int(np.sum(np.unique(edges[:, 0], return_counts=True)[1] == 2)),
               gt_nodes=len(gt_nodes), gt_edges=len(gt_edges),
               gt_edges_with_matched_endpoints=sum(int(a) in available and int(b) in available for a, b in gt_edges),
               mean_matched_error_um=float(np.mean(errors)), full_clip=True)
    save(ROOT / "matches" / arm / f"{name}.npz", matches=np.array(sorted(matches.items()), np.int64).reshape(-1, 2),
         tp_edges=np.array(sorted(tp), np.int64).reshape(-1, 2))
    write(dest, row)
    print("SCORED", arm, name, "edges", row["edge_tp"], row["edge_fp"], row["edge_fn"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    from tools.annotation_selection.common import METRIC_REV, OFFICIAL, graph_hash
    from tools.annotation_selection.metric_adapter import aggregate
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip() != METRIC_REV:
        raise ValueError("Scorer checkout changed")
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Scorer has tracked modifications")
    clips = pilot()
    lock = {a: {c["dataset"]: graph_hash(**{k: graph_for(c["dataset"], a)[k] for k in ("nodes", "edges")})
                for c in clips} for a in args.arms}
    # All requested complete graphs exist and are frozen before any GT is read here.
    key = "-".join(args.arms)
    write(ROOT / "evaluation_locks" / f"{key}.json", dict(config=sha(CONFIG), graphs=lock, arms=args.arms))
    tasks = [(a, c, lock[a][c["dataset"]]) for a in args.arms for c in clips]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, tasks))
    summaries = {}
    for arm in args.arms:
        summaries[arm] = {}
        for embryo in ("pooled", "44b6", "6bba"):
            subset = [r for r in rows if r["arm"] == arm and (embryo == "pooled" or r["embryo"] == embryo)]
            summaries[arm][embryo] = aggregate(subset, [r["dataset"] for r in subset])
    result = dict(config_sha256=sha(CONFIG), clips=clips, frames=600, rows=rows, summaries=summaries,
                  scope="Six reused complete public clips, full official metric. Not all 199 clips or a hidden-test result.",
                  metric_revision=METRIC_REV)
    write(ROOT / "evaluation" / f"summary-{key}.json", result)
    write(RESULTS / f"summary-{key}.json", result)
    for arm in args.arms:
        print(arm, summaries[arm]["pooled"], flush=True)


if __name__ == "__main__":
    main()
