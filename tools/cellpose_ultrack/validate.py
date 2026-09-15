"""Validate completed cohort receipts, benchmark mask parity, and runtime evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import tifffile

from tools.annotation_selection.common import write_json
from tools.cellpose_ultrack.track import REPO, ROOT, HEADER, validate_graph
from tools.detector_screen.cellpose_adapter import sha256


def main():
    panel = json.loads((ROOT / "panel.json").read_text())
    summary = json.loads((ROOT / "evaluation/summary.json").read_text())
    names = [c["dataset"] for c in panel["clips"]]
    if not summary["complete_panel"] or summary["scored_datasets"] != names:
        raise ValueError("Incomplete or changed scored cohort")
    config_hash = sha256(REPO / "configs/cellpose-ultrack-windowed-v1.json")
    if summary["config_sha256"] != config_hash:
        raise ValueError("Final recipe drift")
    assets = json.loads((ROOT / "cellpose/assets.json").read_text())
    source_checkouts = {}
    for source, path, expected in [
        ("cellpose", Path(assets["repo_path"]), assets["preset"]["cellpose_revision"]),
        ("dinov3", Path("/home/mpf/code/kaggle/dinov3"), assets["preset"]["dinov3_revision"]),
        ("ultrack", Path("/home/mpf/code/kaggle/ultrack"), summary["recipe"]["ultrack_revision"]),
    ]:
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if revision != expected or dirty:
            raise ValueError(f"Pinned source checkout changed: {source}")
        source_checkouts[source] = dict(path=str(path), revision=revision, tracked_files_clean=True)
    if sha256(Path(assets["checkpoint_path"])) != assets["preset"]["checkpoint_sha256"]:
        raise ValueError("Cellpose checkpoint changed")
    frame_rows, graph_rows = [], []
    for clip in panel["clips"]:
        name = clip["dataset"]
        directory = ROOT / "tracking" / name
        tracking = json.loads((directory / "result.json").read_text())
        evaluation = json.loads((ROOT / "evaluation" / f"{name}.json").read_text())
        if tracking["config_sha256"] != config_hash or not tracking["full_clip"]:
            raise ValueError("Partial graph or mixed tracking recipe")
        if not tracking["centroid_audit"]["mask_sql_and_serialized_linker_centroids_agree"]:
            raise ValueError("Coordinate audit failed")
        if tracking["graph_sha256"] != sha256(directory / "graph.npz"):
            raise ValueError("Graph changed")
        if tracking["csv_sha256"] != sha256(directory / "submission.csv"):
            raise ValueError("CSV changed")
        with np.load(directory / "graph.npz", allow_pickle=False) as graph:
            validate_graph(graph["nodes"], graph["edges"], clip["shape"])
            if len(graph["nodes"]) != evaluation["num_pred_nodes"] or len(graph["edges"]) != evaluation["selected_edges"]:
                raise ValueError("Scored graph counts changed")
        for t in range(clip["shape"][0]):
            key = f"{name}-t{t:03}"
            frame = json.loads((ROOT / "predictions/cellpose_cpdino_vitb" / f"{key}.json").read_text())
            if frame["preset_sha256"] != assets["preset_sha256"]:
                raise ValueError("Mixed Cellpose inference recipes")
            frame_rows.append(frame)
        if evaluation["edge_tp"] + evaluation["edge_fn"] != evaluation["gt_edges"]:
            raise ValueError("GT edge accounting changed")
        if evaluation["division_tp"] + evaluation["division_fn"] != evaluation["gt_divisions"]:
            raise ValueError("GT division accounting changed")
        graph_rows.append(tracking)
    screen = REPO / "work/detector-screen-20260914"
    if sha256(screen / "panel.json") != panel["source_panel_sha256"]:
        raise ValueError("Original detector benchmark panel changed")
    previous = json.loads((screen / "panel.json").read_text())
    parity = []
    for frame in previous["frames"]:
        if frame["dataset"] not in names or frame["role"] != "pilot":
            continue
        key = frame["key"]
        old_mask = tifffile.imread(screen / "cellpose/volumes" / f"{key}-masks.tif")
        new_mask = tifffile.imread(ROOT / "cellpose/volumes" / f"{key}-masks.tif")
        with np.load(screen / "predictions/cellpose_cpdino_vitb" / f"{key}.npz", allow_pickle=False) as old:
            with np.load(ROOT / "predictions/cellpose_cpdino_vitb" / f"{key}.npz", allow_pickle=False) as new:
                fields = {k: bool(np.array_equal(old[k], new[k])) for k in
                          ("centers_zyx", "instance_ids", "volumes_voxels")}
        match = dict(key=key, mask_equal=bool(np.array_equal(old_mask, new_mask)), fields=fields)
        if not match["mask_equal"] or not all(fields.values()):
            raise ValueError(f"Frozen Cellpose mask/center parity changed: {key}")
        parity.append(match)
    if len(parity) != 12:
        raise ValueError("Expected twelve original pilot frames for parity validation")
    processing = sum(r["timing_seconds"]["total"] for r in frame_rows)
    queue = sum(r["timing_seconds"]["gpu_lock_wait"] for r in frame_rows)
    tracking_seconds = sum(sum(r["stage_seconds"].values()) for r in graph_rows)
    combined_path = ROOT / "submission.csv"
    temporary = ROOT / "submission.csv.tmp"
    combined_rows = 0
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        for name in names:
            with (ROOT / "tracking" / name / "submission.csv").open(newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != HEADER:
                    raise ValueError("Combined CSV input header differs")
                for row in reader:
                    if row["dataset"] != name:
                        raise ValueError("Combined CSV dataset differs")
                    row["id"] = combined_rows
                    writer.writerow(row)
                    combined_rows += 1
    expected_rows = sum(r["selected_nodes"] + r["selected_edges"] for r in graph_rows)
    if combined_rows != expected_rows:
        raise ValueError("Combined CSV row count differs from validated graphs")
    temporary.replace(combined_path)
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  complete_clips=len(names), complete_frames=len(frame_rows),
                  graph_and_csv_hashes_passed=True, gt_edge_and_division_accounting_passed=True,
                  combined_csv=str(combined_path), combined_csv_sha256=sha256(combined_path),
                  combined_csv_rows=combined_rows,
                  benchmark_parity=parity, benchmark_parity_frames=len(parity),
                  cellpose_checkpoint_sha256=assets["preset"]["checkpoint_sha256"],
                  cellpose_source_revision=assets["preset"]["cellpose_revision"],
                  dinov3_source_revision=assets["preset"]["dinov3_revision"],
                  cellpose_preset_sha256=assets["preset_sha256"],
                  cellpose_preset=assets["preset"],
                  cellpose_runtime_versions={p: assets[p] for p in ("torch", "torchvision")},
                  source_checkouts=source_checkouts,
                  cellpose_processing_seconds=processing, cellpose_gpu_queue_seconds=queue,
                  cellpose_mean_processing_seconds=processing / len(frame_rows),
                  cellpose_projected_19900_volume_processing_hours=processing / len(frame_rows) * 19900 / 3600,
                  cellpose_peak_allocated_bytes=max(r["gpu_peak_allocated_bytes"] for r in frame_rows),
                  tracking_stage_seconds=tracking_seconds,
                  largest_tracking_process_rss_mib=max(r["peak_process_rss_mib"] for r in graph_rows),
                  solver_statuses=[dict(dataset=r["dataset"], solves=r["solver"]) for r in graph_rows],
                  test_receipts={p.name: p.read_text().strip() for p in
                                 [ROOT / "tests-full.log", ROOT / "test-hierarchy.log"]})
    write_json(ROOT / "validation.json", result)
    write_json(REPO / "results/cellpose-ultrack-20260914/validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("benchmark_parity", "solver_statuses")}, indent=2))


if __name__ == "__main__":
    main()
