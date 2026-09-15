"""Run pinned ultrack on complete Cellpose mask movies without reading annotations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import resource
import sqlite3
import subprocess
import time

import numpy as np
import tifffile

from tools.detector_screen.cellpose_adapter import sha256, write_json

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "work/cellpose-ultrack-20260914"
HEADER = "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id".split(",")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class LabelMovie:
    """Read one native TIFF at a time through ultrack's ArrayLike interface."""

    def __init__(self, paths, shape):
        self.paths = paths
        self.shape = tuple(shape)
        self.dtype = np.dtype("uint32")

    def __getitem__(self, t):
        labels = tifffile.imread(self.paths[t])
        if labels.shape != self.shape[1:] or labels.dtype.kind not in "iu":
            raise ValueError(f"Invalid native mask: {self.paths[t]}")
        return labels


def validate_graph(nodes, edges, shape):
    if nodes.ndim != 2 or nodes.shape[1] != 5 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Invalid graph shape")
    if nodes.dtype.kind not in "iu" or edges.dtype.kind not in "iu":
        raise ValueError("Submission fields must be integers")
    if len(np.unique(nodes[:, 0])) != len(nodes) or np.any(nodes < 0):
        raise ValueError("Duplicate IDs or negative fields")
    if np.any(nodes[:, 1:] >= np.asarray(shape)):
        raise ValueError("Out-of-bounds time or spatial coordinates")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("Duplicate temporal edges")
    times = {int(row[0]): int(row[1]) for row in nodes}
    for source, target in edges:
        if source not in times or target not in times or times[target] != times[source] + 1:
            raise ValueError("Missing parent or nonconsecutive temporal edge")
    if len(edges):
        if np.max(np.unique(edges[:, 1], return_counts=True)[1]) > 1:
            raise ValueError("Graph contains a merge")
        if np.max(np.unique(edges[:, 0], return_counts=True)[1]) > 2:
            raise ValueError("Graph contains more than two daughters")


def write_csv(path, dataset, nodes, edges):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for index, row in enumerate(nodes):
            writer.writerow([index, dataset, "node", *map(int, row), -1, -1])
        for index, (source, target) in enumerate(edges, len(nodes)):
            writer.writerow([index, dataset, "edge", -1, -1, -1, -1, -1, int(source), int(target)])
    # Check the actual competition serialization, including IDs and rounding.
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError("Submission header mismatch")
        nn, ee = [], []
        for index, row in enumerate(reader):
            if int(row["id"]) != index or row["dataset"] != dataset:
                raise ValueError("CSV index or dataset changed")
            if row["row_type"] == "node":
                nn.append([int(row[k]) for k in ("node_id", "t", "z", "y", "x")])
            elif row["row_type"] == "edge":
                ee.append([int(row[k]) for k in ("source_id", "target_id")])
            else:
                raise ValueError("Unknown CSV row type")
        if not np.array_equal(nodes, np.asarray(nn, dtype=np.int64).reshape(-1, 5)):
            raise ValueError("CSV node round trip changed")
        if not np.array_equal(edges, np.asarray(ee, dtype=np.int64).reshape(-1, 2)):
            raise ValueError("CSV edge round trip changed")


def audit_centroids(config, spacing):
    """Require SQL, serialized linker points, and actual rounded mask means to agree."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    engine = sqla.create_engine(config.data_config.database_path)
    count, largest = 0, 0.0
    with Session(engine) as session:
        rows = session.query(NodeDB.id, NodeDB.z, NodeDB.y, NodeDB.x, NodeDB.pickle).all()
        for node_id, z, y, x, node in rows:
            point = np.asarray(node.centroid, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Invalid serialized mask centroid")
            error = float(np.linalg.norm((point - [z, y, x]) * spacing))
            measured = np.rint(np.asarray(np.nonzero(node.mask)).mean(axis=1) + node.bbox[:3])
            if error > 1e-9 or not np.array_equal(point, measured):
                raise ValueError(f"Mask, linking, and export coordinates disagree: {node_id}")
            count += 1
            largest = max(largest, error)
    engine.dispose()
    return dict(nodes=count, maximum_disagreement_um=largest,
                mask_sql_and_serialized_linker_centroids_agree=True, coordinates_changed=False)


def run_clip(args, clip, recipe):
    import ultrack
    from ultrack.core.solve.solver.mip_solver import MIPSolver
    from ultrack.utils import labels_to_contours

    name = clip["dataset"]
    shape = list(clip["shape"])
    if args.frames:
        shape[0] = args.frames
    directory = args.root / (f"smoke-{args.frames}" if args.frames else "tracking") / name
    directory.mkdir(parents=True, exist_ok=True)
    if Path(ultrack.__file__).resolve().parent != args.ultrack_repo.resolve() / "ultrack":
        raise ValueError("Set PYTHONPATH to the pinned ultrack source checkout")
    revision = subprocess.check_output(["git", "-C", str(args.ultrack_repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != recipe["ultrack_revision"]:
        raise ValueError("ultrack revision mismatch")
    paths, masks, presets = [], [], set()
    for t in range(shape[0]):
        key = f"{name}-t{t:03}"
        mask = args.root / "cellpose/volumes" / f"{key}-masks.tif"
        receipt_path = args.root / "predictions/cellpose_cpdino_vitb" / f"{key}.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["input_shape_zyx"] != shape[1:] or receipt["spacing_um"] != clip["spacing_um"]:
            raise ValueError("Cellpose geometry differs from tracking manifest")
        if receipt["input_sha256"] != sha256(args.root / "images" / name / f"t{t:03}.npy"):
            raise ValueError("Cellpose input image changed")
        presets.add(receipt["preset_sha256"])
        paths.append(mask)
        masks.append(dict(time=t, mask_sha256=sha256(mask), receipt_sha256=sha256(receipt_path),
                          instances=receipt["n_centers"]))
    if len(presets) != 1:
        raise ValueError("Mixed Cellpose recipes")
    assets = json.loads((args.root / "cellpose/assets.json").read_text())
    if assets["preset"]["checkpoint_sha256"] != recipe["cellpose"]["checkpoint_sha256"]:
        raise ValueError("Wrong Cellpose checkpoint")
    cfg = ultrack.MainConfig(data={"working_dir": directory},
                             segmentation=recipe["segmentation"], linking=recipe["linking"],
                             tracking=recipe["tracking"])
    evidence, evidence_stamp = None, {}
    if recipe.get("division_evidence"):
        from tools.cellpose_ultrack.event_costs import bank_fingerprint
        from tools.cellpose_ultrack.learned_divisions import OUT as DIVISION_STUDY_OUT
        settings = recipe["division_evidence"]
        evidence_path = args.root / settings["directory"] / f"{name}.npz"
        evidence_receipt = json.loads(evidence_path.with_suffix(".json").read_text())
        evidence_plan_path = DIVISION_STUDY_OUT / "plan.json"
        evidence_plan = json.loads(evidence_plan_path.read_text())
        configuration_plan_path = Path(recipe.get("configuration_plan", evidence_plan_path))
        configuration_plan = json.loads(configuration_plan_path.read_text())
        if (sha256(evidence_plan_path) != evidence_receipt["inputs"]["plan_sha256"]
                or sha256(args.config) != configuration_plan["configs"][settings["arm"]]["sha256"]
                or settings["scale"] != evidence_plan["evidence_scale"]
                or settings["logit_clip"] != evidence_plan["logit_clip"]):
            raise ValueError("Division evidence settings differ from the declared study")
        if configuration_plan_path != evidence_plan_path and configuration_plan["evidence_plan_sha256"] != sha256(evidence_plan_path):
            raise ValueError("Follow-up study changed the frozen optical evidence plan")
        if evidence_receipt["inputs"]["code_sha256"] != sha256(REPO / "tools/cellpose_ultrack/learned_divisions.py"):
            raise ValueError("Frozen optical adapter changed")
        if (evidence_receipt["sha256"] != sha256(evidence_path)
                or evidence_receipt["inputs"]["clip"] != clip
                or evidence_receipt["inputs"]["source_bank"] != bank_fingerprint(directory / "data.db")):
            raise ValueError("Division evidence does not belong to this frozen candidate bank")
        if evidence_receipt["source_embryo"] == clip["embryo"] or evidence_receipt["annotations_used"]:
            raise ValueError("Wrong optical model source or annotation use")
        with np.load(evidence_path, allow_pickle=False) as data:
            evidence = dict(pairs=data["pairs"], bonus=data["bonus"])
        if settings["arm"] == "pair-control":
            evidence["bonus"] = np.zeros_like(evidence["bonus"])
        elif settings["arm"] != "pair-image":
            raise ValueError("Unknown learned division arm")
        evidence_stamp = dict(division_evidence_sha256=sha256(evidence_path),
                              division_evidence_receipt_sha256=sha256(evidence_path.with_suffix(".json")),
                              division_adapter_sha256=sha256(REPO / "tools/cellpose_ultrack/learned_divisions.py"))
    if recipe.get("division_persistence"):
        if evidence is None or recipe["division_persistence"] != dict(continuation_steps=1):
            raise ValueError("Persistence requires explicit pairs and exactly one continuation step")
        evidence_stamp.update(division_persistence_adapter_sha256=sha256(REPO / "tools/cellpose_ultrack/persistent_divisions.py"),
                              configuration_plan_sha256=sha256(configuration_plan_path))
    fingerprint = digest(dict(clip=clip, frames=shape[0], masks=masks,
                              config=cfg.model_dump(mode="json"), recipe_sha256=sha256(args.config), **evidence_stamp))
    lock = directory / "inputs.json"
    input_record = dict(fingerprint=fingerprint, clip=clip, shape=shape, masks=masks,
                        config=cfg.model_dump(mode="json"), source_revision=revision, **evidence_stamp)
    reuse_path = directory / "reuse.json"
    if reuse_path.exists() and not lock.exists():
        reuse = json.loads(reuse_path.read_text())
        old = reuse["source_inputs"]
        if old["clip"] != clip or old["shape"] != shape or old["masks"] != masks:
            raise ValueError("Reused hypotheses belong to different masks or clip")
        for field in ("segmentation_config", "linking_config"):
            if old["config"][field] != input_record["config"][field]:
                raise ValueError("Reused hypotheses or links have a different recipe")
        if sha256(directory / "data.db") != reuse["initial_database_sha256"]:
            raise ValueError("Reused database changed before its initial validation")
    if lock.exists() and json.loads(lock.read_text()) != input_record:
        raise ValueError("Tracking inputs changed; use a separate study root")
    write_json(lock, input_record)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["fingerprint"] != fingerprint or sha256(directory / "graph.npz") != result["graph_sha256"]:
            raise ValueError("Persisted graph changed")
        print(f"Already tracked: {name}", flush=True)
        return result
    labels = LabelMovie(paths, shape)
    timing_path = directory / "stages.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    started = time.perf_counter()
    if "convert_seconds" not in timings:
        fg, contours = labels_to_contours(labels, foreground_store_or_path=directory / "foreground.zarr",
                                         contours_store_or_path=directory / "contours.zarr", overwrite=True)
        timings["convert_seconds"] = time.perf_counter() - started
        write_json(timing_path, timings)
    else:
        import zarr
        fg = zarr.open_array(directory / "foreground.zarr", mode="r")
        contours = zarr.open_array(directory / "contours.zarr", mode="r")
    if "segment_seconds" not in timings:
        started = time.perf_counter()
        ultrack.segment(fg, contours, cfg, overwrite=True)
        timings["segment_seconds"] = time.perf_counter() - started
        write_json(timing_path, timings)
    centroid_path = directory / "centroid-audit.json"
    if not centroid_path.exists():
        if "link_seconds" in timings:
            raise ValueError("Coordinates must be audited before candidate linking")
        write_json(centroid_path, audit_centroids(cfg, clip["spacing_um"]))
    if "link_seconds" not in timings:
        started = time.perf_counter()
        # No auxiliary detector, learned association model, flow, or GT enters linking.
        ultrack.link(cfg, scale=clip["spacing_um"], overwrite=True)
        timings["link_seconds"] = time.perf_counter() - started
        write_json(timing_path, timings)
    solver_records = []
    original_optimize = MIPSolver.optimize

    def observe_optimize(solver):
        start = time.perf_counter()
        pair_record = {}
        if evidence is not None:
            from tools.cellpose_ultrack.learned_divisions import attach_pair_objective
            pair_record = dict(division_evidence=attach_pair_objective(solver, **evidence))
        if recipe.get("division_persistence"):
            from tools.cellpose_ultrack.persistent_divisions import attach_from_database
            pair_record["division_persistence"] = attach_from_database(solver, directory / "data.db", shape[0] - 1)
        value = original_optimize(solver)
        model = solver._model
        solver_records.append(dict(status=model.status.name, objective=model.objective_value,
                                   objective_bound=model.objective_bound, gap=model.gap,
                                   seconds=time.perf_counter() - start,
                                   variables=model.num_cols, constraints=model.num_rows,
                                   solutions=model.num_solutions, **pair_record))
        return value

    started = time.perf_counter()
    MIPSolver.optimize = observe_optimize
    try:
        ultrack.solve(cfg, overwrite=True)
    finally:
        MIPSolver.optimize = original_optimize
    timings["solve_seconds"] = time.perf_counter() - started
    write_json(timing_path, timings)
    df, _ = ultrack.to_tracks_layer(cfg, include_parents=True, include_node_ids=True)
    nodes = np.column_stack([df["id"].to_numpy(np.int64), df["t"].to_numpy(np.int64),
                             np.rint(df[["z", "y", "x"]].to_numpy()).astype(np.int64)])
    edges = df.loc[df.parent_id >= 0, ["parent_id", "id"]].to_numpy(np.int64).reshape(-1, 2)
    validate_graph(nodes, edges, shape)
    persistence_check = {}
    if recipe.get("division_persistence"):
        from tools.cellpose_ultrack.persistent_divisions import validate_persistence
        persistence_check = dict(division_persistence_check=validate_persistence(nodes, edges, shape[0] - 1))
    # Read actual ultrack SQL exclusions back, including hierarchy exclusions.
    with sqlite3.connect(directory / "data.db") as conn:
        counts = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("nodes", "links", "overlaps")}
        conflicts = conn.execute("SELECT count(*) FROM overlaps o JOIN nodes a ON a.id=o.node_id "
                                 "JOIN nodes b ON b.id=o.ancestor_id WHERE a.selected AND b.selected").fetchone()[0]
        if conflicts:
            raise ValueError("Selected mutually exclusive masks")
    df.to_csv(directory / "observations.csv", index=False)
    np.savez_compressed(directory / "graph.npz", nodes=nodes, edges=edges,
                        float_centers_zyx=df[["z", "y", "x"]].to_numpy())
    write_csv(directory / "submission.csv", name, nodes, edges)
    result = dict(dataset=name, shape=shape, full_clip=shape[0] == clip["shape"][0],
                  fingerprint=fingerprint, graph_sha256=sha256(directory / "graph.npz"),
                  csv_sha256=sha256(directory / "submission.csv"), input_instances=sum(m["instances"] for m in masks),
                  selected_nodes=len(nodes), selected_edges=len(edges),
                  centroid_audit=json.loads(centroid_path.read_text()),
                  predicted_divisions=int(np.sum(np.unique(edges[:, 0], return_counts=True)[1] == 2)),
                  database_counts=counts, selected_overlap_conflicts=conflicts,
                  graph_checks_passed=True, csv_round_trip_passed=True,
                  solver=solver_records, stage_seconds=timings,
                  database_bytes=(directory / "data.db").stat().st_size,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  versions={p: importlib.metadata.version(p) for p in ("numpy", "zarr", "mip", "higra", "sqlalchemy")},
                  ultrack_source=str(ultrack.__file__), ultrack_revision=revision,
                  config_sha256=sha256(args.config),
                  common_stages_reused_from=json.loads(reuse_path.read_text())["source"] if reuse_path.exists() else None,
                  annotations_used=False, **evidence_stamp, **persistence_check)
    write_json(result_path, result)
    print("TRACKED " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=REPO / "configs/cellpose-ultrack-windowed-v1.json")
    parser.add_argument("--ultrack-repo", type=Path, default=Path("/home/mpf/code/kaggle/ultrack"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--frames", type=int, help="Bounded integration smoke only; excluded from full scoring")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    recipe = json.loads(args.config.read_text())
    panel = json.loads((args.root / "panel.json").read_text())
    clips = [c for c in panel["clips"] if not args.datasets or c["dataset"] in args.datasets]
    if not clips or (args.datasets and len(clips) != len(set(args.datasets))):
        raise ValueError("Unknown or empty dataset selection")
    for clip in clips:
        run_clip(args, clip, recipe)


if __name__ == "__main__":
    main()
