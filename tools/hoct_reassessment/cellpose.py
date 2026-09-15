"""Independent Cellpose mask features and upstream HOCT inference, without GT reads."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

from .common import CONFIG, CP, HOCT_SOURCE, OLD, ROOT, arrays, load_model, pilot, read, save, setup, sha, write


def guard():
    """Fail on annotation, native checkpoint, baseline graph, or native feature access."""
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0]).resolve()
        if any(p.endswith(".geff") for p in path.parts):
            raise PermissionError("HOCT inference must not read annotations")
        if OLD in path.parents and not (HOCT_SOURCE in path.parents or OLD / "models/hoct" in path.parents):
            raise PermissionError(f"Standalone inference cannot use legacy evidence: {path.name}")
        if any(p in {"strong-tracker-v3", "segmentation-tracking-v6", "evaluation_matches", "evaluation"} for p in path.parts):
            raise PermissionError("Standalone inference cannot use prior graphs or labels")
    sys.addaudithook(audit)


def candidate_pairs(nodes, positions_um, neighbors=8, max_distance=15.):
    from scipy.spatial import cKDTree
    pairs = set()
    for t in range(int(nodes[:, 1].max())):
        source = np.flatnonzero(nodes[:, 1] == t)
        target = np.flatnonzero(nodes[:, 1] == t + 1)
        if not len(source) or not len(target):
            continue
        for query, reference, reverse in ((source, target, False), (target, source, True)):
            distances, indices = cKDTree(positions_um[reference]).query(
                positions_um[query], k=min(neighbors, len(reference)), distance_upper_bound=max_distance)
            distances = np.asarray(distances).reshape(len(query), -1)
            indices = np.asarray(indices).reshape(len(query), -1)
            for i, j in zip(*np.nonzero(np.isfinite(distances))):
                a, b = int(nodes[query[i], 0]), int(nodes[reference[indices[i, j]], 0])
                pairs.add((b, a) if reverse else (a, b))
    return np.asarray(sorted(pairs), np.int64).reshape(-1, 2)


def frame_features(labels, image):
    setup()
    from skimage.measure import regionprops
    from hoct.features.features import normalize_image, _border_dist_nd
    normalized = normalize_image(image)
    regions = regionprops(labels, intensity_image=normalized)
    positions = np.array([r.centroid for r in regions], np.float32).reshape(-1, 3)
    border = _border_dist_nd(positions, labels.shape)
    properties = np.array([[r.equivalent_diameter_area, r.intensity_min, r.intensity_max,
                            r.intensity_mean, r.intensity_std, *r.inertia_tensor.ravel(), b]
                           for r, b in zip(regions, border)], np.float32).reshape(-1, 15)
    if not np.isfinite(properties).all():
        raise ValueError("Nonfinite actual-mask features")
    return positions, properties, np.array([r.label for r in regions], np.int64)


def prepare_one(name):
    import tifffile
    setup()
    guard()
    clip = next(c for c in pilot() if c["dataset"] == name)
    config = read(CONFIG)["cellpose"]
    output = ROOT / "cellpose/features" / f"{name}.npz"
    assets = read(CP / "cellpose/assets.json")
    if assets["preset"]["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Unexpected detector checkpoint")
    sources = []
    for t in range(clip["shape"][0]):
        key = f"{name}-t{t:03}"
        image_path = CP / "images" / name / f"t{t:03}.npy"
        mask_path = CP / "cellpose/volumes" / f"{key}-masks.tif"
        receipt_path = CP / "predictions/cellpose_cpdino_vitb" / f"{key}.json"
        receipt = read(receipt_path)
        if receipt["input_sha256"] != sha(image_path):
            raise ValueError("Cellpose image changed")
        if receipt["input_shape_zyx"] != clip["shape"][1:] or receipt["spacing_um"] != clip["spacing_um"]:
            raise ValueError("Cellpose geometry changed")
        sources.append(dict(t=t, image=sha(image_path), mask=sha(mask_path), receipt=sha(receipt_path)))
    inputs = dict(config=sha(CONFIG), code=sha(__file__), sources=sources)
    if output.with_suffix(".json").exists():
        old = read(output.with_suffix(".json"))
        if old["inputs"] != inputs or old["sha256"] != sha(output):
            raise ValueError("Stale Cellpose features")
        return name
    started = time.monotonic()
    node_parts, pos_parts, prop_parts, label_parts = [], [], [], []
    count = 0
    for t in range(clip["shape"][0]):
        key = f"{name}-t{t:03}"
        labels = tifffile.imread(CP / "cellpose/volumes" / f"{key}-masks.tif")
        image = np.load(CP / "images" / name / f"t{t:03}.npy", allow_pickle=False)
        if labels.shape != tuple(clip["shape"][1:]) or image.shape != labels.shape:
            raise ValueError("Native image/mask shape mismatch")
        positions, props, label_ids = frame_features(labels, image)
        raw = arrays(CP / "predictions/cellpose_cpdino_vitb" / f"{key}.npz")
        if not np.array_equal(np.rint(positions).astype(int), np.rint(raw["centers_zyx"]).astype(int)):
            raise ValueError("Mask centroids changed relative to frozen detector outputs")
        node_parts.append(np.column_stack([np.arange(count, count + len(positions)), np.full(len(positions), t),
                                          np.rint(positions)]).astype(np.int64))
        pos_parts.append(positions)
        prop_parts.append(props)
        label_parts.append(label_ids)
        count += len(positions)
    nodes, positions, properties = map(np.concatenate, (node_parts, pos_parts, prop_parts))
    pairs = candidate_pairs(nodes, positions * clip["spacing_um"], config["n_neighbors"], config["distance_um"])
    save(output, nodes=nodes, positions=positions, properties=properties, pairs=pairs, mask_labels=np.concatenate(label_parts))
    write(output.with_suffix(".json"), dict(dataset=name, inputs=inputs, sha256=sha(output), nodes=len(nodes),
          edges=len(pairs), seconds=time.monotonic()-started, annotation_access_blocked=True,
          native_model_and_graph_access_blocked=True, all_cellpose_centers_preserved=True))
    print("CELLPOSE FEATURES", name, len(nodes), len(pairs), round(time.monotonic()-started, 1), flush=True)
    return name


def feature_graph(data, shape):
    setup()
    import polars as pl
    import tracksdata as td
    from hoct.features import REGIONPROPS, add_delta_t
    graph = td.graph.IndexedRXGraph()
    for key in ["z", "y", "x", *REGIONPROPS]:
        tensor = key == "inertia_tensor"
        graph.add_node_attr_key(key, pl.Array(pl.Float32, (3, 3)) if tensor else pl.Float32,
                                np.zeros((3, 3), np.float32) if tensor else 0.)
    attributes = []
    for node, pos, props in zip(data["nodes"], data["positions"], data["properties"]):
        attributes.append(dict(t=int(node[1]), **dict(zip("zyx", pos.tolist())),
            equivalent_diameter_area=float(props[0]), intensity_min=float(props[1]), intensity_max=float(props[2]),
            intensity_mean=float(props[3]), intensity_std=float(props[4]), inertia_tensor=props[5:14].reshape(3, 3),
            border_dist=float(props[14])))
    ids = graph.bulk_add_nodes(attributes)
    mapping = {int(node[0]): int(gid) for node, gid in zip(data["nodes"], ids)}
    graph.bulk_add_edges([dict(source_id=mapping[int(a)], target_id=mapping[int(b)]) for a, b in data["pairs"]])
    add_delta_t(graph)
    graph.metadata.update(shape=tuple(shape), was_2d=False, scale=(1., 1.625, .40625, .40625),
                          feature_units="native voxels", candidate_distance_units="micrometers")
    return graph, mapping


def score_one(task):
    name, model_name, execution = task
    setup()
    guard()
    from hoct.data import TiledRoiDataset
    from hoct.data._transforms import Standardize
    from hoct._api import _MEAN, _STD
    from hoct.features import REGIONPROPS
    from hoct.inference import _predict
    from hoct.tracking import ILPSolverConfig
    from tracksdata.functional import TilingScheme

    logging.getLogger("hoct").setLevel(logging.ERROR)
    logging.getLogger("tracksdata").setLevel(logging.ERROR)
    clip = next(c for c in pilot() if c["dataset"] == name)
    config = read(CONFIG)["cellpose"]
    path = ROOT / "cellpose/features" / f"{name}.npz"
    score_directory = "scores" if execution == "cpu" else "scores_cooperative_cuda"
    output = ROOT / "cellpose" / score_directory / model_name / f"{name}.npz"
    inputs = dict(features=sha(path), feature_receipt=sha(path.with_suffix(".json")), config=sha(CONFIG),
                  code=sha(__file__), upstream=sha(_predict.__file__), checkpoint=read(CONFIG)["models"][model_name])
    if execution == "cooperative_cuda":
        from .runtime import CooperativeModel
        inputs["runtime"] = sha(sys.modules[CooperativeModel.__module__].__file__)
    if output.with_suffix(".json").exists():
        old = read(output.with_suffix(".json"))
        if old["inputs"] != inputs or old["sha256"] != sha(output):
            raise ValueError("Stale Cellpose HOCT scores")
        return str(output)
    data = arrays(path)
    graph, mapping = feature_graph(data, clip["shape"])
    model = load_model(model_name)
    if execution == "cooperative_cuda":
        model = CooperativeModel(model)
    scheme = TilingScheme(tile_shape=tuple(config["tile_shape_tzyx"]), overlap_shape=tuple(config["tile_overlap_tzyx"]))
    dataset = TiledRoiDataset(graph, REGIONPROPS, scheme, dict_transforms=[Standardize(_MEAN, _STD)])
    solver_config = ILPSolverConfig(**config["decoder"], tracklet_solver=False, timeout=120.)
    started = time.monotonic()
    hook_calls = []
    def score_only(**kwargs):
        assert kwargs["graph"] is graph
        hook_calls.append(True)
        return graph
    # The upstream API always calls its solver. Intercept only that last call:
    # feature transforms, model, overlapping-window medians and normalization run upstream.
    try:
        with patch.object(_predict, "solve_tracking", side_effect=score_only):
            _predict.model_predict(model, dataset, solver_config=solver_config, return_solution=False, prefetch=False)
    finally:
        if execution == "cooperative_cuda":
            model.release()
    assert len(hook_calls) == 1
    reverse = {gid: nid for nid, gid in mapping.items()}
    edge_values = {(reverse[int(a)], reverse[int(b)]): float(s) for a, b, s in
                   graph.edge_attrs(attr_keys=["source_id", "target_id", "similarity"]).select("source_id", "target_id", "similarity").iter_rows()}
    node_values = {reverse[int(i)]: float(p) for i, p in graph.node_attrs(attr_keys=["node_id", "orphan_prob"]).select("node_id", "orphan_prob").iter_rows()}
    similarity = np.array([edge_values[tuple(p)] for p in data["pairs"]], np.float64)
    orphan = np.array([node_values[int(n[0])] for n in data["nodes"]], np.float64)
    if np.any((similarity < 0) | (similarity > 1)) or not np.isfinite(similarity).all():
        raise ValueError("Upstream inference left candidates unscored")
    if not np.isfinite(orphan).all() or np.any((orphan < 0) | (orphan > 1)):
        raise ValueError("Invalid orphan probabilities")
    ni = {int(n[0]): i for i, n in enumerate(data["nodes"])}
    totals = np.zeros(len(data["nodes"]))
    targets = np.array([ni[int(b)] for b in data["pairs"][:, 1]])
    np.add.at(totals, targets, similarity)
    used = np.unique(targets)
    error = float(np.max(np.abs(totals[used] + orphan[used] - 1)))
    if error > 2e-5:
        raise ValueError("Parent probabilities do not sum to one including orphan")
    save(output, similarity=similarity, orphan=orphan)
    write(output.with_suffix(".json"), dict(dataset=name, model=model_name, inputs=inputs, sha256=sha(output),
          seconds=time.monotonic()-started, candidate_edges=len(similarity), normalization_max_error=error,
          execution=model.receipt() if execution == "cooperative_cuda" else {"device": "CPU FP32"},
          annotation_access_blocked=True, native_model_and_graph_access_blocked=True,
          scoring="Upstream TiledRoiDataset and model_predict; only final solver intercepted"))
    print("CELLPOSE HOCT", model_name, name, round(time.monotonic()-started, 1), flush=True)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "score"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+", default=["general_v1", "ctc_v0"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execution", choices=["cpu", "cooperative_cuda"], default="cpu")
    args = parser.parse_args()
    names = args.datasets or read(CONFIG)["pilot_clips"]
    if args.execution == "cooperative_cuda" and args.workers != 1:
        raise ValueError("Use one shared-queue GPU worker")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = names if args.stage == "prepare" else [(n, m, args.execution) for n in names for m in args.models]
        list(pool.map(prepare_one if args.stage == "prepare" else score_one, tasks))


if __name__ == "__main__":
    main()
