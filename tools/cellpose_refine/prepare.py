"""Create separate source supervision and annotation-free inference manifests."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
import zarr

from .common import (DATA, EMBRYOS, REPO, RESULTS, SCREEN, SPACING, WORK,
                     assert_source, config, json_hash, now, save_npz, sha, write_json)


def jitter_queries(gt, shape, key, cfg):
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 5)
    rng = np.random.default_rng(int(hashlib.sha256((str(cfg["jitter_seed"]) + key).encode()).hexdigest()[:16], 16))
    d = cdist(gt[:, 2:] * SPACING, gt[:, 2:] * SPACING)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1) if len(gt) else np.empty(0)
    queries, target, weights, ids, kinds = [], [], [], [], []
    for i, node in enumerate(gt):
        center = node[2:]
        radius = min(cfg["jitter_radius_um"], nearest[i] * cfg["jitter_neighbor_fraction"])
        for j in range(cfg["jitters_per_gt"] + 1):
            if j == 0:
                delta = np.zeros(3)
            else:
                v = rng.normal(size=3)
                delta = v / np.linalg.norm(v) * radius * rng.uniform() ** (1 / 3)
            q = np.clip(center + delta / SPACING, 0, np.asarray(shape) - 1)
            queries.append(q); target.append((center - q) * SPACING)
            weights.append(cfg["zero_offset_weight"] if j == 0 else cfg["jitter_weight"])
            ids.append(int(node[0])); kinds.append(j != 0)
    return {"queries": np.asarray(queries, dtype=np.float64).reshape(-1, 3),
            "target_um": np.asarray(target, dtype=np.float32).reshape(-1, 3),
            "weight": np.asarray(weights, dtype=np.float32), "gt_ids": np.asarray(ids, dtype=np.int64),
            "kind": np.asarray(kinds, dtype=np.int8)}


def natural_targets(queries, gt, cfg):
    target = np.zeros((len(queries), 3), dtype=np.float32)
    weight = np.zeros(len(queries), dtype=np.float32)
    gt_ids = np.full(len(queries), -1, dtype=np.int64)
    if not len(queries) or not len(gt):
        return dict(target_um=target, weight=weight, gt_ids=gt_ids)
    distance = cdist(queries * SPACING, gt[:, 2:] * SPACING)
    best_gt = distance.argmin(axis=1)
    best_query = distance.argmin(axis=0)
    for g, q in enumerate(best_query):
        other = np.delete(distance[q], g)
        margin = float(other.min() - distance[q, g]) if len(other) else np.inf
        if best_gt[q] != g or distance[q, g] > cfg["natural_max_match_um"] or margin < cfg["natural_min_other_gt_margin_um"]:
            continue
        delta = (gt[g, 2:] - queries[q]) * SPACING
        delta /= max(1., np.linalg.norm(delta) / cfg["target_clip_radius_um"])
        target[q] = delta; weight[q] = cfg["natural_query_weight"]; gt_ids[q] = int(gt[g, 0])
    return dict(target_um=target, weight=weight, gt_ids=gt_ids)


def main():
    cfg = config()
    if (WORK / "feature-lock.json").exists():
        old = json.loads((WORK / "feature-lock.json").read_text())
        if old["config_sha256"] != json_hash(cfg):
            raise ValueError("Study already has a different feature/data lock")
        print("Preparation already locked", flush=True)
        return
    panel = json.loads((SCREEN / "panel.json").read_text())
    baseline_hashes = json.loads((SCREEN / "evaluation/output-hashes/cellpose_cpdino_vitb.json").read_text())
    images = {r["key"]: r for r in panel["frames"]}
    inference = []
    for row in panel["frames"]:
        r = row.copy()
        r["prediction_path"] = str(SCREEN / "predictions/cellpose_cpdino_vitb" / (r["key"] + ".npz"))
        r["prediction_sha256"] = sha(r["prediction_path"])
        assert r["prediction_sha256"] == baseline_hashes[r["key"]]["prediction_sha256"]
        r["query_count"] = len(np.load(r["prediction_path"], allow_pickle=False)["centers_zyx"])
        r["cache_path"] = str(WORK / "features/inference" / r["embryo"] / (r["key"] + ".npz"))
        inference.append(r)
    write_json(WORK / "inputs/inference.json", {"scope": "annotation-free frozen Cellpose proposals", "rows": inference})
    source_summaries = {}
    for embryo in EMBRYOS:
        jitter_rows, natural_rows, source_files = [], [], []
        n_gt = 0
        for path in sorted(DATA.glob(embryo + "*.geff")):
            dataset = path.stem
            meta_path = DATA / (dataset + ".zarr") / "zarr.json"
            meta = json.loads(meta_path.read_text())
            actual_spacing = meta["attributes"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1:]
            np.testing.assert_array_equal(actual_spacing, SPACING)
            array_meta = DATA / (dataset + ".zarr/0/zarr.json")
            shape = json.loads(array_meta.read_text())["shape"]
            assert shape == [100, 64, 256, 256]
            graph = zarr.open_group(str(path), mode="r")
            nodes = np.column_stack([graph["nodes/ids"][:], *[graph[f"nodes/props/{axis}/values"][:] for axis in "tzyx"]]).astype(np.int64)
            source_files.append({"dataset": dataset, "nodes_sha256": hashlib.sha256(nodes.tobytes()).hexdigest(),
                                 "image_metadata_sha256": sha(meta_path), "array_metadata_sha256": sha(array_meta), "nodes": len(nodes)})
            for t in cfg["source_training_times"]:
                gt = nodes[nodes[:, 1] == t]
                key = f"{dataset}-t{t:03}"
                arrays = jitter_queries(gt, shape[1:], key, cfg)
                query_path = WORK / "queries/jitter" / embryo / (key + ".npz")
                save_npz(query_path, **arrays)
                r = dict(key=key, dataset=dataset, embryo=embryo, time=t, shape=shape[1:], spacing_um=actual_spacing,
                         image_path=images.get(key, {}).get("image_path"), raw_path=str(DATA / (dataset + ".zarr")),
                         query_path=str(query_path), query_sha256=sha(query_path), query_count=len(arrays["queries"]),
                         cache_path=str(WORK / "features/jitter" / embryo / (key + ".npz")))
                jitter_rows.append(r); n_gt += len(gt)
            for row in inference:
                if row["dataset"] != dataset:
                    continue
                gt = nodes[nodes[:, 1] == row["time"]]
                with np.load(row["prediction_path"], allow_pickle=False) as p:
                    queries = p["centers_zyx"].astype(np.float64)
                arrays = natural_targets(queries, gt, cfg)
                target_path = WORK / "queries/natural" / embryo / (row["key"] + ".npz")
                save_npz(target_path, **arrays)
                natural_rows.append({**row, "target_path": str(target_path), "target_sha256": sha(target_path),
                                     "supervised_queries": int((arrays["weight"] > 0).sum())})
        assert_source(embryo, jitter_rows + natural_rows)
        exposure = {"labels": [embryo], "unlabeled_images": [embryo], "calibration": [],
                    "teacher_construction": [], "normalization_fit": []}
        manifest = {"source_embryo": embryo, "adaptation_exposure": exposure,
                    "inherited_exposure": "cpdino supervised/pretraining acquisitions unverified",
                    "jitter_rows": jitter_rows, "natural_rows": natural_rows, "source_files": source_files}
        write_json(WORK / "inputs" / ("train-" + embryo + ".json"), manifest)
        source_summaries[embryo] = {"clips": len(source_files), "jitter_frames": len(jitter_rows),
                                  "jitter_gt_nodes": n_gt, "jitter_queries": sum(r["query_count"] for r in jitter_rows),
                                  "natural_frames": len(natural_rows), "natural_supervised_queries": sum(r["supervised_queries"] for r in natural_rows),
                                  "manifest_sha256": sha(WORK / "inputs" / ("train-" + embryo + ".json")),
                                  "adaptation_exposure": exposure, "source_files": source_files}
    lock = {"created_utc": now(), "config": cfg, "config_sha256": json_hash(cfg), "source_summaries": source_summaries,
            "inference_manifest_sha256": sha(WORK / "inputs/inference.json"),
            "feature_code_sha256": {f: sha(REPO / "tools/cellpose_refine" / f) for f in ("common.py", "features.py", "prepare.py")},
            "assessment_panel_sha256": sha(SCREEN / "panel.json"), "status": "data and feature recipe locked before extraction/training"}
    write_json(WORK / "feature-lock.json", lock)
    write_json(RESULTS / "registry.json", lock)
    print(json.dumps({e: {k: v for k, v in s.items() if k != "source_files"} for e, s in source_summaries.items()}), flush=True)


if __name__ == "__main__":
    main()
