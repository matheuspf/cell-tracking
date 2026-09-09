"""Prepare portable labels and previews for the downloaded Biohub resources.

Images remain in the original archive. Coordinates carry explicit units.
This script does not train a model, submit predictions, or alter source files.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from PIL import Image
import zarr

ROOT = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
ARCHIVE = ROOT.parent / "biohub-forum-archive"
SYNTH = ARCHIVE / "downloads/kaggle/biohub_synthetic"
ZOO = ARCHIVE / "downloads/virtual-embryo-zoo"
REAL = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
PREP = ROOT / "prepared"
NATIVE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
XY_STRIDE = np.array([1, 4, 4], dtype=np.float32)


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def split_for(key):
    # Entire independent synthetic examples stay together; this is not real CV.
    return "synthetic_holdout" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10 == 0 else "synthetic_train"


def tracklets(t, edges):
    """A new segment starts at a root or immediately after a fork."""
    n = len(t)
    incoming = np.bincount(edges[:, 1], minlength=n)
    outgoing = np.bincount(edges[:, 0], minlength=n)
    assert incoming.max(initial=0) <= 1
    parent = np.full(n, -1, np.int64)
    parent[edges[:, 1]] = edges[:, 0]
    segment = np.full(n, -1, np.int64)
    count = 0
    for i in np.argsort(t, kind="stable"):
        p = parent[i]
        if p < 0 or outgoing[p] != 1:
            segment[i] = count
            count += 1
        else:
            segment[i] = segment[p]
    return segment, outgoing


def synthetic():
    manifest = json.loads((SYNTH / "manifest.json").read_text())
    records = []
    totals = {"nodes": 0, "edges": 0, "division_parents": 0, "static_centroids": 0, "clone_time_collisions": 0}
    for kind, key in (("static", "static"), ("sequence", "sequences")):
        for item in manifest[key]:
            source = SYNTH / item["file"]
            out = PREP / "synthetic" / (source.stem + "_labels.npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            with np.load(source, allow_pickle=False) as a:
                if kind == "static":
                    pos = a["centroids"].astype(np.float32)
                    assert np.isfinite(pos).all() and (pos >= 0).all() and (pos < [64, 256, 256]).all()
                    np.savez_compressed(out, zyx_native=pos, zyx_pooled=pos / XY_STRIDE, zyx_um=pos * NATIVE_UM)
                    totals["static_centroids"] += len(pos)
                    fields = {"centroids": len(pos), "image_shape": [64, 256, 256], "image_key": "volume"}
                else:
                    n = a["nodes"]
                    t = n[:, 0].astype(np.int64)
                    xyz = n[:, 1:4].astype(np.float32)
                    e = a["edges"].astype(np.int64)
                    div = a["divisions"].astype(np.int64)
                    clone = n[:, 4].astype(np.int64)
                    assert np.array_equal(t, n[:, 0]) and np.isfinite(xyz).all()
                    assert e.min(initial=0) >= 0 and e.max(initial=0) < len(n)
                    assert np.all(t[e[:, 1]] == t[e[:, 0]] + 1)
                    assert (xyz >= 0).all() and (xyz < [64, 256, 256]).all()
                    segment, degrees = tracklets(t, e)
                    assert np.array_equal(np.flatnonzero(degrees >= 2), np.sort(div))
                    assert len(np.unique(np.column_stack([t, segment]), axis=0)) == len(t)
                    collisions = len(t) - len(np.unique(np.column_stack([t, clone]), axis=0))
                    totals["clone_time_collisions"] += collisions
                    np.savez_compressed(out, node_id=np.arange(len(n), dtype=np.int64), t=t,
                        zyx_native=xyz, zyx_pooled=xyz / XY_STRIDE, zyx_um=xyz * NATIVE_UM,
                        edges=e, division_parent_ids=div, tracklet_id=segment, source_clone_id=clone)
                    for label, value in (("nodes", len(n)), ("edges", len(e)), ("division_parents", len(div))):
                        totals[label] += value
                    fields = {"nodes": len(n), "edges": len(e), "division_parents": len(div),
                              "image_shape": [item["T"], 64, 64, 64], "image_key": "volumes"}
            records.append({"sample_id": source.stem, "kind": kind, "source": str(Path("..") / ARCHIVE.name / source.relative_to(ARCHIVE)),
                "labels": str(out.relative_to(ROOT)), "split": split_for(source.stem), **fields})
    dump(PREP / "synthetic_manifest.json", records)
    dump(PREP / "synthetic_audit.json", totals)
    print("Synthetic labels:", totals, flush=True)
    return records, totals


def zoo_graph(path):
    """Decode track points and direct parents; do not use closure entries as edges."""
    g = zarr.open_group(str(path), mode="r")
    ptr = np.asarray(g["tracks_to_points/indptr"][:], dtype=np.int64)
    point_ids = np.asarray(g["tracks_to_points/indices"][:], dtype=np.int64)
    pos = np.asarray(g["tracks_to_points/data"][:], dtype=np.float32)
    ntracks = len(ptr) - 1
    max_points = g["points"].shape[1] // int(g["points"].attrs["values_per_point"])
    t = point_ids // max_points
    track = np.repeat(np.arange(ntracks, dtype=np.int64), np.diff(ptr))
    assert np.isfinite(pos).all() and (pos > -9000).all()
    assert len(np.unique(point_ids)) == len(point_ids), "Point belongs to multiple tracklets"
    assert np.all(np.diff(t)[np.diff(track) == 0] > 0), "Track points must be ordered in time"
    ind = np.asarray(g["tracks_to_tracks/indices"][:], dtype=np.int64)
    par = np.asarray(g["tracks_to_tracks/data"][:], dtype=np.int64)
    ip = np.asarray(g["tracks_to_tracks/indptr"][:], dtype=np.int64)
    parent_track = np.full(ntracks, -1, np.int64)
    for k in range(ntracks):
        a, b = ip[k:k+2]
        j = np.searchsorted(ind[a:b], k)
        assert j < b-a and ind[a+j] == k, "Missing diagonal parent metadata"
        value = par[a+j]
        parent_track[k] = value - 1 if value > 0 else -1
    assert parent_track.max(initial=-1) < ntracks
    # The parent ID encoded for a column must be consistent in every closure row.
    decoded = np.where(par > 0, par - 1, -1)
    assert np.array_equal(decoded, parent_track[ind])
    within = np.flatnonzero((np.diff(track) == 0) & (np.diff(t) == 1))
    within_gaps = int(np.count_nonzero((np.diff(track) == 0) & (np.diff(t) != 1)))
    valid_child = np.flatnonzero((parent_track >= 0) & (np.diff(ptr) > 0))
    start = ptr[valid_child]
    end = ptr[parent_track[valid_child] + 1] - 1
    parent_nonempty = np.diff(ptr)[parent_track[valid_child]] > 0
    adjacent = parent_nonempty & (t[start] == t[end] + 1)
    e = np.vstack([np.column_stack([within, within + 1]), np.column_stack([end[adjacent], start[adjacent]])])
    degrees = np.bincount(e[:, 0], minlength=len(t))
    div = np.flatnonzero(degrees >= 2)
    result = {"source": str(Path("..") / ARCHIVE.name / path.relative_to(ARCHIVE)), "frames": int(g["points"].shape[0]),
              "nodes": len(t), "tracklets": ntracks, "edges": len(e), "division_parents": len(div),
              "within_track_gaps": within_gaps, "nonadjacent_parent_links": int((~adjacent).sum()),
              "units": "source coordinate units; physical scale and frame duration require source calibration",
              "coordinate_fields": list(g["points"].attrs["fields"]),
              "image_available": False, "label_quality": "source tracking results; not assumed manually verified"}
    out = PREP / "zoo" / (path.name.replace("tracks_", "").replace("_attributes_bundle.zarr", "") + "_graph.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, node_id=np.arange(len(t), dtype=np.int64), source_point_id=point_ids,
        t=t, zyx_source=pos, tracklet_id=track, parent_tracklet=parent_track,
        edges=e, division_parent_ids=div)
    result["prepared"] = str(out.relative_to(ROOT))
    print("Zoo:", out.name, result["nodes"], "nodes", result["edges"], "edges; gaps", within_gaps, flush=True)
    return result


def riken_preview():
    path = ROOT / "cache/zebrafish_animal_c_bd5.h5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = ARCHIVE / "downloads/ssbd/bdml/zebrafish_animal_c_bdml3.0.zip"
        with zipfile.ZipFile(archive) as zf:
            members = [x for x in zf.namelist() if x.endswith("/" + path.name) or x == path.name]
            if len(members) != 1:
                raise ValueError("Expected exactly one animal C HDF5 file")
            import shutil
            with zf.open(members[0]) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    with h5py.File(path, "r") as f:
        frames = sorted((int(k) for k in f["data"] if k.isdigit()))
        counts = [int(f[f"data/{k}/object/0"].shape[0]) for k in frames]
        positions, times, ids = [], [], []
        for t in frames[:10]:
            a = f[f"data/{t}/object/0"][:]
            positions.append(np.column_stack([a[k].astype(np.float32) for k in ["z", "y", "x"]]))
            times.append(np.full(len(a), t, dtype=np.int64))
            ids.append(a["ID"].astype("U16"))
        out = PREP / "riken_animal_c_first10_frames.npz"
        np.savez_compressed(out, t=np.concatenate(times), zyx_um=np.concatenate(positions),
                            measurement_id=np.concatenate(ids))
        feature_def = [{"id": int(x["fID"]), "name": x["name"].decode(), "unit": x["fUnit"].decode()} for x in f["data/featureDef"][:]]
    info = {"inspected_sample": path.name, "frames": len(frames), "point_measurements": sum(counts),
            "first10_frames_prepared": str(out.relative_to(ROOT)), "time_step_seconds": 90,
            "space_unit": "micrometer", "feature_definitions": feature_def,
            "explicit_temporal_edges_found": False,
            "inspection_scope": "Full structural/count audit of animal C, 10 frames exported; other six archives inventoried only"}
    dump(PREP / "riken_audit.json", info)
    return info


def image_url(volume):
    a = np.asarray(volume)
    lo, hi = np.quantile(a, [0.01, 0.995])
    projection = np.max(a, axis=0)
    px = np.clip((projection-lo)/max(hi-lo, 1), 0, 1)
    img = Image.fromarray(np.uint8(px*255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def previews():
    with np.load(SYNTH / "sequences/seq_0000.npz", allow_pickle=False) as a:
        v, n, e, div = a["volumes"], a["nodes"], a["edges"], a["divisions"]
        seq = [{"image": image_url(v[t]), "centroids": (n[n[:, 0] == t, 1:4] / XY_STRIDE).round(3).tolist()} for t in range(len(v))]
        parent = int(next(i for i in div if n[i, 0] == 2))
        daughters = e[e[:, 0] == parent, 1]
        fork = {"parent_id": parent, "daughter_ids": daughters.tolist(), "parent": n[parent, :4].tolist(), "daughters": n[daughters, :4].tolist()}
    with np.load(SYNTH / "static/vol_00000.npz", allow_pickle=False) as a:
        static = {"image": image_url(a["volume"]), "centroids": a["centroids"].round(3).tolist()}
    p = REAL / "train/44b6_0113de3b.zarr"
    a = zarr.open_group(str(p), mode="r")["0"][0]
    g = zarr.open_group(str(p.with_suffix('.geff')), mode='r')
    t = g['nodes/props/t/values'][:]
    xyz = np.column_stack([g[f'nodes/props/{ax}/values'][:] for ax in ['z','y','x']])
    real = {"image": image_url(a), "centroids": xyz[t == 0].tolist(), "sample": p.stem}
    return {"sequence": seq, "static": static, "real": real, "fork": fork}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-only", action="store_true",
                        help="Prepare labels without local HTML reference files or competition previews")
    args = parser.parse_args()
    PREP.mkdir(parents=True, exist_ok=True)
    records, totals = synthetic()
    zoo = [zoo_graph(p) for p in sorted(ZOO.glob("*attributes_bundle.zarr"))]
    dump(PREP / "zoo_manifest.json", zoo)
    riken = riken_preview()
    if args.labels_only:
        print("Prepared external labels; no competition inputs or presentation references required", flush=True)
        return
    visual = previews()
    baseline = json.loads((ROOT / "reference/local-train-audit.json").read_text())
    catalog = {"snapshot": "2026-09-08", "synthetic": totals,
               "synthetic_counts": {"static": 1539, "sequences": 2174},
               "synthetic_splits": {k: sum(x['split'] == k for x in records) for k in ['synthetic_train','synthetic_holdout']},
               "zoo": zoo, "riken": riken, "competition": baseline,
               "previews": visual, "native_um_per_voxel_zyx": NATIVE_UM.tolist()}
    dump(ROOT / "catalog.json", catalog)
    print("Prepared catalog and preview data", flush=True)


if __name__ == "__main__":
    main()
