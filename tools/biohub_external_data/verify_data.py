"""Validate prepared outputs and independent CSV correspondence."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from data_adapter import load_synthetic, points_to_heatmap, prediction_rows, CSV_COLUMNS

ROOT = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
ARCHIVE = ROOT.parent / "biohub-forum-archive"


def validate_graph(a):
    n = len(a["node_id"])
    t, e = a["t"], a["edges"]
    assert np.array_equal(a["node_id"], np.arange(n))
    assert e.ndim == 2 and e.shape[1] == 2
    assert np.all((e >= 0) & (e < n))
    assert np.all(t[e[:, 1]] == t[e[:, 0]] + 1)
    incoming = np.bincount(e[:, 1], minlength=n)
    assert incoming.max(initial=0) <= 1
    degree = np.bincount(e[:, 0], minlength=n)
    assert np.array_equal(np.sort(a["division_parent_ids"]), np.flatnonzero(degree >= 2))


def compare_original_csv(species, filename):
    source = ARCHIVE / "downloads/virtual-embryo-zoo" / filename
    with source.open() as f:
        rows = list(csv.DictReader(f))
    tracks = {}
    for d in rows:
        tracks.setdefault(int(d["track_id"]), []).append(d)
    original = {
        tuple(sorted((int(d["t"]), float(d["z"]), float(d["y"]), float(d["x"])) for d in ds)): k
        for k, ds in tracks.items()
    }
    with np.load(ROOT / f"prepared/zoo/{species}_graph.npz", allow_pickle=False) as a:
        t, xyz, track, parents = a["t"], a["zyx_source"], a["tracklet_id"], a["parent_tracklet"]
    ptr = np.r_[0, np.cumsum(np.bincount(track))]
    remap = {}
    for k in range(len(ptr)-1):
        u, v = ptr[k:k+2]
        signature = tuple(sorted(tuple(x) for x in np.column_stack([t[u:v], xyz[u:v]])))
        remap[k] = original[signature]
    assert len(rows) == len(t) and len(remap) == len(tracks)
    for k, source_id in remap.items():
        expected = int(tracks[source_id][0]["parent_track_id"])
        p = int(parents[k])
        assert (-1 if p < 0 else remap[p]) == expected
    return {"source": filename, "nodes": len(t), "tracklets": len(remap),
            "exact_time_coordinates_and_parent_relationships": True}


def main():
    report = {"date": "2026-09-08", "status": "passed", "checks": {}}
    checks = report["checks"]
    manifest = json.loads((ROOT / "prepared/synthetic_manifest.json").read_text())
    expected = json.loads((ROOT / "prepared/synthetic_audit.json").read_text())
    totals = {"nodes": 0, "edges": 0, "division_parents": 0, "static_centroids": 0}
    for r in manifest:
        assert (ROOT / r["source"]).is_file()
        with np.load(ROOT / r["labels"], allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        native, pooled, physical = (a[k] for k in ["zyx_native", "zyx_pooled", "zyx_um"])
        assert np.isfinite(native).all()
        assert np.all((native >= 0) & (native < [64, 256, 256]))
        np.testing.assert_allclose(pooled * [1, 4, 4], native, rtol=1e-6)
        np.testing.assert_allclose(physical, native * [1.625, 0.40625, 0.40625], rtol=1e-6)
        if r["kind"] == "sequence":
            validate_graph(a)
            assert len(np.unique(np.column_stack([a["t"], a["tracklet_id"]]), axis=0)) == len(a["t"])
            for key, value in [("nodes", len(a["t"])), ("edges", len(a["edges"])), ("division_parents", len(a["division_parent_ids"]))]:
                totals[key] += value
        else:
            totals["static_centroids"] += len(native)
    assert len(manifest) == 3713
    assert all(totals[k] == expected[k] for k in totals)
    checks["all_synthetic_labels"] = {"files": len(manifest), "coordinate_roundtrip": True,
                                    "graph_integrity_and_unique_tracklets": True, **totals}
    print("All 3,713 synthetic label files verified", flush=True)

    seq = load_synthetic("seq_0000")
    static = load_synthetic("vol_00000")
    native_static = load_synthetic("vol_00000", grid="native")
    assert seq["image"].shape == (6, 64, 64, 64)
    assert static["image"].shape == (1, 64, 64, 64)
    np.testing.assert_array_equal(static["image"], native_static["image"][:, :, ::4, ::4])
    for sample in [seq, static, native_static]:
        np.testing.assert_allclose(sample["zyx_image"] * sample["voxel_um_zyx"], sample["zyx_um"], rtol=1e-6)
    assert not seq["division_target_observed"][seq["t"] == 5].any()
    assert seq["is_division_parent"].sum() == 47
    try:
        load_synthetic("seq_0000", grid="native")
        raise AssertionError("Unavailable native images accepted")
    except ValueError:
        pass
    # Equal physical offsets along Z and X must have the same Gaussian value.
    h = points_to_heatmap((25, 64, 64), np.array([[12., 32., 32.]]), [1.625, .40625, .40625])
    assert h[12, 32, 32] == 1.0
    np.testing.assert_allclose(h[13, 32, 32], h[12, 32, 36])
    checks["image_loader"] = {"static_stride_equivalence": True, "sequence_no_double_sampling": True,
                             "physical_point_alignment": True, "last_frame_division_mask": True,
                             "anisotropic_heatmap": True}
    rows = list(prediction_rows("DEMO", seq["t"], seq["zyx_image"] * [1, 4, 4], seq["edges"]))
    assert all(len(row) == len(CSV_COLUMNS) for row in rows)
    assert [r[0] for r in rows] == list(range(len(rows)))
    assert len(rows) == len(seq["t"]) + len(seq["edges"])
    assert rows[0][5:8] == np.rint(seq["zyx_native"][0]).astype(int).tolist()
    for i, (u, v) in enumerate(seq["edges"]):
        assert rows[len(seq["t"])+i][8:] == [int(u), int(v)]
    try:
        list(prediction_rows("DEMO", [0, 2], np.zeros((2, 3)), [[0, 1]]))
        raise AssertionError("Non-adjacent edge accepted")
    except ValueError:
        pass
    checks["csv_writer"] = {"node_and_edge_fields": True, "native_roundtrip": True,
                            "invalid_temporal_edge_rejected": True, "demo_rows": len(rows)}

    zoo = json.loads((ROOT / "prepared/zoo_manifest.json").read_text())
    checks["zoo_graphs"] = []
    for r in zoo:
        with np.load(ROOT / r["prepared"], allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        validate_graph(a)
        assert len(a["t"]) == r["nodes"] and len(a["edges"]) == r["edges"]
        assert r["within_track_gaps"] == 0 and r["nonadjacent_parent_links"] == 0
        assert np.isfinite(a["zyx_source"]).all()
        checks["zoo_graphs"].append({"file": r["prepared"], "nodes": r["nodes"], "edges": r["edges"],
                                   "division_parents": r["division_parents"], "graph_integrity": True})
        print("Verified", r["prepared"], flush=True)
    checks["independent_source_csv_comparison"] = [
        compare_original_csv("ascidian", "ascidian_tracks_withSize.csv"),
        compare_original_csv("elegans", "elegans_tracks.csv"),
    ]
    with np.load(ROOT / "prepared/riken_animal_c_first10_frames.npz", allow_pickle=False) as f:
        assert len(np.unique(f["t"])) == 10
        assert np.isfinite(f["zyx_um"]).all()
        checks["riken_preview"] = {"frames": 10, "points": len(f["t"]), "finite_coordinates": True,
                                   "temporal_edges_included": False}
    hashes = []
    for p in sorted((ROOT / "prepared").rglob("*")):
        if p.is_file():
            with p.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            hashes.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                           "sha256": digest})
    (ROOT / "prepared_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    checks["prepared_files"] = {"count": len(hashes), "bytes": sum(x["bytes"] for x in hashes),
                                "checksum_manifest": "prepared_checksums.json"}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("All preparation checks passed", flush=True)


if __name__ == "__main__":
    main()
