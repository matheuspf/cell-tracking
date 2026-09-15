"""Create small, reproducible trajectory and measurement examples for the guide.

Reads already prepared graphs and animal C's cached HDF5 in read-only mode.
The output is for presentation, not a new training-label release. Coordinate
values are rounded to five decimals only in the presentation JSON.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
LABELS = {
    "ascidian": "Ascidian", "drosophila": "Fruit fly", "elegans": "C. elegans",
    "mouse": "Mouse", "tribolium": "Flour beetle", "zebrafish": "Zebrafish",
}


def bbox(points):
    return {"min": np.round(points.min(axis=0).astype(float), 5).tolist(),
            "max": np.round(points.max(axis=0).astype(float), 5).tolist()}


def zoo_example(meta):
    path = ROOT / meta["prepared"]
    species = path.stem.removesuffix("_graph")
    with np.load(path, allow_pickle=False) as source:
        times = source["t"]
        positions = source["zyx_source"]
        tracks = source["tracklet_id"]
        parent_tracks = source["parent_tracklet"]
        edges = source["edges"]
        division_ids = source["division_parent_ids"]
        source_point_ids = source["source_point_id"]

    # Prefer an observed fork away from the ends of the recording. Selection
    # uses geometry only for presentation; it creates no nearest-neighbour links.
    candidates = division_ids[(times[division_ids] >= 7) &
                              (times[division_ids] < meta["frames"] - 8)]
    if len(candidates):
        anchor_time = int(np.median(times[candidates]))
        candidates = candidates[np.abs(times[candidates] - anchor_time) ==
                                np.abs(times[candidates] - anchor_time).min()]
        anchor_time = int(times[candidates[0]])
        candidates = candidates[times[candidates] == anchor_time]
        centre = np.median(positions[times == anchor_time], axis=0)
        anchor_id = int(candidates[np.argmin(((positions[candidates] - centre) ** 2).sum(axis=1))])
    else:
        anchor_time = meta["frames"] // 2
        candidates = np.flatnonzero(times == anchor_time)
        centre = np.median(positions[candidates], axis=0)
        anchor_id = int(candidates[np.argmin(((positions[candidates] - centre) ** 2).sum(axis=1))])

    start, end = max(0, anchor_time - 7), min(meta["frames"] - 1, anchor_time + 8)
    window_ids = np.flatnonzero((times >= start) & (times <= end))
    present_tracks = set(map(int, np.unique(tracks[window_ids])))
    # A family here means tracks connected through the recorded direct parent
    # relation inside this window. The original graph has already passed the
    # adjacent-frame parent-edge checks; assert that again for the sampled edges.
    root_by_track = {}

    def root(track):
        track = int(track)
        chain = []
        current = track
        while current not in root_by_track and int(parent_tracks[current]) in present_tracks:
            if current in chain:
                raise ValueError("Cycle in prepared track parent metadata")
            chain.append(current)
            current = int(parent_tracks[current])
        result = root_by_track.get(current, current)
        root_by_track[current] = result
        for previous in chain:
            root_by_track[previous] = result
        return result

    for track in present_tracks:
        root(track)
    root_lookup = np.full(len(parent_tracks), -1, dtype=np.int64)
    for track, value in root_by_track.items():
        root_lookup[track] = value
    family = root_lookup[tracks[window_ids]]
    family_counts = Counter(map(int, family))
    seeds = np.flatnonzero(times == anchor_time)
    distances = ((positions[seeds] - positions[anchor_id]) ** 2).sum(axis=1)
    ordered_families = list(dict.fromkeys(map(int, root_lookup[tracks[seeds[np.argsort(distances, kind="stable")]]])))
    anchor_family = int(root_lookup[tracks[anchor_id]])
    ordered_families.remove(anchor_family)
    ordered_families.insert(0, anchor_family)
    selected_families, count = [], 0
    for family_id in ordered_families:
        next_count = family_counts[family_id]
        if count + next_count > 1200:
            continue
        selected_families.append(family_id)
        count += next_count
        if count >= 640:
            break
    selected_ids = window_ids[np.isin(family, selected_families)]
    selected_ids = selected_ids[np.lexsort((selected_ids, times[selected_ids]))]
    membership = np.zeros(len(times), dtype=bool)
    membership[selected_ids] = True
    selected_edges = edges[membership[edges[:, 0]] & membership[edges[:, 1]]]
    counts = Counter(map(int, selected_edges[:, 0]))
    displayed_divisions = [node for node, degree in counts.items() if degree >= 2]
    assert 300 <= len(selected_ids) <= 1200, (species, len(selected_ids))
    assert np.all(times[selected_edges[:, 1]] == times[selected_edges[:, 0]] + 1)
    assert set(displayed_divisions).issubset(set(map(int, division_ids)))
    if len(candidates) and len(division_ids):
        assert anchor_id in displayed_divisions, "Selected anchor fork must retain both daughters"
    assert np.isfinite(positions[selected_ids]).all()
    nodes = []
    for node in selected_ids:
        z, y, x = np.round(positions[node].astype(float), 5)
        nodes.append({"id": int(node), "source_point_id": int(source_point_ids[node]),
                      "t": int(times[node]), "z": float(z), "y": float(y), "x": float(x),
                      "tracklet_id": int(tracks[node]),
                      "family_id": int(root_lookup[tracks[node]])})
    result = {
        "species": species, "label": LABELS[species], "source": meta["source"],
        "prepared_source": meta["prepared"],
        "units": "source coordinate units (physical scale not calibrated)",
        "time_unit": "source frame index; physical frame duration unverified",
        "full": {k: meta[k] for k in ["frames", "nodes", "tracklets", "edges", "division_parents"]},
        "window": {"start": start, "end": end}, "anchor_node_id": anchor_id,
        "anchor_time": anchor_time, "nodes": nodes, "edges": selected_edges.tolist(),
        "division_parent_ids": sorted(displayed_divisions),
        "bbox_zyx": bbox(positions[selected_ids]),
        "sample": {"nodes": len(nodes), "edges": len(selected_edges),
                   "tracklets": len(np.unique(tracks[selected_ids])),
                   "families": len(selected_families), "division_parents": len(displayed_divisions)},
        "selection_note": "Deterministic spatial neighbourhood around a recorded fork where available; complete connected track families within a 16-frame window, subject to a 1,200-observation budget. This is a selected illustration, not a representative random sample.",
        "label_note": "Edges come from the prepared reconstruction of source track/parent metadata. No geometric links are invented. Experimental tracking output is not assumed manually verified ground truth.",
        "projection_note": "XY projection discards depth. A 2D crossing can involve cells separated in Z. Species panels use separate source-unit bounds and are not comparable physical scales.",
    }
    if not meta["division_parents"]:
        result["selection_note"] += " Mouse has no recorded forks in this export; continuous tracklets are shown. This does not establish biological absence of division."
    print(species, result["sample"], flush=True)
    return result


def riken_example():
    with np.load(ROOT / "prepared/riken_animal_c_first10_frames.npz", allow_pickle=False) as source:
        times, positions, ids = source["t"], source["zyx_um"], source["measurement_id"]
    audit = json.loads((ROOT / "prepared/riken_audit.json").read_text())
    frames = []
    for time in np.unique(times):
        indices = np.flatnonzero(times == time)
        points = []
        for index in indices:
            z, y, x = np.round(positions[index].astype(float), 5)
            points.append({"id": str(ids[index]), "z": float(z), "y": float(y), "x": float(x)})
        frames.append({"t": int(time), "source_object_t": int(time) + 1,
                       "elapsed_seconds": int(time - times.min()) * 90,
                       "count": len(points), "points": points})
    features = []
    with h5py.File(ROOT / "cache/zebrafish_animal_c_bd5.h5", "r") as source:
        raw_objects = source["data/0/object/0"][:]
        raw_features = source["data/0/feature/0"][:]
        by_id = {}
        for feature in raw_features:
            by_id.setdefault(feature["ID"].decode(), {})[int(feature["fID"])] = float(feature["value"])
        for measurement in raw_objects[:10]:
            measurement_id = measurement["ID"].decode()
            values = by_id[measurement_id]

            def ordered_values(keys):
                return [round(values[key], 5) if key in values else None for key in keys]

            bounds_min = ordered_values([4, 3, 2])
            bounds_max = ordered_values([7, 6, 5])
            fwhm_min = ordered_values([10, 9, 8])
            fwhm_max = ordered_values([13, 12, 11])

            def widths(low, high):
                return [round(b - a, 5) if a is not None and b is not None else None
                        for a, b in zip(low, high)]

            record = {
                "measurement_id": measurement_id, "t": 0,
                **{key: float(measurement[key]) for key in ["z", "y", "x"]},
                "mean_intensity_au": values.get(0), "com_intensity_au": values.get(1),
                "bbox_min_zyx_um": bounds_min, "bbox_max_zyx_um": bounds_max,
                "bbox_size_zyx_um": widths(bounds_min, bounds_max),
                "fwhm_min_zyx_um": fwhm_min, "fwhm_max_zyx_um": fwhm_max,
                "fwhm_size_zyx_um": widths(fwhm_min, fwhm_max),
                "source_feature_ids_present": sorted(values),
            }
            assert int(measurement["t"]) == 1
            matching = np.flatnonzero((ids == measurement_id) & (times == 0))
            assert len(matching) == 1
            assert np.allclose(positions[matching[0]], [record[k] for k in ["z", "y", "x"]])
            features.append(record)
    result = {
        "source": "prepared/riken_animal_c_first10_frames.npz",
        "features_source": "cache/zebrafish_animal_c_bd5.h5:data/0/feature/0",
        "units": "µm", "time_step_seconds": 90,
        "frames": frames, "total_displayed_measurements": sum(frame["count"] for frame in frames),
        "bbox_zyx": bbox(positions), "features": features,
        "feature_definitions": audit["feature_definitions"],
        "notes": [
            "All 1,312 measurements from the ten prepared frames are shown; this is an early excerpt of animal C, not all 821 frames.",
            "Display frame 0 corresponds to object time index 1 in the HDF5. Elapsed time is relative to the first displayed frame, not an absolute developmental age.",
            "Measurement IDs identify observations. Their persistence between frames is not established, so this viewer has no temporal edges or inferred identities.",
            "Bounding boxes describe segmented nucleus cores in the source; neither their original masks nor paired microscopy images are present in this prepared dataset.",
            "FWHM is the width of an intensity distribution measured at half its maximum, not a cell boundary. Missing source feature values remain null; no dimensions are invented.",
            "Fluorescence values are arbitrary units and require acquisition-specific normalization before comparison with Biohub images.",
        ],
    }
    assert result["total_displayed_measurements"] == 1312
    print("riken", result["total_displayed_measurements"], "measurements;", len(features), "feature examples", flush=True)
    return result


def main():
    metadata = json.loads((ROOT / "prepared/zoo_manifest.json").read_text())
    result = {"schema_version": 1, "coordinate_presentation_decimals": 5,
              "zoo": [zoo_example(meta) for meta in metadata], "riken": riken_example()}
    output = ROOT / "trajectory_examples.json"
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    verification = {
        "passed": True,
        "checks": ["All Zoo edges retained from prepared graph; endpoints within displayed window",
                   "Every Zoo edge connects adjacent frames",
                   "Displayed forks agree with prepared graph; both daughters retained at fork anchor",
                   "Every Zoo example has 300–1,200 observations and finite coordinates",
                   "RIKEN feature IDs join uniquely to first-frame measurement positions",
                   "All 1,312 exported RIKEN observations preserved; no temporal edges generated",
                   "JSON contains no NaN or Infinity"],
        "zoo": {item["species"]: item["sample"] for item in result["zoo"]},
        "riken_frame_counts": [frame["count"] for frame in result["riken"]["frames"]],
        "riken_feature_examples": len(result["riken"]["features"]),
        "output_bytes": output.stat().st_size,
    }
    (ROOT / "checks").mkdir(exist_ok=True)
    (ROOT / "checks/trajectory-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print("Wrote", output, output.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
