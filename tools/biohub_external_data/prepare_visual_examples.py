"""Build bounded, source-backed image examples for the offline Biohub guide.

Run in the cell-tracking Conda environment with preparation's isolated packages
on PYTHONPATH. Source microscopy and labels are read only. PNGs are display
derivatives: native spatial sampling, fixed contrast within a depth/film example.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import zarr

from data_adapter import load_synthetic, points_to_heatmap

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "work/biohub-data-guide"
ARCHIVE = ROOT.parent / "biohub-forum-archive"
SYNTH = ARCHIVE / "downloads/kaggle/biohub_synthetic"
REAL = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
NATIVE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)


def relative(path):
    return os.path.relpath(path, ROOT)


def contrast(volume):
    low, high = np.quantile(volume, [0.01, 0.998])
    if high <= low:
        high = low + 1
    return {"low": float(low), "high": float(high),
            "method": "1st–99.8th percentile of this complete volume; fixed for every slice"}


def display(array, levels):
    scaled = np.clip((array.astype(np.float32) - levels["low"]) /
                     (levels["high"] - levels["low"]), 0, 1)
    return np.rint(scaled * 255).astype(np.uint8)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def check_volume(volume, points, spacing):
    assert volume.ndim == 3 and volume.dtype == np.uint16
    assert points.ndim == 2 and points.shape[1] == 3
    assert np.isfinite(points).all() and (points >= 0).all()
    assert (points < np.array(volume.shape)).all()
    assert np.array(spacing).shape == (3,) and (np.array(spacing) > 0).all()


def record(volume, points, *, sample, title, kind, t, spacing, source, caption):
    check_volume(volume, points, spacing)
    levels = contrast(volume)
    return {"id": sample + "_t" + str(t), "title": title, "kind": kind,
            "sample": sample, "t": int(t), "shape_zyx": list(volume.shape),
            "voxel_um_zyx": list(map(float, spacing)),
            "image": png(display(volume.max(axis=0), levels)),
            "points_zyx": points.tolist(), "count": len(points),
            "caption": caption, "source_relative": relative(source), "contrast": levels,
            "projection": "Maximum intensity through the full Z depth"}


def static_example(item, title):
    source = ROOT / item["source"]
    with np.load(source, allow_pickle=False) as data:
        volume, points = data["volume"], data["centroids"]
        spacing = data["voxel_um"]
    assert len(points) == item["centroids"]
    result = record(volume, points, sample=item["sample_id"], title=title,
                    kind="synthetic", t=0, spacing=spacing, source=source,
                    caption="Synthetic microscopy with every released simulated cell centre. "
                    "Markers project all depths onto XY; overlapping markers can represent separate cells in Z.")
    return result, volume, points


def real_example(sample, t=None):
    source = REAL / "train" / (sample + ".zarr")
    image_group = zarr.open_group(str(source), mode="r")
    graph = zarr.open_group(str(source.with_suffix(".geff")), mode="r")
    times = np.asarray(graph["nodes/props/t/values"][:])
    assert np.equal(times, times.astype(int)).all()
    if t is None:
        t = int(np.argmax(np.bincount(times.astype(int))))
    points = np.column_stack([graph[f"nodes/props/{axis}/values"][:] for axis in "zyx"])[times == t]
    volume = image_group["0"][t]
    scales = image_group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    spacing = next(x["scale"][-3:] for x in scales if x["type"] == "scale")
    assert len(points) > 0
    result = record(volume, points, sample=sample,
                    title=f"Real embryo {sample.split('_')[0]} · frame {t}", kind="real",
                    t=t, spacing=spacing, source=source,
                    caption="Real competition microscopy. Markers show only available GEFF annotations: "
                    "unmarked fluorescence is not a confirmed negative. This is a display of one local training patch.")
    result["label_source_relative"] = relative(source.with_suffix(".geff"))
    return result, volume, points


def depth_example(info, volume, points):
    levels = info["contrast"]
    rendered = display(volume, levels)
    result = {key: value for key, value in info.items() if key != "image"}
    result.update(slices=[png(frame) for frame in rendered],
                  default_z=int(np.clip(np.rint(np.median(points[:, 0])), 0, volume.shape[0]-1)),
                  image_shape=list(volume.shape),
                  caption="One native XY plane at a time; use the Z slider. Only points within 0.5 Z voxels "
                  "of the selected plane should be drawn. The contrast limits stay fixed across all slices. "
                  + ("Real labels are sparse." if info["kind"] == "real" else
                     "The simulated labels describe centres, not cell boundaries."))
    result.pop("projection")
    return result


def heatmap_example(sample):
    data = load_synthetic(sample, grid="pooled")
    image = data["image"][0]
    points = data["zyx_image"]
    target = points_to_heatmap(image.shape, points, data["voxel_um_zyx"], sigma_um=2.0)
    assert target.shape == image.shape and np.isfinite(target).all()
    assert target.min() >= 0 and target.max() <= 1 and target.max() > .5
    levels = contrast(image)
    return {"sample": sample, "sigma_um": 2.0, "shape_zyx": list(image.shape),
            "voxel_um_zyx": data["voxel_um_zyx"].tolist(),
            "image": png(display(image.max(axis=0), levels)),
            "target_image": png(np.rint(target.max(axis=0)*255).astype(np.uint8)),
            "points_zyx": points.tolist(), "count": len(points),
            "target_range": [float(target.min()), float(target.max())],
            "target_reduction": "maximum of Gaussian bumps, truncated at 3 sigma",
            "source_relative": relative(SYNTH / "static" / (sample + ".npz")),
            "caption": "A generated detector training target from the actual synthetic centre labels. "
            "Gaussian sigma is 2 µm, a chosen hyperparameter; this is not a segmentation mask "
            "and is unrelated to the metric's matching tolerance. Images and targets are max-Z projections "
            "on the 64³ grid, with Y/X labels divided by four."}


def division_example():
    data = load_synthetic("seq_0000", grid="pooled")
    image, coords, times, edges = data["image"], data["zyx_image"], data["t"], data["edges"]
    # Prefer a legible actual fork with a context parent and one observed successor
    # for each daughter; later daughter separation makes the 3D event easier to see.
    options = []
    for parent in data["division_parent_ids"]:
        if times[parent] != 2:
            continue
        before = edges[edges[:, 1] == parent, 0]
        daughters = edges[edges[:, 0] == parent, 1]
        after = [edges[edges[:, 0] == d, 1] for d in daughters]
        if len(before) != 1 or len(daughters) != 2 or any(len(x) != 1 for x in after):
            continue
        later = np.array([a[0] for a in after])
        nodes = np.r_[before, [parent], daughters, later]
        if (coords[nodes] < 6).any() or (coords[nodes] > 57).any():
            continue
        separation = np.linalg.norm((coords[later[0]] - coords[later[1]])[1:])
        options.append((separation, int(parent), before, daughters, later))
    assert options, "No suitable observed fork found in seq_0000"
    _, parent, before, daughters, later = max(options, key=lambda x: x[0])
    selected = np.r_[before, [parent], daughters, later]
    lower = np.maximum(0, np.floor(coords[selected].min(axis=0) - [4, 6, 6]).astype(int))
    upper = np.minimum(image.shape[1:], np.ceil(coords[selected].max(axis=0) + [4, 6, 6]).astype(int) + 1)
    section = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    crops = image[(slice(1, 5), *section)]
    levels = contrast(crops)
    levels["method"] = "1st–99.8th percentile over all four cropped volumes; identical contrast for every frame"
    frames = []
    selected_by_time = [(before, ["context"]), (np.array([parent]), ["parent"]),
                        (daughters, ["daughter_a", "daughter_b"]),
                        (later, ["daughter_a", "daughter_b"])]
    for j, (ids, roles) in enumerate(selected_by_time):
        t = j + 1
        assert np.all(times[ids] == t)
        frames.append({"t": t, "image": png(display(crops[j].max(axis=0), levels)),
                       "points": [{"node_id": int(i), "role": role,
                                   "zyx_pooled": coords[i].tolist(),
                                   "zyx_crop": (coords[i] - lower).tolist(),
                                   "xy_crop": (coords[i] - lower)[[2, 1]].tolist(),
                                   "tracklet_id": int(data["tracklet_id"][i]),
                                   "source_clone_id": int(data["source_clone_id"][i])}
                                  for i, role in zip(ids, roles)]})
    chosen_edges = edges[np.isin(edges[:, 0], selected) & np.isin(edges[:, 1], selected)]
    assert len(chosen_edges) == 5 and np.all(times[chosen_edges[:, 1]] == times[chosen_edges[:, 0]] + 1)
    assert len(set(data["source_clone_id"][selected].tolist())) == 1
    assert len(set(data["tracklet_id"][np.r_[[parent], daughters]].tolist())) == 3
    return {"sample": "seq_0000", "parent_id": parent, "daughter_ids": daughters.tolist(),
            "context_id": int(before[0]), "edges": chosen_edges.tolist(), "frames": frames,
            "crop_origin_zyx": lower.tolist(), "crop_shape_zyx": (upper-lower).tolist(),
            "voxel_um_zyx": data["voxel_um_zyx"].tolist(), "contrast": levels,
            "source_relative": relative(SYNTH / "sequences/seq_0000.npz"),
            "caption": "An actual released synthetic parent–daughter fork, with context and one later frame. "
            "All panels use the same XY crop, Z slab and contrast. Each panel projects the listed Z slab "
            "onto XY; 2D appearance alone does not prove a division. The source graph supplies that label. "
            "The parent and both daughters share one source clone ID, while prepared tracklet IDs split at the fork."}


def main():
    manifest = json.loads((ROOT / "prepared/synthetic_manifest.json").read_text())
    ordered = sorted((row for row in manifest if row["kind"] == "static"), key=lambda row: row["centroids"])
    examples = [static_example(ordered[i], title) for i, title in
                [(0, "Synthetic · fewer centres"), (len(ordered)//2, "Synthetic · median centre count"),
                 (len(ordered)-1, "Synthetic · more centres")]]
    # Fixed sample choices make subsequent guide builds reproducible.
    examples += [real_example("44b6_0113de3b", 0),
                 real_example("6bba_05b6850b"), real_example("6bba_05db0fb1")]
    payload = {"schema_version": 1, "gallery": [item[0] for item in examples],
               "depth": [depth_example(*examples[1]), depth_example(*examples[-1])],
               "heatmap": heatmap_example(examples[1][0]["sample"]), "division": division_example(),
               "display_note": "PNG previews preserve native spatial pixels but convert uint16 intensity "
               "to contrast-enhanced 8-bit grayscale. These are explanatory views, not quantitative intensity comparisons.",
               "selection_note": "The three static examples span minimum, median and maximum released centroid "
               "counts. Real examples are selected local training frames, not a random or statistically representative sample."}
    output = ROOT / "visual_examples.json"
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    # Round-trip every embedded PNG to check geometry, and retain a compact audit.
    images = 0
    for info in payload["gallery"]:
        decoded = Image.open(io.BytesIO(base64.b64decode(info["image"].split(",", 1)[1])))
        assert decoded.size == tuple(info["shape_zyx"][:0:-1])
        images += 1
    for info in payload["depth"]:
        assert len(info["slices"]) == info["shape_zyx"][0]
        for url in info["slices"]:
            decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            assert decoded.size == tuple(info["shape_zyx"][:0:-1])
            images += 1
    heat = payload["heatmap"]
    for key in ("image", "target_image"):
        decoded = Image.open(io.BytesIO(base64.b64decode(heat[key].split(",", 1)[1])))
        assert decoded.size == tuple(heat["shape_zyx"][:0:-1])
        images += 1
    for frame in payload["division"]["frames"]:
        decoded = Image.open(io.BytesIO(base64.b64decode(frame["image"].split(",", 1)[1])))
        assert decoded.size == tuple(payload["division"]["crop_shape_zyx"][:0:-1])
        images += 1
    audit = {"passed": True, "gallery_examples": len(payload["gallery"]),
             "gallery_counts": {item["id"]: item["count"] for item in payload["gallery"]},
             "depth_examples": len(payload["depth"]), "pngs_checked": images,
             "output_bytes": output.stat().st_size, "division_parent": payload["division"]["parent_id"],
             "division_edge_count": len(payload["division"]["edges"]), "heatmap_sigma_um": heat["sigma_um"],
             "checks": ["finite in-bounds centroids", "source counts", "native ZYX voxel scales",
                        "fixed contrast within slices and division frames", "all PNG dimensions", "heatmap range",
                        "real source fork endpoints", "shared clone and distinct daughter tracklets"]}
    (ROOT / "checks").mkdir(exist_ok=True)
    (ROOT / "checks/visual-examples-verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
