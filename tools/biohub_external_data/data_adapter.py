"""Load prepared Biohub training labels with matching image coordinates.

Only NumPy is needed. See README.md for provenance, split and label limitations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"
NATIVE_UM_ZYX = np.array([1.625, 0.40625, 0.40625], np.float32)
STRIDE_ZYX = np.array([1, 4, 4], np.float32)
CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def load_synthetic(sample_id: str, *, grid: str = "pooled", source_root=None):
    """Return float32 TZYX images in [0,1] and aligned point/graph targets.

    grid="pooled": 64³, XY stride sampling for static images; sequence images
    are already sampled. Source-root can relocate the biohub_synthetic folder.
    Native sequence images were not released, so grid="native" rejects them.
    """
    if grid not in ("pooled", "native"):
        raise ValueError("grid must be pooled or native")
    records = json.loads((ROOT / "prepared/synthetic_manifest.json").read_text())
    record = next((r for r in records if r["sample_id"] == sample_id), None)
    if record is None:
        raise KeyError(sample_id)
    source = (ROOT / record["source"]).resolve()
    if source_root is not None:
        source = Path(source_root) / source.parent.name / source.name
    if record["kind"] == "sequence" and grid == "native":
        raise ValueError("Only pooled sequence images are available; use grid='pooled'")
    with np.load(source, allow_pickle=False) as original:
        image = original[record["image_key"]]
    with np.load(ROOT / record["labels"], allow_pickle=False) as labels:
        out = {key: labels[key] for key in labels.files}
    if record["kind"] == "static":
        image = image[None]
        if grid == "pooled":
            image = image[:, :, ::4, ::4]
        count = len(out["zyx_native"])
        out.update(node_id=np.arange(count, dtype=np.int64), t=np.zeros(count, dtype=np.int64))
    out["image"] = image.astype(np.float32) / 65535.0
    out["zyx_image"] = out["zyx_" + grid]
    out["voxel_um_zyx"] = NATIVE_UM_ZYX * (STRIDE_ZYX if grid == "pooled" else 1)
    out["sample_id"], out["kind"], out["split"] = sample_id, record["kind"], record["split"]
    if record["kind"] == "sequence":
        division = np.zeros(len(out["node_id"]), dtype=bool)
        division[out["division_parent_ids"]] = True
        out["is_division_parent"] = division
        # Last-frame cells have no observed future; they are not negative examples.
        out["division_target_observed"] = out["t"] < image.shape[0] - 1
    return out


def points_to_heatmap(shape_zyx, points_zyx, voxel_um_zyx, sigma_um=2.0):
    """Optional Gaussian centre target; it is not a segmentation mask.

    sigma_um is a training hyperparameter, not the evaluator's matching radius.
    Preserve sub-voxel centres, truncate at 3 sigma, combine by maximum.
    """
    shape = np.asarray(shape_zyx, dtype=np.int64)
    spacing = np.asarray(voxel_um_zyx, dtype=np.float32)
    points = np.asarray(points_zyx, dtype=np.float32)
    if shape.shape != (3,) or (shape <= 0).any():
        raise ValueError("shape_zyx must contain three positive dimensions")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("voxel spacing must be finite and positive")
    if not np.isfinite(sigma_um) or sigma_um <= 0:
        raise ValueError("sigma_um must be finite and positive")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N×3 array")
    if (points < 0).any() or (points >= shape).any():
        raise ValueError("point outside image grid")
    sigma = sigma_um / spacing
    heatmap = np.zeros(tuple(shape), dtype=np.float32)
    for centre in points:
        lo = np.maximum(0, np.floor(centre - 3 * sigma).astype(int))
        hi = np.minimum(shape, np.ceil(centre + 3 * sigma).astype(int) + 1)
        axes = [((np.arange(lo[k], hi[k]) - centre[k]) / sigma[k]) ** 2 for k in range(3)]
        patch = np.exp(-0.5 * (axes[0][:, None, None] + axes[1][None, :, None] + axes[2][None, None, :]))
        section = tuple(slice(lo[k], hi[k]) for k in range(3))
        np.maximum(heatmap[section], patch, out=heatmap[section])
    return heatmap


def prediction_rows(dataset_id, t, zyx_native, edges, *, row_offset=0):
    """CSV rows for ONE graph. edges index node rows; t and XYZ must be native.

    The caller must predict all hidden test folders and concatenate these rows
    with increasing row_offset. This helper does not run inference.
    """
    t = np.asarray(t)
    xyz = np.asarray(zyx_native)
    edges = np.asarray(edges)
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty folder basename")
    if t.ndim != 1 or xyz.shape != (len(t), 3) or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("expected t[N], zyx_native[N,3], edges[E,2]")
    if not np.isfinite(t).all() or not np.equal(t, np.floor(t)).all() or (t < 0).any():
        raise ValueError("timepoints must be nonnegative integers")
    if not np.isfinite(xyz).all() or (xyz < 0).any():
        raise ValueError("centroids must be finite and nonnegative")
    if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
        raise ValueError("edge endpoints must be integers")
    edges = edges.astype(np.int64)
    if (edges < 0).any() or (edges >= len(t)).any():
        raise ValueError("edge endpoint outside node array")
    if not np.all(t[edges[:, 1]] == t[edges[:, 0]] + 1):
        raise ValueError("edges must connect adjacent frames")
    integer_xyz = np.rint(xyz).astype(np.int64)
    for i, (time, pos) in enumerate(zip(t, integer_xyz)):
        yield [row_offset + i, dataset_id, "node", i, int(time), *pos.tolist(), -1, -1]
    for j, (source, target) in enumerate(edges):
        yield [row_offset + len(t) + j, dataset_id, "edge", -1, -1, -1, -1, -1, int(source), int(target)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="seq_0000")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--export-demo", type=Path)
    args = parser.parse_args()
    sample = load_synthetic(args.sample, source_root=args.source_root)
    print(json.dumps({k: (list(v.shape) if isinstance(v, np.ndarray) else v) for k, v in sample.items()}, indent=2))
    if args.export_demo:
        if sample["kind"] != "sequence":
            raise ValueError("Use a sequence to demonstrate graph rows")
        args.export_demo.parent.mkdir(parents=True, exist_ok=True)
        with args.export_demo.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(prediction_rows("SYNTHETIC_FORMAT_DEMO", sample["t"], sample["zyx_native"], sample["edges"]))
        print("Wrote synthetic schema example; this is not a competition submission.")


if __name__ == "__main__":
    main()
