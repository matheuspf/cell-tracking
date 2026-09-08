#!/usr/bin/env python3
"""Read-only, metadata-only training inventory. Never an exact true-cell census.

Reads JSON and array SHAPES, not microscopy chunks or GT coordinates. The full
GEFF integrity/coordinate/overlap audit is a separate Codex implementation task.
Supports local Zarr v2/v3 metadata; unknown formats fail rather than invent data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_DATA = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return obj


def metadata_file(path: Path, array: bool = False) -> Path:
    names = ("zarr.json", ".zarray") if array else ("zarr.json", ".zattrs")
    for name in names:
        file = path / name
        if file.is_file():
            return file
    raise FileNotFoundError(f"no supported {'array' if array else 'group'} metadata: {path}")


def locate_estimates(obj: object, prefix: str = "") -> list[tuple[str, object]]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "estimated_number_of_nodes":
                found.append((path, value))
            else:
                found.extend(locate_estimates(value, path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(locate_estimates(value, f"{prefix}[{i}]"))
    return found


def shape_of(path: Path) -> tuple[list[int], Path]:
    meta_path = metadata_file(path, array=True)
    shape = read_json(meta_path).get("shape")
    if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError(f"invalid shape: {path}")
    return shape, meta_path


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    train = root / "train"
    images = {p.stem: p for p in train.glob("*.zarr") if p.is_dir()}
    graphs = {p.stem: p for p in train.glob("*.geff") if p.is_dir()}
    errors, rows = [], []
    if not images:
        errors.append("no training Zarr directories found")
    if images.keys() != graphs.keys():
        errors.append("image/GEFF stem sets differ")
    for name in sorted(images.keys() & graphs.keys()):
        try:
            image_shape, image_meta = shape_of(images[name] / "0")
            node_shape, node_meta = shape_of(graphs[name] / "nodes/ids")
            edge_shape, edge_meta = shape_of(graphs[name] / "edges/ids")
            if len(node_shape) != 1 or len(edge_shape) != 2 or edge_shape[1] != 2:
                raise ValueError("unexpected GEFF ID-array shape")
            graph_meta = metadata_file(graphs[name])
            estimates = locate_estimates(read_json(graph_meta))
            if len(estimates) != 1:
                raise ValueError(f"expected exactly one total-node estimate; got {len(estimates)}")
            estimate_path, raw_estimate = estimates[0]
            if isinstance(raw_estimate, bool):
                raise ValueError("boolean total-node estimate")
            estimate = float(raw_estimate)
            if not math.isfinite(estimate) or estimate <= 0:
                raise ValueError("total-node estimate must be finite and positive")
            files = (image_meta, node_meta, edge_meta, graph_meta)
            rows.append({"dataset": name, "embryo_prefix_unverified": name.split("_")[0],
                         "image_shape": image_shape, "annotated_nodes_shape_count": node_shape[0],
                         "annotated_edges_shape_count": edge_shape[0], "estimated_total": estimate,
                         "estimate_key_path": estimate_path,
                         "reference_coverage_estimate": node_shape[0] / estimate,
                         "estimate_below_annotation_count": estimate < node_shape[0],
                         "metadata_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in files}})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    complete = not errors and len(rows) == len(images)
    n_ann = sum(r["annotated_nodes_shape_count"] for r in rows)
    n_est = sum(r["estimated_total"] for r in rows)
    return {"status": "metadata_complete" if complete else "incomplete",
            "estimand": "annotated observations / supplied estimated observations, NOT exact cell prevalence",
            "unique_biological_cell_coverage": None,
            "image_count": len(images), "geff_count": len(graphs), "rows": rows, "errors": errors,
            "reference_coverage_estimate_pooled": n_ann / n_est if complete and n_est else None,
            "warning": "array shape is not an ID-integrity check; overlap can repeat biological observations"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "work/annotation-selection-v1/metadata.json")
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    if output == data_root or data_root in output.parents:
        parser.error("output must not be inside raw data")
    result = audit(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {len(result['rows'])}/{result['image_count']} samples -> {output}")
    raise SystemExit(0 if result["status"] == "metadata_complete" else 2)


if __name__ == "__main__":
    main()
