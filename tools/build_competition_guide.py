#!/usr/bin/env python3
"""Build a portable, offline HTML guide with a small real-data microscopy preview.

Run in the cell-tracking environment. Reads only the six requested timepoints;
never writes to the input directory. The HTML embeds all images and annotations.
"""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import zarr

from competition_paths import DATA_ROOT, REPO_ROOT


def jpeg(array: np.ndarray, low: float, high: float) -> str:
    pixels = (np.clip((array.astype(float) - low) / (high - low), 0, 1) * 255).astype("uint8")
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build(data_root: Path, sample: str, output: Path) -> None:
    image_path = data_root / "train" / f"{sample}.zarr" / "0"
    graph_path = data_root / "train" / f"{sample}.geff"
    metadata = json.loads((image_path / "zarr.json").read_text())
    graph_metadata = json.loads((graph_path / "zarr.json").read_text())
    geff = graph_metadata["attributes"]["geff"]
    axes = [axis["name"] for axis in geff["axes"]]
    if axes != ["t", "z", "y", "x"] or len(metadata["shape"]) != 4:
        raise ValueError("Preview requires the verified T,Z,Y,X layout")
    image = zarr.open_array(image_path, mode="r")
    graph = zarr.open_group(graph_path, mode="r")
    times = [t for t in [0, 20, 40, 60, 80, 99] if t < image.shape[0]]
    planes = list(range(0, image.shape[1], 4))
    # A partial preview cache must contain every requested chunk. Zarr normally
    # fills missing chunks with zeros, which would misrepresent unavailable data.
    chunks = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    if chunks != [1, *image.shape[1:]]:
        raise ValueError("Preview expects one full timepoint per chunk")
    for t in times:
        if not (image_path / "c" / str(t) / "0/0/0").is_file():
            raise FileNotFoundError(f"Missing preview timepoint: {t}")
    volumes = [image[t] for t in times]
    low, high = np.quantile(np.concatenate([v.ravel()[::64] for v in volumes]), [.01, .999])
    if high <= low:
        raise ValueError("Preview has no usable intensity range")
    ids = graph["nodes/ids"][:]
    coords = {axis: graph[f"nodes/props/{axis}/values"][:] for axis in axes}
    nodes = [dict(id=str(node_id), **{axis: int(coords[axis][i]) for axis in axes})
             for i, node_id in enumerate(ids)]
    edges = [[str(a), str(b)] for a, b in graph["edges/ids"][:]]
    # Inventory is an official file-list snapshot, not a claim that the full
    # archive has already been downloaded or extracted on this machine.
    inventory_path = REPO_ROOT / "work/data-api-files.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())["files"]
        samples = {row["path"].split("/")[1][:-5] for row in inventory
                   if row["path"].startswith("train/") and ".zarr/" in row["path"]}
    else:
        samples = {p.stem for p in (DATA_ROOT / "train").glob("*.zarr")}
    if len(samples) != 199:
        raise ValueError("This dated guide expects the verified 199-clip inventory")
    manifest = json.loads((REPO_ROOT / "reference/manifest.json").read_text())
    payload = {
        "sample": sample, "shape": list(image.shape), "dtype": str(image.dtype),
        "chunks": chunks, "times": times, "planes": planes,
        "scale": {axis["name"]: axis["scale"] for axis in geff["axes"]},
        "nodes": nodes, "edges": edges,
        "estimatedNodes": geff["extra"]["estimated_number_of_nodes"],
        "embryos": dict(sorted(Counter(s.split("_")[0] for s in samples).items())),
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "referenceSync": manifest.get("last_sync"),
        "normalization": {"low": float(low), "high": float(high)},
        "frames": [{"t": t, "mip": jpeg(v.max(axis=0), low, high),
                    "slices": [jpeg(v[z], low, high) for z in planes]}
                   for t, v in zip(times, volumes)],
        "csvHeader": (data_root / "sample_submission.csv").read_text().splitlines()[0],
    }
    template = (REPO_ROOT / "docs/guide/template.html").read_text()
    marker = "__GUIDE_DATA_JSON__"
    if template.count(marker) != 1:
        raise ValueError("Template must contain exactly one data marker")
    rendered = template.replace(marker, json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(f"Built {output} ({output.stat().st_size / 1024**2:.2f} MiB)")
    print(f"Real preview: {sample}; {len(times)} frames; {len(planes)} planes/frame; "
          f"{len(nodes)} annotated nodes; {len(edges)} annotated edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--sample", default="44b6_0113de3b")
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/cell-tracking/guide/index.html"))
    args = parser.parse_args()
    build(args.data_root, args.sample, args.output)
