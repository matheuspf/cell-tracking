"""Freeze complete clips before tracking; export only image inputs for inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from tools.detector_screen.cellpose_adapter import sha256, write_json

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "work/cellpose-ultrack-20260914"
SCREEN = REPO / "work/detector-screen-20260914"
DATA = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scope", choices=["pilot", "assessment", "all"], default="pilot")
    args = parser.parse_args()
    source = json.loads((SCREEN / "panel.json").read_text())
    names = sorted({f["dataset"] for f in source["frames"] if f["role"] == args.scope})
    if args.scope == "all":
        names = sorted(p.stem for p in DATA.glob("*.zarr"))
    if not names:
        raise ValueError("Empty panel")
    frames, clips = [], []
    for name in names:
        image_path = DATA / f"{name}.zarr"
        metadata = json.loads((image_path / "zarr.json").read_text())
        ms = metadata["attributes"]["multiscales"][0]
        axes = [a["name"].lower() for a in ms["axes"]]
        if axes != list("tzyx"):
            raise ValueError(f"Unexpected axes: {axes}")
        spacing = ms["datasets"][0]["coordinateTransformations"][0]["scale"][1:]
        arr = zarr.open_group(image_path, mode="r")["0"]
        if len(arr.shape) != 4 or str(arr.dtype) != "uint16":
            raise ValueError(f"Unexpected image: {arr.shape}, {arr.dtype}")
        clips.append(dict(dataset=name, embryo=name.split("_")[0], shape=list(arr.shape),
                          spacing_um=spacing, image_path=str(image_path),
                          image_metadata_sha256=sha256(image_path / "zarr.json"),
                          array_metadata_sha256=sha256(image_path / "0/zarr.json")))
        for t in range(arr.shape[0]):
            path = args.root / "images" / name / f"t{t:03}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                previous = SCREEN / "images" / name / path.name
                if previous.exists():
                    path.symlink_to(previous)
                else:
                    np.save(path, np.asarray(arr[t]), allow_pickle=False)
            frames.append(dict(key=f"{name}-t{t:03}", dataset=name, embryo=name.split("_")[0],
                               time=t, role="tracking", image_path=str(path),
                               spacing_um=spacing, shape=list(arr.shape[1:])))
        print(f"Prepared {name}: {arr.shape[0]} complete frames", flush=True)
    panel = dict(schema=1, scope=args.scope, frames=frames, clips=clips,
                 selection="All frames of the previously frozen detector-screen clip cohort; no outcome-based selection.",
                 source_panel_sha256=sha256(SCREEN / "panel.json"),
                 inference_annotations="No GT, estimated cell counts, or existing tracker outputs in this manifest.")
    path = args.root / "panel.json"
    if path.exists() and json.loads(path.read_text()) != panel:
        raise ValueError("Existing panel differs; use a separate output root")
    write_json(path, panel)
    print(f"Frozen {len(clips)} clips / {len(frames)} frames at {path}", flush=True)


if __name__ == "__main__":
    main()
