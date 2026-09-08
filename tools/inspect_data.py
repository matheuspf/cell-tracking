#!/usr/bin/env python3
"""Inventory Zarr/GEFF metadata and CSV schemas without loading full volumes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from competition_paths import DATA_ROOT, REPO_ROOT, require_data_root


def inspect() -> dict:
    root = require_data_root()
    result = {"data_root": str(root.resolve()), "splits": {}}
    for split in ("train", "test"):
        images = sorted((root / split).glob("*.zarr"))
        graphs = sorted((root / split).glob("*.geff"))
        shapes = Counter()
        dtypes = Counter()
        for image in images:
            meta = json.loads((image / "0/zarr.json").read_text())
            shapes[str(meta["shape"])] += 1
            dtypes[meta["data_type"]] += 1
        result["splits"][split] = {
            "image_volumes": len(images),
            "annotation_graphs": len(graphs),
            "embryo_prefixes": sorted({p.name.split("_")[0] for p in images}),
            "shapes": dict(shapes),
            "dtypes": dict(dtypes),
            "unpaired_training_images": [p.stem for p in images if split == "train"
                                         and not p.with_suffix(".geff").exists()],
        }
    with (root / "sample_submission.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        result["sample_submission"] = {
            "columns": reader.fieldnames, "rows": len(rows),
            "row_types": dict(Counter(r["row_type"] for r in rows)),
            "datasets": sorted({r["dataset"] for r in rows}),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work/data-inventory.json")
    args = parser.parse_args()
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
