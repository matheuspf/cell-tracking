"""Annotation-free verification that cached images equal the canonical raw input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import zarr

from .common import DATA, RESULTS, WORK, now, sha, write_json


def main():
    started = time.perf_counter()
    rows = json.loads((WORK / "inputs/inference.json").read_text())["rows"]
    verified = []
    for i, row in enumerate(rows):
        image_sha = sha(row["image_path"])
        old = json.loads(Path(row["prediction_path"]).with_suffix(".json").read_text())
        assert image_sha == old["input_sha256"], row["key"]
        cached = np.load(row["image_path"], allow_pickle=False)
        raw_path = DATA / (row["dataset"] + ".zarr")
        raw = np.asarray(zarr.open_group(raw_path, mode="r")["0"][row["time"]])
        np.testing.assert_array_equal(cached, raw, err_msg=row["key"])
        pixel_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        sidecar = Path(row["cache_path"]).with_suffix(".json")
        if sidecar.exists():
            assert json.loads(sidecar.read_text())["volume_sha256"] == pixel_sha
        verified.append({"key": row["key"], "image_file_sha256": image_sha,
                         "canonical_pixel_sha256": pixel_sha})
        if (i + 1) % 50 == 0:
            print(json.dumps({"canonical_images_verified": i + 1, "total": len(rows)}), flush=True)
    payload = {"created_utc": now(), "status": "passed", "frames": len(rows),
               "seconds": time.perf_counter() - started, "annotation_access": False,
               "checks": "Every cached image matches the original detector input hash and its canonical Zarr timepoint exactly; existing feature-volume hashes agree",
               "verified": verified}
    write_json(WORK / "raw-input-audit.json", payload)
    write_json(RESULTS / "raw-input-audit.json", {k: v for k, v in payload.items() if k != "verified"} |
               {"detail_sha256": sha(WORK / "raw-input-audit.json")})


if __name__ == "__main__":
    main()
