#!/usr/bin/env python3
"""Run ACTUAL Ultrack/CBC on a tiny synthetic movie. Not a biological benchmark."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import time
from contracts import validate_observations
from preflight import admitted, existing_parent
import shutil


def execute(output: Path) -> dict:
    # Lazy imports keep --help useful before V700 provisioning. Missing packages
    # must fail actual execution; they must never produce a skipped/pass receipt.
    import numpy as np
    import ultrack
    from ultrack import MainConfig, segment, link, solve, to_tracks_layer
    from ultrack.utils import labels_to_contours
    from ultrack.core.export import tracks_to_zarr

    shape = (4, 12, 32, 40)
    labels = np.zeros(shape, dtype=np.uint16)
    for t in range(shape[0]):
        labels[t, 3:7, 12:18, 10+t:16+t] = 1
    image = labels.astype(np.float32)
    config = MainConfig.model_validate({
        "data": {"working_dir": str(output / "db"), "database": "sqlite"},
        "segmentation": {"min_area": 20, "max_area": 500, "min_frontier": 0.1, "n_workers": 1},
        "linking": {"max_distance": 4.0, "max_neighbors": 3, "n_workers": 1},
        "tracking": {"solver_name": "CBC", "n_threads": 1, "time_limit": 60,
                     "link_function": "identity", "bias": 0.0, "window_size": None,
                     "appear_weight": -0.1, "disappear_weight": -0.1, "division_weight": -0.1},
    })
    signed = np.asarray([-2.0, 0.0, 1.0])
    if not np.array_equal(config.tracking_config.apply_link_function(signed), signed):
        raise AssertionError("Signed weights were transformed unexpectedly")
    start = time.monotonic()
    foreground, contours = labels_to_contours(
        labels, sigma=None,
        foreground_store_or_path=output / "foreground.zarr",
        contours_store_or_path=output / "contours.zarr",
    )
    segment(foreground=foreground, contours=contours, config=config)
    # A channel sequence, not a bare TZYX array. Scale is spatial ZYX in um.
    link(config=config, images=(image,), scale=(1.625, 0.40625, 0.40625))
    solve(config=config, use_annotations=False, use_ground_truth_match=False)
    frame, _ = to_tracks_layer(config=config, include_parents=True, include_node_ids=True)
    check = validate_observations(frame.to_dict("records"), shape, integer_coordinates=False)
    if check["nodes"] != 4 or len(check["edges"]) != 3 or check["divisions"] != 0:
        raise AssertionError(f"Synthetic continuation not recovered: {check}")
    selected = tracks_to_zarr(config=config, tracks_df=frame,
                              store_or_path=output / "selected.zarr", overwrite=False)
    if tuple(selected.shape) != shape:
        raise AssertionError("Selected masks changed the TZYX lattice")
    exact = all(np.array_equal(np.asarray(selected[t]) > 0, labels[t] > 0) for t in range(shape[0]))
    if not exact:
        raise AssertionError("Synthetic object support was not exactly retained")
    return {"status": "passed", "actual_ultrack_execution": True,
            "synthetic_only": True, "competition_score": None,
            "ultrack_version": getattr(ultrack, "__version__", None),
            "ultrack_source": str(Path(ultrack.__file__).resolve()),
            "requested_solver": "CBC", "wall_seconds": time.monotonic()-start,
            "graph": check, "exact_synthetic_mask_support": exact,
            "signatures": {fn.__name__: str(inspect.signature(fn))
                           for fn in (labels_to_contours, segment, link, solve, to_tracks_layer, tracks_to_zarr)},
            "remaining_tests": ["division", "hierarchy conflict", "empty frames", "window boundaries",
                                "real native evidence", "actual installed source/solver provenance"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New admitted local directory; never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    if not admitted(shutil.disk_usage(existing_parent(output)).free, 0.125):
        parser.error("Smoke needs 8 GiB free reserve plus 0.125 GiB incremental capacity")
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = execute(output)
    except Exception as exc:
        report = {"status": "failed", "actual_ultrack_execution": None,
                  "synthetic_only": True, "competition_score": None,
                  "error_type": type(exc).__name__, "error": str(exc)}
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
