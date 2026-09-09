"""Measure geometry in all prepared synthetic sequences, without training.

This records descriptive physical distances, not a motion prior or loss choice.
The published calibration constants and actual exported displacements are distinct.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "work/biohub-data-guide"


def distance_summary(chunks):
    values = np.concatenate(chunks)
    return {
        "count": len(values),
        "units": "micrometre",
        "quantiles": dict(zip(
            ["min", "p10", "median", "p90", "max"],
            np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
        )),
    }


def main():
    paths = sorted((ROOT / "prepared/synthetic").glob("seq_*_labels.npz"))
    assert len(paths) == 2174, "This audit describes the complete captured release"
    continuations, sisters = [], []
    starts, ends = 0, 0
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            xyz, edges, times = arrays["zyx_um"], arrays["edges"], arrays["t"]
        degree = np.bincount(edges[:, 0], minlength=len(times))
        assert np.all(degree <= 2)
        assert np.all(times[edges[:, 1]] == times[edges[:, 0]] + 1)
        continuation = edges[degree[edges[:, 0]] == 1]
        continuations.append(np.linalg.norm(xyz[continuation[:, 1]] - xyz[continuation[:, 0]], axis=1))
        fork = edges[degree[edges[:, 0]] == 2]
        fork = fork[np.argsort(fork[:, 0], kind="stable")]
        assert np.array_equal(fork[::2, 0], fork[1::2, 0])
        sisters.append(np.linalg.norm(xyz[fork[::2, 1]] - xyz[fork[1::2, 1]], axis=1))
        incoming = np.bincount(edges[:, 1], minlength=len(times))
        starts += int(np.count_nonzero((times > 0) & (incoming == 0)))
        ends += int(np.count_nonzero((times < 5) & (degree == 0)))
    report = {
        "audited_on": datetime.now(timezone.utc).date().isoformat(),
        "source": "all 2174 prepared synthetic sequence graphs from 2026-09-08 release",
        "nondivision_edges": distance_summary(continuations),
        "sister_separation_at_first_daughter_frame": distance_summary(sisters),
        "nodes_after_frame_zero_without_parent": starts,
        "nodes_before_last_frame_without_child": ends,
        "meaning": "Descriptive label geometry only; not new simulation or model evaluation; no reweighting or training calibration chosen.",
    }
    destination = ROOT / "checks/synthetic-motion-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination.relative_to(ROOT.parent.parent))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
