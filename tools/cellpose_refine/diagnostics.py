"""Evaluation-only GT strata and signed errors; never imported by the trainer."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .common import SPACING


def division_neighborhoods(gt):
    result = {}
    for name, clip in gt["clips"].items():
        times = {int(n[0]): int(n[1]) for n in clip["nodes"]}
        outgoing, adjacent = defaultdict(set), defaultdict(set)
        for a, b in clip["edges"]:
            outgoing[int(a)].add(int(b))
            adjacent[int(a)].add(int(b)); adjacent[int(b)].add(int(a))
        selected = set()
        for parent, daughters in outgoing.items():
            if len(daughters) < 2:
                continue
            visited, frontier = {parent}, {parent}
            for _ in range(2):
                frontier = {b for a in frontier for b in adjacent[a]} - visited
                visited |= frontier
            selected |= {n for n in visited if abs(times[n] - times[parent]) <= 2}
        result[name] = selected
    return result


def make_strata(frames, gt):
    """Definitions are fixed before any target score is opened."""
    divisions = division_neighborhoods(gt)
    labels = {}
    for frame in frames:
        volume = np.load(frame["image_path"], allow_pickle=False)
        q1, q99 = np.percentile(volume, [1., 99.])
        nodes = np.asarray(gt["frames"][frame["key"]]["gt_nodes"], dtype=np.int64).reshape(-1, 5)
        by_id = {}
        for node in nodes:
            point = node[2:].astype(float)
            index = node[2:].astype(int)
            lower = np.maximum(index - [1, 4, 4], 0)
            upper = np.minimum(index + [2, 5, 5], volume.shape)
            local = volume[tuple(slice(a, b) for a, b in zip(lower, upper))]
            grids = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lower, upper)], indexing="ij"), axis=-1)
            inside = np.linalg.norm((grids - point) * SPACING, axis=-1) <= 2.
            peak = float(local[inside].max())
            brightness = (peak - q1) / max(float(q99 - q1), 1.)
            margin = float(np.minimum(point * SPACING, (np.asarray(volume.shape) - 1 - point) * SPACING).min())
            zfraction = point[0] / (volume.shape[0] - 1)
            axial = "z_lower_quarter" if zfraction < .25 else "z_upper_quarter" if zfraction > .75 else "z_middle_half"
            tags = ["faint_peak" if brightness < .5 else "brighter_peak", "within_3um_center_bound" if margin <= 3 else "interior", axial]
            if int(node[0]) in divisions[frame["dataset"]]:
                tags.append("annotated_division_neighborhood")
            by_id[int(node[0])] = tags
        labels[frame["key"]] = by_id
    return labels


def signed_and_stratified(rows, gt_labels, embryo):
    selected = [r for r in rows if embryo == "pooled" or r["embryo"] == embryo]
    totals = defaultdict(int)
    recovered = defaultdict(lambda: defaultdict(int))
    signed = []
    for row in selected:
        for tags in gt_labels[row["key"]].values():
            for tag in tags:
                totals[tag] += 1
        for node, tags in gt_labels[row["key"]].items():
            for radius in ("3.0", "7.0"):
                if node in row["_found_ids"][radius]:
                    for tag in tags:
                        recovered[tag][radius] += 1
        for record in row["_records"].values():
            signed.append(np.asarray(record["predicted_um"]) - record["gt_um"])
    delta = np.asarray(signed, dtype=float).reshape(-1, 3)
    return {
        "strata": {tag: {"gt": n, "matched": {r: recovered[tag][r] for r in ("3.0", "7.0")},
                          "recall": {r: recovered[tag][r] / n for r in ("3.0", "7.0")}}
                   for tag, n in totals.items()},
        "signed_error_on_matched_at_7": {"n": len(delta), "axis_order": "ZYX", "unit": "micrometers",
            "mean": delta.mean(axis=0).tolist() if len(delta) else None,
            "median": np.median(delta, axis=0).tolist() if len(delta) else None,
            "mean_absolute": np.abs(delta).mean(axis=0).tolist() if len(delta) else None},
    }


DEFINITIONS = {
    "faint_peak": "Maximum raw intensity in a 2um ball about GT, normalized by full-frame 1st/99th percentiles, is below0.5. Diagnostic only; not a fitted cutoff.",
    "within_3um_center_bound": "GT is within3um of any valid center-coordinate bound (0 or shape-1), in physical units.",
    "z_groups": "Lower/middle/upper quarters/half of local crop Z. Global optical depth is unavailable; these are not true acquisition-depth strata.",
    "annotated_division_neighborhood": "Within two undirected GT graph hops and ±2frames of an annotated parent with at least two outgoing daughters.",
    "limits": "Sparse GT, repeated observations and only two reused embryos; matched-only signed errors have conditional denominators. Strata overlap. No temporal predictions are evaluated here.",
}
