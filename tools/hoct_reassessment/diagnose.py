"""Evaluation-only endpoint/candidate attribution and matched comparator changes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from .common import CONFIG, ROOT, RESULTS, arrays, pilot, read, write


def run():
    out = {}
    arms = [f"cellpose_{m}{control}_cuda" for m in ("general_v1", "ctc_v0") for control in ("", "_no_division")]
    if (ROOT / "evaluation/cellpose_source_residual_cuda").exists():
        arms += ["cellpose_source_residual_cuda", "cellpose_source_residual_no_division_cuda"]
    for arm in arms:
        counts = Counter()
        clips = []
        for clip in pilot():
            name = clip["dataset"]
            pred = arrays(ROOT / "graphs" / arm / f"{name}.npz")
            matched = arrays(ROOT / "matches" / arm / f"{name}.npz")
            match = dict(map(tuple, matched["matches"]))
            reverse = {int(g): int(p) for p, g in match.items()}
            bank = set(map(tuple, arrays(ROOT / "cellpose/features" / f"{name}.npz")["pairs"]))
            edges = set(map(tuple, pred["edges"]))
            true_edges = set(map(tuple, matched["tp_edges"]))
            gt = zarr.open_group(Path(clip["image_path"]).with_suffix(".geff"), mode="r")
            gt_edges = np.asarray(gt["edges/ids"][:])
            out_gt, in_gt = set(gt_edges[:, 0]), set(gt_edges[:, 1])
            causes = Counter()
            for a, b in gt_edges:
                source, target = reverse.get(int(a)), reverse.get(int(b))
                if (source, target) in edges:
                    continue
                if source is None or target is None:
                    causes["missing_matched_endpoint"] += 1
                elif (source, target) not in bank:
                    causes["outside_candidate_bank"] += 1
                else:
                    causes["candidate_present_not_selected"] += 1
            fp_causes = Counter()
            for a, b in edges - true_edges:
                ma, mb = int(a) in match, int(b) in match
                # A matched endpoint is evaluable only where GT records a link
                # in that direction; sparse track starts/ends stay unknown.
                if match.get(int(a)) not in out_gt and match.get(int(b)) not in in_gt:
                    continue
                fp_causes["both_matched_wrong_identity" if ma and mb else "touches_unmatched_endpoint"] += 1
            score = read(ROOT / "evaluation" / arm / f"{name}.json")
            assert sum(causes.values()) == score["edge_fn"]
            assert sum(fp_causes.values()) == score["edge_fp"]
            counts.update(causes)
            counts.update(fp_causes)
            clips.append(dict(dataset=name, missing_edges=dict(causes), false_edges=dict(fp_causes)))
        out[arm] = dict(counts=dict(counts), clips=clips)
    write(ROOT / "diagnostics.json", out)
    write(RESULTS / "diagnostics.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    run()
