"""Post-prediction audit of optical forks, never used by inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import zarr

from tools.annotation_selection.common import write_json
from tools.annotation_selection.metric_adapter import make_graph
from tools.cellpose_ultrack.learned_divisions import OUT, WORK, paths
from tools.cellpose_ultrack.track import ROOT
from tracking_cellmot import division_metrics as official


def main():
    root,_ = paths("pair-image")
    panel = json.loads((ROOT / "panel.json").read_text())
    result = []
    for clip in panel["clips"]:
        name = clip["dataset"]
        with np.load(root / "tracking" / name / "graph.npz",allow_pickle=False) as d:
            nodes,edges = d["nodes"],d["edges"]
        pred,ids = make_graph(nodes,edges)
        gt = zarr.open_group(Path(clip["image_path"]).with_suffix(".geff"),mode="r")
        gn = np.column_stack([gt["nodes/ids"][:],*[gt[f"nodes/props/{a}/values"][:] for a in "tzyx"]]).astype(np.int64)
        ge = np.asarray(gt["edges/ids"][:],dtype=np.int64)
        truth,_ = make_graph(gn,ge)
        score = official.score_divisions(pred,truth,tuple(clip["spacing_um"]),7.)
        tp,fp = {int(ids[i]) for i in score.tp_forks},{int(ids[i]) for i in score.fp_forks}
        measured = json.loads((root / "evaluation" / f"{name}.json").read_text())
        if len(tp)!=measured["division_tp"] or len(fp)!=measured["division_fp"]:
            raise ValueError("Division audit and measured score differ")
        children = defaultdict(list)
        for p,c in edges:
            children[int(p)].append(int(c))
        positions = {int(n[0]):n for n in nodes}
        with np.load(WORK / "division-evidence" / f"{name}.npz",allow_pickle=False) as d:
            optical = {tuple(map(int,p)):float(v) for p,v in zip(d["pairs"],d["optical_logit"],strict=True)}
        def depth(start):
            current = {start}
            for i in range(5):
                current = {c for p in current for c in children[p]}
                if not current:
                    return i
            return 5
        for p in sorted(tp|fp):
            cs = sorted(children[p])
            if len(cs)!=2:
                raise ValueError("Official division is not a binary fork")
            ds = [depth(c) for c in cs]
            result.append(dict(dataset=name,parent=p,time=int(positions[p][1]),kind="TP" if p in tp else "FP",
                               daughter_depths_capped_at_5=ds,optical_logit=optical[(p,*cs)],
                               missing_immediate_daughter_persistence=min(ds)==0,
                               missing_two_step_daughter_persistence=min(ds)<2,
                               timepoints_remaining=99-int(positions[p][1])))
    summary = {}
    for kind in ("TP","FP"):
        rows = [r for r in result if r["kind"]==kind]
        summary[kind] = dict(events=len(rows),no_immediate_persistence=sum(r["missing_immediate_daughter_persistence"] for r in rows),
                             less_than_two_step_persistence=sum(r["missing_two_step_daughter_persistence"] for r in rows),
                             minimum_depth_histogram=dict(Counter(min(r["daughter_depths_capped_at_5"]) for r in rows)))
    write_json(OUT / "branch-audit.json",dict(summary=summary,events=result,
               scope="Official evaluable forks in the frozen pair-image graph. Path depth is a diagnostic, not a biological label or an inference-time GT feature."))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
