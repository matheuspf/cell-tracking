"""Freeze full graphs, then use the unmodified official scorer and sparse labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import time

import numpy as np
import zarr

from .common import CONFIG, ROOT, RESULTS, arrays, clips, read, save, sha, write


def graph(arm,name):
    path=ROOT/"graphs"/arm/f"{name}.npz"
    assert read(path.with_suffix(".json"))["sha256"]==sha(path)
    return arrays(path)


def candidates(arm,name):
    if arm.startswith("organoid"):
        return set(map(tuple,arrays(ROOT/"organoid/candidates"/f"{name}.npz")["pairs"]))
    output=set()
    for t in range(99):
        values=arrays(ROOT/"cellect/scores"/name/f"t{t:03}.npz")
        for a,targets in zip(values["source_ids"],values["target_ids"]):
            output.update((int(a),int(b)) for b in targets)
    return output


def one(task):
    arm,clip,frozen=task
    from tools.annotation_selection.common import graph_hash
    from tools.annotation_selection.metric_adapter import evaluate_graph
    from tools.cellpose_ultrack.evaluate import assert_matching
    from tools.cellpose_ultrack.track import validate_graph
    name=clip["dataset"]
    data=graph(arm,name);nodes,edges=data["nodes"],data["edges"]
    assert graph_hash(nodes,edges)==frozen
    validate_graph(nodes,edges,clip["shape"])
    path=Path(clip["image_path"]).with_suffix(".geff")
    truth=zarr.open_group(path,mode="r")
    gt_nodes=np.column_stack([truth["nodes/ids"][:],*[truth[f"nodes/props/{k}/values"][:] for k in "tzyx"]]).astype(np.int64)
    gt_edges=np.asarray(truth["edges/ids"][:],np.int64).reshape(-1,2)
    metadata=truth.attrs["geff"]
    scale=[next(a["scale"] for a in metadata["axes"] if a["name"]==axis) for axis in "zyx"]
    np.testing.assert_array_equal(scale,clip["spacing_um"])
    estimate=metadata["extra"]["estimated_number_of_nodes"]
    inputs=dict(graph=frozen,gt=graph_hash(gt_nodes,gt_edges),metadata=sha(path/"zarr.json"),
                metric=read(CONFIG)["metric_revision"],evaluator=sha(__file__))
    dest=ROOT/"evaluation"/arm/f"{name}.json"
    if dest.exists():
        old=read(dest);assert old["inputs"]==inputs;return old
    start=time.monotonic()
    row,matches,tp=evaluate_graph(name,nodes,edges,gt_nodes,gt_edges,scale,estimate)
    error=assert_matching(matches,nodes,gt_nodes,np.array(scale))
    reverse={g:p for p,g in matches.items()};bank=candidates(arm,name);selected=set(map(tuple,edges))
    miss=Counter()
    for a,b in gt_edges:
        p,q=reverse.get(int(a)),reverse.get(int(b))
        if (p,q) in selected:continue
        if p is None or q is None:miss["unmatched_endpoint"]+=1
        elif (p,q) not in bank:miss["outside_candidate_bank"]+=1
        else:miss["candidate_present_not_selected"]+=1
    false=Counter();out_gt=set(gt_edges[:,0]);in_gt=set(gt_edges[:,1])
    for a,b in selected-set(tp):
        if matches.get(int(a)) not in out_gt and matches.get(int(b)) not in in_gt:continue
        false["both_matched_wrong_identity" if int(a) in matches and int(b) in matches else "touches_unmatched_endpoint"]+=1
    assert sum(miss.values())==row["edge_fn"] and sum(false.values())==row["edge_fp"]
    row.update(arm=arm,embryo=clip["embryo"],inputs=inputs,full_clip=True,seconds=time.monotonic()-start,
               gt_nodes=len(gt_nodes),gt_edges=len(gt_edges),predicted_edges=len(edges),
               predicted_divisions=int(np.sum(np.unique(edges[:,0],return_counts=True)[1]==2)),
               mean_matched_error_um=float(np.mean(error)),missing_edges=dict(miss),false_edges=dict(false))
    save(ROOT/"matches"/arm/f"{name}.npz",matches=np.array(sorted(matches.items()),np.int64).reshape(-1,2),
         tp_edges=np.array(sorted(tp),np.int64).reshape(-1,2))
    write(dest,row)
    print("SCORED",arm,name,row["edge_tp"],row["edge_fp"],row["edge_fn"],flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families",nargs="+",default=["organoid","cellect"],choices=["organoid","cellect"])
    args=parser.parse_args()
    from tools.annotation_selection.common import OFFICIAL,METRIC_REV,graph_hash
    from tools.annotation_selection.metric_adapter import aggregate
    assert read(CONFIG)["metric_revision"]==METRIC_REV
    assert subprocess.check_output(["git","-C",str(OFFICIAL),"rev-parse","HEAD"],text=True).strip()==METRIC_REV
    assert not subprocess.check_output(["git","-C",str(OFFICIAL),"status","--porcelain","--untracked-files=no"],text=True).strip()
    for family in args.families:
        assert (RESULTS/f"{family}-validation.json").exists(),"Validate models before evaluation"
    assert (RESULTS/"flow-validation.json").exists()
    arms=[family+suffix for family in args.families for suffix in ("","_no_division")]
    cohort=clips()
    lock={arm:{clip["dataset"]:graph_hash(**graph(arm,clip["dataset"])) for clip in cohort} for arm in arms}
    key="-".join(arms)
    write(ROOT/"evaluation-locks"/f"{key}.json",dict(config=sha(CONFIG),graphs=lock))
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(one,[(arm,clip,lock[arm][clip["dataset"]]) for arm in arms for clip in cohort]))
    summaries={}
    for arm in arms:
        summaries[arm]={}
        for embryo in ("pooled","44b6","6bba"):
            subset=[r for r in rows if r["arm"]==arm and (embryo=="pooled" or r["embryo"]==embryo)]
            summaries[arm][embryo]=aggregate(subset,[r["dataset"] for r in subset])
    result=dict(config_sha256=sha(CONFIG),metric_revision=METRIC_REV,frames=600,clips=cohort,rows=rows,summaries=summaries,
                scope="Six complete reused development clips. Frozen external models; no local fitting. Not unseen-embryo validation.")
    write(RESULTS/f"summary-{key}.json",result)
    for arm,s in summaries.items():print(arm,s["pooled"],flush=True)


if __name__=="__main__":main()
