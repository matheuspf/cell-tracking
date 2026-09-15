"""Graph decoding for released association models, independent of evaluation."""
from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
import types

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .common import BANK, CONFIG, ORG, ROOT, RESULTS, arrays, bank, clips, guard, read, save, sha, write


def solve_flow(graph, *, allow_divisions=True, time_limit=300.):
    """Exact binary flow objective encoded by OrganoidTracker's DPCT graph.

    Node states are 0/1; one incoming link or birth supplies a selected node.
    A division supplies its second outgoing link and excludes termination.
    Costs include the original omitted-detection constant in the reported value.
    """
    hypotheses=graph["segmentationHypotheses"]
    links=graph["linkingHypotheses"]
    n,e=len(hypotheses),len(links)
    if not n:
        return set(),set(),dict(optimal=True,objective=0.,gap=0.)
    index={h["id"]:i for i,h in enumerate(hypotheses)}
    # Variable order: selected nodes, births, deaths, divisions, edges.
    c=np.zeros(4*n+e,np.float64);upper=np.ones_like(c)
    constant=0.
    rr=[];cc=[];vv=[]
    def add(row,col,value):rr.append(row);cc.append(col);vv.append(value)
    for i,h in enumerate(hypotheses):
        constant+=h["features"][0][0]
        c[i]=h["features"][1][0]-h["features"][0][0]
        c[n+i]=h["appearanceFeatures"][1][0]
        c[2*n+i]=h["disappearanceFeatures"][1][0]
        if allow_divisions and "divisionFeatures" in h:
            c[3*n+i]=h["divisionFeatures"][1][0]
        else:upper[3*n+i]=0
        add(i,i,-1);add(i,n+i,1)
        add(n+i,i,-1);add(n+i,2*n+i,1);add(n+i,3*n+i,-1)
        add(2*n+i,3*n+i,1);add(2*n+i,i,-1)
        add(3*n+i,3*n+i,1);add(3*n+i,2*n+i,1)
    for j,h in enumerate(links):
        a,b=index[h["src"]],index[h["dest"]]
        c[4*n+j]=h["features"][1][0]-h["features"][0][0]
        constant+=h["features"][0][0]
        add(b,4*n+j,1);add(n+a,4*n+j,1)
    A=coo_matrix((vv,(rr,cc)),shape=(4*n,len(c))).tocsc()
    lower=np.r_[np.zeros(2*n),np.full(2*n,-np.inf)]
    upper_constraint=np.r_[np.zeros(3*n),np.ones(n)]
    started=time.monotonic()
    result=milp(c,integrality=np.ones(len(c),np.uint8),bounds=Bounds(np.zeros(len(c)),upper),
                constraints=LinearConstraint(A,lower,upper_constraint),
                options=dict(time_limit=time_limit,mip_rel_gap=0.,presolve=True,threads=2))
    if not result.success or result.mip_gap>1e-8:
        raise RuntimeError(f"OrganoidTracker flow failed to reach optimum: {result.message}; gap={getattr(result,'mip_gap',None)}")
    rounded=np.rint(result.x)
    assert np.max(abs(rounded-result.x))<1e-5
    lhs=A@rounded
    assert np.all(lhs>=lower-1e-7) and np.all(lhs<=upper_constraint+1e-7)
    selected={h["id"] for i,h in enumerate(hypotheses) if rounded[i]}
    edges={(h["src"],h["dest"]) for j,h in enumerate(links) if rounded[4*n+j]}
    return selected,edges,dict(optimal=True,objective=float(result.fun+constant),gap=float(result.mip_gap),
                              seconds=time.monotonic()-started,variables=len(c),constraints=A.shape[0],
                              solver="SciPy milp / HiGHS, original DPCT costs and binary flow constraints")


def upstream_graph(data,clip,frames):
    # The actual upstream graph construction can be used without installing
    # DPCT. Its two solver methods deliberately fail if accidentally called.
    def forbidden(*args,**kwargs):raise RuntimeError("DPCT backend unavailable; use the verified exact flow adapter")
    sys.modules["dpct"]=types.SimpleNamespace(trackFlowBased=forbidden,trackMagnusson=forbidden)
    sys.path.insert(0,str(ORG))
    from organoid_tracker.linking import dpct_linker
    from .organoid import experiment
    ex,positions=experiment(data,clip["spacing_um"])
    # A metadata-only image loader suffices for the upstream boundary penalty.
    class ShapeLoader:
        def get_image_size_zyx(self):return tuple(clip["shape"][1:])
        def uncached(self):return self
        def first_time_point_number(self):return 0
        def last_time_point_number(self):return int(clip["shape"][0])-1
    ex.images.image_loader(ShapeLoader())
    for frame in frames:
        for nid,p,penalty in zip(frame["node_ids"],frame["division_probability"],frame["division_penalty"]):
            ex.positions.set_position_data(positions[int(nid)],"division_probability",float(p))
            ex.positions.set_position_data(positions[int(nid)],"division_penalty",float(penalty))
        for (a,b),p,penalty in zip(frame["pairs"],frame["link_probability"],frame["link_penalty"]):
            ex.links.add_link(positions[int(a)],positions[int(b)])
            ex.links.set_link_data(positions[int(a)],positions[int(b)],"link_probability",float(p))
            ex.links.set_link_data(positions[int(a)],positions[int(b)],"link_penalty",float(penalty))
    random.seed(read(CONFIG)["organoid"]["seed"])
    for name in ("appearance_penalty","disappearance_penalty"):
        dpct_linker.calculate_appearance_penalty(ex,.01,name=name,buffer_distance=8,only_top=True)
    ids=dpct_linker._PositionToId()
    graph,has_divisions,_=dpct_linker._create_dpct_graph(ids,ex.links,ex.positions,0,clip["shape"][0]-1)
    reverse={p:n for n,p in positions.items()}
    mapping={h["id"]:reverse[ids.position(h["id"])] for h in graph["segmentationHypotheses"]}
    return graph,mapping


def cellect_edges(score,allow_divisions):
    proposals=[]
    for row,a in enumerate(score["source_ids"]):
        probability=score["similarity"][row]
        # The sixth independent sigmoid denotes no match among the five choices.
        if int(np.argmax(probability))>=5:continue
        candidates={}
        for j,b in enumerate(score["target_ids"][row]):
            distance=float(score["distances"][row,j]);p=float(probability[j])
            if int(b) not in candidates or p>candidates[int(b)][0]:candidates[int(b)]=(p,distance)
        ordered=sorted(candidates.items(),key=lambda item:(-item[1][0],item[0]))
        size=float(score["source_sizes"][row]);quality=float(score["division"][row])
        ordered=[(b,p,d) for b,(p,d) in ordered if d<max(size,5)*6][:2]
        if not ordered:continue
        b,p,d=ordered[0]
        if p>.1:proposals.append((int(a),b,p))
        if allow_divisions and len(ordered)>1:
            b,p,d=ordered[1]
            if quality>.8 or (p>.5 and quality>.5 and abs(d-ordered[0][2])<1.2*size):
                proposals.append((int(a),b,p))
    incoming={}
    for a,b,p in sorted(proposals,key=lambda value:(-value[2],value[0],value[1])):
        if b not in incoming:incoming[b]=(a,b)
    return np.array(sorted(incoming.values()),np.int64).reshape(-1,2)


def write_graph(clip,arm,nodes,edges,inputs,detail):
    from tools.cellpose_ultrack.track import validate_graph,write_csv
    dest=ROOT/"graphs"/arm/f"{clip['dataset']}.npz"
    if dest.with_suffix(".json").exists():
        old=read(dest.with_suffix(".json"));assert old["inputs"]==inputs and old["sha256"]==sha(dest)
        return
    validate_graph(nodes,edges,clip["shape"])
    save(dest,nodes=nodes,edges=edges)
    write_csv(dest.with_suffix(".csv"),clip["dataset"],nodes,edges)
    write(dest.with_suffix(".json"),dict(inputs=inputs,sha256=sha(dest),csv_sha256=sha(dest.with_suffix(".csv")),
          nodes=len(nodes),edges=len(edges),baseline_edges_used=False,biohub_fitting=False,**detail))
    print("GRAPH",arm,clip["dataset"],len(nodes),len(edges),flush=True)


def run(family,names=None):
    guard()
    for clip in clips():
        name=clip["dataset"]
        if names and name not in names:continue
        data=bank(name)
        if family=="organoid":
            paths=[ROOT/"organoid/frames"/name/f"t{t:03}.npz" for t in range(clip["shape"][0])]
            frames=[arrays(p) for p in paths]
            graph,mapping=upstream_graph(data,clip,frames)
        else:
            paths=[ROOT/"cellect/scores"/name/f"t{t:03}.npz" for t in range(clip["shape"][0]-1)]
            frames=[arrays(p) for p in paths]
        inputs=dict(config=sha(CONFIG),code=sha(__file__),bank=sha(BANK/f"{name}.npz"),frames=[sha(p) for p in paths])
        for divisions in (True,False):
            arm=family+("" if divisions else "_no_division")
            if family=="organoid":
                selected,linked,detail=solve_flow(graph,allow_divisions=divisions)
                ids={mapping[i] for i in selected}
                nodes=data["nodes"][np.isin(data["nodes"][:,0],list(ids))]
                edges=np.array(sorted((mapping[a],mapping[b]) for a,b in linked),np.int64).reshape(-1,2)
                detail.update(input_nodes=len(data["nodes"]),selected_nodes=len(nodes),upstream_pruned_links=len(graph["linkingHypotheses"]))
            else:
                nodes=data["nodes"]
                edges=np.concatenate([cellect_edges(frame,divisions) for frame in frames])
                detail=dict(all_observations_preserved=True,decoder="Released ranking/distance/division gates; deterministic incoming tie resolution")
            write_graph(clip,arm,nodes,edges,inputs,dict(division_enabled=divisions,**detail))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family",choices=["organoid","cellect"])
    parser.add_argument("--datasets",nargs="+")
    args=parser.parse_args();run(args.family,args.datasets)


if __name__=="__main__":main()
