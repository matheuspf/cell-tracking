"""GPL-2.0-only: CELLECT's released features and temporal matcher at Cellpose centers.

Ranking and gates follow pinned CELLECT inference.py. External instances replace
the detector's candidate grouping; this is an association transfer experiment.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import sys
import time

import numpy as np
import torch

from .common import BANK, CEL, CONFIG, ROOT, RESULTS, Images, Lease, arrays, bank, clips, guard, read, save, sha, verify_repo, write


def load_models():
    verify_repo(CEL,"cellect")
    sys.path.insert(0,str(CEL))
    architecture=importlib.import_module("unetext3Dn_con7")
    backbone=architecture.UNet3D(2,6).eval()
    matcher=architecture.EXNet(64,8).eval()
    config=read(CONFIG)["cellect"]
    weights={}
    for key,model in (("backbone",backbone),("matcher",matcher)):
        path=CEL/"model"/config[key]
        weights[key]=dict(path=str(path),sha256=sha(path))
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    # Execute the unchanged pure feature function only, avoiding the training
    # loss module's unnecessary import-time CUDA allocations.
    source=CEL/"recoloss.py"
    tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="sort_feature")
    scope={"torch":torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),scope)
    write(RESULTS/"cellect-models.json",dict(weights=weights,revision=config["revision"],
          feature_source_sha256=sha(source),backbone_parameters=sum(p.numel() for p in backbone.parameters()),
          matcher_parameters=sum(p.numel() for p in matcher.parameters()),scope="No Biohub fitting"))
    return backbone,matcher,scope["sort_feature"]


def tiles(shape):
    patch=np.array([256,256,32]); step=np.array([248,248,28])
    shape=np.maximum(np.array(shape),patch)
    axes=[]
    for length,size,stride in zip(shape,patch,step):
        axes.append(sorted({min(i*int(stride),int(length-size)) for i in range(int(np.ceil(length/stride)))}))
    from itertools import product
    return np.array(list(product(*axes)),np.int64),patch,shape


def tile_owners(points,starts,patch):
    local=points[None]-starts[:,None]
    inside=np.all((local>=0)&(local<patch),axis=2)
    margin=np.minimum(local,patch-1-local)*np.array([1,1,4])
    score=margin.min(2)+1e-6*margin.sum(2)
    score[~inside]=-np.inf
    assert np.all(inside.any(0)),"A Cellpose center lacks a covering tile"
    return score.argmax(0)


def sample_outputs(outputs,indices):
    _,_,features,division,size=outputs
    y,x,z=indices.T
    # Same channel lookup as upstream vsup256infer at integer candidate centers.
    f=features[0,:,y,x,z].T
    d=division[0,:,y,x,z].T.sigmoid()
    s=size[0,0,y,x,z]
    return f,d,s


def extract_frame(clip,t,data,model,lease,images):
    name=clip["dataset"]
    dest=ROOT/"cellect/features"/name/f"t{t:03}.npz"
    inputs=dict(bank=sha(BANK/f"{name}.npz"),config=sha(CONFIG),code=sha(__file__))
    if dest.with_suffix(".json").exists():
        old=read(dest.with_suffix(".json"));assert old["inputs"]==inputs and old["sha256"]==sha(dest)
        return
    started=time.monotonic()
    ii=np.flatnonzero(data["nodes"][:,1]==t)
    native_shape=tuple(clip["shape"][1:])
    first,second=images.raw(t),images.raw(t+1)
    assert first.shape==second.shape==native_shape
    raw=np.stack([first.transpose(1,2,0),second.transpose(1,2,0)]).astype(np.float32)
    raw=np.clip(raw,0,65535)
    positive=raw[raw>0]
    minimum=float(positive.min()) if len(positive) else 0.
    raw=np.log1p(np.maximum(raw,minimum)+1900).astype(np.float32)
    starts,patch,padded=tiles(raw.shape[1:])
    if tuple(padded)!=raw.shape[1:]:
        raw=np.pad(raw,[(0,0),*[(0,int(n-old)) for n,old in zip(padded,raw.shape[1:])]],constant_values=float(raw.min()))
    points=data["nodes"][ii][:,[3,4,2]].astype(np.int64)
    owners=tile_owners(points,starts,patch)
    f=np.empty((len(ii),64),np.float32);d=np.empty((len(ii),2),np.float32);s=np.empty(len(ii),np.float32)
    done=0
    for k,start in enumerate(starts):
        jj=np.flatnonzero(owners==k)
        if not len(jj): continue
        a,b,c=map(int,start)
        batch=np.ascontiguousarray(raw[:,a:a+256,b:b+256,c:c+32][None])
        lease.acquire()
        with torch.inference_mode():
            tensor=torch.from_numpy(batch).cuda()
            output=model(tensor)
            coords=torch.as_tensor(points[jj]-start,device="cuda",dtype=torch.long)
            values=sample_outputs(output,coords)
            for dest_array,value in zip((f,d,s),values):dest_array[jj]=value.cpu().numpy()
            del tensor,output,values,coords
        lease.finish_batch();done+=1
    assert np.isfinite(f).all() and np.isfinite(d).all() and np.isfinite(s).all()
    assert np.all(np.linalg.norm(f,axis=1)>0),"Unusable zero feature at Cellpose center"
    save(dest,node_ids=data["nodes"][ii,0],features=f,division=d,sizes=s,owner=owners)
    write(dest.with_suffix(".json"),dict(inputs=inputs,sha256=sha(dest),nodes=len(ii),patches=done,
          seconds=time.monotonic()-started,runtime=lease.receipt(),duplicate_last_frame=t==clip["shape"][0]-1))
    print("CELLECT FEATURES",name,t,len(ii),done,round(time.monotonic()-started,2),flush=True)


def score_transition(clip,t,data,matcher,sort_feature):
    from scipy.spatial import cKDTree
    name=clip["dataset"]
    paths=[ROOT/"cellect/features"/name/f"t{tt:03}.npz" for tt in (t,t+1)]
    dest=ROOT/"cellect/scores"/name/f"t{t:03}.npz"
    inputs=dict(features=[sha(p) for p in paths],config=sha(CONFIG),code=sha(__file__))
    if dest.with_suffix(".json").exists():
        old=read(dest.with_suffix(".json"));assert old["inputs"]==inputs and old["sha256"]==sha(dest)
        return
    a,b=map(arrays,paths)
    ni={int(n[0]):i for i,n in enumerate(data["nodes"])}
    pp=[]
    for value in (a,b):
        index=np.array([ni[int(i)] for i in value["node_ids"]])
        points=data["positions"][index][:,[1,2,0]].astype(np.float32)
        points[:,2]*=float(clip["spacing_um"][0]/clip["spacing_um"][1])
        pp.append(torch.from_numpy(points))
    count=min(5,len(a["node_ids"]),len(b["node_ids"]))
    assert count>0
    _,indices=cKDTree(pp[1].numpy()).query(pp[0].numpy(),k=count)
    indices=np.asarray(indices).reshape(len(a["node_ids"]),count)
    if count<5:indices=np.pad(indices,((0,0),(0,5-count)),mode="edge")
    nn=torch.from_numpy(indices)
    with torch.inference_mode():
        features,offsets,sizes,divisions=sort_feature(torch.from_numpy(b["features"]),pp[1],pp[0],nn,
             torch.from_numpy(b["sizes"]),torch.from_numpy(a["sizes"]),torch.from_numpy(b["division"]),torch.from_numpy(a["division"]),n=5)
        logits=matcher(torch.from_numpy(a["features"])[:,None],features,offsets,sizes,divisions)
        assert logits.shape==(len(nn),8) and torch.isfinite(logits).all()
        similarity=logits[:,:6].sigmoid().numpy()
        division=logits[:,-2:].softmax(1).numpy()[:,1]
    save(dest,source_ids=a["node_ids"],target_ids=b["node_ids"][indices],similarity=similarity,
         division=division,distances=offsets.norm(dim=2).numpy(),source_sizes=a["sizes"],logits=logits.numpy())
    write(dest.with_suffix(".json"),dict(inputs=inputs,sha256=sha(dest),source_nodes=len(nn),raw_output_channels=8,
          six_independent_match_scores=True,last_two_channels="softmax division flag"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+")
    parser.add_argument("--frames",type=int,default=100)
    args=parser.parse_args()
    guard();torch.set_num_threads(2)
    backbone,matcher,sort_feature=load_models()
    lease=Lease([backbone])
    try:
        for clip in clips():
            if args.datasets and clip["dataset"] not in args.datasets:continue
            data=bank(clip["dataset"]);images=Images(clip["dataset"],clip["shape"][0])
            for t in range(min(args.frames,clip["shape"][0])):
                extract_frame(clip,t,data,backbone,lease,images)
            lease.release()
            for t in range(min(args.frames,clip["shape"][0])-1):
                score_transition(clip,t,data,matcher,sort_feature)
    finally:
        lease.release();write(ROOT/"cellect/runtime.json",lease.receipt())


if __name__=="__main__":
    main()
