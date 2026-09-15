"""Released OrganoidTracker2 link/division models on fixed Cellpose observations."""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from .common import BANK, CONFIG, ORG, ROOT, RESULTS, Images, Lease, bank, clips, guard, read, save, sha, verify_repo, write


def setup():
    verify_repo(ORG, "organoid")
    sys.path.insert(0, str(ORG))
    import _keras_environment
    _keras_environment.activate()
    import torch
    torch.set_num_threads(2)


def load_models():
    setup()
    import keras
    from keras.src.backend.torch.core import device_scope
    from organoid_tracker.neural_network.link_detection_cnn.link_predictor import LinkModel
    from organoid_tracker.neural_network.division_detection_cnn.division_predictor import DivisionModel
    models = []
    for kind, cls in (("links", LinkModel), ("divisions", DivisionModel)):
        folder = ROOT / "models/organoid" / f"model_{kind}"
        expected = "link_model_sha256" if kind == "links" else "division_model_sha256"
        assert sha(folder / "model.keras") == read(CONFIG)["organoid"][expected]
        settings = read(folder / "settings.json")
        with device_scope("cpu"):
            model = keras.saving.load_model(folder / "model.keras", compile=False, safe_mode=True)
        models.append(cls(model, tuple(settings["time_window"]), tuple(settings["patch_shape_zyx"]),
                          float(settings["platt_scaling"]), float(settings["platt_intercept"])/np.log(10)))
    return models


def experiment(data, spacing):
    from organoid_tracker.core.experiment import Experiment
    from organoid_tracker.core.position import Position
    from organoid_tracker.core.resolution import ImageResolution
    ex = Experiment()
    ex.images.set_resolution(ImageResolution(float(spacing[2]), float(spacing[1]), float(spacing[0]), 1.))
    positions = {}
    for node, point in zip(data["nodes"], data["positions"]):
        nid, t = map(int, node[:2])
        p = Position(x=float(point[2]), y=float(point[1]), z=float(point[0]), time_point_number=t)
        ex.positions.add(p)
        positions[nid] = p
    assert len(ex.positions) == len(positions), "Coincident Cellpose positions must retain identity"
    return ex, positions


def prepare(clip):
    from organoid_tracker.linking.nearest_neighbor_linker import nearest_neighbor
    name = clip["dataset"]
    dest = ROOT / "organoid/candidates" / f"{name}.npz"
    inputs = dict(bank=sha(BANK/f"{name}.npz"), code=sha(__file__), config=sha(CONFIG))
    if dest.with_suffix(".json").exists():
        old=read(dest.with_suffix(".json"))
        assert old["inputs"]==inputs and old["sha256"]==sha(dest)
        from .common import arrays
        return arrays(dest)
    data = bank(name)
    ex, positions = experiment(data, clip["spacing_um"])
    inverse = {p:n for n,p in positions.items()}
    links = nearest_neighbor(ex, tolerance=2)
    pairs = np.array(sorted((inverse[a], inverse[b]) for a,b in links.find_all_links()), np.int64).reshape(-1,2)
    save(dest, **data, pairs=pairs)
    write(dest.with_suffix(".json"), dict(inputs=inputs, sha256=sha(dest), nodes=len(data["nodes"]), pairs=len(pairs)))
    return dict(**data, pairs=pairs)


def resize_patch(patch, shape):
    from skimage.transform import resize
    return np.stack([resize(patch[...,i], shape, order=0, clip=False, preserve_range=True, anti_aliasing=False)
                     for i in range(patch.shape[-1])], axis=-1).astype(np.float32)


def coord_inputs(a, b, distances, device="cpu"):
    import torch
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)
    d = torch.as_tensor(distances, dtype=torch.float32, device=device)
    z,y,x = a.shape[1:4]
    grids = torch.meshgrid(*(torch.arange(-n//2,n//2,device=device,dtype=torch.float32) for n in (z,y,x)), indexing="ij")
    left=[(grid[None]-d[:,k,None,None,None]).abs()[...,None]/n for k,(grid,n) in enumerate(zip(grids,(z,y,x)))]
    right=[(grid[None]+d[:,k,None,None,None]).abs()[...,None]/n for k,(grid,n) in enumerate(zip(grids,(z,y,x)))]
    return {"input_1":torch.cat([a,*left],-1),"input_2":torch.cat([*right[::-1],b],-1),"input_distances":d}


def calibrated(raw, model):
    raw = np.asarray(raw, np.float64).reshape(-1)
    likelihood = model.platt_intercept + model.platt_scaling*(np.log10(raw+1e-10)-np.log10(1-raw+1e-10))
    from scipy.special import expit
    return expit(likelihood*np.log(10)), -likelihood


def score_frame(clip, t, data, models, lease, image_cache):
    import torch
    from keras.src.backend.torch.core import device_scope
    from organoid_tracker.core import TimePoint
    from organoid_tracker.neural_network.division_detection_cnn.division_predictor import _split_into_patches as division_patches
    from organoid_tracker.neural_network.link_detection_cnn.link_predictor import _extract_patch_array_normalized
    name=clip["dataset"]
    dest=ROOT/"organoid/frames"/name/f"t{t:03}.npz"
    inputs=dict(candidates=sha(ROOT/"organoid/candidates"/f"{name}.npz"), config=sha(CONFIG), code=sha(__file__))
    if dest.with_suffix(".json").exists():
        old=read(dest.with_suffix(".json")); assert old["inputs"]==inputs and old["sha256"]==sha(dest)
        return
    started=time.monotonic()
    link_model, div_model=models
    scale=tuple(np.asarray(clip["spacing_um"])/read(CONFIG)["organoid"]["model_spacing_zyx_um"])
    ii=np.flatnonzero(data["nodes"][:,1]==t)
    ex,positions=experiment(dict(nodes=data["nodes"][ii],positions=data["positions"][ii]),clip["spacing_um"])
    patches=list(division_patches(image_cache,TimePoint(t),positions.values(),div_model.time_window,
                     patch_shape_zyx_px=div_model.patch_shape_zyx,scale_factors_zyx=scale,intensity_quantiles=(.01,.99)))
    batch_size=read(CONFIG)["organoid"]["batch_size"]
    raw_div=[]
    for start in range(0,len(patches),batch_size):
        batch=np.stack([resize_patch(p.array,div_model.patch_shape_zyx) for p in patches[start:start+batch_size]])
        lease.acquire()
        with device_scope("cuda:0"), torch.inference_mode():
            output=div_model.keras_model(torch.as_tensor(batch,device="cuda:0"),training=False)
            raw_div.extend(output.cpu().numpy().reshape(-1))
            del output
        lease.finish_batch()
    raw_div=np.array(raw_div,np.float32)
    dp,dc=calibrated(raw_div,div_model)
    nodes=data["nodes"]
    node_index={int(n[0]):i for i,n in enumerate(nodes)}
    mask=np.array([nodes[node_index[int(a)],1]==t for a,b in data["pairs"]])
    pairs=data["pairs"][mask]
    raw_links=[]
    if len(pairs):
        all_ids=np.unique(pairs)
        ids=np.array([node_index[int(n)] for n in all_ids])
        _,pos=experiment(dict(nodes=nodes[ids],positions=data["positions"][ids]),clip["spacing_um"])
        # Extract and resize once per center per two-frame context. CoordConv
        # remains edge-specific, exactly as in the released predictor.
        full_images={TimePoint(t+dt):image_cache.get_image(TimePoint(t+dt)) for dt in range(link_model.time_window[0],link_model.time_window[1]+1)}
        low=min(float(np.quantile(im.array,.01)) for im in full_images.values())
        high=max(float(np.quantile(im.array,.99)) for im in full_images.values())
        assert high>low
        native_patch=tuple(int(n/s) for n,s in zip(link_model.patch_shape_zyx,scale))
        cache={}
        for nid in all_ids:
            p=pos[int(nid)]
            patch=_extract_patch_array_normalized(full_images,p,native_patch,low,high)
            cache[int(nid)]=resize_patch(patch,link_model.patch_shape_zyx)
        for start in range(0,len(pairs),batch_size):
            part=pairs[start:start+batch_size]
            a=np.stack([cache[int(i)] for i,j in part]); b=np.stack([cache[int(j)] for i,j in part])
            delta=np.array([np.round((data["positions"][node_index[int(j)]].astype(float)-data["positions"][node_index[int(i)]])*scale) for i,j in part])
            lease.acquire()
            with device_scope("cuda:0"), torch.inference_mode():
                inputs_gpu=coord_inputs(a,b,delta,"cuda:0")
                output=link_model.keras_model(inputs_gpu,training=False)
                raw_links.extend(output.cpu().numpy().reshape(-1))
                del output, inputs_gpu
            lease.finish_batch()
    raw_links=np.array(raw_links,np.float32)
    lp,lc=calibrated(raw_links,link_model)
    assert all(np.isfinite(v).all() for v in (dp,dc,lp,lc))
    save(dest,node_ids=nodes[ii,0],raw_division=raw_div,division_probability=dp,division_penalty=dc,
         pairs=pairs,raw_links=raw_links,link_probability=lp,link_penalty=lc)
    write(dest.with_suffix(".json"),dict(inputs=inputs,sha256=sha(dest),seconds=time.monotonic()-started,nodes=len(ii),edges=len(pairs),runtime=lease.receipt()))
    print("ORGANOID",name,t,len(ii),len(pairs),round(time.monotonic()-started,2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+")
    parser.add_argument("--frames",type=int,default=100)
    args=parser.parse_args()
    guard(); models=load_models()
    lease=Lease([m.keras_model for m in models])
    try:
        for clip in clips():
            if args.datasets and clip["dataset"] not in args.datasets: continue
            data=prepare(clip)
            images=Images(clip["dataset"],clip["shape"][0])
            for t in range(min(args.frames,clip["shape"][0])):
                score_frame(clip,t,data,models,lease,images)
    finally:
        lease.release()
        write(ROOT/"organoid/runtime.json",lease.receipt())


if __name__=="__main__":
    main()
