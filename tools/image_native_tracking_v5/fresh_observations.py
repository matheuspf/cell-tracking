"""Image/model-only version of the measured streamed observation producer.

Its arrays are compared to the study producer on fresh complete clips. No study
graph, source label, cached feature, count estimate, or inventory is read here.
"""
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from . import native_adapter as native
from .observations import peaks,regions

def observe(model,base_nodes,image_path):
    frames=native.Frames(image_path)
    if tuple(frames.shape)!=(100,64,256,256):
        raise ValueError('V5 observation model validated for 100x64x256x256; use explicit C0 fallback for this shape')
    nodes={};confidence={};owners={};fs={};ft={};next_id=int(base_nodes[:,0].max())+1 if len(base_nodes) else 0
    jobs={}
    with torch.no_grad(),ThreadPoolExecutor(max_workers=2) as pool:
        for t in range(99):
            features,det=model.encode(frames.pair(t))
            for j in range(2):
                tt=t+j
                if tt in nodes:continue
                old=base_nodes[base_nodes[:,1]==tt]
                prob=det[j][0,0].sigmoid().cpu().numpy();im=frames.frame(tt).numpy()
                new,cf,owner=peaks(prob,im,old,next_id,tt);next_id+=len(new)
                nodes[tt]=np.concatenate([old,new]);confidence[tt]=np.concatenate([np.ones(len(old),np.float32),cf])
                owners[tt]=np.concatenate([np.full(len(old),-1,np.int64),owner])
                jobs[tt]=pool.submit(regions,im,prob,nodes[tt])
            inputs=native.pair_inputs(nodes[t],nodes[t+1])
            fs[t]=model._index_features(features[:,0],inputs[0][0],inputs[0][2])[0].cpu().numpy()
            ft[t+1]=model._index_features(features[:,1],inputs[1][0],inputs[1][2])[0].cpu().numpy()
            del features,det
        props={t:job.result() for t,job in jobs.items()}
    stacked=np.concatenate([nodes[t] for t in range(100)])
    return dict(nodes=stacked,oldmask=np.isin(stacked[:,0],base_nodes[:,0]),
        features_source=np.concatenate([fs.get(t,np.zeros((len(nodes[t]),32),np.float32)) for t in range(100)]).astype(np.float16),
        features_target=np.concatenate([ft.get(t,np.zeros((len(nodes[t]),32),np.float32)) for t in range(100)]).astype(np.float16),
        properties=np.concatenate([props[t][0] for t in range(100)]),valid_region=np.concatenate([props[t][1] for t in range(100)]),
        collisions=np.concatenate([props[t][2] for t in range(100)]),optical_centroids=np.concatenate([props[t][3] for t in range(100)]),
        confidence=np.concatenate([confidence[t] for t in range(100)]),split_owner=np.concatenate([owners[t] for t in range(100)]))

if __name__=='__main__':
    import time,sys
    from .common import *
    model=native.load();records=[]
    for row in read(OUT/'density_selection.json')['pilots']:
        name=row['dataset'];base=arrays(OUT/'pilots/C0'/name/'predictions'/f'{name}.npz');start=time.monotonic()
        fresh=observe(model,base['nodes'],DATA/'train'/f'{name}.zarr');old=arrays(OUT/'observations'/f'{name}.npz')
        exact={k:bool(np.array_equal(v,old[k],equal_nan=True)) for k,v in fresh.items()}
        assert all(exact.values()),exact
        records.append(dict(dataset=name,fields=exact,seconds=time.monotonic()-start))
    write(OUT/'fresh_observation_parity.json',dict(clips=records,all_arrays_exact=True,source='fresh full-image computation; cached comparison only after prediction'))
    print(records,flush=True)
