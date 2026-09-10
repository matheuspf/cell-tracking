"""Frozen installed DeepCenter full-field evidence for novel observation selection."""
import time
from types import SimpleNamespace
import torch,zarr
from .common import *

def load():
    from strong_tracker_v3.replay import repair_namespace
    ctx=SimpleNamespace(full=V1/'public_harmonic_full',data=DATA,out=OUT/'deepcenter_runtime',v2=V2)
    ns=repair_namespace(ctx,image_dir=DATA/'train',fresh_heatmaps=True)
    bundle=ns['_heatmap_state']['bundle'];assert bundle is not None
    return ns,bundle

def score_nodes(ns,bundle,nodes,image_path):
    arr=zarr.open_group(str(image_path),mode='r')['0'];cfg=bundle['cfg'];factor=int(getattr(cfg,'pool_factor',4))
    confidence=np.zeros(len(nodes),np.float32);new_peak_counts=[]
    from scipy.ndimage import maximum_filter
    with torch.no_grad():
        for t in range(arr.shape[0]):
            raw=np.asarray(arr[t]);image=ns['_dc_normalize_dynamic_range'](ns['_dc_pool_frame_xy'](raw,factor),cfg)
            heat=bundle['model'](torch.from_numpy(image)[None,None].to(bundle['device'])).sigmoid()[0,0].cpu().numpy()
            ii=np.flatnonzero(nodes[:,1]==t);xyz=np.clip((nodes[ii,2:]/[1,factor,factor]).astype(int),0,np.array(heat.shape)-1)
            # Local maximum handles native/DeepCenter pooling-phase differences without moving scored centers.
            local=maximum_filter(heat,size=3);confidence[ii]=local[tuple(xyz.T)]
            new_peak_counts.append(int(((heat==local)&(heat>.5)).sum()))
    return confidence,new_peak_counts

def run():
    ns,bundle=load()
    for row in inventory():
        name=row['dataset'];dest=OUT/'deepcenter'/f'{name}.npz'
        if dest.exists():continue
        p=OUT/'observations'/f'{name}.npz'
        while not p.exists():time.sleep(10)
        c=arrays(p);start=time.monotonic();dc,counts=score_nodes(ns,bundle,c['nodes'],DATA/'train'/f'{name}.zarr')
        save(dest,confidence=dc)
        write(dest.with_suffix('.json'),dict(dataset=name,full_frames=100,full_field=True,seconds=time.monotonic()-start,
            native_proposals=len(c['nodes'])-int(c['oldmask'].sum()),deepcenter_peaks=int(sum(counts)),
            new_native_peaks_with_deepcenter_gt05=int(((~c['oldmask'])&(dc>.5)).sum()),
            checkpoint_sha256=sha(bundle['path']),observation_sha256=sha(p),sha256=sha(dest),
            use='continuous optical likelihood for a preregistered detector-agreement control; no label or count-estimate input'))
        print('DeepCenter full field',name,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':run()
