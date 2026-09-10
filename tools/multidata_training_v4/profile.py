"""GT-free runtime profile on a fixed image, without comparative graph scoring."""
from .infer import model,geometry,image_scores,refine
from .common import *
from .adapters import temporal_patches
from .proposals import build
import time
import torch
import zarr

def run():
    torch.set_num_threads(2);name='44b6_87bba6c4';row=next(r for r in inputs() if r['dataset']==name)
    g=arrays(V3/'selected_predictions'/f'{name}.npz');nodes=g['nodes'];timings={}
    start=time.monotonic();image=zarr.open_group(row['image_path'],mode='r')['0'][:];timings['image_read']=time.monotonic()-start
    start=time.monotonic();a,c,x=build(nodes[:,1],nodes[:,2:]);timings['geometry_proposals']=time.monotonic()-start
    m=model('G_synthetic');start=time.monotonic();gl,ll,gate=geometry(m,x);timings['one_G_model']=time.monotonic()-start;del m
    selected=np.flatnonzero(gate>.5);used=np.unique(np.r_[a[selected],c[selected].reshape(-1)]);used=used[used>=0]
    start=time.monotonic();pp,vv=temporal_patches(image,nodes[used,1],nodes[used,2:],row['physical_scale']);timings['image_patches']=time.monotonic()-start
    rev=np.full(len(nodes),-1,int);rev[used]=np.arange(len(used));m=model('I_synthetic')
    start=time.monotonic();image_scores(m,x,a,c,selected,pp,vv,rev);timings['one_I_model']=time.monotonic()-start;del m
    m=model('D_synthetic');start=time.monotonic();_,det=refine(nodes,image,row['physical_scale'],m);timings['one_D_refinement']=time.monotonic()-start
    write(OUT/'inference_runtime_profile.json',dict(created=now(),dataset=name,nodes=len(nodes),gated_parents=len(selected),image_centers=len(used),
        seconds=timings,detector=det,labels_read=False,target_scores_computed=False,concurrent_training=True,
        model_hashes={n:sha(OUT/'models'/f'{n}.pt') for n in ['G_synthetic','I_synthetic','D_synthetic']}))
    print(timings,flush=True)
