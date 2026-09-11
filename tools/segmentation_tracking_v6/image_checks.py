"""Actual pilot image reads and orthogonal center overlays; no substitute masks."""
import time
import numpy as np
import zarr
from .common import *

def run():
    from .ultrack_adapter import UltrackAdapter
    from .providers import CellposeAdapter
    start=time.monotonic();rows=[]
    for pilot in read_json(OUT/'preflight.json')['pilots']:
        name=pilot['dataset'];path=DATA/'train'/f'{name}.zarr';meta=image_metadata(path)
        arr=zarr.open_group(path,mode='r')['0'];frame_stats=[]
        for t in pilot['frames']:
            im=np.asarray(arr[t])
            frame_stats.append(dict(frame=t,shape=list(im.shape),dtype=str(im.dtype),minimum=int(im.min()),
                maximum=int(im.max()),mean=float(im.mean()),p99=float(np.percentile(im,99))))
        t=pilot['frames'][len(pilot['frames'])//2];im=np.asarray(arr[t]);graph=c0(name)
        centers=graph['nodes'][graph['nodes'][:,1]==t,2:]
        # The display is a raw image/C0 optical check, never a learned-mask overlay.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        center=np.array(im.shape)//2;vmax=np.percentile(im,99.5)
        axes[0].imshow(im[center[0]],vmin=0,vmax=vmax,cmap='gray')
        axes[1].imshow(im[:,center[1],:],vmin=0,vmax=vmax,cmap='gray',aspect=4)
        axes[2].imshow(im[:,:,center[2]],vmin=0,vmax=vmax,cmap='gray',aspect=4)
        for ax,(normal,u,v,title) in zip(axes,[(0,2,1,'XY'),(1,2,0,'XZ'),(2,1,0,'YZ')]):
            p=centers[np.abs(centers[:,normal]-center[normal])<=1]
            ax.scatter(p[:,u],p[:,v],s=12,facecolors='none',edgecolors='#00c9bb')
            ax.set_title(title+' · raw image + C0 centers');ax.set_xlabel('native voxels')
        fig.suptitle('Source optical check; no learned masks available')
        destination=SCRATCH/'overlays'/f'{name}.png';destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=130);plt.close(fig)
        rows.append(dict(dataset=name,embryo=pilot['embryo'],image=meta,frames=frame_stats,
                         overlay=str(destination),overlay_sha256=sha(destination),learned_mask_overlay=False))
    attempts=[]
    for backend,call in [('Cellpose',lambda:CellposeAdapter('/root/.cellpose/models/cpsam')),
                         ('Ultrack',UltrackAdapter)]:
        try:call()
        except Blocked as exc: attempts.append(dict(backend=backend,status='blocked',reason=str(exc)))
        else: raise RuntimeError('Availability changed; rerun preflight before any recipe selection')
    attempts.insert(0,dict(backend='FOCUS-3D',status='blocked',reason='No infer_volume Python source in /root/FOCUS-3D; weights verified separately'))
    write(OUT/'image_checks.json',dict(real_frames_read=32,clips=rows,attempts=attempts,
        actual_learned_segmenter_frames=0,actual_ultrack_frames=0,seconds=time.monotonic()-start,
        recipe_status='Neither of the two registered physical recipes could be selected',
        interpretation='Raw image and C0 checks only. Merge/duplicate/tile-seam quality, sparse point containment and mask storage/time remain unmeasured.'))

if __name__=='__main__':run()
