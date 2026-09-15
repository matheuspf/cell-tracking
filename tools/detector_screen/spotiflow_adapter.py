"""Frozen nuclear-scale Spotiflow screen with explicit pooling-coordinate origin."""
import argparse, fcntl, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from spotiflow.model import Spotiflow

ROOT=Path(__file__).resolve().parents[2]/'work/detector-screen-20260914'
LOCK=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['pilot','assessment','all'],default='all');ap.add_argument('--limit',type=int);args=ap.parse_args()
    torch.set_num_threads(4)
    model=Spotiflow.from_folder(str(ROOT/'assets/spotiflow/synth_3d'),map_location='cpu')
    cfg=dict(checkpoint='synth_3d/best.pt',checkpoint_sha256=hashlib.sha256((ROOT/'assets/spotiflow/synth_3d/best.pt').read_bytes()).hexdigest(),
        normalization='upstream percentiles 1/99.8',xy_pool=4,pool_origin_zyx=[0,1.5,1.5],min_distance=1,subpixel=True,
        proposal_threshold=.05,default_threshold=.3,input_spacing_um=[1.625]*3,
        scope='Fixed nuclear-scale transfer screen on same isotropic sampling as incumbent. Checkpoint trained on much smaller fluorescent spots. No training or label-based scaling.')
    (ROOT/'spotiflow').mkdir(exist_ok=True);(ROOT/'spotiflow/config.json').write_text(json.dumps(cfg,indent=2))
    rows=[r for r in json.loads((ROOT/'panel.json').read_text())['frames'] if args.role=='all' or r['role']==args.role]
    if args.limit:rows=rows[:args.limit]
    for row in rows:
        outputs=[ROOT/'predictions'/m/(row['key']+'.npz') for m in ['spotiflow','spotiflow_loose']]
        if all(p.exists() for p in outputs):continue
        raw=np.load(row['image_path']);z,y,x=raw.shape
        image=raw.astype(np.float32).reshape(z,y//4,4,x//4,4).mean(axis=(2,4))
        with LOCK.open('a') as lock:
            wait=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX);wait=time.monotonic()-wait
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            centers,details=model.predict(image,prob_thresh=.05,n_tiles=(1,1,1),min_distance=1,subpix=True,verbose=False,device='cuda')
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();model.cpu();torch.cuda.empty_cache()
            elapsed=time.monotonic()-start
            fcntl.flock(lock,fcntl.LOCK_UN)
        centers=np.asarray(centers,dtype=np.float64).reshape(-1,3)*[1,4,4]+[0,1.5,1.5]
        scores=np.asarray(details.prob,dtype=np.float64).reshape(-1)
        for dest,threshold in zip(outputs,[.3,.05]):
            keep=scores>=threshold;dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,centers_zyx=centers[keep],scores=scores[keep])
            dest.with_suffix('.json').write_text(json.dumps(dict(key=row['key'],seconds=elapsed,gpu_wait_seconds=wait,
                peak_cuda_bytes=peak,threshold=threshold,candidates=int(keep.sum()),input_shape=list(image.shape),
                execution='One shared neural pass for default and loose-threshold outputs.'),indent=2))
        del details;gc.collect()
        print(row['key'],len(centers),int((scores>=.3).sum()),round(elapsed,3),flush=True)

if __name__=='__main__':main()
