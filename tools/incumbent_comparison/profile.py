"""Resource/gradient preflight of the real upstream training step from scratch."""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from .common import WORK,OUT,GPU_LOCK,read,write,sha,now,modules,source_guard,source_hashes
from .data import dataset


def run(source):
    training,_=modules();source_guard(source)
    from tools.detector_screen.cellpose_adapter import gpu_lock
    manifest=read(WORK/'metadata'/source/'index.json')
    names=[r['dataset'] for r in manifest['clips']]
    ds=dataset(names)
    # Deterministic coverage across the source; no target data selects batches.
    indices=np.linspace(0,len(ds)-1,16,dtype=int).tolist()
    loader=DataLoader(Subset(ds,indices),batch_size=8,shuffle=False,num_workers=2)
    torch.manual_seed(314159);torch.cuda.manual_seed_all(314159)
    model=training.UNetNodeTransformer(training.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),32,32)
    before={name:next(module.parameters()).detach().clone() for name,module in [('encoder',model.unet),('detector',model.detect_head),('association',model.transformer)]}
    started=time.perf_counter()
    result={'created_utc':now(),'source':source,'batch_size':8,'updates':4,'seed':314159,'native_source':source_hashes(),'script_sha256':sha(__file__),'retained_training':False}
    with gpu_lock(GPU_LOCK) as queue:
        torch.cuda.reset_peak_memory_stats();model.to('cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        try:
            edge,det=training.train_epoch(model,loader,optimizer,torch.device('cuda'),det_loss_weight=1.,det_neg_weight=.01,max_iters=4,pool_kernel_um=5.)
            torch.cuda.synchronize()
            changed={name:float((next(module.parameters()).detach().cpu()-before[name]).norm()) for name,module in [('encoder',model.unet),('detector',model.detect_head),('association',model.transformer)]}
            assert changed['encoder']>0 and changed['detector']>0
            result.update(status='passed',mean_edge_loss=edge,mean_detection_loss=det,parameter_changes_l2=changed)
        except torch.cuda.OutOfMemoryError as error:
            result.update(status='oom',error=str(error))
        finally:
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          seconds_excluding_queue=time.perf_counter()-started-queue,gpu_queue_seconds=queue,
                          source_windows=len(ds),batches_per_source_epoch=(len(ds)+7)//8)
            del optimizer;model.to('cpu');torch.cuda.empty_cache()
    write(OUT/f'profile-{source}.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('44b6','6bba'),required=True);a=p.parse_args();run(a.source)
