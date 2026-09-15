"""Bounded FP32 normalization cache; exact upstream augmentation precedes FP16."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import zarr

from .common import DATA,WORK,OUT,read,write,sha,now,modules


def prepare(source):
    training,_=modules()
    index=read(WORK/'metadata'/source/'index.json');records=[];started=time.perf_counter()
    for i,row in enumerate(index['clips']):
        name=row['dataset'];path=WORK/'frames'/source/f'{name}.npy';receipt=path.with_suffix('.json')
        image_meta=DATA/f'{name}.zarr/zarr.json'
        if receipt.exists():
            record=read(receipt)
            assert record['image_metadata_sha256']==sha(image_meta) and record['sha256']==sha(path)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            meta=torch.load(row['path'],map_location='cpu',weights_only=True)['video_meta']
            shape=tuple(meta['image_shape']);assert shape==(100,64,64,64)
            array=zarr.open_group(str(meta['zarr_path']),mode='r')['0']
            temporary=path.with_suffix('.tmp.npy')
            frames=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=shape)
            for t in range(0,shape[0],4):
                raw=array[t:t+4,::1,::4,::4].astype(np.float32)
                frames[t:t+4]=np.maximum((raw-meta['q_low'])/(meta['q_high']-meta['q_low']+1e-6),0.)
            frames.flush();del frames;temporary.replace(path)
            # Compare cached bytes after the original dataset's FP16 round trip.
            payload=torch.load(row['path'],map_location='cpu',weights_only=True)
            vm=dict(payload['video_meta']);vm['zarr_path']=Path(vm['zarr_path'])
            windows=[training.FrameWindowData(**w) for w in payload['windows']]
            original=training.FrameWindowDataset([(training.VideoMeta(**vm),windows)])
            cached=np.load(path,mmap_mode='r',allow_pickle=False);tested=[]
            for k in sorted({0,max(0,len(original)-1)}):
                if not len(original):continue
                item=original[k];t=int(item['t_start'])
                torch.testing.assert_close(item['imgs'],torch.from_numpy(np.array(cached[t:t+2])).half(),rtol=0,atol=0)
                tested.append(t)
            record={'dataset':name,'embryo':source,'path':str(path),'shape':list(shape),'dtype':'float32',
                    'sha256':sha(path),'image_metadata_sha256':sha(image_meta),'metadata_sha256':row['sha256'],
                    'normalization':'Original q0.001/q0.999 and clamp minimum0; augmentation occurs in FP32 before original FP16 input round trip.',
                    'exact_upstream_window_checks':tested,'created_utc':now()}
            write(receipt,record)
        records.append(record)
        if i%10==0 or i==len(index['clips'])-1:
            print(json.dumps({'source':source,'cached_clips':i+1,'total_clips':len(index['clips']),'seconds':time.perf_counter()-started}),flush=True)
    write(WORK/'frames'/source/'index.json',{'created_utc':now(),'source':source,'records':records})
    write(OUT/f'frames-{source}.json',{'source':source,'clips':len(records),'bytes':sum(Path(r['path']).stat().st_size for r in records),
         'cache_manifest_sha256':sha(WORK/'frames'/source/'index.json'),'exact_upstream_windows_verified':sum(len(r['exact_upstream_window_checks']) for r in records)})


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self,original,source,seed,epoch):
        self.original=original;self.seed=seed;self.epoch=epoch;self.cache={}
        self.augmentations=modules()[0].DEFAULT_AUGMENTATIONS
        self.source=source
    def __len__(self):return len(self.original)
    def __getitem__(self,idx):
        meta,vm=self.original._data[idx];name=vm.zarr_path.stem
        assert self.source=='all' or name.startswith(self.source+'_')
        if name not in self.cache:
            path=WORK/'frames'/name[:4]/f'{name}.npy'
            self.cache[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        t=meta['t_start'];imgs=torch.from_numpy(np.array(self.cache[name][t:t+meta['n_frames']]))
        coords=meta['coords'];masks=meta['masks']
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,idx]))
        for aug in self.augmentations:imgs,coords,masks=aug(imgs,coords,masks,rng=rng)
        return {**meta,'coords':coords,'masks':masks,'imgs':imgs.half(),
                'image_shape':torch.tensor(vm.image_shape,dtype=torch.long),
                'voxel_size':torch.tensor(vm.voxel_size,dtype=torch.float32),
                'downsample':torch.tensor(vm.downsample,dtype=torch.float32)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('44b6','6bba'),required=True);a=p.parse_args();prepare(a.source)
