"""Exact upstream window metadata, serialized separately for each embryo."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import DATA,WORK,OUT,WEIGHTS,read,write,save,sha,now,modules,source_hashes,verify_sources


def prepare():
    training,_=modules();split=read(WEIGHTS/'split_manifest.json')
    sources=source_hashes();records=[];started=time.perf_counter()
    # Original split order is preserved, including the non-independent monitor.
    for i,name in enumerate(split['train']):
        path=WORK/'metadata'/name[:4]/f'{name}.pt';receipt=path.with_suffix('.json')
        raw=DATA/f'{name}.geff';image_meta=DATA/f'{name}.zarr/zarr.json'
        input_files={str(p):sha(p) for p in sorted(raw.rglob('*')) if p.is_file()}
        input_files[str(image_meta)]=sha(image_meta)
        if receipt.exists():
            record=read(receipt)
            assert record['native_source']==sources and record['inputs']==input_files and record['sha256']==sha(path)
        else:
            vm,windows=training.load_dataset_windows(DATA/name,window_size=2,downsample=(1,4,4))
            metadata=asdict(vm);metadata['zarr_path']=str(vm.zarr_path)
            save(path,{'video_meta':metadata,'windows':[asdict(w) for w in windows]})
            record={'dataset':name,'embryo':name[:4],'path':str(path),'sha256':sha(path),'windows':len(windows),
                    'max_nodes':max((max(w.node_counts) for w in windows),default=0),
                    'times':[w.t_start for w in windows],'native_source':sources,'inputs':input_files}
            write(receipt,record)
        records.append(record)
        if i%10==0 or i==len(split['train'])-1:
            print(json.dumps({'metadata_clips':i+1,'total_clips':len(split['train']),'seconds':time.perf_counter()-started}),flush=True)
    for embryo in ('44b6','6bba'):
        rows=[r for r in records if r['embryo']==embryo]
        write(WORK/'metadata'/embryo/'index.json',{'created_utc':now(),'embryo':embryo,'clips':rows,'native_source':sources})
    manifest={'created_utc':now(),'split_sha256':sha(WEIGHTS/'split_manifest.json'),'native_source':sources,
              'clip_records':[{k:r[k] for k in ('dataset','embryo','path','sha256','windows','max_nodes')} for r in records],
              'train':split['train'],'monitor':split['test'],'max_nodes_all':max(r['max_nodes'] for r in records),
              'total_windows':sum(r['windows'] for r in records)}
    write(OUT/'data-manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('total_windows','max_nodes_all')}),flush=True)


def video_data(names):
    training,_=modules();result=[]
    for name in names:
        path=WORK/'metadata'/name[:4]/f'{name}.pt';record=read(path.with_suffix('.json'))
        assert record['sha256']==sha(path);verify_sources(record['native_source'])
        payload=torch.load(path,map_location='cpu',weights_only=True)
        vm=dict(payload['video_meta']);vm['zarr_path']=Path(vm['zarr_path'])
        result.append((training.VideoMeta(**vm),[training.FrameWindowData(**w) for w in payload['windows']]))
    return result


def dataset(names,augmentations=None,max_nodes=None):
    training,_=modules()
    return training.FrameWindowDataset(video_data(names),max_nodes=max_nodes,augmentations=augmentations)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare',));p.parse_args();prepare()
