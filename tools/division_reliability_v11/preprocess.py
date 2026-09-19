"""Exact, nonlearned source-fit frame cache with explicit raw parents."""
from pathlib import Path
import sys
import time
from .common import DATA, WORK, REPO, Blocked, read, write, sha, now


def run(source):
    split=read(WORK/'source_partitions.json')[source]
    folder=WORK/'preprocessed'/source;folder.mkdir(parents=True,exist_ok=True)
    parents=[DATA/'train'/f'{n}{s}' for n in split['fit'] for s in ('.zarr','.geff')]
    specification=dict(source=source,clips=split['fit'],data_code=sha(Path(__file__).with_name('data.py')),
                       construction_code=sha(Path(__file__)),learned_statistics=False,
                       split_sha256=sha(WORK/'source_partitions.json'))
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=parents,outputs=[folder],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
                  *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import zarr
    import shutil
    from .data import normalize,background,labels
    started=time.monotonic();receipts=[]
    for k,name in enumerate(split['fit']):
        dest=folder/name;dest.mkdir(exist_ok=True)
        if (dest/'receipt.json').exists():
            receipt=read(dest/'receipt.json')
            if receipt['specification']!=specification:raise Blocked('Preprocessing cache specification changed')
            receipts.append(receipt);continue
        if shutil.disk_usage(folder).free<11*2**30:raise Blocked('Insufficient durable cache space')
        image=zarr.open_array(str(DATA/'train'/f'{name}.zarr/0'),mode='r')
        x=np.lib.format.open_memmap(dest/'images.npy',mode='w+',dtype=np.float32,shape=(100,64,64,64))
        b=np.lib.format.open_memmap(dest/'background.npy',mode='w+',dtype=np.bool_,shape=x.shape)
        quantiles=[]
        for t in range(100):
            raw=np.asarray(image[t]);coarse,q=normalize(raw)
            x[t]=coarse;b[t]=background(raw,coarse);quantiles.append(q)
        x.flush();b.flush();del x,b
        nodes,edges=labels(DATA/'train'/f'{name}.geff')
        np.save(dest/'nodes.npy',nodes);np.save(dest/'edges.npy',edges)
        parity=[]
        x=np.load(dest/'images.npy',mmap_mode='r');b=np.load(dest/'background.npy',mmap_mode='r')
        for t in (0,49,99):
            raw=np.asarray(image[t]);coarse,q=normalize(raw)
            if not np.array_equal(x[t],coarse) or not np.array_equal(b[t],background(raw,coarse)):
                raise Blocked('Frame cache differs from raw preprocessing')
            parity.append(t)
        receipt=dict(status='passed',dataset=name,specification=specification,parity_frames=parity,
                     files={p.name:sha(p) for p in dest.glob('*.npy')},frame_quantiles=quantiles,
                     source_annotation_files={str(p.relative_to(DATA)):sha(p) for p in (DATA/'train'/f'{name}.geff').rglob('*') if p.is_file()})
        write(dest/'receipt.json',receipt,immutable=True);receipts.append(receipt)
        print(f'{source} deterministic cache {k+1}/{len(split["fit"])}',flush=True)
    result=dict(status='passed',specification=specification,guard=guard,clips=len(receipts),
                completed_utc=now(),wall_seconds=time.monotonic()-started,
                clip_receipt_hashes={n:sha(folder/n/'receipt.json') for n in split['fit']})
    write(folder/'manifest.json',result,immutable=True)
    return result


class SourcePairs:
    def __init__(self,folder,clips):
        import numpy as np
        self.np=np;self.cache={}
        manifest=read(folder/'manifest.json')
        if manifest['status']!='passed' or manifest['specification']['clips']!=clips:
            raise Blocked('Wrong preprocessed source-fit population')
        if manifest['specification']['data_code']!=sha(Path(__file__).with_name('data.py')):
            raise Blocked('Source preprocessing code changed')
        for n in clips:
            if sha(folder/n/'receipt.json')!=manifest['clip_receipt_hashes'][n]:raise Blocked('Cache parent receipt changed')
            receipt=read(folder/n/'receipt.json')
            for name,digest in receipt['files'].items():
                if sha(folder/n/name)!=digest:raise Blocked('Source preprocessing array changed: '+n+'/'+name)
            self.cache[n]={k:np.load(folder/n/(k+'.npy'),mmap_mode='r') for k in ('images','background','nodes','edges')}

    def pair(self,name,t):
        from .data import centers_target
        np=self.np;c=self.cache[name];query=[];targets=[];positives=[];backgrounds=[]
        for j in (t,t+1):
            points=c['nodes'][c['nodes'][:,1]==j];y,known=centers_target((64,64,64),points[:,2:].astype(np.float32))
            targets.append(y);positives.append(known);backgrounds.append(c['background'][j]&~known);query.append(points)
        return dict(images=c['images'][t:t+2,None].copy(),targets=np.stack(targets),positives=np.stack(positives),
                    backgrounds=np.stack(backgrounds),query=query,edges=c['edges'],dataset=name,time=t)
