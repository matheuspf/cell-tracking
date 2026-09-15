"""Exact compact triplanes, without allocating unused seven-frame 3D crops."""
from collections import OrderedDict

import numpy as np
from scipy.ndimage import zoom

from .crops import SHAPE, prediction_tracklet
from .frame_crops import quantized_frames


def planes(volume, position):
    axes = [int(c)+np.arange(n)-n//2 for c,n in zip(position,SHAPE)]
    ok = [(a>=0)&(a<s) for a,s in zip(axes,volume.shape)]
    aa = [np.clip(a,0,s-1) for a,s in zip(axes,volume.shape)]
    middle = [n//2 for n in SHAPE]
    return [
        volume[aa[0][middle[0]],aa[1][:,None],aa[2][None,:]]*
            (ok[0][middle[0]]&ok[1][:,None]&ok[2][None,:]),
        volume[aa[0][:,None],aa[1][middle[1]],aa[2][None,:]]*
            (ok[0][:,None]&ok[1][middle[1]]&ok[2][None,:]),
        volume[aa[0][:,None],aa[1][None,:],aa[2][middle[2]]]*
            (ok[0][:,None]&ok[1][None,:]&ok[2][middle[2]]),
    ]


def sample(images,nodes,pred,succ,indices):
    indices = list(map(int,indices))
    if not hasattr(images,'_compact_quantized_frames'):
        images._compact_quantized_frames = OrderedDict()
    cache = images._compact_quantized_frames
    patch = np.zeros((len(indices),3,3,12,12),np.uint8)
    valid = np.zeros((len(indices),3,4),np.float32)
    grouped = {}
    for k,i in enumerate(indices):
        positions,tracked = prediction_tracklet(nodes,pred,succ,i)
        for j,dt in enumerate([-1,0,1]):
            t = int(nodes[i,1])+dt
            if 0<=t<images.shape[0]:
                grouped.setdefault(t,[]).append((k,j,positions[dt],tracked[dt]))
    for t,entries in sorted(grouped.items()):
        if t not in cache:
            raw = images.raw(t)
            cache[t] = quantized_frames(raw,*images.quantiles[t])[0]
            while len(cache)>images.max_frames:
                cache.popitem(last=False)
        cache.move_to_end(t)
        for i,j,position,tracked in entries:
            if not np.equal(position,np.rint(position)).all():
                raise ValueError('Exact compact sampler requires integer tracklet centers')
            patch[i,j] = np.stack([zoom(p,(12/p.shape[0],12/p.shape[1]),order=1)
                                    for p in planes(cache[t],position)])
            valid[i,j,:2] = [1.,float(tracked)]
            for k,scale in enumerate([1,2]):
                axes = [c+scale*(np.arange(n)-(n-1)/2) for c,n in zip(position,SHAPE)]
                counts = [((a>=0)&(a<=s-1)).sum() for a,s in zip(axes,images.shape[1:])]
                valid[i,j,k+2] = np.prod(counts)/np.prod(SHAPE)
    return patch,valid
