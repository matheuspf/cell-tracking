"""Exact integer-center inference crops, without generic coordinate interpolation.

The fine grid lands at half integers in all axes and the coarse grid at integers.
Thus the reference trilinear samples are exact eight-voxel means or direct reads.
Fractional diagnostic centers retain the general reference implementation.
"""
import numpy as np
from .scenes import Scenes,FloatImages,SHAPE,OFFSETS,crop


def integer_crop(images,t,position):
    p=np.asarray(position)
    if not np.isfinite(p).all() or not np.equal(p,np.rint(p)).all():return crop(images,t,position)
    p=p.astype(np.int64);shape=np.asarray(SHAPE)
    out=np.zeros((7,2,2,*SHAPE),np.uint8)
    for j,dt in enumerate(OFFSETS):
        frame=images.raw(t+dt)
        if frame is None:continue
        dims=np.asarray(frame.shape);lo,hi=images.quantiles[t+dt]
        lower=p-shape//2;start=np.maximum(lower,0);stop=np.minimum(lower+shape+1,dims)
        block=np.zeros(tuple(shape+1),np.float64)
        if (stop>start).all():
            block[tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(start,lower,stop))]=\
                frame[tuple(slice(int(a),int(b)) for a,b in zip(start,stop))]
        # Double accumulation and a final float32 cast reproduce scipy's
        # interpolation output; all eight weights are exactly 1/8.
        value=np.zeros(SHAPE,np.float64)
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):value+=block[z:z+16,y:y+64,x:x+64]*.125
        value=value.astype(np.float32)
        axes=[(lower[k]+np.arange(n)>=0)&(lower[k]+np.arange(n)+1<dims[k]) for k,n in enumerate(SHAPE)]
        valid=axes[0][:,None,None]&axes[1][None,:,None]&axes[2][None,None,:]
        out[j,0,0]=np.rint(255*np.clip((value-lo)/max(1.,hi-lo),0,1)*valid).astype(np.uint8)
        out[j,0,1]=valid.astype(np.uint8)*255
        q=[p[k]+2*np.arange(n)-(n-1) for k,n in enumerate(SHAPE)]
        axes=[(q[k]>=0)&(q[k]<dims[k]) for k in range(3)]
        valid=axes[0][:,None,None]&axes[1][None,:,None]&axes[2][None,None,:]
        value=frame[np.ix_(*(q[k].clip(0,dims[k]-1) for k in range(3)))].astype(np.float32,copy=False)
        out[j,1,0]=np.rint(255*np.clip((value-lo)/max(1.,hi-lo),0,1)*valid).astype(np.uint8)
        out[j,1,1]=valid.astype(np.uint8)*255
    return out


class InferenceScenes(Scenes):
    def __init__(self,rows,persist=False):
        if persist:raise ValueError('Inference crops are streamed, never cached as trainable state')
        super().__init__(rows,persist=False)

    def get(self,name,anchor,position,t):
        if name not in self.images:
            self.images[name]=FloatImages(self.rows[name]['image_path'],self.stats,max_frames=9)
            while len(self.images)>2:self.images.popitem(last=False)
        self.images.move_to_end(name)
        return integer_crop(self.images[name],t,position)
