"""Explicit image grids and intensity; masks describe observations, not padding."""
import numpy as np
from scipy.ndimage import map_coordinates

def native_refinement(offset,spacing,cap_um=.5):
    """Invert the /3 native-voxel training target, then bound physical motion."""
    delta=np.asarray(offset)*3.
    norm=np.linalg.norm(delta*np.asarray(spacing),axis=-1)
    return delta*np.minimum(1.,cap_um/np.maximum(norm,1e-8))[...,None]

def planes(image,points,spacing=(1,1,1),physical_step=None):
    """Sample original pixels at explicit coordinates; never pool a sequence twice."""
    points=np.asarray(points,np.float32)
    yy,xx=np.meshgrid(np.arange(12)-5.5,np.arange(12)-5.5,indexing='ij')
    step=np.ones(3) if physical_step is None else physical_step/np.asarray(spacing)
    offsets=np.stack([np.stack([np.zeros_like(xx),yy*step[1],xx*step[2]]),
        np.stack([yy*step[0],np.zeros_like(xx),xx*step[2]]),
        np.stack([yy*step[0],xx*step[1],np.zeros_like(xx)])])
    coords=points[:,None,:,None,None]+offsets[None]
    values=map_coordinates(image,coords.transpose(2,0,1,3,4).reshape(3,-1),order=1,mode='nearest',prefilter=False)
    return values.reshape(len(points),3,12,12).astype(np.float32)

def temporal_patches(image,t,points,spacing):
    n=len(t);out=np.zeros((n,3,3,12,12),np.uint8);valid=np.zeros((n,3),np.uint8)
    # Fixed actual-image normalization (labels and count estimates not inputs).
    for f in np.unique(np.clip(np.asarray(t)[:,None]+[-1,0,1],0,image.shape[0]-1)):
        frame=np.asarray(image[int(f)],np.float32)
        low,hi=np.quantile(frame[::2,::4,::4],[.02,.995]);norm=np.clip((frame-low)/max(hi-low,1),0,1)
        for j,offset in enumerate([-1,0,1]):
            ix=np.flatnonzero(np.asarray(t)+offset==f)
            if len(ix):
                out[ix,j]=np.rint(planes(norm,points[ix],spacing,physical_step=1.625)*255).astype(np.uint8);valid[ix,j]=1
    return out,valid

def synthetic_record(raw,labels,kind):
    if kind=='sequence':
        image=raw['volumes'];points=labels['zyx_native']/[1,4,4];spacing=np.asarray(raw['voxel_um_pooled'])
        assert image.shape==(6,64,64,64)
        assert image.dtype==np.uint16 and np.array_equal(spacing,[1.625]*3)
        assert np.allclose(points,labels['zyx_pooled'])
        assert np.allclose(raw['nodes'][:,1:4],labels['zyx_native'])
        assert np.array_equal(raw['edges'],labels['edges'])
        return image,points,spacing
    image=raw['volume'];points=labels['zyx_native'];assert image.shape==(64,256,256)
    assert image.dtype==np.uint16 and np.array_equal(raw['voxel_um'],[1.625,.40625,.40625])
    assert np.allclose(raw['centroids'],points)
    return image[None],points,np.asarray(raw['voxel_um'])
