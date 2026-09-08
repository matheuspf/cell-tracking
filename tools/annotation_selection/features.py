"""Label-free feature schema and bounded patch extraction."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

GROUPS = {
    'geometry_time': ['t_norm','z_norm','y_norm','x_norm','boundary_um'],
    'appearance_quality': ['response','intensity','local_contrast','density10','nearest_um'],
    'predicted_temporal': ['tracklet_length','age','remaining','incoming_um','outgoing_um','predicted_fork'],
}
GROUPS['all'] = sum(list(GROUPS.values()), [])
FEATURES = GROUPS['all']
VERSION = 'image_geometry_temporal_v1'


def assert_features(columns):
    if set(columns) - set(FEATURES):
        raise ValueError(f'Forbidden feature columns: {set(columns)-set(FEATURES)}')


def frame_features(t, xyz, responses, image, background, shape, scale, q, *, sampling_xyz=None):
    n=len(xyz)
    physical=xyz*np.asarray(scale)
    if n:
        tree=cKDTree(physical)
        dist=tree.query(physical,k=2)[0][:,1] if n>1 else np.full(n,100.)
        density=tree.query_ball_point(physical,10.,return_length=True)-1
    else:
        dist=density=np.empty(0)
    f=np.zeros((n,len(FEATURES)),np.float32)
    f[:,0]=t/max(shape[0]-1,1)
    f[:,1:4]=xyz/(np.asarray(shape[1:])-1)
    f[:,4]=np.minimum(physical,(np.asarray(shape[1:])-1-xyz)*scale).min(axis=1)
    f[:,5]=responses
    if n:
        sample=xyz if sampling_xyz is None else sampling_xyz
        f[:,6]=image[tuple(sample.T)]/q
        ijk=sample//[1,2,2]
        f[:,7]=(image[tuple(sample.T)]-background[tuple(ijk.T)])/q
    f[:,8]=density
    f[:,9]=np.minimum(dist,100.)
    return f


def patches(image, xyz, q):
    # 3 panels, each 32 x 32; 0.8125 um pixels (26 um field), at predicted centers.
    offsets=np.arange(32,dtype=np.float32)-15.5
    u,v=np.meshgrid(offsets,offsets,indexing='ij')
    zero=np.zeros_like(u)
    panels=[np.stack([zero,u*2,v*2]),np.stack([u*.5,zero,v*2]),np.stack([u*.5,v*2,zero])]
    out=np.empty((len(xyz),3,32,32),np.uint8)
    for start in range(0,len(xyz),128):
        points=xyz[start:start+128].T[:,:,None,None]
        for i,panel in enumerate(panels):
            coords=points+panel[:,None]
            vals=map_coordinates(image,coords,order=1,mode='reflect',prefilter=False).astype(np.float32)
            out[start:start+128,i]=np.round(np.clip(vals/q,0,1)*255).astype(np.uint8)
    return out
