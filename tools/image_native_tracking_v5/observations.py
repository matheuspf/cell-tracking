"""Label-blind native full-frame peak discovery and image-supported watershed regions."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from skimage.measure import regionprops

SPACING=np.array([1.625,1.625,1.625])

def peaks(prob,image,old,next_id,t):
    # Exhaustive over the full native field. Plateaus/flat background have no evidence.
    smooth=ndi.gaussian_filter(image,.65)
    background=ndi.gaussian_filter(image,2.5)
    maxima=(prob==ndi.maximum_filter(prob,size=3))&(prob>.15)&(smooth-background>.025)
    coords=np.argwhere(maxima)
    if len(coords)==0:return np.empty((0,5),np.int64),np.empty(0,np.float32),np.empty(0,np.int64)
    order=np.lexsort((coords[:,2],coords[:,1],coords[:,0],-prob[tuple(coords.T)]));coords=coords[order]
    old_grid=old[:,2:]/[1,4,4]
    if len(old):
        dist,owner=cKDTree(old_grid*SPACING).query(coords*SPACING)
    else:dist=np.full(len(coords),np.inf);owner=np.zeros(len(coords),int)
    # A detector peak closer than 2.5um to an incumbent is a duplicate observation.
    keep=dist>=2.5;coords=coords[keep];dist=dist[keep];owner=owner[keep]
    accepted=[]
    for c in coords:
        if not accepted or np.min(np.linalg.norm((np.asarray(accepted)-c)*SPACING,axis=1))>=3.25:accepted.append(c)
    coords=np.asarray(accepted,int).reshape(-1,3)
    if not len(coords):return np.empty((0,5),np.int64),np.empty(0,np.float32),np.empty(0,np.int64)
    strength=prob[tuple(coords.T)]
    node=np.column_stack([np.arange(next_id,next_id+len(coords)),np.full(len(coords),t),coords*[1,4,4]]).astype(np.int64)
    # Origin owner supports grouping alternatives. It is image-derived, never a GT join.
    dist,idx=cKDTree(old_grid*SPACING).query(coords*SPACING) if len(old) else (np.full(len(coords),np.inf),np.zeros(len(coords),int))
    group=np.array([int(old[i,0]) if d<6.5 else -1 for d,i in zip(dist,idx)],np.int64)
    return node,strength.astype(np.float32),group

def regions(image,prob,nodes):
    grid=np.clip((nodes[:,2:]/[1,4,4]).astype(int),0,np.array(image.shape)-1)
    smooth=ndi.gaussian_filter(image,.65);background=ndi.gaussian_filter(image,2.5)
    foreground=(prob>.025)&(smooth>.03)&((smooth-background)>.005)
    markers=np.zeros(image.shape,np.int32);valid=np.zeros(len(nodes),bool);collision=np.zeros(len(nodes),bool)
    for i,c in enumerate(grid):
        if markers[tuple(c)]:collision[i]=True;continue
        if not foreground[tuple(c)]:continue
        markers[tuple(c)]=i+1;valid[i]=True
    labels=watershed(-smooth,markers,mask=foreground)
    # Bound watershed expansion to six native voxels from seeds, while retaining variable morphology.
    if markers.any():
        distance=ndi.distance_transform_edt(markers==0)
        labels[distance>6]=0
    props=np.full((len(nodes),15),np.nan,np.float32)
    optical=np.full((len(nodes),3),np.nan,np.float32)
    # Match official HOCT per-frame intensity normalization and physical shape units.
    normalized=(image-image.min())/(np.quantile(image,.999)-image.min()+1e-7)
    for r in regionprops(labels,intensity_image=normalized,spacing=tuple(SPACING)):
        i=r.label-1
        if r.num_pixels<3:valid[i]=False;continue
        border=1-min(1,float(np.min(np.minimum(grid[i],np.asarray(image.shape)-grid[i])))/5)
        props[i]=[r.equivalent_diameter_area,r.intensity_min,r.intensity_max,r.intensity_mean,r.intensity_std,*r.inertia_tensor.ravel(),border]
        optical[i]=r.centroid
    valid &= np.isfinite(props).all(axis=1)
    return props,valid,collision,optical
