"""Persistent bbox-local occupancy, image-only ownership, and matched features."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import find_objects
from .contracts import Region, bbox_iou, intersection_voxels, daughter_union_features
from .common import reserve, sha

BOX_FEATURES = ['box_available','bbox_iou','bbox_log_volume_ratio','bbox_extent_change',
                'source_border','target_border']
MASK_FEATURES = ['mask_available','mask_iou','source_coverage','target_coverage',
                 'log_physical_volume_ratio','physical_shape_change',
                 'masked_mean_change','masked_std_change']

@dataclass(frozen=True)
class Observation:
    region: Region
    mean_intensity: float
    std_intensity: float
    border_faces: tuple[bool, ...]
    confidence: float | None = None

    def metadata(self):
        r=self.region
        return dict(clip=r.dataset,frame=r.time,provider=r.provider,instance_id=r.instance,
            grid_id=r.grid_id,bbox_zyx=list(r.bbox),spacing_um=list(r.spacing),
            native_center=r.center().tolist(),inside_center=r.center(inside=True).tolist(),
            volume_um3=r.volume_um3,voxel_count=r.voxel_count,
            shape_eigenvalues_um2=np.linalg.eigvalsh(r.covariance_um2()).tolist(),
            inside_mask_intensity_mean=self.mean_intensity,inside_mask_intensity_std=self.std_intensity,
            border_faces=list(self.border_faces),model_confidence=self.confidence,
            mask_sha256=r.mask_sha256())

def extract(labels, image, *, clip, frame, provider, spacing, grid_id='native-origin-zero'):
    labels=np.asarray(labels);image=np.asarray(image)
    if labels.ndim!=3 or labels.dtype.kind not in 'iu' or (labels<0).any() or image.shape!=labels.shape:
        raise ValueError('Segmenter must return nonnegative integer native-grid ZYX labels')
    if not np.isfinite(image).all(): raise ValueError('Nonfinite image')
    values,inverse=np.unique(labels,return_inverse=True)
    compact=inverse.reshape(labels.shape).astype(np.int32)+1
    compact[labels==0]=0
    slices=find_objects(compact)
    observations=[]
    for i,label in enumerate(values):
        if label==0: continue
        section=slices[i];mask=labels[section]==label
        low=[s.start for s in section];high=[s.stop for s in section]
        region=Region(clip,int(frame),provider,int(label),grid_id,tuple(low+high),mask,tuple(spacing))
        intensity=image[section][mask].astype(np.float64)
        faces=tuple([low[k]==0 for k in range(3)]+[high[k]==labels.shape[k] for k in range(3)])
        observations.append(Observation(region,float(intensity.mean()),float(intensity.std()),faces))
    return observations

def save_frame(path, observations, *, shape, provenance):
    """One compressed occupancy cache per selected frame, shared by all arms."""
    path=Path(path);reserve(path)
    if path.suffix!='.npz':raise ValueError('Mask cache requires an explicit .npz suffix')
    metadata=[];packed=[];offset=0
    keys=[o.region.key for o in observations]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate frame-local instance')
    for obj in observations:
        a=np.packbits(obj.region.mask.ravel());row=obj.metadata()
        row['mask_reference']=dict(file=path.name,byte_start=offset,byte_stop=offset+len(a),
                                   shape=list(obj.region.mask.shape),bitorder='big')
        offset+=len(a);packed.append(a);metadata.append(row)
    receipt=dict(shape=list(shape),objects=metadata,provenance=provenance,
                 label_identity='frame-local, never a track ID')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Selected masks are immutable: '+str(path))
    np.savez_compressed(path,packed=np.concatenate(packed) if packed else np.empty(0,np.uint8),
                        metadata=np.asarray(json.dumps(receipt,allow_nan=False)))
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,objects=len(metadata))

def load_frame(path):
    with np.load(path,allow_pickle=False) as z:
        data=z['packed'];receipt=json.loads(str(z['metadata']))
    result=[]
    for o in receipt['objects']:
        ref=o['mask_reference'];size=int(np.prod(ref['shape']))
        mask=np.unpackbits(data[ref['byte_start']:ref['byte_stop']])[:size].reshape(ref['shape']).astype(bool)
        region=Region(o['clip'],o['frame'],o['provider'],o['instance_id'],o['grid_id'],
                      tuple(o['bbox_zyx']),mask,tuple(o['spacing_um']))
        if region.mask_sha256()!=o['mask_sha256']: raise ValueError('Mask content hash mismatch')
        result.append(Observation(region,o['inside_mask_intensity_mean'],o['inside_mask_intensity_std'],
                                  tuple(o['border_faces']),o['model_confidence']))
    return result,receipt

def attach(nodes, observations):
    """Strict image-only containment; every shared mask remains unowned.

    Return row-index ownership. C0 coordinates, IDs, and candidate scores are
    untouched. Outside/ambiguous detections do not get fabricated masks.
    """
    nodes=np.asarray(nodes);claims={};candidates={}
    for i,n in enumerate(nodes):
        point=n[2:]
        if not np.equal(point,np.rint(point)).all(): raise ValueError('C0 center must be an integer voxel')
        for j,obj in enumerate(observations):
            r=obj.region
            if r.time!=int(n[1]): continue
            local=point.astype(int)-r.bbox[:3]
            if (local>=0).all() and (local<r.mask.shape).all() and r.mask[tuple(local)]:
                candidates.setdefault(i,[]).append(j);claims.setdefault(j,[]).append(i)
    owner={};flags={}
    for i in range(len(nodes)):
        cand=candidates.get(i,[])
        if not cand: flags[i]='missing'
        elif len(cand)>1: flags[i]='overlapping_hypotheses'
        elif len(claims[cand[0]])>1: flags[i]='shared_or_merged'
        else: owner[i]=observations[cand[0]];flags[i]='owned'
    return owner,flags

def pair_features(a, b, arm):
    if arm not in ['P0','B0','M0']: raise ValueError(arm)
    if arm=='P0': return np.empty(0,np.float32)
    length=len(BOX_FEATURES)+(len(MASK_FEATURES) if arm=='M0' else 0)
    if a is None or b is None: return np.zeros(length,np.float32)
    x,y=a.region,b.region
    if y.time!=x.time+1: raise ValueError('Pair is not consecutive')
    ex=np.asarray(x.bbox[3:])-x.bbox[:3];ey=np.asarray(y.bbox[3:])-y.bbox[:3]
    features=[1.,bbox_iou(x,y),np.log(np.prod(ey)/np.prod(ex)),
              np.linalg.norm(np.log(ey/ex)),float(any(a.border_faces)),float(any(b.border_faces))]
    if arm=='M0':
        inter=intersection_voxels(x,y);union=x.voxel_count+y.voxel_count-inter
        shape_x=np.linalg.eigvalsh(x.covariance_um2());shape_y=np.linalg.eigvalsh(y.covariance_um2())
        features += [1.,inter/union,inter/x.voxel_count,inter/y.voxel_count,
            np.log(y.volume_um3/x.volume_um3),np.linalg.norm(np.log((shape_y+.01)/(shape_x+.01))),
            (b.mean_intensity-a.mean_intensity)/(abs(a.mean_intensity)+abs(b.mean_intensity)+1.),
            (b.std_intensity-a.std_intensity)/(a.std_intensity+b.std_intensity+1.)]
    return np.asarray(features,np.float32)

def division_features(parent, a, b, future_a=None, future_b=None):
    """Soft evidence only; absence and nonconservation are not rejection rules."""
    if parent is None or a is None or b is None: return None
    if a.region.time!=parent.region.time+1: raise ValueError('Nonconsecutive division')
    result=daughter_union_features(replace(parent.region,time=a.region.time),a.region,b.region)
    separation=np.linalg.norm((a.region.center()-b.region.center())*a.region.spacing)
    result['daughter_separation_um']=float(separation)
    result['persistent_separation_um']=None
    if future_a is not None and future_b is not None:
        if future_a.region.time!=a.region.time+1 or future_b.region.time!=a.region.time+1:
            raise ValueError('Wrong persistence frame')
        result['persistent_separation_um']=float(min(separation,np.linalg.norm(
            (future_a.region.center()-future_b.region.center())*a.region.spacing)))
    return result

def validate_hierarchy_selection(regions):
    """Selected ancestor/split or overlapping instances cannot both survive."""
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            if a.time==b.time and intersection_voxels(a,b)>0:
                raise ValueError('Conflicting hierarchy hypotheses selected together')
    return True
