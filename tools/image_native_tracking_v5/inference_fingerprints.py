"""Model-input fingerprints; no study paths, labels or graph cache reads."""
import hashlib,json
import numpy as np

OBSERVATIONS=['nodes','oldmask','features_source','features_target','properties','valid_region',
    'collisions','optical_centroids','confidence','split_owner']

def array_hash(value):
    a=np.ascontiguousarray(value)
    if np.issubdtype(a.dtype,np.floating) and np.isnan(a).any():
        a=a.copy();a[np.isnan(a)]=np.nan
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()

def inputs(c,bank,model_scores,config,joint,calibration,dc):
    result={'observation_'+k:array_hash(c[k]) for k in OBSERVATIONS}
    result.update({k:array_hash(bank[k]) for k in ['pairs','native_logits']})
    family=config['family'];population='P0' if config['pop']=='P0' else 'P1'
    key=population if family in ['N1','N2'] else 'H0' if family=='H0' else f"H1_{config['seed']}" if family in ['H1','H2'] else 'Hctc' if family=='Hctc' else None
    if key is not None:result['model_scores']=array_hash(model_scores[key])
    if config['pop']=='PDC':result['DeepCenter_confidence']=array_hash(dc)
    result['configuration']=hashlib.sha256(json.dumps([config,joint,calibration],sort_keys=True).encode()).hexdigest()
    return result
