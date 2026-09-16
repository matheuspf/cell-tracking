"""Native learner calibration, fitted only on its registered source development clips."""
import torch
from .common import *
from . import native_adapter as native
from .score_models import encode_at_nodes,native_scores
from .banks import cached
from .supervision import supported_pairs
from .joint_calibration import fit as fit_joint
from .calibrate import logistic

def fit(source,stage,seed):
    dest=OUT/'calibration'/f'{source}_{stage}_{seed}.json'
    if dest.exists():return read(dest)
    joint=fit_joint(source);model=native.load(OUT/'models/native'/f'{source}_{stage}_{seed}.pt');frozen=native.load()
    xx=[];yy=[];prior=[];shards=[]
    rows=sorted([r for r in inventory() if r['embryo']==source],key=lambda r:r['dataset'])[:10]
    for r in rows:
        name=r['dataset'];c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask'];nodes=c['nodes'][m];base=graph(name)
        common=cached(frozen,name,source);pairs=common['pairs'];labels,cov=supported_pairs(name,nodes,pairs);known=labels>=0
        if stage=='N2':fs,ft=encode_at_nodes(model,nodes,DATA/'train'/f'{name}.zarr')
        else:fs,ft=c['features_source'][m],c['features_target'][m]
        scores=native_scores(model,nodes,pairs,fs,ft);old=set(map(tuple,base['edges']))
        xx.append(scores[known]);yy.append(labels[known]);prior.append(np.array([tuple(e) in old for e in pairs[known]],np.float32))
        shards.append(dict(dataset=name,supported_rows=int(known.sum())))
    cal=logistic(np.concatenate(xx)[:,None],np.concatenate(yy),fixed=joint['config']['incumbent_edge_bonus']*np.concatenate(prior))
    result=dict(source=source,stage=stage,seed=seed,calibration=cal,source_shards=shards,created=now(),
        checkpoint_sha256=sha(OUT/'models/native'/f'{source}_{stage}_{seed}.pt'))
    write(dest,result);print('source native calibration',source,stage,seed,cal,flush=True);return result

if __name__=='__main__':
    import sys
    source,stage,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
    if (OUT/'calibration'/f'{source}_{stage}_{seed}.json').exists():fit(source,stage,seed)
    else:
        with gpu_aux():fit(source,stage,seed)
