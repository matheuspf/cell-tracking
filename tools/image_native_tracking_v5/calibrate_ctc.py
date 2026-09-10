import torch
from .common import *
from .hoct_adapter import load
from .score_models import hoct_scores
from .supervision import supported_pairs
from .joint_calibration import fit as joint_fit
from .calibrate import logistic

def fit(source):
    dest=OUT/'calibration'/f'{source}_Hctc.json'
    if dest.exists():return
    joint=joint_fit(source);model=load('ctc_v0');xx=[];yy=[];fixed=[]
    rows=sorted([r for r in inventory() if r['embryo']==source],key=lambda r:r['dataset'])[:10]
    for row in rows:
        name=row['dataset'];c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask'];nodes=c['nodes'][m]
        pairs=arrays(OUT/'banks'/source/'P0'/f'{name}.npz')['pairs'];labels,cov=supported_pairs(name,nodes,pairs)
        scores,r=hoct_scores(model,nodes,pairs,c['properties'][m],c['valid_region'][m]);scores=scores['H0'];known=(labels>=0)&np.isfinite(scores)
        old=set(map(tuple,graph(name)['edges']));xx.append(scores[known]);yy.append(labels[known]);fixed.append(np.array([tuple(e) in old for e in pairs[known]],float))
    cal=logistic(np.concatenate(xx)[:,None],np.concatenate(yy),fixed=joint['config']['incumbent_edge_bonus']*np.concatenate(fixed))
    write(dest,dict(source=source,calibration=cal,source_clips=[r['dataset'] for r in rows],created=now(),checkpoint_sha256=sha(OUT/'models/hoct/ctc_v0.pt')))

if __name__=='__main__':
    import sys
    fit(sys.argv[1])
