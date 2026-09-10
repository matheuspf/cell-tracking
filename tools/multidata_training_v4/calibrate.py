"""Pre-target scaling; external sanity diagnostics are not biological calibration."""
import torch
import pandas as pd
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize_scalar
from .common import *
from .sampling import paths
from .models import Geometry,ImageEvent

def load_model(name):
    cls=Geometry if name.startswith('G_') else ImageEvent
    model=cls().cuda().eval();model.load_state_dict(torch.load(OUT/'models'/f'{name}.pt',map_location='cuda',weights_only=False)['model']);return model

def sample(domain,component,part='validation',limit=2048):
    ps=paths(domain,component,part);data=[]
    # Whole examples stay in their prepared partition. Uniform coverage across examples.
    for p in ps:
        d=arrays(p);count=max(1,limit//len(ps));ix=np.linspace(0,len(d['target'])-1,min(count,len(d['target']))).astype(int)
        data.append({k:d[k][ix] for k in ['x','target']+(['patch','valid'] if component=='I' else [])})
    return {k:np.concatenate([d[k] for d in data]) for k in data[0]}

def scores(model,d):
    out=[]
    with torch.no_grad():
        for start in range(0,len(d['target']),32):
            b={k:torch.as_tensor(v[start:start+32],device='cuda') for k,v in d.items() if k!='target'}
            if 'patch' in b:b['patch']=b['patch'].float()/255
            out.append(model(b)[0].cpu().numpy())
    return np.concatenate(out)

def diagnostics(logits,y,temp):
    m=y>=0;l=logits[m]/temp;y=y[m];p=softmax(l,axis=1);pred=p.argmax(1)
    nll=float(np.mean(logsumexp(l,axis=1)-l[np.arange(len(y)),y]));rows=[]
    confidence=1-p[:,0]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        ix=(confidence>=lo)&(confidence<=hi if hi==1 else confidence<hi)
        if ix.any():rows.append(dict(low=float(lo),high=float(hi),n=int(ix.sum()),predicted=float(confidence[ix].mean()),observed=float((y[ix]>0).mean())))
    return dict(rows=len(y),nll=nll,accuracy=float((pred==y).mean()),positive=int((y>0).sum()),
        positive_pair_correct=int(((pred==y)&(y>0)).sum()),false_pair=int(((pred!=y)&(pred>0)).sum()),bins=rows)

def run():
    existing=OUT/'calibration.json'
    if existing.exists():
        old=read(existing)
        assert old['procedure_code_sha256']==sha(Path(__file__))
        assert old['dataset_index_sha256']==sha(OUT/'runtime_dataset_index.jsonl')
        for name,record in old['models'].items():assert record['model_sha256']==sha(OUT/'models'/f'{name}.pt')
        print('reuse frozen calibration',flush=True);return
    from .provenance import model_manifest,equivalent_histories
    model_manifest();equivalent_histories()
    torch.set_num_threads(2);result={};diagnostic=[]
    for source in ['44b6','6bba']:
        for component in ['G','I']:
            # The same external generator diagnostic is used for all data arms.
            val=sample('synthetic',component);test=sample('synthetic',component,'test')
            for arm in ['C1','C2','C3','C4','C5','C6','C1short']:
                name=f'{component}_{arm}_{source}';model=load_model(name);l=scores(model,val);y=val['target'];m=y>=0
                def objective(t):return float((logsumexp(l[m]/t,axis=1)-l[m,np.maximum(y[m],0)]/t).mean())
                temp=float(minimize_scalar(objective,bounds=(.5,8.),method='bounded').x)
                result[name]=dict(temperature=temp,calibration=diagnostics(l,y,temp),generator_test=diagnostics(scores(model,test),test['target'],temp),
                    model_sha256=sha(OUT/'models'/f'{name}.pt'))
                if component=='G':
                    zoo=sample('zebrafish','G','test');result[name]['zoo_weak_test']=diagnostics(scores(model,zoo),zoo['target'],temp)
                diagnostic.append(dict(model=name,temperature=temp,**{k:v for k,v in result[name]['generator_test'].items() if k!='bins'}))
                print('calibrated',name,round(temp,3),flush=True);del model
    write(OUT/'calibration.json',dict(created=now(),models=result,primary_margin=4.,descriptive_margins=[2.,6.],gate_probability=.5,
        procedure_code_sha256=sha(Path(__file__)),dataset_index_sha256=sha(OUT/'runtime_dataset_index.jsonl'),
        independent_source_inner_groups=False,real_biological_posterior=False,
        caveat='Temperature fitted on unbalanced observed generator validation bags; not biological prevalence. All controls share this diagnostic/calibration access. Real-only means no external neural parameter updates. Primary margin fixed at 4.0.'))
    pd.DataFrame(diagnostic).to_csv(OUT/'transfer_diagnostics.csv',index=False)
    status('W450',state='all source directions calibrated before comparative outcomes')
