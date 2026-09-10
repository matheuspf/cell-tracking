"""Deterministic balanced exposure cycles with exact visited-example accounting."""
import torch
import numpy as np
from .common import *

class Bank:
    def __init__(self,paths,component,seed=SEED):
        self.paths=paths;self.component=component;parts=[];sid=[];groups=[]
        keys=['patch','target','offset','positive'] if component=='D' else ['x','target','link_y','weight','group']+(['patch','valid'] if component=='I' else [])
        for i,p in enumerate(paths):
            d=arrays(p);n=len(d['target']);parts.append({k:d[k] for k in keys});sid.extend([i]*n)
            groups.extend(d.get('group',np.arange(n)).tolist())
        raw={k:np.concatenate([d[k] for d in parts]) for k in keys}
        self.sid=np.array(sid);self.groups=np.array(groups);self.seen=np.zeros(len(sid),bool)
        self.data={k:torch.as_tensor(v,device='cuda') for k,v in raw.items() if k!='group'}
        rng=np.random.default_rng(seed)
        if component=='D':self.strata=[rng.permutation(len(sid))]
        else:self.strata=[rng.permutation(np.flatnonzero(raw['target']>0)),rng.permutation(np.flatnonzero(raw['target']==0))]
        assert all(len(a)>0 for a in self.strata), (component,paths[:1],list(map(len,self.strata)))
        self.rows=0
    def batch(self,step,n):
        k=len(self.strata);counts=[n//k]*k;counts[-1]+=n-sum(counts)
        ix=np.concatenate([s[(np.arange(c)+step*c)%len(s)] for s,c in zip(self.strata,counts)])
        self.seen[ix]=1;self.rows+=len(ix);idx=torch.as_tensor(ix,device='cuda')
        result={k:v[idx] for k,v in self.data.items()}
        if 'patch' in result:result['patch']=result['patch'].float()/255.
        return result
    def coverage(self):
        seen=np.flatnonzero(self.seen);y=self.data['target'].cpu().numpy()
        return dict(available_examples=len(self.paths),consumed_unique_examples=len(np.unique(self.sid[seen])),
            available_rows=len(self.sid),consumed_unique_groups=len(set(zip(self.sid[seen],self.groups[seen]))),
            consumed_unique_positive_groups=len(set(zip(self.sid[seen[y[seen]>0]],self.groups[seen[y[seen]>0]]))),
            sampled_rows=self.rows)

def paths(domain,component,partition='train'):
    if domain=='synthetic':
        kind='static' if component=='D' else 'sequence'
        return [OUT/'cache/synthetic'/f'{r["sample"]}.npz' for r in records() if r['source']=='synthetic_'+kind and r['partition']==partition]
    if domain in ['zebrafish','ascidian']:
        assert component=='G','Image-free geometry must not enter image models'
        return [OUT/'cache/zoo'/f'{domain}_{partition}.npz']
    if domain.startswith('rendered_zoo_'):
        assert component=='I','Rendered point spread images are an optical-only source'
        return [OUT/'cache'/domain/f'{partition}.npz']
    assert domain in ['44b6','6bba']
    return sorted((OUT/'cache'/('dreal' if component=='D' else 'real')).glob(domain+'_*.npz'))

def records():return [json.loads(s) for s in (OUT/'runtime_dataset_index.jsonl').read_text().splitlines()]
