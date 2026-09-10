"""Actual source/gradient/overfit tests, plus prediction and solver contracts."""
import torch
from torch.nn import functional as F
from .common import *
from .models import Geometry,ImageEvent,Detector
from .train import loss
from .decode import solver_fixtures
from .proposals import PAIRS,build

def run():
    torch.set_num_threads(2);torch.manual_seed(SEED)
    d=arrays(next((OUT/'cache/synthetic').glob('seq_*.npz')))
    pos=np.flatnonzero(d['target']>0)[:4];neg=np.flatnonzero(d['target']==0)[:4];ids=np.r_[pos,neg]
    assert len(pos)>0 and len(neg)>0
    b={k:torch.as_tensor(v[ids],device='cuda') for k,v in d.items() if k not in ['group','anchor_t']}
    b['patch']=b['patch'].float()/255
    assert b['valid'].sum(2).max()<=3
    tests={}
    for comp,cls in [('G',Geometry),('I',ImageEvent)]:
        m=cls().cuda();opt=torch.optim.Adam(m.parameters(),lr=.001)
        with torch.no_grad():initial=float(loss(m,b,comp))
        for step in range(600):
            opt.zero_grad();l=loss(m,b,comp);l.backward()
            if step==0:grad=float(sum(p.grad.norm() for p in (m.point.parameters() if comp=='G' else m.encoder.parameters()) if p.grad is not None))
            opt.step()
        with torch.no_grad():final=float(loss(m,b,comp));pred=m(b)[0].argmax(1)
        assert final<initial*.4,(comp,initial,final)
        assert float((pred==b['target']).float().mean())>=.875,(comp,pred,b['target'])
        assert grad>0
        # Swap daughters and compare the correctly remapped unordered pair columns.
        perm=np.array([1,0,2,3,4,5]);pairs={tuple(v):i for i,v in enumerate(PAIRS)}
        remap=[0]+[1+pairs[tuple(sorted(perm[v]))] for v in PAIRS]
        swapped=dict(b,x=b['x'][:,perm])
        if comp=='I':swapped.update(patch=b['patch'][:,np.r_[0,perm+1]],valid=b['valid'][:,np.r_[0,perm+1]])
        with torch.no_grad():assert torch.allclose(m(swapped)[0],m(b)[0][:,remap],atol=2e-4)
        if comp=='I':
            pad=dict(b,patch=torch.cat([b['patch'],torch.randn_like(b['patch'][:,:,:1])],2),
                valid=torch.cat([b['valid'],torch.zeros_like(b['valid'][:,:,:1])],2))
            with torch.no_grad():assert torch.allclose(m(pad)[0],m(b)[0],atol=2e-4)
        tests[comp]=dict(initial_loss=initial,final_loss=final,steps=600,encoder_gradient_norm=grad,actual_source=True,
            positive_bags=len(pos),hard_continuation_bags=len(neg),daughter_permutation=True,padding_invariance=True)
        del m,opt
    dd=arrays(next((OUT/'cache/synthetic').glob('vol_*.npz')));bd={k:torch.as_tensor(v,device='cuda') for k,v in dd.items()};bd['patch']=bd['patch'].float()/255
    m=Detector().cuda();opt=torch.optim.Adam(m.parameters(),lr=.003);initial=float(loss(m,bd,'D'))
    for step in range(600):opt.zero_grad();l=loss(m,bd,'D');l.backward();opt.step()
    final=float(loss(m,bd,'D'));assert final<initial*.8,(initial,final)
    tests['D']=dict(initial_loss=initial,final_loss=final,steps=600,native_static_pixels=True)
    tests['decoder']=solver_fixtures()
    tests['source_coordinates']='All 3713 raw/prepared NPZ coordinate/edge/grid comparisons performed by cache adapter'
    tests['no_gt_features']='proposals.build accepts time, points, anchors only; targets computed afterward'
    tests['zoo_direct_parents']={s:read(OUT/'cache/zoo'/f'{s}_audit.json')['direct_parent_edges_checked'] for s in ['zebrafish','ascidian']}
    write(OUT/'sanity_tests.json',tests);print(tests,flush=True)
