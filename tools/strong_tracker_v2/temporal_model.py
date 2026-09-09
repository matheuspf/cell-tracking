"""Compact five-frame triplanar encoder; inference has no annotation dependency."""
import numpy as np
import torch
from torch import nn

from .common import adjacency


class Encoder(nn.Module):
    def __init__(self,features):
        super().__init__()
        def branch():
            layers=[];c=20
            for w in [64,96,192,256]:
                layers += [nn.Conv2d(c,w,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,w),nn.SiLU()]
                c=w
            return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.local=branch();self.context=branch()
        self.head=nn.Sequential(nn.Linear(512+features,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,image,features):
        local=image[:,:,8:24,8:24]
        return self.head(torch.cat([self.local(local),self.context(image),features],dim=1)).squeeze(1)


def temporal_indices(nodes,edges):
    _,pred,succ=adjacency(nodes,edges)
    index=np.repeat(np.arange(len(nodes))[:,None],5,axis=1)
    valid=np.zeros((len(nodes),5),np.uint8);valid[:,2]=1
    for i in range(len(nodes)):
        for direction,offsets in [(pred,[1,0]),(succ,[3,4])]:
            cur=i
            for pos in offsets:
                if len(direction[cur])!=1:break
                cur=direction[cur][0];index[i,pos]=cur;valid[i,pos]=1
    return index,valid


def make_batch(patches,index,valid,features,ids,mean,std,device,augment=False):
    # The panels retain their explicit XY/XZ/YZ geometry across all five frames.
    # No flips, panel rotations or rolled borders are used.
    pix=np.ascontiguousarray(patches[index[ids]].reshape(len(ids),15,32,32))
    xx=torch.from_numpy(pix).to(device=device,dtype=torch.float32).div_(255.)
    masks=torch.from_numpy(valid[ids].astype(np.float32)).to(device)[:,:,None,None].expand(-1,-1,32,32)
    if augment:
        gain=.8+.4*torch.rand((len(ids),1,1,1),device=device)
        xx=(xx*gain).clamp_(0,1)
    xx=torch.cat([xx,masks],dim=1)
    ff=torch.from_numpy(np.ascontiguousarray((features[ids]-mean)/std)).to(device)
    return xx,ff
