"""Factorized compact fork policy; upstream association stays frozen."""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(*widths):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if i < len(widths)-2:layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class CompactEncoder(nn.Module):
    """Same frame/temporal family as pipeline_error_training.models.CompactEncoder."""
    def __init__(self):
        super().__init__()
        self.frame=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),
                                nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),
                                nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),
                                nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.temporal=mlp(131,128,128)

    def forward(self,patch,valid):
        n,t=patch.shape[:2]
        present=valid[...,0]>0
        f=patch.new_zeros((n,t,64))
        if present.any():f[present]=self.frame(patch[present])
        w=present.to(f.dtype);offset=torch.arange(t,device=f.device,dtype=f.dtype)-1
        mean=(f*w[...,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
        moment=(f*w[...,None]*offset[None,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
        context=torch.stack([w.sum(1)/3,(w*offset).sum(1)/3,(w*offset.square()).sum(1)/3],-1)
        return self.temporal(torch.cat([mean,moment,context],-1))


class CompactPolicy(nn.Module):
    def __init__(self,parent_width,action_width,identity_width):
        super().__init__()
        self.encoder=CompactEncoder()
        self.occurrence=mlp(128*3+parent_width,128,64,1)
        self.action=mlp(128*3+action_width,128,64,1)
        self.identity=mlp(128*3+identity_width,128,64,1)

    def parent_score(self,parent,daughters,scalars):
        if len(daughters):
            mean,maximum=daughters.mean(0),daughters.max(0).values
        else:mean=maximum=torch.zeros_like(parent)
        return self.occurrence(torch.cat([parent,mean,maximum,scalars],-1)).squeeze(-1)

    def action_scores(self,parent,a,b,scalars):
        return self.action(torch.cat([parent,a+b,(a-b).abs(),scalars],-1)).squeeze(-1)


def supported_losses(occurrence,actions,labels,*,complete=True):
    """Negative groups learn occurrence; unknown rows have zero supervised gradient."""
    positive=labels==1;known=labels>=0
    target=1 if complete and positive.any() else 0 if complete and len(labels)>0 and (labels==0).all() else -1
    occ=F.binary_cross_entropy_with_logits(occurrence,occurrence.new_tensor(float(target))) if target>=0 else occurrence*0
    rank=torch.logsumexp(actions[known],0)-torch.logsumexp(actions[positive],0) if complete and positive.any() else actions.sum()*0
    return occ,rank,dict(occurrence=int(target>=0),ranking=int(complete and positive.any()),target=target)
