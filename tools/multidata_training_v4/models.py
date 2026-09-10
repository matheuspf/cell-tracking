"""Geometry and optical towers have separate inputs, losses, and provenance."""
import torch
from torch import nn
from .proposals import PAIRS

def mlp(*widths):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2:layers.append(nn.SiLU())
    return nn.Sequential(*layers)

class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=mlp(16,256,256,128)
        self.context=mlp(128,512,512,128)
        self.pair=mlp(384,512,256,1)
        self.null=mlp(128,128,1);self.link=mlp(256,128,1);self.gate=mlp(128,128,1)
        self.register_buffer('pairs',torch.tensor(PAIRS))
    def encode(self,x):
        mask=x[:,:,11:12];p=self.point(x)*mask
        c=self.context(p.sum(1)/mask.sum(1).clamp_min(1))
        return p,c
    def score_encoded(self,p,c,x):
        mask=x[:,:,11:12]
        u,v=p[:,self.pairs[:,0]],p[:,self.pairs[:,1]]
        s=self.pair(torch.cat([u+v,(u-v).abs(),c[:,None].expand(-1,15,-1)],-1)).squeeze(-1)
        valid=(mask[:,self.pairs[:,0],0]*mask[:,self.pairs[:,1],0])>0
        s=s.masked_fill(~valid,-1e4)
        links=self.link(torch.cat([p,c[:,None].expand(-1,6,-1)],-1)).squeeze(-1)
        return torch.cat([self.null(c),s],1),links,self.gate(c).squeeze(-1)
    def forward(self,batch):
        p,c=self.encode(batch['x']);return self.score_encoded(p,c,batch['x'])

class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
    def forward(self,x):return self.net(x)

class Detector(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=FrameEncoder();self.center=mlp(64,64,1);self.offset=mlp(64,64,3)
    def forward(self,b):
        z=self.encoder(b['patch']);return self.center(z).squeeze(-1),self.offset(z).tanh()

class ImageEvent(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=FrameEncoder()
        self.temporal=mlp(131,128,128);self.pair=mlp(384,256,128,1)
        self.null=mlp(128,128,1);self.gate=mlp(128,128,1);self.link=mlp(256,128,1)
        self.register_buffer('pairs',torch.tensor(PAIRS))
    def encode(self,patch,valid):
        b,n,t=patch.shape[:3]
        f=self.encoder(patch.reshape(-1,3,12,12)).reshape(b,n,t,64)
        valid=valid.float();w=valid.unsqueeze(-1);offset=torch.arange(t,device=patch.device).float()-1
        mean=(f*w).sum(2)/w.sum(2).clamp_min(1)
        moment=(f*w*offset[None,None,:,None]).sum(2)/w.sum(2).clamp_min(1)
        # Masked pooling works for arbitrary padding; occupancy/moment use actual frame offsets.
        context=torch.stack([valid.sum(2)/3,(valid*offset).sum(2)/3,(valid*offset.square()).sum(2)/3],-1)
        return self.temporal(torch.cat([mean,moment,context],-1))
    def score_encoded(self,z,x):
        p=z[:,0];d=z[:,1:]
        u,v=d[:,self.pairs[:,0]],d[:,self.pairs[:,1]]
        s=self.pair(torch.cat([p[:,None].expand(-1,15,-1),u+v,(u-v).abs()],-1)).squeeze(-1)
        m=x[:,:,11]
        s=s.masked_fill((m[:,self.pairs[:,0]]*m[:,self.pairs[:,1]])==0,-1e4)
        links=self.link(torch.cat([p[:,None].expand(-1,6,-1),d],-1)).squeeze(-1)
        return torch.cat([self.null(p),s],1),links,self.gate(p).squeeze(-1)
    def forward(self,batch):
        return self.score_encoded(self.encode(batch['patch'],batch['valid']),batch['x'])
