"""A scene is encoded once per optimizer update, shared across complete edits."""
import torch
from torch import nn
from torch.nn import functional as F

from .features import FEATURE_DIM


def sample_maps(maps,grid):
    """Trilinear sampling with deterministic indexed backward on CUDA.

    Coordinates have no gradients. This avoids grid_sample's nondeterministic
    CUDA 3D backward while retaining gradients into every shared feature map.
    """
    n,c,d,h,w=maps.shape
    q=grid.reshape(n,-1,3)
    xyz=(q+1)*q.new_tensor([w-1,h-1,d-1])/2
    lower=xyz.floor().long();fraction=xyz-lower
    out=maps.new_zeros((n,c,q.shape[1]),dtype=torch.float32)
    flat=maps.flatten(2).float()
    for z in (0,1):
        for y in (0,1):
            for x in (0,1):
                offset=lower.new_tensor([x,y,z]);index=lower+offset
                good=((index>=0)&(index<index.new_tensor([w,h,d]))).all(-1)
                ix=index[...,0].clamp(0,w-1)+w*index[...,1].clamp(0,h-1)+w*h*index[...,2].clamp(0,d-1)
                weight=torch.where(offset.bool(),fraction,1-fraction).prod(-1)*good
                values=torch.gather(flat,2,ix[:,None].expand(-1,c,-1))
                out=out+values*weight[:,None]
    return out


def mlp(*sizes):
    layers=[]
    for i,(a,b) in enumerate(zip(sizes,sizes[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(sizes)-2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def query_grids(q):
    """Separate raw-mask and CNN-lattice coordinates, in normalized XYZ order.

    Same-padded odd convolutions place feature centers at raw indices 4*j in Z
    and 8*j in Y/X. The crop center is raw index (7.5,31.5,31.5), so its feature
    coordinate is (1.875,3.9375,3.9375), rather than the feature-array midpoint.
    """
    center=q.new_tensor([7.5,31.5,31.5])
    stride=q.new_tensor([4.,8.,8.]);last=q.new_tensor([3.,7.,7.])
    raw=torch.stack([(q/center/f).flip(-1) for f in (1.,2.)])
    feature=torch.stack([(2*((q/f+center)/stride)/last-1).flip(-1) for f in (1.,2.)])
    return raw,feature


class SceneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(
            nn.Conv3d(2,16,3,stride=(1,2,2),padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv3d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU())
        self.query=nn.Linear(128+8,128)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',
                                         batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.LayerNorm(128)
        self.forward_calls=0

    def maps(self,scene):
        self.forward_calls+=1
        b,t,s,c,z,y,x=scene.shape
        return self.spatial(scene.reshape(b*t*s,c,z,y,x)).reshape(b,t,s,64,4,8,8)

    def tokens(self,maps,scene,queries,times):
        """Queries are voxel offsets from the stationary parent scene center."""
        result=[]
        for k,(q,qt) in enumerate(zip(queries,times)):
            count=len(q)
            raw_grid,feature_grid=query_grids(q)
            raw_grid=raw_grid[None].expand(7,-1,-1,-1).reshape(14,1,count,1,3)
            feature_grid=feature_grid[None].expand(7,-1,-1,-1).reshape(14,1,count,1,3)
            # Sampling retains spatial maps; outside coordinates are zero, never
            # clamped onto a different cell. Validity is queried independently.
            z=sample_maps(maps[k].reshape(14,64,4,8,8),feature_grid.float())
            valid=F.grid_sample(scene[k,:,:,1:2].reshape(14,1,16,64,64).float(),raw_grid.float(),
                                align_corners=True,padding_mode='zeros')[:,0,0,:,0]
            z=z.reshape(7,2,64,count).permute(3,0,1,2)
            valid=valid.reshape(7,2,count).permute(2,0,1)
            z=(z*valid[...,None]).flatten(-2)
            rel=(q*q.new_tensor([1.625,.40625,.40625])/26.)[:,None].expand(-1,7,-1)
            dt=torch.arange(-3,4,device=q.device,dtype=q.dtype)[None,:,None].expand(count,-1,-1)/3
            own=qt[:,None,None].expand(-1,7,-1)/3
            present=(valid.max(-1,keepdim=True).values>0).float()
            features=torch.cat((z,valid,rel,dt,own,present),-1)
            z=self.temporal(self.query(features))
            weights=present
            result.append(self.output((z*weights).sum(1)/weights.sum(1).clamp_min(1)))
        return result


class ActionModel(nn.Module):
    def __init__(self,image=True):
        super().__init__()
        self.image=image
        self.encoder=SceneEncoder() if image else None
        self.action=mlp(FEATURE_DIM+(128*6 if image else 0),128,64)
        self.score=nn.Linear(64,1)
        self.risk=nn.Linear(64,1)
        self.identity=nn.Linear(64,1)
        self.keep=mlp(128 if image else 13,64,1)
        for head in (self.score,self.risk,self.identity,self.keep[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def scores(self,batch,z=None):
        features=batch['features']
        if self.image:
            p,a,b=batch['event_index'].unbind(-1)
            event=torch.cat((z[p],z[a]+z[b],(z[a]-z[b]).abs()),-1)
            edits=torch.einsum('acq,qh->ach',batch['incidence'],z).flatten(1)
            features=torch.cat((features,event,edits),-1)
            context=z[batch['event_index'][0,0]]
        else:
            # Frozen image-derived native parent evidence, same definition as
            # the complete action descriptors, not an old model prediction.
            context=features[0,116:129]
        h=self.action(features)
        null=self.keep(context).squeeze(-1)
        gain=self.score(h).squeeze(-1)-null
        gain=torch.where(batch['keep'],torch.zeros_like(gain),gain)
        return dict(gain=gain.float(),risk=self.risk(h).squeeze(-1).float(),
                    identity=self.identity(h).squeeze(-1).float())

    def forward(self,batches,scenes=None):
        if not self.image:
            lengths=[len(b['features']) for b in batches]
            h=self.action(torch.cat([b['features'] for b in batches]))
            null=self.keep(torch.stack([b['features'][0,116:129] for b in batches])).squeeze(-1)
            logits=self.score(h).squeeze(-1).split(lengths)
            risks=self.risk(h).squeeze(-1).split(lengths)
            identities=self.identity(h).squeeze(-1).split(lengths)
            return [dict(gain=torch.where(b['keep'],torch.zeros_like(s),s-n).float(),
                         risk=r.float(),identity=i.float())
                    for b,s,n,r,i in zip(batches,logits,null,risks,identities)]
        if self.image:
            maps=self.encoder.maps(scenes)
            zs=self.encoder.tokens(maps,scenes,[b['query_voxels'] for b in batches],
                                   [b['query_time'] for b in batches])
        else:
            zs=[None]*len(batches)
        return [self.scores(b,z) for b,z in zip(batches,zs)]


def loss(output,batch,risk_weight=1.):
    gain=output['gain'].float()
    known=batch['supported'].bool()
    keep=batch['keep'].bool()
    zero=gain.sum()*0+output['risk'].sum()*0+output['identity'].sum()*0
    if not known.any():
        return zero,dict(event=zero,risk=zero,identity=zero,total=zero)
    good=known & (batch['utility']>1e-12)
    candidates=known | keep
    if not good.any():
        good=keep
    event=torch.logsumexp(gain[candidates],0)-torch.logsumexp(gain[good],0)
    # Known zero utility prefers keep. Unknown alternatives are absent from
    # both softmax and risk loss; identity has its own independent support mask.
    risk=F.binary_cross_entropy_with_logits(output['risk'][known],(batch['utility'][known]>1e-12).float())
    identity_mask=(batch['identity']>=0)&~keep
    identity=F.binary_cross_entropy_with_logits(output['identity'][identity_mask],
                batch['identity'][identity_mask].float()) if identity_mask.any() else zero
    total=event+risk_weight*risk+.25*identity
    return total,dict(event=event,risk=risk,identity=identity,total=total)
