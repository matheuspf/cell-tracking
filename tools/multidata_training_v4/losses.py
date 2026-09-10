import torch
from torch.nn import functional as F

def event_loss(logits,links,gate,batch):
    y=batch['target'];mask=y>=0
    # One observation bag, one loss unit, regardless of alternatives.
    weight=batch['weight']
    ce=F.cross_entropy(logits,y.clamp_min(0),reduction='none')
    loss=(ce*mask*weight).sum()/(mask*weight).sum().clamp_min(1)
    ly=batch['link_y'];lm=ly>=0
    ll=F.binary_cross_entropy_with_logits(links,ly.clamp_min(0).float(),reduction='none')
    loss+=.35*((ll*lm).sum(1)/lm.sum(1).clamp_min(1)*weight).mean()
    gy=(y>0).float();gl=F.binary_cross_entropy_with_logits(gate,gy,reduction='none')
    loss+=.35*(gl*mask*weight).sum()/(mask*weight).sum().clamp_min(1)
    return loss
