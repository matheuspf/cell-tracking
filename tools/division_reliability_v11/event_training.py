"""Differentiable 32-group mixed objective and auditable source sampler."""
from collections import OrderedDict,Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import read,sha,Blocked
from .features import normalize
from .models import supported_losses


class Groups:
    def __init__(self,root,clips,normalizer=None,*,max_cache_bytes=8*2**30):
        self.root=root;self.clips=clips;self.normalizer=normalizer;self.loaded=OrderedDict()
        self.max_cache_bytes=max_cache_bytes;self.cache_bytes=0
        self.positive={};self.negative={};self.identity={};self.receipts={};self.visits=Counter()
        for clip in clips:
            r=read(root/clip/'receipt.json');self.receipts[clip]=r
            if sha(root/clip/'training.npz')!=r['training_sha256']:raise Blocked('Event fit bank bytes changed')
            # GEFF IDs persist across overlapping spatial clips. Each biological
            # event gets one unit, then one compatible (clip, anchor) is sampled.
            for event,anchors in r['positive_events'].items():
                self.positive.setdefault(event,[]).extend((clip,a) for a in anchors)
            with np.load(root/clip/'training.npz') as f:
                offset=f['offset'];risk=f['risks'];ip=f['identity_pairs']
                self.negative[clip]=[i for i,(a,b) in enumerate(zip(offset[:-1],offset[1:])) if b>a and (risk[a:b]==0).all()]
                self.identity[clip]=sorted(set(ip[:,1].tolist()))
        self.negative={k:v for k,v in self.negative.items() if v}
        self.identity={k:v for k,v in self.identity.items() if v}
        if not self.positive:raise Blocked('No reachable positive source event pool')
        if not self.negative:raise Blocked('No completely supported negative source group pool')
        if not self.identity:raise Blocked('No supported incoming identity pool')

    def load(self,clip):
        if clip not in self.loaded:
            with np.load(self.root/clip/'training.npz') as f:self.loaded[clip]={k:f[k] for k in f.files}
            self.cache_bytes+=sum(a.nbytes for a in self.loaded[clip].values())
            # Only immutable source observation arrays are cached. Every
            # trainable encoder activation is recomputed with gradients.
            while len(self.loaded)>1 and self.cache_bytes>self.max_cache_bytes:
                _,old=self.loaded.popitem(last=False);self.cache_bytes-=sum(a.nbytes for a in old.values())
        self.loaded.move_to_end(clip);return self.loaded[clip]

    def group(self,kind,clip,key):
        d=self.load(clip)
        if kind=='identity':
            selected=np.flatnonzero(d['identity_pairs'][:,1]==key)
            pairs=d['identity_pairs'][selected];ids=np.unique(pairs);ix={int(j):i for i,j in enumerate(ids)}
            x=d['identity_x'][selected]
            if self.normalizer:x=normalize(x,*self.normalizer['identity'])
            return dict(kind=kind,patch=d['patch'][ids],valid=d['valid'][ids],
                        pairs=np.array([[ix[int(a)],ix[int(b)]] for a,b in pairs]),x=x,y=d['identity_y'][selected])
        a,b=map(int,d['offset'][key:key+2]);events=d['events'][a:b];ids=np.unique(events)
        ix={int(j):i for i,j in enumerate(ids)};x=d['action_x'][a:b];parent=d['parent_x'][key]
        if self.normalizer:x=normalize(x,*self.normalizer['action']);parent=normalize(parent,*self.normalizer['parent'])
        return dict(kind='event',patch=d['patch'][ids],valid=d['valid'][ids],
                    parent=ix[int(d['parent'][key])],pairs=np.array([[ix[int(a)],ix[int(b)]] for _,a,b in events]),
                    x=x,parent_x=parent,y=d['risks'][a:b])

    def choose(self,rng,kind,hard=None):
        if kind=='positive':
            keys=sorted(self.positive);event=keys[int(rng.integers(len(keys)))];anchors=self.positive[event]
            clip,key=anchors[int(rng.integers(len(anchors)))]
            probability=1/(len(keys)*len(anchors));event_identity=event
        else:
            pool=self.identity if kind=='identity' else hard if kind=='hard' and hard else self.negative
            clips=sorted(pool);clip=clips[int(rng.integers(len(clips)))];options=pool[clip]
            key=options[int(rng.integers(len(options)))];probability=1/(len(clips)*len(options));event_identity=None
        visit=f'{kind}:{clip}:{key}';self.visits[visit]+=1
        receipt=dict(slot=kind,clip=clip,key=int(key),selection_probability=probability,
                     event_identity=event_identity,hard_fallback=kind=='hard' and not bool(hard),
                     within_group_action_selection='all unique legal actions with sparse loss masks')
        return self.group('identity' if kind=='identity' else 'event',clip,key),receipt

    def batch(self,rng,step,horizon,hard=None):
        slots=['identity']*32 if step<=horizon//10 else ['identity']*16+['positive']*8+['negative']*4+['hard']*4
        samples=[self.choose(rng,k,hard) for k in slots]
        return [s[0] for s in samples],[s[1] for s in samples]


def mixed_update(model,optimizer,samples,lr):
    """Accumulate 32 groups; each loss has its own eligible-group denominator."""
    if len(samples)!=32:raise ValueError('Registered effective event batch is 32 groups')
    counts=dict(identity=sum(s['kind']=='identity' for s in samples),occurrence=0,ranking=0,consistency=len(samples))
    for s in samples:
        if s['kind']=='event':
            y=s['y'];counts['occurrence']+=int((y==1).any() or (len(y)>0 and (y==0).all()))
            counts['ranking']+=int((y==1).any())
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=lr
    total=dict(identity=0.,occurrence=0.,ranking=0.,consistency=0.)
    for sample in samples:
        device=next(model.parameters()).device
        patch=torch.as_tensor(sample['patch'],device=device,dtype=torch.float32)/255
        valid=torch.as_tensor(sample['valid'],device=device,dtype=torch.float32)
        z=model.encoder(patch,valid)
        # Fixed label-free same-image intensity perturbation. No spatial warp,
        # new biological label, or trainable-embedding cache is introduced.
        scale=.9+.2*torch.rand((len(patch),1,1,1,1),device=device)
        # Multiplication preserves zero padding in partially visible triplanes.
        augmented=(patch*scale).clamp(0,1)*valid[:,:,0,None,None,None]
        za=model.encoder(augmented,valid)
        consistency=F.mse_loss(za,z.detach())
        loss=.1*consistency/counts['consistency'];total['consistency']+=float(consistency.detach())/counts['consistency']
        pair=torch.as_tensor(sample['pairs'],device=device,dtype=torch.long)
        x=torch.as_tensor(sample['x'],device=device,dtype=torch.float32)
        y=torch.as_tensor(sample['y'],device=device)
        if sample['kind']=='identity':
            a,b=z[pair[:,0]],z[pair[:,1]]
            logits=model.identity(torch.cat([a,b,(a-b).abs(),x],-1)).squeeze(-1)
            known=y>=0
            identity=F.binary_cross_entropy_with_logits(logits[known],y[known].float()) if known.any() else logits.sum()*0
            loss=loss+identity/max(1,counts['identity']);total['identity']+=float(identity.detach())/max(1,counts['identity'])
        else:
            p=z[sample['parent']];daughters=torch.unique(pair)
            occurrence=model.parent_score(p,z[daughters],torch.as_tensor(sample['parent_x'],device=device))
            action=model.action_scores(p.expand(len(pair),-1),z[pair[:,0]],z[pair[:,1]],x)
            occ,rank,_=supported_losses(occurrence,action,y)
            loss=loss+occ/max(1,counts['occurrence'])+rank/max(1,counts['ranking'])
            total['occurrence']+=float(occ.detach())/max(1,counts['occurrence']);total['ranking']+=float(rank.detach())/max(1,counts['ranking'])
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
    if not torch.isfinite(norm):raise Blocked('Nonfinite compact gradient')
    encoder_gradient=sum(float(p.grad.detach().square().sum()) for p in model.encoder.parameters() if p.grad is not None)**.5
    optimizer.step()
    return dict(losses=total,denominators=counts,gradient_norm=float(norm),encoder_gradient_norm=encoder_gradient,lr=lr)


def fit_normalizer(root,clips):
    sums={};squares={};counts={}
    for clip in clips:
        with np.load(root/clip/'training.npz') as f:
            for kind,key in [('action','action_x'),('parent','parent_x'),('identity','identity_x')]:
                x=f[key].astype(np.float64)
                sums[kind]=sums.get(kind,0)+x.sum(0);squares[kind]=squares.get(kind,0)+(x*x).sum(0);counts[kind]=counts.get(kind,0)+len(x)
    output={}
    for key in sums:
        if not counts[key]:raise Blocked('Empty fit-only normalization census: '+key)
        mean=sums[key]/counts[key];std=np.sqrt(np.maximum(0,squares[key]/counts[key]-mean**2))
        output[key]=[mean.tolist(),np.maximum(std,1e-4).tolist()]
    return dict(values=output,counts=counts,std_floor=1e-4,clip=[-10,10],fitted_partition='source-fit')
