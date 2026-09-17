"""Grouped out-of-fit small heads; frozen image-encoder exposure stays explicit.

This is the prespecified low-support calibration fallback, not another joint
image model or a substitute for any 4,096-update main fit.
"""
from collections import Counter,OrderedDict,defaultdict
import contextlib
import copy
import json
import time
import numpy as np
import torch

from .common import WORK,RESULTS,read_json,write_json,append_json,sha,digest,save_arrays,atomic_torch_save
from .contracts import LR,lr_factor
from .dataset import SourceDataset
from .model import ActionModel,loss
from .resources import Lease,Monitor

FOLDS=3
HEAD_UPDATES=4096


def plan():
    value=dict(folds=FOLDS,head_updates=HEAD_UPDATES,effective_batch=32,
        fold_seed=20260916,fold_assignment='SHA256 of source overlap group, modulo three',
        small_head='Same complete-action MLP, keep, score, risk and identity heads, reset from zero-output initialization',
        image_encoder='Frozen at the source checkpoint; may have seen inner fold labels in joint fitting',
        encoder_exposure='Not clean OOF; only the small head is out of fit, upstream P0 remains exposed',
        optimizer='Same AdamW, schedule, losses and representative mixture as the main fit',
        trigger='Fewer than five positive or negative max-selected held-source overlap groups',
        no_target_access=True,augmentation=False,trainable_encoder_cache=False,
        fallback='Fixed zero boundary with calibration_unestablished if a fold lacks either supported class or OOF support remains too small')
    write_json(RESULTS/'small_head_oof_plan.json',value,immutable=True)
    return value


def fold_of(source,group):
    return int(digest(['small-head-oof',20260916,source,group])[:16],16)%FOLDS


def subset(base,keys):
    data=copy.copy(base)
    data.anchors={k:base.anchors[k] for k in keys}
    data.groups=defaultdict(list);data.random_groups=defaultdict(list);data.positive_groups=defaultdict(list)
    for key,a in data.anchors.items():
        data.groups[a['group']].append(key)
        if a['random_included']:data.random_groups[a['group']].append(key)
        if a['positive']:data.positive_groups[a['group']].append(key)
    data.ordinary={g:ks for g,ks in data.random_groups.items() if g not in data.positive_groups}
    data.ordinary_keys=sorted(data.ordinary);data.positive_keys=sorted(data.positive_groups)
    data.random_keys=sorted(data.random_groups);data.visits=Counter();data.anchor_visits=Counter();data.mined=[]
    return data


class FrozenPoints:
    def __init__(self,source_model,base,root,amp,checkpoint_sha256):
        self.model,self.base,self.root,self.amp=source_model,base,root,amp
        if self.model.image:self.model.encoder.requires_grad_(False)
        self.fingerprint=digest([checkpoint_sha256,sha(RESULTS/'small_head_oof_plan.json')])
        self.cache=OrderedDict()

    def path(self,key):
        a=self.base.anchors[key]
        return self.root/a['dataset']/(str(a['anchor'])+'.npz')

    @torch.no_grad()
    def prepare(self):
        if not self.model.image:return
        iterator=iter(sorted(self.base.anchors));finished=False
        while not finished:
            with Lease('small_head_oof/frozen_encoder/'+self.base.source,required_gib=3.):
                self.model.cuda().eval();start=time.monotonic()
                while True:
                    try:key=next(iterator)
                    except StopIteration:finished=True;break
                    path=self.path(key)
                    if not path.exists():
                        b,scene=self.base.batch(dict(key=key),'cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):
                            maps=self.model.encoder.maps(scene[None])
                            z=self.model.encoder.tokens(maps,scene[None],[b['query_voxels']],[b['query_time']])[0]
                        save_arrays(path,points=z.float().cpu().numpy(),fingerprint=np.array(self.fingerprint))
                    if time.monotonic()-start>=30:break
                self.model.cpu();torch.cuda.empty_cache()

    def get(self,key,device):
        if not self.model.image:return None
        if key not in self.cache:
            with np.load(self.path(key),allow_pickle=False) as f:
                if str(f['fingerprint'])!=self.fingerprint:raise ValueError('Frozen encoder cache provenance drift')
                self.cache[key]=f['points']
            while len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return torch.as_tensor(self.cache[key],device=device)


def batch(data,points,key,device):
    b,_=data.batch(dict(key=key),device)
    return b,points.get(key,device)


def run(source_model,held,root,amp,checkpoint_sha256):
    from .calibration import fit
    from .train import optimizer_to
    plan();final=root/'summary.json'
    if final.exists():return read_json(final)
    original_image=source_model.image
    base=SourceDataset(held.source,image=original_image)
    base.anchors.update(held.anchors)
    # All metadata retains its original fit/calibration provenance. The OOF
    # groups unite complete clips and their known exact-frame overlaps.
    split=read_json(RESULTS/'split_manifest.json')['directions'][held.source]
    group={key:split[a['dataset']]['overlap_group'] for key,a in base.anchors.items()}
    folds={key:fold_of(held.source,g) for key,g in group.items()}
    fold_data=[]
    for fold in range(FOLDS):
        train=subset(base,[k for k in base.anchors if folds[k]!=fold])
        validation=subset(base,[k for k in base.anchors if folds[k]==fold])
        if not train.positive_keys or not train.ordinary_keys or not validation.anchors:
            result=fit([]);result.update(reason='Grouped small-head OOF infeasible: an intact overlap fold lacks positive/negative training support or held anchors',
                source=held.source,fold=fold,attempted_grouped_oof=True)
            write_json(final,result);return result
        fold_data.append((train,validation))
    # The immutable source image reader must include calibration clips as well.
    if original_image:
        from .scenes import Scenes
        base.rows=base.rows+held.rows;base.scenes=Scenes(base.rows)
    points=FrozenPoints(source_model,base,root/'frozen_points',amp,checkpoint_sha256)
    rows=[];receipts=[];device='cuda' if original_image else 'cpu'
    with Monitor(root/'resources.json'):
        points.prepare()
        # Subsequent fitting reads frozen arrays only. No cached representation
        # belongs to any trainable encoder in these explicitly head-only fits.
        for data in [base,*[d for pair in fold_data for d in pair]]:data.image=False
        for fold,(train,validation) in enumerate(fold_data):
            folder=root/f'fold-{fold}';done=folder/'complete.json'
            if done.exists():
                receipt=read_json(done);rows.extend(receipt['rows']);receipts.append({k:v for k,v in receipt.items() if k!='rows'});continue
            seed=20260916+fold;torch.manual_seed(seed)
            model=ActionModel(image=original_image)
            if original_image:
                model.encoder.load_state_dict(source_model.encoder.state_dict())
                model.encoder.requires_grad_(False)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=1e-4)
            resume=folder/'resume.pt';step=0;started=time.monotonic()
            if resume.exists():
                saved=torch.load(resume,map_location='cpu',weights_only=False)
                if saved['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('OOF source checkpoint drift')
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['step']
            while step<HEAD_UPDATES:
                context=Lease('small_head_oof/train/'+held.source,required_gib=3.) if original_image else contextlib.nullcontext()
                with context:
                    model.to(device).train();optimizer_to(optimizer,device);block=time.monotonic()
                    while step<HEAD_UPDATES:
                        optimizer.zero_grad(set_to_none=True);total_value=0.
                        samples=train.samples(step,seed,'G30')
                        for start in range(0,32,8):
                            chunk=samples[start:start+8];pairs=[batch(train,points,s['key'],device) for s in chunk]
                            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bool(original_image and amp)):
                                outputs=([model.scores(b,z) for b,z in pairs] if original_image else model([b for b,z in pairs]))
                            terms=[loss(o,b,risk_weight=2*s['risk_design_weight'])[0]/32 for o,(b,z),s in zip(outputs,pairs,chunk)]
                            v=torch.stack(terms).sum();v.backward();total_value+=float(v.detach())
                        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                        for g in optimizer.param_groups:g['lr']=LR*lr_factor(step)
                        optimizer.step();step+=1
                        append_json(folder/'history.jsonl',dict(step=step,total=total_value,gradient_norm=float(norm)))
                        if step%128==0 or time.monotonic()-block>=30:break
                    model.cpu();optimizer_to(optimizer,'cpu')
                    if original_image:torch.cuda.empty_cache()
                if step%128==0 or step==HEAD_UPDATES:
                    atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                        checkpoint_sha256=checkpoint_sha256),resume)
            fold_rows=[];keys=iter(sorted(k for k,a in validation.anchors.items() if a['random_included']));finished=False
            while not finished:
                context=Lease('small_head_oof/predict/'+held.source,required_gib=3.) if original_image else contextlib.nullcontext()
                with context,torch.no_grad():
                    model.to(device).eval();block=time.monotonic()
                    while True:
                        try:key=next(keys)
                        except StopIteration:finished=True;break
                        b,z=batch(validation,points,key,device)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bool(original_image and amp)):
                            out=model.scores(b,z) if original_image else model([b])[0]
                        gains=out['gain'].cpu().numpy();indices=np.flatnonzero(~b['keep'].cpu().numpy())
                        if len(indices):
                            i=indices[np.argmax(gains[indices])]
                            if bool(b['supported'][i]):
                                fold_rows.append(dict(key=key,group=group[key],fold=fold,gain=float(gains[i]),
                                    target=int(b['utility'][i]>1e-12),weight=1/validation.anchors[key]['random_probability'],
                                    selected_action=int(i),null_gain=0.))
                        if time.monotonic()-block>=30:break
                    model.cpu()
                    if original_image:torch.cuda.empty_cache()
            receipt=dict(fold=fold,head_updates=step,train_overlap_groups=len({group[k] for k in train.anchors}),
                held_overlap_groups=len({group[k] for k in validation.anchors}),
                overlap=set(group[k] for k in train.anchors)&set(group[k] for k in validation.anchors),
                model_sha256=sha(resume),seconds=time.monotonic()-started,rows=fold_rows)
            assert not receipt['overlap'];receipt['overlap']=[]
            write_json(done,receipt);rows.extend(fold_rows);receipts.append({k:v for k,v in receipt.items() if k!='rows'})
            print(f'Small-head OOF {held.source} fold {fold+1}/{FOLDS}: {step} updates, {len(fold_rows)} held decisions',flush=True)
    result=fit(rows)
    if result['status']=='held_source_fitted':result['status']='grouped_small_head_oof_fitted'
    result.update(rows=rows,source=held.source,partition='grouped_small_head_oof',attempted_grouped_oof=True,
        fold_receipts=receipts,checkpoint_sha256=checkpoint_sha256,post_maximization=True,target_used=False,
        image_encoder_exposure='Frozen original source encoder includes original source-fit labels; not clean image OOF' if original_image else 'Frozen inherited P0/native features remain exposed',
        calibration_scope='Only small heads are grouped out of fit; reused acquisition groups remain exploratory')
    write_json(final,result);return result
