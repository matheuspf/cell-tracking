"""Real division-event classifiers; source event bags, never membership labels."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from .common import digest, now, sha, write_json
from .event_proposals import EVENT_FEATURES


def network(geometry_dim=len(EVENT_FEATURES)):
    import torch
    from torch import nn
    class TemporalEventNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.image=nn.Sequential(nn.Conv2d(36,16,3,padding=1),nn.SiLU(),
                nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
            self.geometry=nn.Sequential(nn.Linear(geometry_dim,32),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(56,32),nn.SiLU(),nn.Linear(32,1))
        def forward(self,crops,features):
            return self.head(torch.cat([self.image(crops.reshape(-1,36,12,12)),self.geometry(features)],dim=1)).squeeze(1)
    return TemporalEventNetwork()


def fit_logistic(x,y,weights,path,metadata):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(x)
    w=weights.astype(float).copy()
    # Equal event-class objective; these scores are not calibrated prevalence.
    w[y==0]*=w[y==1].sum()/max(1e-12,w[y==0].sum())
    model=LogisticRegression(C=.1,max_iter=1500,solver='lbfgs',random_state=20260909)
    model.fit(scaler.transform(x),y,sample_weight=w)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,scaler=scaler,metadata=metadata,features=EVENT_FEATURES),path)
    pred=model.predict_proba(scaler.transform(x))[:,1]
    receipt=dict(**metadata,model_sha256=sha(path),training_rows=len(x),positive_rows=int(y.sum()),
                 objective='equal event-class loss; positive bag total weight one; scores not posterior calibrated',
                 source_weighted_bce=float(np.average(-y*np.log(np.maximum(pred,1e-8))-(1-y)*np.log(np.maximum(1-pred,1e-8)),weights=w)),
                 actual_iterations=model.n_iter_.tolist())
    write_json(path.with_suffix('.json'),receipt)
    return receipt


def fit_image(x,y,bags,crops,crop_index,path,metadata,seed=20260909,steps=10000):
    import torch
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    if not torch.cuda.is_available():raise RuntimeError('Event image training requires CUDA; no silent CPU fallback')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(seed);scaler=StandardScaler().fit(x)
    xt=torch.as_tensor(scaler.transform(x).astype(np.float32),device='cuda')
    ct=torch.as_tensor(crops,device='cuda')
    ci=torch.as_tensor(crop_index,device='cuda')
    yt=torch.as_tensor(y.astype(np.float32),device='cuda')
    model=network(x.shape[1]).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    groups=[defaultdict(list),defaultdict(list)]
    for k,(label,bag) in enumerate(zip(y,bags)):groups[int(label)][str(bag)].append(k)
    group_keys=[sorted(g) for g in groups]
    if not all(group_keys):raise ValueError('Source event fit needs supported positive and negative groups')
    group_order=[rng.permutation(len(g)) for g in group_keys];group_cursor=[0,0]
    within={key:rng.permutation(members) for g in groups for key,members in g.items()}
    within_cursor={key:0 for key in within};seen=np.zeros(len(x),bool);curves=[];tic=time.perf_counter()
    for step in range(1,steps+1):
        batch=[]
        for label in [0,1]:
            for _ in range(32):
                cursor=group_cursor[label]
                if cursor==len(group_order[label]):group_order[label]=rng.permutation(len(group_keys[label]));cursor=0
                key=group_keys[label][group_order[label][cursor]];group_cursor[label]=cursor+1
                j=within_cursor[key]
                if j==len(within[key]):within[key]=rng.permutation(groups[label][key]);j=0
                batch.append(within[key][j]);within_cursor[key]=j+1
        seen[batch]=True;index=torch.as_tensor(batch,device='cuda')
        xb=ct[ci[index]].float()/255.
        # Geometry is daughter-permutation invariant by construction; the image
        # branch is parent-centered and cannot identify an ordering of daughters.
        logits=model(xb,xt[index]);loss=F.binary_cross_entropy_with_logits(logits,yt[index])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step==1 or step%500==0:
            torch.cuda.synchronize();row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-tic,
                positive_rows_seen=int(np.sum(seen&(y==1))),positive_groups=len(groups[1]),negative_groups=len(groups[0]))
            curves.append(row);print('event_image',metadata['source_embryo'],seed,row,flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-tic
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),optimizer=optimizer.state_dict(),mean=scaler.mean_,scale=scaler.scale_,
        metadata=metadata,features=EVENT_FEATURES,seed=seed,actual_steps=steps,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state),path)
    receipt=dict(**metadata,created=now(),model_sha256=sha(path),seed=seed,actual_steps=steps,seconds=seconds,
        cuda_device=torch.cuda.get_device_name(0),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        unique_positive_groups=len(groups[1]),unique_negative_groups=len(groups[0]),positive_rows=int(y.sum()),
        positive_rows_seen=int(np.sum(seen&(y==1))),all_positive_alternatives_seen=bool(seen[y==1].all()),
        source_learning_curve=curves,objective='uniform event bags and supported nondivision groups; equal class risk, not calibrated prevalence',
        upstream='compact 9-frame encoder trained from scratch on source-only division event labels')
    write_json(path.with_suffix('.json'),receipt)
    del model,optimizer,xt,ct,ci,yt
    torch.cuda.empty_cache()
    return receipt


def predict_logistic(path,features):
    import joblib
    artifact=joblib.load(path)
    return artifact['model'].predict_proba(artifact['scaler'].transform(features))[:,1].astype(np.float32)


def load_image(path):
    import torch
    a=torch.load(path,map_location='cpu',weights_only=False)
    model=network(len(a['features'])).cuda().eval();model.load_state_dict(a['state_dict'])
    return model,a


def predict_image(loaded,features,crops,crop_index,batch=2048):
    import torch
    model,a=loaded;result=np.empty(len(features),np.float32)
    with torch.inference_mode():
        # The image branch is parent-centered. Encode each unique crop once;
        # candidate geometry/path alternatives share this exact image embedding.
        image_features=torch.empty((len(crops),24),device='cuda')
        for start in range(0,len(crops),batch):
            c=torch.as_tensor(crops[start:start+batch],device='cuda').float()/255.
            image_features[start:start+batch]=model.image(c.reshape(-1,36,12,12))
        for start in range(0,len(features),batch):
            stop=min(len(features),start+batch)
            x=torch.as_tensor(((features[start:stop]-a['mean'])/a['scale']).astype(np.float32),device='cuda')
            ids=torch.as_tensor(crop_index[start:stop],device='cuda')
            logits=model.head(torch.cat([image_features[ids],model.geometry(x)],dim=1)).squeeze(1)
            result[start:stop]=logits.sigmoid().cpu().numpy()
    return result
