"""Sparse source-only probing of genuine frozen HOCT edge embeddings."""
import time,sys
import torch
import torch.nn.functional as F
from .common import *
from .candidates import bank,motion_config
from .supervision import supported_pairs
from .hoct_adapter import load,iter_embeddings
from .cache import score_features
from . import native_adapter as native
from .banks import cached as cached_bank

def extract_sources():
    torch.set_num_threads(2);model=load();native_model=native.load();motion=motion_config()
    for row in inventory():
        name=row['dataset'];dest=OUT/'hoct_training_features'/f'{name}.npz'
        if dest.exists():continue
        observation=OUT/'observations'/f'{name}.npz'
        while not observation.exists():time.sleep(10)
        start=time.monotonic();c=arrays(observation);m=c['oldmask'];nodes=c['nodes'][m]
        # Build the training bank using THIS source's statistics, never the opposite source's radius.
        common=cached_bank(native_model,name,row['embryo']);pairs=common['pairs']
        labels,coverage=supported_pairs(name,nodes,pairs);x=[];y=[];native_rows=[];ids=[];raw=[];meta_rows=[]
        nl=common['native_logits']
        covered=np.zeros(len(pairs),bool)
        for ii,ll,xx,meta in iter_embeddings(model,nodes,pairs,c['properties'][m],c['valid_region'][m]):
            covered[ii]=True;ok=labels[ii]>=0
            if ok.any():x.append(xx[ok].astype(np.float16));y.append(labels[ii[ok]]);ids.append(ii[ok]);raw.append(ll[ok]);native_rows.append(nl[ii[ok]])
            meta_rows.append(meta)
        assert x,name
        save(dest,features=np.concatenate(x),labels=np.concatenate(y),native_logits=np.concatenate(native_rows),
            raw_logits=np.concatenate(raw),pair_indices=np.concatenate(ids))
        write(dest.with_suffix('.json'),dict(dataset=name,source=row['embryo'],created=now(),seconds=time.monotonic()-start,
            checkpoint_sha256=sha(OUT/'models/hoct/general_v1.pt'),observations_sha256=sha(observation),source_motion=motion[row['embryo']],
            candidate_universe_sha256=digest(pairs.tolist()),coverage=coverage,feature_covered_edges=int(covered.sum()),
            source_positive_feature_rows=int(sum((v==1).sum() for v in y)),source_negative_feature_rows=int(sum((v==0).sum() for v in y)),
            missing_positive_features=int(((labels==1)&~covered).sum()),max_window_edges=max(z['edges'] for z in meta_rows),
            max_window_nodes=max(z['nodes'] for z in meta_rows),sha256=sha(dest),deduplicated=True,consistency='disabled; C0 absent edges never hard negatives'))
        print('HOCT source features',name,round(time.monotonic()-start,1),'s',sum(len(v) for v in y),'supported edges',flush=True)

def fit(source,seed=20260910):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    dest=OUT/'models/hoct'/f'{source}_probe_{seed}.pt'
    if dest.with_suffix('.json').exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha256'];return
    shards=[];x=[];y=[];nl=[];rl=[];c0=[]
    for row in inventory():
        if row['embryo']!=source:continue
        p=OUT/'hoct_training_features'/f"{row['dataset']}.npz"
        while not p.exists():time.sleep(10)
        c=arrays(p);x.append(c['features']);y.append(c['labels']);nl.append(c['native_logits']);rl.append(c['raw_logits']);shards.append(dict(dataset=row['dataset'],sha256=sha(p)))
        pairs=arrays(OUT/'banks'/source/'P0'/f"{row['dataset']}.npz")['pairs'];old=set(map(tuple,graph(row['dataset'])['edges']))
        c0.append(np.array([tuple(e) in old for e in pairs[c['pair_indices']]],np.float32))
    X=torch.tensor(np.concatenate(x).astype(np.float32),device='cuda');Y=torch.tensor(np.concatenate(y),dtype=torch.float32,device='cuda')
    backbone=load();params=dict(backbone.named_parameters());head=torch.nn.Linear(X.shape[1],1).cuda()
    assert params['head.weight'].shape==head.weight.shape
    with torch.no_grad():head.weight.copy_(params['head.weight']);head.bias.copy_(params['head.bias'])
    initial=head.weight.detach().cpu().clone();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    batch=8192;steps=2000;history=[];start=time.monotonic();order=torch.randperm(len(X),device='cuda');cursor=0;seen=torch.zeros(len(X),dtype=torch.bool,device='cuda')
    for step in range(steps):
        if cursor>=len(X):order=torch.randperm(len(X),device='cuda');cursor=0
        ii=order[cursor:cursor+batch];cursor+=len(ii);seen[ii]=True
        optimizer.zero_grad(set_to_none=True);log=head(X[ii])[:,0]
        loss=F.binary_cross_entropy_with_logits(log,Y[ii]);loss.backward();gn=float(head.weight.grad.norm())
        optimizer.step()
        if step%100==0 or step==steps-1:
            with torch.no_grad():full=F.binary_cross_entropy_with_logits(head(X)[:,0],Y).item()
            history.append(dict(step=step+1,loss=full,seconds=time.monotonic()-start,gradient_norm=gn,seen_rows=int(seen.sum())))
            print(source,'HOCT probe',seed,step+1,full,flush=True)
    state=dict(weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu(),source=source,seed=seed,steps=steps)
    torch.save(state,dest);delta=float((state['weight']-initial).norm());assert delta>0
    with torch.no_grad():pred=head(X)[:,0].cpu().numpy()
    save(OUT/'source_calibration'/f'{source}_H_{seed}.npz',labels=Y.cpu().numpy(),H0=np.concatenate(rl),H1=pred,N0=np.concatenate(nl),c0=np.concatenate(c0))
    write(dest.with_suffix('.json'),dict(source=source,seed=seed,steps=steps,seconds=time.monotonic()-start,history=history,
        architecture='official general_v1 frozen 6,252,593 parameter JIT; genuine 288-dimensional edge embeddings + ProbedModel-equivalent linear head',
        head_parameter_change_l2=delta,encoder_changed=False,trainable_modules=['head.weight','head.bias'],
        supported_positives=int(Y.sum()),supported_negatives=int((Y==0).sum()),seen_rows=int(seen.sum()),total_rows=len(X),
        source_shards=shards,sha256=sha(dest),consistency_control='H1_no_hard_ILP_consistency; unknown edges masked',
        upstream_provenance='official public weights, biological exposure not independently certified',source_dev='resubstitution, fixed 2000 updates'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','fit']);p.add_argument('--source',choices=['44b6','6bba']);p.add_argument('--seed',type=int,default=20260910)
    a=p.parse_args()
    if a.action=='extract':extract_sources()
    else:fit(a.source,a.seed)
