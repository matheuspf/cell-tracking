"""Actual installed association and temporal 3D encoder optimization."""
import time,random,resource,collections
import torch
import torch.nn.functional as F
from .common import *
from . import native_adapter as native
from .supervision import native_catalog

def tensor_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class SourceData:
    def __init__(self,source):self.source=source;self.cache=collections.OrderedDict();self.frames={}
    def get(self,row):
        name=row['dataset'];assert name.startswith(self.source+'_')
        if name not in self.cache:
            c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask']
            self.cache[name]=(c['nodes'][m],c['features_source'][m].astype(np.float32),c['features_target'][m].astype(np.float32))
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(name);nodes,fs,ft=self.cache[name];t=row['t']
        ai=np.flatnonzero(nodes[:,1]==t);bi=np.flatnonzero(nodes[:,1]==t+1)
        a,b=nodes[ai],nodes[bi];am={int(n[0]):i for i,n in enumerate(a)};bm={int(n[0]):i for i,n in enumerate(b)}
        positive=np.array([(am[x],bm[y]) for x,y in row['positive']],np.int64)
        return a,b,(fs[ai],ft[bi]),positive
    def images(self,row):
        name=row['dataset']
        if name not in self.frames:
            self.frames={name:native.Frames(DATA/'train'/f'{name}.zarr')}
        return self.frames[name].pair(row['t'])

def objective(log,positive,teacher=None):
    p=torch.as_tensor(positive,device=log.device,dtype=torch.long).reshape(-1,2)
    if not log.numel():return log.sum()*0
    target=torch.full((log.shape[1],),-1,device=log.device,dtype=torch.long)
    target[p[:,1]]=p[:,0]
    # Every daughter chooses its own parent; two columns can have the same parent.
    loss=F.cross_entropy(log.T,target,ignore_index=-1) if len(p) else log.sum()*0
    count=torch.bincount(p[:,0],minlength=log.shape[0]);forks=count[p[:,0]]>1
    if forks.any():loss=loss+.25*(-log.log_softmax(0)[p[forks,0],p[forks,1]]).mean()
    if teacher is not None:
        with torch.no_grad():
            q=teacher.softmax(0);conf=q.max(0).values;unknown=target<0
            # Soft stability only when the unchanged model has a confident incoming choice.
            mask=unknown&(conf>.98)
        if mask.any():loss=loss+.02*F.kl_div(log[:,mask].log_softmax(0),q[:,mask],reduction='batchmean')
    return loss

def _fit(source,stage,seed=20260910,steps=None,tiny=False):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    assert stage in ['N1','N2'];steps=steps or (8000 if stage=='N1' else 12000)
    key=f'{source}_{stage}_{seed}'+('_tiny' if tiny else '')
    dest=OUT/'models/native'/f'{key}.pt';meta=dest.with_suffix('.json')
    if meta.exists():assert sha(dest)==read(meta)['sha256'];return
    previous=OUT/'models/native'/f'{source}_N1_{seed}.pt'
    if stage=='N2' and not tiny:assert previous.exists()
    model=native.load(previous if stage=='N2' and not tiny else None)
    model.eval() # Freeze BN running statistics and use deterministic native dropout at inference setting.
    for p in model.parameters():p.requires_grad_(False)
    for p in model.transformer.parameters():p.requires_grad_(True)
    if stage=='N2':
        for p in model.unet.parameters():p.requires_grad_(True)
    encoder_before=tensor_hash(model.unet);initial_tensor=model.unet.encoder_blocks[0][0].weight.detach().cpu().clone()
    head_before=tensor_hash(model.transformer)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    groups=[dict(params=model.transformer.parameters(),lr=1e-4)]
    if stage=='N2':groups.append(dict(params=model.unet.parameters(),lr=1e-5))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    catalog=native_catalog(source);allcatalog=catalog
    if tiny:catalog=[r for r in catalog if (OUT/'observations'/f"{r['dataset']}.npz").exists()][:4]
    assert catalog
    data=SourceData(source);teacher=native.load()
    source_windows=len(catalog);batch=4;order=[];rng=np.random.default_rng(seed);epoch=0;cursor=0;seen=set();seen_pos=set()
    def refill():
        nonlocal order,epoch,cursor
        # Shuffle clips, then contiguous four-window blocks: bounded reads and all transitions eligible.
        grouped={}
        for i,r in enumerate(catalog):grouped.setdefault(r['dataset'],[]).append(i)
        names=list(grouped);rng.shuffle(names);order=[]
        for name in names:
            blocks=[grouped[name][i:i+4] for i in range(0,len(grouped[name]),4)];rng.shuffle(blocks)
            order.extend(i for block in blocks for i in block)
        epoch+=1;cursor=0
    refill();history=[];grad_receipt={};start=time.monotonic();torch.cuda.reset_peak_memory_stats();start_step=0
    cp=OUT/'models/native'/f'{key}.resume.pt'
    if cp.exists() and not tiny:
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert (state['seed'],state['source'],state['stage'])==(seed,source,stage)
        model.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer']);start_step=state['step']
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        order=state['order'];cursor=state['cursor'];epoch=state['epoch'];history=state['history']
        seen=set(map(tuple,state['seen']));seen_pos=set(map(tuple,state['seen_pos']))
        start-=history[-1]['seconds'];grad_receipt=state['grad_receipt']
        print('Resumed complete optimizer/RNG state',key,'at step',start_step,flush=True)
    # Tiny overfit validates real losses/gradients before a production budget is spent.
    for step in range(start_step,steps):
        optimizer.zero_grad(set_to_none=True);losses=[]
        for accumulation in range(batch):
            if cursor>=len(order):refill()
            i=order[cursor];cursor+=1;r=catalog[i];a,b,feat,positive=data.get(r)
            if stage=='N1':
                log=native.logits(model,a,b,features=feat)
                with torch.no_grad():teacherlog=native.logits(teacher,a,b,features=feat)
            else:
                images=data.images(r)
                log=native.logits(model,a,b,images=images)
                with torch.no_grad():teacherlog=native.logits(teacher,a,b,features=feat)
            loss=objective(log,positive,teacherlog);assert torch.isfinite(loss), (key,step)
            (loss/batch).backward();losses.append(float(loss.detach()))
            seen.add((r['dataset'],r['t']));seen_pos.update((r['dataset'],x,y) for x,y in r['positive'])
            del log,loss
        if step==0:
            grad_receipt={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm);optimizer.step()
        if step%20==0 or step==steps-1:
            h=dict(step=step+1,loss=float(np.mean(losses)),gradient_norm=float(norm),seconds=time.monotonic()-start,
                seen_windows=len(seen),seen_positives=len(seen_pos),gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            history.append(h)
            write(OUT/'training'/f'{key}_progress.json',dict(key=key,source=source,stage=stage,seed=seed,steps=step+1,target_steps=steps,history=history))
            print(key,step+1,round(h['loss'],5),round(h['seconds'],1),flush=True)
        if step%250==249 and not tiny:
            reserve();cp=OUT/'models/native'/f'{key}.resume.pt';cp.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,seed=seed,source=source,stage=stage,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order=order,cursor=cursor,epoch=epoch,
                history=history,seen=list(seen),seen_pos=list(seen_pos),grad_receipt=grad_receipt),cp.with_suffix('.tmp'));cp.with_suffix('.tmp').replace(cp)
    encoder_after=tensor_hash(model.unet);head_after=tensor_hash(model.transformer)
    delta=float((model.unet.encoder_blocks[0][0].weight.detach().cpu()-initial_tensor).norm())
    assert head_before!=head_after
    if stage=='N1':assert encoder_before==encoder_after
    else:assert encoder_before!=encoder_after and delta>0 and grad_receipt['unet.encoder_blocks.0.0.weight']>0
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),source=source,stage=stage,seed=seed,steps=steps,trained_modules=trainable),dest)
    write(meta,dict(key=key,source=source,stage=stage,seed=seed,steps=steps,effective_batch_windows=batch,seconds=time.monotonic()-start,
        native_image_optimizer_updates=steps if stage=='N2' else 0,encoder_before=encoder_before,encoder_after=encoder_after,first_encoder_tensor_l2_change=delta,
        head_before=head_before,head_after=head_after,first_step_gradients=grad_receipt,trainable_modules=trainable,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        source_windows=source_windows,source_positive_edges=sum(len(r['positive']) for r in catalog),seen_windows=len(seen),seen_positives=len(seen_pos),
        missed_windows=source_windows-len(seen),history=history,sha256=sha(dest),tiny=tiny,
        precision='float32; BN statistics and dropout fixed to inference mode',source_dev='source resubstitution only; fixed budget, no target tuning',
        teacher_stability_weight=.02,teacher_confidence_min=.98,teacher_targets='soft incoming distributions, unsupported targets only; no hard nondivision or birth labels',
        code_sha256=sha(Path(__file__))))

def fit(source,stage,seed=20260910,steps=None,tiny=False):
    # Per-fit exclusion lets an already registered source run use the auxiliary
    # GPU lane without racing the main queue's checkpoint or optimizer state.
    import fcntl
    key=f'{source}_{stage}_{seed}'+('_tiny' if tiny else '')
    lock=OUT/'training_locks'/f'{key}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _fit(source,stage,seed,steps,tiny)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',choices=['44b6','6bba']);p.add_argument('stage',choices=['N1','N2']);
    p.add_argument('--seed',type=int,default=20260910);p.add_argument('--steps',type=int);p.add_argument('--tiny',action='store_true')
    a=p.parse_args();fit(a.source,a.stage,a.seed,a.steps,a.tiny)
