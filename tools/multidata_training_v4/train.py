"""Resumable actual optimizer steps, balanced exposure cycles, source-only adaptation."""
import gc
import time
import random
import resource
import torch
from torch.nn import functional as F
from .common import *
from .models import Geometry,ImageEvent,Detector
from .losses import event_loss
from .sampling import Bank,paths

MODELS={'G':Geometry,'I':ImageEvent,'D':Detector}

def probability_consistency(student,teacher,component):
    """Bound consistency when confidently negative raw logits have no lower bound."""
    if component=='D':student,teacher=student.sigmoid(),teacher.sigmoid()
    else:student,teacher=student.softmax(-1),teacher.softmax(-1)
    return F.mse_loss(student,teacher)

def detector_loss(output,b):
    center,offset=output
    return F.binary_cross_entropy_with_logits(center,b['target'])+.5*(F.smooth_l1_loss(offset,b['offset'],reduction='none').mean(1)*b['positive']).sum()/b['positive'].sum().clamp_min(1)

def loss(model,b,component):
    o=model(b)
    return detector_loss(o,b) if component=='D' else event_loss(*o,b)

def augment(batch,seed,noise=.025):
    b=dict(batch)
    if 'patch' in b:
        # Appearance only: the same gain/blur/noise law applies to all views/daughters.
        p=b['patch'];gain=.6+.8*torch.rand((len(p),)+ (1,)*(p.ndim-1),device=p.device)
        flat=p.reshape(-1,3,12,12);blur=F.avg_pool2d(flat,3,1,1).reshape_as(p)
        b['patch']=((.8*p+.2*blur)*gain+torch.randn_like(p)*noise).clamp(0,1)
    return b

def validation(model,bank,component):
    model.eval();n=min(len(bank.sid),2048);idx=torch.linspace(0,len(bank.sid)-1,n,device='cuda').long()
    total=0.;correct=0;pos=0;tp=0;fp=0;cnt=0
    with torch.no_grad():
        for ids in idx.split(64 if component!='I' else 16):
            b={k:v[ids] for k,v in bank.data.items()}
            if 'patch' in b:b['patch']=b['patch'].float()/255
            o=model(b);l=detector_loss(o,b) if component=='D' else event_loss(*o,b)
            total+=float(l)*len(ids);cnt+=len(ids)
            if component!='D':
                pred=o[0].argmax(1);y=b['target'];m=y>=0
                correct+=int(((pred==y)&m).sum());pos+=int((y>0).sum());tp+=int(((pred==y)&(y>0)).sum());fp+=int(((pred>0)&(pred!=y)&m).sum())
    model.train()
    return dict(loss=total/max(cnt,1),rows=cnt,correct=correct,positive=pos,positive_pair_correct=tp,false_pair=fp)

def fit(name,component,domains,steps,init=None,source=None,replay=None,randomized=False,seed=SEED,calibration_source=None):
    dest=OUT/'models'/f'{name}.pt';receipt=dest.with_suffix('.json')
    calibration_source=calibration_source or (source if randomized else None)
    noise=.025
    if randomized:
        assert calibration_source in ['44b6','6bba']
        backgrounds=[]
        for p in paths(calibration_source,'D'):
            d=arrays(p);bg=d['patch'][d['positive']==0].astype(float)/255
            if len(bg):backgrounds.append(float(bg.std()))
        noise=float(np.clip(np.median(backgrounds) if backgrounds else .025,.005,.08))
    config=dict(name=name,component=component,domains=domains,steps=steps,init=init,source=source,replay=replay,randomized=randomized,seed=seed,
        appearance_calibration_source=calibration_source,appearance_noise_std=noise,
        batch_size=64 if component!='I' else 16,adapt_real_fraction=.75,consistency_replay_for_real_only=True)
    if source:config['consistency_space']='bounded_probability_v1' if not replay else 'external_supervised_replay'
    if component in ['G','I'] and any(d in ['44b6','6bba','zebrafish','ascidian'] or d.startswith('rendered_zoo_') for d in domains+(replay or [])):
        config['sparse_edge_mask_version']='unknown_second_child_masked_v2'
    if receipt.exists():
        old=read(receipt);assert old['config']==config;assert sha(dest)==old['checkpoint_sha256'];print('reuse',name,flush=True);return
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
    model=MODELS[component]().cuda();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    if init:
        state=torch.load(OUT/'models'/f'{init}.pt',map_location='cuda',weights_only=False)['model']
        if component=='I' and init.startswith('D_'):model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')})
        else:model.load_state_dict(state)
    encoder=list(model.encoder.parameters()) if hasattr(model,'encoder') else list(model.point.parameters())+list(model.context.parameters())
    encids={id(p) for p in encoder};heads=[p for p in model.parameters() if id(p) not in encids]
    lr=.0008 if component=='D' else .0003
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':lr/10 if source else lr},{'params':heads,'lr':lr}],weight_decay=.001)
    all_domains=list(dict.fromkeys(([source] if source else domains)+(replay or [])))
    banks={d:Bank(paths(d,component),component,seed) for d in all_domains}
    val_domain='synthetic' if 'synthetic' in all_domains else ('zebrafish' if 'zebrafish' in all_domains else None)
    vb=Bank(paths(val_domain,component,'validation'),component,seed) if val_domain else None
    cursor=0;elapsed=0.;history=[]
    if dest.exists():
        state=torch.load(dest,map_location='cuda',weights_only=False);assert state['config']==config
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);cursor=state['step'];elapsed=state['elapsed'];history=state['history']
        torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        for d,b in banks.items():b.seen=state['seen'][d];b.rows=state['rows'][d]
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'encoder' in k or 'point' in k}
    batch_size=config['batch_size'];running=0.;loader=0.;start=time.monotonic();gradient_norm=None
    for step in range(cursor,steps):
        warm=source is not None and step<500
        for p in encoder:p.requires_grad_(not warm)
        tload=time.monotonic()
        if source:
            b=banks[source].batch(step,batch_size*3//4)
            rd=(replay or [source])[step%len(replay or [source])];r=banks[rd].batch(step,batch_size//4)
        else:
            domain=domains[step%len(domains)];b=banks[domain].batch(step//len(domains),batch_size);r=None
        if randomized:b=augment(b,step,noise)
        loader+=time.monotonic()-tload
        optimizer.zero_grad(set_to_none=True);l=loss(model,b,component)
        if r is not None:
            if replay:
                if randomized:r=augment(r,step,noise)
                l=.75*l+.25*loss(model,r,component)
            else:
                # Exactly matched supervised source rows in adaptation. Remaining real-only
                # compute uses consistency on the source image/geometry, without reading labels.
                with torch.no_grad():teacher=model(r)[0].detach()
                rr=augment(r,step)
                if component=='G':rr=dict(r,x=r['x']+torch.randn_like(r['x'])*.005);rr['x'][:,:,11:14]=r['x'][:,:,11:14]
                student=model(rr)[0]
                l=.75*l+.25*probability_consistency(student,teacher,component)
        if not torch.isfinite(l):raise RuntimeError('Nonfinite loss '+name)
        l.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();running+=float(l)
        if gradient_norm is None and not warm:gradient_norm=float(sum(p.grad.norm() for p in encoder if p.grad is not None))
        if (step+1)%250==0 or step+1==steps:
            entry=dict(step=step+1,loss=running/min(250,step+1-cursor),elapsed_seconds=elapsed+time.monotonic()-start)
            running=0.;history.append(entry)
            if (step+1)%1000==0 or step+1==steps:
                if vb:entry['validation']=validation(model,vb,component)
                state=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,elapsed=entry['elapsed_seconds'],history=history,
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),seen={d:b.seen for d,b in banks.items()},rows={d:b.rows for d,b in banks.items()})
                dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.tmp.pt');torch.save(state,temp);temp.replace(dest)
                print(name,step+1,'loss',round(entry['loss'],5),'seconds',round(entry['elapsed_seconds'],1),flush=True)
                write(OUT/f'active_training_{source or "pretraining"}_{os.getpid()}.json',dict(model=name,**entry))
    delta=sum(float((v.detach().cpu()-initial[k]).square().sum()) for k,v in model.state_dict().items() if k in initial)
    result=dict(config=config,checkpoint_sha256=sha(dest),actual_updates=steps,trainable_parameters=sum(p.numel() for p in model.parameters()),
        coverage={d:b.coverage() for d,b in banks.items()},encoder_parameter_squared_change=delta,encoder_gradient_norm=gradient_norm,
        elapsed_seconds=elapsed+time.monotonic()-start,loader_seconds=loader,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,history=history,
        direct_sources=all_domains,initialization_exposure=init,sparse_source_supervised_rows=steps*(batch_size*3//4) if source else None,
        note='Balanced observation-bag objective is not biological prevalence calibration; primary supervised source exposure matched across arms')
    write(receipt,result);print('completed',name,result['coverage'],flush=True)
    del model,optimizer,banks,vb;gc.collect();torch.cuda.empty_cache()

def adapt(source):
    for tag,pre,ext in [('real','D_real_'+source,None),('synthetic','D_synthetic',['synthetic'])]:
        fit('D_'+tag+'_adapt_'+source,'D',[source],8000,init=pre,source=source,replay=ext)
    for arm in ['C1','C2','C3','C4','C5','C6','C1short']:
        ginit={'C1':'G_real_'+source,'C2':'G_synthetic','C3':'G_fish','C4':'G_combined','C5':'G_multispecies','C6':'G_combined','C1short':None}[arm]
        iinit=('I_real_'+source if arm in ['C1','C3'] else ('I_randomized_'+source if arm=='C6' else 'I_synthetic')) if arm!='C1short' else None
        replay={'C1':None,'C2':['synthetic'],'C3':None,'C4':['synthetic','zebrafish'],'C5':['synthetic','zebrafish','synthetic','zebrafish','synthetic','zebrafish','ascidian','ascidian'],
            'C6':['synthetic','zebrafish'],'C1short':None}[arm]
        fit('G_'+arm+'_'+source,'G',[source],8000,init=ginit,source=source,replay=replay)
        fit('I_'+arm+'_'+source,'I',[source],8000,init=iinit,source=source,replay=['synthetic'] if arm in ['C2','C4','C5','C6'] else None,randomized=arm=='C6')
    write(OUT/f'adaptation_complete_{source}.json',dict(source=source,completed=True,created=now()))

def run(phase=None):
    if phase in ['44b6','6bba']:
        adapt(phase);return
    assert (OUT/'sanity_tests.json').exists(),'Run actual sanity/gradient/solver tests first'
    status('W420-W440',state='actual CUDA training')
    fit('D_synthetic','D',['synthetic'],12000)
    for source in ['44b6','6bba']:fit('D_real_'+source,'D',[source],12000)
    for tag,domains in [('synthetic',['synthetic']),('fish',['zebrafish']),('combined',['synthetic','zebrafish']),
                        ('multispecies',['synthetic','zebrafish','synthetic','zebrafish','synthetic','zebrafish','ascidian','ascidian'])]:
        fit('G_'+tag,'G',domains,20000)
    for source in ['44b6','6bba']:fit('G_real_'+source,'G',[source],20000)
    fit('I_synthetic','I',['synthetic'],20000,init='D_synthetic')
    for source in ['44b6','6bba']:
        fit('I_real_'+source,'I',[source],20000,init='D_real_'+source)
        fit('I_randomized_'+source,'I',['synthetic'],20000,init='D_synthetic',randomized=True,calibration_source=source)
    import subprocess,sys
    processes=[];logs=[]
    for source in ['44b6','6bba']:
        log=OUT/'logs'/f'train_adapt_{source}.log';log.parent.mkdir(parents=True,exist_ok=True)
        stream=log.open('a');logs.append(stream)
        processes.append(subprocess.Popen([sys.executable,'-m','multidata_training_v4','train','--phase',source],stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for stream in logs:stream.close()
    assert not any(codes),codes
    receipts=[read(p) for p in sorted((OUT/'models').glob('*.json'))]
    write(OUT/'training_receipts.json',receipts)
    write(OUT/'checkpoint_manifest.json',{p.stem:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted((OUT/'models').glob('*.pt'))})
    import pandas as pd
    pd.DataFrame([dict(model=r['config']['name'],**h) for r in receipts for h in r['history']]).to_csv(OUT/'learning_curves.csv',index=False)
    status('W440',state='all primary fits complete')
