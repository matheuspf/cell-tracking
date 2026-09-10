"""GT-free bounded prediction; no training caches or external data are inputs."""
from strong_tracker_v3.inference import deny_annotations
GUARD=deny_annotations()
import os
import sys
from pathlib import Path
EXTERNAL_GUARD={'blocked_reads':0,'installed_before_dependencies':not any(k in sys.modules for k in ['torch','numpy','zarr'])}
def _deny_external_inputs(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        text=str(Path(os.fsdecode(args[0])).absolute().resolve())
        if any(x in text for x in ['biohub-forum-archive','biohub-data-guide','/cache/real/','/cache/dreal/','/cache/synthetic/','/cache/zoo/']):
            EXTERNAL_GUARD['blocked_reads']+=1
            raise PermissionError('External/training data unavailable in inference: '+text)
sys.addaudithook(_deny_external_inputs)
import time
import torch
from scipy.special import logsumexp,expit
from scipy.optimize import linear_sum_assignment
from .common import *
from .models import Geometry,ImageEvent,Detector
from .proposals import build,PAIRS
from .adapters import temporal_patches,planes,native_refinement
from .decode import apply

ARMS=['C1short','C1','C2','C3','C4','C5','C6']

def model(name,root=None):
    root=OUT/'models' if root is None else Path(root)
    cls=Geometry if name.startswith('G_') else ImageEvent if name.startswith('I_') else Detector
    m=cls().cuda().eval();m.load_state_dict(torch.load(root/f'{name}.pt',map_location='cuda',weights_only=False)['model']);return m

@torch.no_grad()
def geometry(m,x):
    n=len(x);logits=np.full((n,16),-1e4,np.float32);logits[:,0]=0;links=np.zeros((n,6),np.float32);gate=np.zeros(n,np.float32)
    for start in range(0,n,256):
        xx=torch.as_tensor(x[start:start+256],device='cuda');p,c=m.encode(xx);gg=m.gate(c).squeeze(-1)
        links[start:start+len(xx)]=m.link(torch.cat([p,c[:,None].expand(-1,6,-1)],-1)).squeeze(-1).cpu().numpy()
        gate[start:start+len(xx)]=gg.sigmoid().cpu().numpy()
        keep=gg>0
        if keep.any():logits[start+torch.where(keep)[0].cpu().numpy()]=m.score_encoded(p[keep],c[keep],xx[keep])[0].cpu().numpy()
    return logits,links,gate

@torch.no_grad()
def image_scores(m,x,anchors,candidates,selected,patches,valid,reverse):
    encoded=[]
    for start in range(0,len(patches),128):
        pp=torch.as_tensor(patches[start:start+128],device='cuda').float()/255
        vv=torch.as_tensor(valid[start:start+128],device='cuda')
        encoded.append(m.encode(pp[:,None],vv[:,None])[:,0])
    embedding=torch.cat(encoded);result=np.full((len(anchors),16),-1e4,np.float32);result[:,0]=0
    for start in range(0,len(selected),512):
        rows=selected[start:start+512];ix=np.c_[anchors[rows],np.maximum(candidates[rows],0)]
        ind=reverse[ix];ind=np.maximum(ind,0)
        z=embedding[torch.as_tensor(ind,device='cuda')];xx=torch.as_tensor(x[rows],device='cuda')
        result[rows]=m.score_encoded(z,xx)[0].cpu().numpy()
    return result

def event_choices(anchors,candidates,logits,gate,margin=4.):
    best=np.argmax(logits[:,1:],axis=1)+1;rest=logits.copy();score=rest[np.arange(len(rest)),best].copy()
    rest[np.arange(len(rest)),best]=-1e4;gain=score-logsumexp(rest,axis=1)-margin
    selected=np.flatnonzero((gain>0)&(gate>.5))
    pairs=PAIRS[best[selected]-1];ev=np.c_[anchors[selected],candidates[selected,pairs[:,0]],candidates[selected,pairs[:,1]]]
    good=np.all(ev>=0,axis=1)
    return ev[good],gain[selected][good]

def relink(nodes,anchors,candidates,links):
    edges=[]
    for t in np.unique(nodes[anchors,1]):
        rows=np.flatnonzero(nodes[anchors,1]==t);a=anchors[rows];nxt=np.flatnonzero(nodes[:,1]==t+1)
        rev={int(v):i for i,v in enumerate(nxt)}
        # One continuation or an explicit null; event decoder adds a second daughter later.
        cost=np.zeros((len(a),len(nxt)+len(a)),np.float32);cost[:,:len(nxt)]=1e4
        for i,row in enumerate(rows):
            for j,c in enumerate(candidates[row]):
                if c>=0:cost[i,rev[int(c)]]=-links[row,j]
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if j<len(nxt) and cost[i,j]<0:edges.append((nodes[a[i],0],nodes[nxt[j],0]))
    return np.array(edges,np.int64).reshape(-1,2)

@torch.no_grad()
def refine(nodes,image,spacing,detector):
    out=nodes.copy();center=[];shifts=[]
    for t in np.unique(nodes[:,1]):
        ix=np.flatnonzero(nodes[:,1]==t);frame=np.asarray(image[int(t)],np.float32)
        lo,hi=np.quantile(frame[::2,::4,::4],[.02,.995]);im=np.clip((frame-lo)/max(hi-lo,1),0,1)
        p=planes(im,nodes[ix,2:]);pred=[];prob=[]
        for start in range(0,len(p),256):
            pp=torch.as_tensor(p[start:start+256],device='cuda');score,offset=detector({'patch':pp});pred.append(offset.cpu().numpy());prob.append(score.sigmoid().cpu().numpy())
        # Training stores the native-voxel displacement divided by three.
        # Invert that normalization before applying the frozen physical cap.
        delta=native_refinement(np.concatenate(pred),spacing);pp=np.concatenate(prob);delta[pp<.15]=0
        pts=np.clip(np.rint(nodes[ix,2:]+delta),0,np.array(image.shape[1:])-1).astype(np.int64)
        out[ix,2:]=pts;center.extend(pp);shifts.extend(np.linalg.norm((pts-nodes[ix,2:])*spacing,axis=1))
    return out,dict(mean_center_score=float(np.mean(center)),changed_centers=int(np.count_nonzero(shifts)),
        mean_realized_shift_um=float(np.mean(shifts)),max_realized_shift_um=float(np.max(shifts)),node_count_unchanged=True)

def policies():
    return ['C0','decoder_only','original_objective_C4',*ARMS,'C4_Gonly','C4_m2','C4_m6','C1_relinked','C4_relinked','D_real','D_synthetic','D_synthetic_C4']

def transform(nodes,edges,image,spacing,source,variant,models_root,calibration,evidence_callback=None):
    """Single graph fresh-image path. Source fit is explicit, never inferred from a name."""
    assert source in ['44b6','6bba']
    if not len(nodes):return nodes.copy(),edges.copy(),dict(identity=True,reason='empty_detection_population',weights_loaded=[])
    if variant in ['C0','decoder_only','original_objective_C4']:
        return nodes.copy(),edges.copy(),dict(identity=True,weights_loaded=[])
    arm='C4' if variant in ['C4_Gonly','C4_m2','C4_m6','C4_relinked','D_synthetic_C4'] else 'C1' if variant in ['D_real','D_synthetic','C1_relinked'] else variant
    loaded=[];detector_stats=None
    if variant.startswith('D_'):
        det='real' if variant=='D_real' else 'synthetic';key=f'D_{det}_adapt_{source}'
        detector=model(key,models_root);nodes,detector_stats=refine(nodes,image,spacing,detector);loaded.append(key);del detector
    a,c,x=build(nodes[:,1],nodes[:,2:].astype(float));key=f'G_{arm}_{source}';gm=model(key,models_root)
    gl,ll,gate=geometry(gm,x);loaded.append(key);del gm
    logits=gl/calibration['models'][key]['temperature']
    if variant!='C4_Gonly':
        sel=np.flatnonzero(gate>.5);used=np.unique(np.r_[a[sel],c[sel].reshape(-1)]);used=used[used>=0]
        if len(used):
            pp,vv=temporal_patches(image,nodes[used,1],nodes[used,2:],spacing);rev=np.full(len(nodes),-1,int);rev[used]=np.arange(len(used))
            key=f'I_{arm}_{source}';im=model(key,models_root);il=image_scores(im,x,a,c,sel,pp,vv,rev);loaded.append(key);del im
            logits=.5*(logits+il/calibration['models'][key]['temperature'])
    if variant.startswith('D_') or variant.endswith('_relinked'):edges=relink(nodes,a,c,ll)
    ev,gain=event_choices(a,c,logits,gate,{'C4_m2':2.,'C4_m6':6.}.get(variant,4.));ee,stats=apply(nodes,edges,ev,gain)
    if evidence_callback is not None:evidence_callback(dict(anchors=a,candidates=c,gate=gate,margin_events=ev))
    selected=np.flatnonzero(gate>.5)
    budget=dict(parent_observations=len(a),after_gate=len(selected),daughter_options=int((c[selected]>=0).sum()),
        pair_options=int(np.sum((c[selected][:,PAIRS[:,0]]>=0)&(c[selected][:,PAIRS[:,1]]>=0))),after_margin=len(ev),**stats)
    return nodes,ee,dict(weights_loaded=loaded,decoder=stats,detector=detector_stats,annotation_reads=0,candidate_budget=budget)

def one(row,models,calibration):
    import zarr
    name=row['dataset'];start=time.monotonic();receipt=OUT/'prediction_receipts'/f'{name}.json'
    if receipt.exists():
        d=read(receipt)
        for v,h in d['hashes'].items():assert sha(OUT/'predictions'/v/f'{name}.npz')==h
        return d
    g=arrays(V3/'selected_predictions'/f'{name}.npz');nodes,edges=g['nodes'],g['edges']
    # One bounded ~0.8 GiB volume avoids repeatedly decompressing the same clip
    # for the detector controls and each changed-coordinate association rebuild.
    image=zarr.open_group(row['image_path'],mode='r')['0'][:]
    source='6bba' if row['embryo']=='44b6' else '44b6'
    a,c,x=build(nodes[:,1],nodes[:,2:].astype(float));gs={};metrics=[];results={};union=[];margin_events={}
    for arm in ARMS:
        l,links,gate=geometry(models['G_'+arm+'_'+source],x);gs[arm]=(l,links,gate);union.extend(np.flatnonzero(gate>.5))
    save(OUT/'candidate_evidence'/f'{name}.npz',anchors=a,candidates=c,
        **{f'gate_{arm}':gs[arm][2] for arm in ARMS})
    selected=np.unique(union).astype(int)
    used=np.unique(np.r_[a[selected],c[selected].reshape(-1)]);used=used[used>=0]
    if len(used):patches,valid=temporal_patches(image,nodes[used,1],nodes[used,2:],row['physical_scale'])
    reverse=np.full(len(nodes),-1,np.int64);reverse[used]=np.arange(len(used))
    for arm in ARMS:
        gl,links,gate=gs[arm];sel=np.flatnonzero(gate>.5)
        il=image_scores(models['I_'+arm+'_'+source],x,a,c,sel,patches,valid,reverse) if len(used) else gl
        tempg=calibration['models']['G_'+arm+'_'+source]['temperature'];tempi=calibration['models']['I_'+arm+'_'+source]['temperature']
        logits=.5*(gl/tempg+il/tempi);ev,gain=event_choices(a,c,logits,gate)
        margin_events[arm]=ev
        ee,dd=apply(nodes,edges,ev,gain);results[arm]=(nodes,ee)
        budget=dict(parent_observations=len(a),after_gate=len(sel),daughter_options=int((c[sel]>=0).sum()),
            pair_options=int(np.sum((c[sel][:,PAIRS[:,0]]>=0)&(c[sel][:,PAIRS[:,1]]>=0))))
        metrics.append(dict(variant=arm,**budget,after_margin=len(ev),**dd))
        if arm in ['C1','C4']:
            re=relink(nodes,a,c,links);re,stats=apply(nodes,re,ev,gain);results[arm+'_relinked']=(nodes,re)
            metrics.append(dict(variant=arm+'_relinked',**budget,after_margin=len(ev),rebuilt_associations=True,**stats))
        if arm=='C4':
            for margin in [2.,6.]:
                evm,gainm=event_choices(a,c,logits,gate,margin);em,ddm=apply(nodes,edges,evm,gainm);variant=f'C4_m{int(margin)}';results[variant]=(nodes,em)
                metrics.append(dict(variant=variant,**budget,after_margin=len(evm),**ddm))
            evg,gg=event_choices(a,c,gl/tempg,gate);eg,ddg=apply(nodes,edges,evg,gg);results['C4_Gonly']=(nodes,eg)
            metrics.append(dict(variant='C4_Gonly',**budget,after_margin=len(evg),**ddg))
    results['decoder_only']=(nodes,apply(nodes,edges,np.empty((0,3),int),[])[0])
    results['original_objective_C4']=(nodes,edges.copy())
    for v in ['C0','decoder_only','original_objective_C4']:
        metrics.append(dict(variant=v,parent_observations=len(a),after_gate=0,after_margin=0,proposed=0,selected=0))
    detector_stats={}
    for det in ['real','synthetic']:
        nn,ds=refine(nodes,image,row['physical_scale'],models['D_'+det+'_adapt_'+source]);detector_stats[det]=ds
        aa,cc,xx=build(nn[:,1],nn[:,2:].astype(float))
        for arm in (['C1','C4'] if det=='synthetic' else ['C1']):
            gl,ll,gate=geometry(models['G_'+arm+'_'+source],xx);sel=np.flatnonzero(gate>.5)
            uu=np.unique(np.r_[aa[sel],cc[sel].reshape(-1)]);uu=uu[uu>=0]
            if len(uu):
                pp,vv=temporal_patches(image,nn[uu,1],nn[uu,2:],row['physical_scale']);rev=np.full(len(nn),-1,int);rev[uu]=np.arange(len(uu))
                il=image_scores(models['I_'+arm+'_'+source],xx,aa,cc,sel,pp,vv,rev)
            else:il=gl
            logits=.5*(gl/calibration['models']['G_'+arm+'_'+source]['temperature']+il/calibration['models']['I_'+arm+'_'+source]['temperature'])
            ev,gain=event_choices(aa,cc,logits,gate);ee=relink(nn,aa,cc,ll);ee,dd=apply(nn,ee,ev,gain)
            variant='D_'+det+('_C4' if arm=='C4' else '');results[variant]=(nn,ee)
            metrics.append(dict(variant=variant,parent_observations=len(aa),after_gate=len(sel),after_margin=len(ev),
                daughter_options=int((cc[sel]>=0).sum()),pair_options=int(np.sum((cc[sel][:,PAIRS[:,0]]>=0)&(cc[sel][:,PAIRS[:,1]]>=0))),
                rebuilt_associations=True,**dd))
    save(OUT/'margin_candidates'/f'{name}.npz',**margin_events)
    from strong_tracker_v3.common import validate
    hashes={}
    for variant,(nn,ee) in results.items():
        validate(nn,ee,row['image_shape']);p=OUT/'predictions'/variant/f'{name}.npz';save(p,nodes=nn,edges=ee);hashes[variant]=sha(p)
    d=dict(dataset=name,source_model=source,seconds=time.monotonic()-start,hashes=hashes,candidate_budget=metrics,detector=detector_stats,
        annotation_reads=0,blocked_annotation_reads=GUARD['blocked_reads'],external_dataset_reads=0,blocked_external_reads=EXTERNAL_GUARD['blocked_reads'],
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    write(receipt,d);print('predicted',name,'seconds',round(d['seconds'],1),'union gate',len(selected),flush=True);return d

def predict_source(source,cal):
    names=[f'{c}_{arm}_{source}' for c in ['G','I'] for arm in ARMS]+[f'D_{det}_adapt_{source}' for det in ['real','synthetic']]
    models={n:model(n) for n in names}
    for row in inputs():
        if row['embryo']!=source:one(row,models,cal)
    write(OUT/f'prediction_source_receipt_{source}.json',dict(source=source,completed=True,annotation_guard=GUARD,external_guard=EXTERNAL_GUARD))

def run(phase=None):
    torch.set_num_threads(2);cal=read(OUT/'calibration.json')
    if phase in ['44b6','6bba']:
        predict_source(phase,cal);return
    import pandas as pd
    config=dict(variants=policies(),candidate_daughters=6,pair_options=15,gate=.5,primary_margin=4,
        source_direction={'44b6':'6bba','6bba':'44b6'},calibration_sha256=sha(OUT/'calibration.json'),
        inference_code_sha256={name:sha(Path(__file__).parent/name) for name in ['infer.py','models.py','adapters.py','proposals.py','decode.py']},
        model_hashes=read(OUT/'checkpoint_manifest.json'),before_target_outcomes=True)
    config_path=OUT/'inference_config.json'
    if config_path.exists():assert {k:v for k,v in read(config_path).items() if k!='created'}==config
    else:write(config_path,dict(created=now(),**config))
    status('W450-W460',state='freezing all planned complete predictions before target outcomes')
    import subprocess
    processes=[]
    for source in ['44b6','6bba']:
        processes.append(subprocess.Popen([sys.executable,'-m','multidata_training_v4','infer','--phase',source]))
    codes=[p.wait() for p in processes];assert not any(codes),codes
    receipts=[read(p) for p in sorted((OUT/'prediction_receipts').glob('*.json'))]
    pd.DataFrame([dict(dataset=r['dataset'],**b) for r in receipts for b in r['candidate_budget']]).to_csv(OUT/'candidate_budget.csv',index=False)
    write(OUT/'prediction_generation_receipt.json',dict(samples=len(receipts),annotation_reads=0,blocked_reads=GUARD['blocked_reads'],
        external_guard=EXTERNAL_GUARD,annotation_guard=GUARD,
        source_workers=[read(OUT/f'prediction_source_receipt_{s}.json') for s in ['44b6','6bba']],
        total_clip_seconds=sum(r['seconds'] for r in receipts),calibration_sha256=sha(OUT/'calibration.json')))
