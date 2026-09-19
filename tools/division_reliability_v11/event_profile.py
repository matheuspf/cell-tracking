"""Real legal source groups, source crop parity, and compact/linear pilot fits."""
from pathlib import Path
import sys
import tempfile
import fcntl
import time
from .common import REPO,WORK,DATA,RESULTS,Blocked,read,write,now,sha


def run(source,seed,clip):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Fit source required')
    prediction=WORK/'source_inference-overfit'/source/str(seed)/clip
    folder=WORK/'event_profile'/source/str(seed);folder.mkdir(parents=True,exist_ok=True)
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    lock=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock').open('a+')
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,DATA/'train'/f'{clip}.zarr',DATA/'train'/f'{clip}.geff'],outputs=[folder],
                  code_roots=[REPO/'tools',REPO/'handover',REPO/'work/annotation-selection-v1/official',
                              Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    from .actions import Bank
    from .data import labels
    from .features import NAMES,action_features
    from .linear import fit
    from .models import CompactPolicy,supported_losses
    from pipeline_error_training.labels import SourceLabels
    from pipeline_error_training.crops import Images,sample_native,compact_view
    from pipeline_error_training.compact_inference_crops import sample
    torch.set_num_threads(4);torch.manual_seed(seed)
    tic=time.monotonic()
    with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
    bank=Bank(nodes,edges,scores,100)
    gt,gt_edges=labels(DATA/'train'/f'{clip}.geff')
    evaluator=SourceLabels(nodes,edges,gt,gt_edges,[1.625,.40625,.40625])
    # Label-guided *source diagnostic selection*, never new deployment anchors.
    possible=set()
    for roles in evaluator.roles.values():
        if roles:
            parents,_=roles
            possible.update(parents)
            possible.update(j for p in parents for j in bank.near[p])
    possible.update(evaluator.matches)
    selected={};examined=0
    for p in sorted(possible & bank.expanded):
        group=bank.parent(p);examined+=1
        if not group['complete'] or not group['forks']:continue
        risks=[evaluator.decision(a)['metric_fork_target'] for a in group['forks']]
        target=1 if 1 in risks else 0 if all(y==0 for y in risks) else -1
        if target in (0,1) and target not in selected:selected[target]=(group,risks)
        if len(selected)==2:break
    result=dict(status='started',source=source,seed=seed,clip=clip,guard=guard,examined_source_parents=examined,
                supported_parent_types=sorted(selected),source_label_guided_fixture=True,not_population_prevalence=True,
                production_fit=False,gpu_lease_seconds=0.,feature_names=list(NAMES))
    if len(selected)!=2:
        result.update(status='blocked',reason='No positive and completely supported negative group in the actual clean pilot bank')
        write(folder/'receipt.json',result);raise Blocked(result['reason'])
    images=Images(DATA/'train'/f'{clip}.zarr',max_frames=9)
    samples=[];parity=[]
    for label,(group,risks) in sorted(selected.items()):
        indices=sorted({group['parent'],*[j for d in group['forks'] for j in d.event[1:3]]})
        patch,valid=sample(images,nodes,bank.pred,bank.succ,indices)
        # Real source parity at actual prediction centers, no historical crop cache.
        for k in (0,len(indices)-1):
            reference,masks=sample_native(images,nodes,bank.pred,bank.succ,indices[k])
            if not np.array_equal(compact_view(reference),patch[k]) or not np.array_equal(masks[1:4],valid[k]):
                raise Blocked('Compact crop adapter differs from original native crop')
            parity.append(dict(node_id=int(nodes[indices[k],0]),pixels_exact=True,masks_exact=True))
        index={p:i for i,p in enumerate(indices)}
        x,parent=action_features(bank,group,confidence,patch.astype(np.float32)/255,valid,index)
        samples.append(dict(group=group,risks=np.asarray(risks),patch=patch,valid=valid,indices=index,x=x,parent=parent))
    start=time.monotonic()
    linear=fit(np.stack([s['parent'] for s in samples]),[s['x'] for s in samples],
               [s['risks'] for s in samples],[1.,1.],[True,True])
    result.update(linear_seconds=time.monotonic()-start,linear=linear,crop_parity=parity,
                  actual_legal_actions=[len(s['risks']) for s in samples])
    fcntl.flock(lock,fcntl.LOCK_EX);lease=time.monotonic()
    try:
        free,total=torch.cuda.mem_get_info();allowed=min(20*2**30-(total-free),free-2*2**30)
        if allowed<2*2**30:raise Blocked('No compact pilot GPU allowance')
        torch.cuda.set_per_process_memory_fraction(allowed/total)
        model=CompactPolicy(len(samples[0]['parent']),len(NAMES),len(NAMES)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        times=[];denominators=[]
        for step in range(3):
            start=time.monotonic();optimizer.zero_grad(set_to_none=True);losses=[]
            for s in samples:
                z=model.encoder(torch.tensor(s['patch'],device='cuda',dtype=torch.float32)/255,torch.tensor(s['valid'],device='cuda'))
                p=s['indices'][s['group']['parent']]
                daughters=sorted({s['indices'][j] for d in s['group']['forks'] for j in d.event[1:3]})
                a=model.parent_score(z[p],z[daughters],torch.tensor(s['parent'],device='cuda'))
                ii=torch.tensor([[s['indices'][d.event[k]] for k in (1,2)] for d in s['group']['forks']],device='cuda')
                b=model.action_scores(z[p].expand(len(ii),-1),z[ii[:,0]],z[ii[:,1]],torch.tensor(s['x'],device='cuda'))
                occ,rank,denom=supported_losses(a,b,torch.tensor(s['risks'],device='cuda'))
                losses.extend([occ,rank]);denominators.append(denom)
            loss=torch.stack(losses).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
            if time.monotonic()-lease>55:raise Blocked('Compact pilot exceeded short lease allowance')
        result.update(status='passed',two_group_update_seconds=times,loss=float(loss.detach()),gradient_norm=float(norm),
                      denominators=denominators,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      limitation='Two real occurrence/action groups only; this is not a production 32-group mixed/identity/mining throughput measurement.')
    finally:
        result['gpu_lease_seconds']=time.monotonic()-lease;fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        result.update(wall_seconds=time.monotonic()-tic,finished_utc=now());write(folder/'receipt.json',result)
    return result
