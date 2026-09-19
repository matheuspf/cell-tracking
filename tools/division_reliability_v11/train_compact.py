"""Registered final-only C11 fit and one frozen source-fit mining pass."""
from pathlib import Path
import fcntl
import sys
import tempfile
import time
import random
import os
from .common import REPO,DATA,WORK,RESULTS,Blocked,read,write,sha,now


def mine(model,groups,normalizer,source,seed,folder,resources):
    import heapq
    import numpy as np
    from .actions import Bank,utilities
    from .policy import score_bank
    import shutil
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'frozen_checkpoint.pt').exists():shutil.copy2(folder.parent/'resume.pt',folder/'frozen_checkpoint.pt')
    signature=dict(checkpoint_sha256=sha(folder/'frozen_checkpoint.pt'),
                   source_bank_receipts={n:sha(groups.root/n/'receipt.json') for n in groups.clips},
                   policy_code_sha256=sha(Path(__file__).with_name('policy.py')),
                   normalizer_sha256=sha(folder.parents[1]/'linear/normalizer.json'))
    write(folder/'signature.json',signature,immutable=True)
    pool={};summary={}
    for clip in groups.clips:
        output=folder/clip;output.mkdir(parents=True,exist_ok=True)
        if (output/'receipt.json').exists():
            old=read(output/'receipt.json');summary[clip]=old
            if old['selected_groups']:pool[clip]=old['selected_groups']
            continue
        prediction=WORK/'predictions/C00'/source/str(seed)/clip
        if sha(prediction/'graph.npz')!=groups.receipts[clip]['graph_sha256']:raise Blocked('Mining C00 graph differs from the frozen fit bank')
        with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
        bank=Bank(nodes,edges,scores,100)
        parents=groups.receipts[clip]['group_parents']
        negative={parents[i]:i for i in groups.negative.get(clip,[])}
        known=set(parents);hard=[];unknown=[]
        for g,a,b in score_bank(bank,confidence,DATA/'train'/f'{clip}.zarr',normalizer,resources,model=model,progress=output/'progress.json'):
            score=float(utilities(bank,g,a,b,0).max());p=g['parent']
            if p in negative:
                item=(score,-p,negative[p])
                if len(hard)<256:heapq.heappush(hard,item)
                elif item>hard[0]:heapq.heapreplace(hard,item)
            elif p not in known:
                item=(score,-p)
                if len(unknown)<20:heapq.heappush(unknown,item)
                elif item>unknown[0]:heapq.heapreplace(unknown,item)
        if hard:pool[clip]=[g for _,_,g in sorted(hard,reverse=True)]
        summary[clip]=dict(selected_parent_groups=len(hard),supported_negative_pool=len(negative),
                           selected_groups=pool.get(clip,[]),
                           unknown_top_groups=[dict(raw_gain=s,parent=-p) for s,p in sorted(unknown,reverse=True)],
                           unknown_used_as_negative=False)
        write(output/'receipt.json',summary[clip])
    result=dict(pool=pool,summary=summary,criterion='maximum raw complete-fork gain per parent',
                per_clip_parent_cap=256,passes=1,source_fit_only=True,signature=signature,finished_utc=now())
    write(folder/'receipt.json',result,immutable=True);return pool


def run(source,seed,*,resume=True,defer_mining=False):
    from .readiness import require_production
    require_production('train-compact');lock=read(RESULTS/'execution_lock.json');horizon=lock['event_updates']
    clips=read(WORK/'source_partitions.json')[source]['fit'];bank=WORK/'banks'/source/str(seed)/'fit'
    linear=WORK/'fits'/source/str(seed)/'linear'
    if read(linear/'model.json')['status']!='fitted':raise Blocked('C01 must be fitted before C11 spending')
    folder=WORK/'fits'/source/str(seed)/'compact';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.json').exists():
        result=read(folder/'final.json')
        if result['weights_sha256']!=sha(folder/'final.pt') or result['lock_identity']!=lock['identity']:raise Blocked('Compact final integrity failure')
        return result
    implementation={n:sha(Path(__file__).with_name(n)) for n in ('train_compact.py','event_training.py','models.py','features.py','policy.py','actions.py','mining.py')}
    write(folder/'implementation.json',implementation,immutable=True)
    owner=(folder/'owner.lock').open('a+')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise Blocked('Compact cell already has an owner') from exc
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .resources import Resources,ROOT
    resources=Resources('event_mining',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[bank,linear,*[DATA/'train'/f'{n}.zarr' for n in clips],
                          *[WORK/'predictions/C00'/source/str(seed)/n for n in clips]],outputs=[folder,ROOT],
                  code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    from .event_training import Groups,mixed_update
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    from .upstream import learning_rate
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.init();random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    normalizer=read(linear/'normalizer.json')['values'];groups=Groups(bank,clips,normalizer)
    from .provenance import digest
    bank_identity=digest({clip:sha(bank/clip/'receipt.json') for clip in clips})
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed);step=0;hard=None;mined=False;last_saved=0;last_time=time.monotonic()
    if (folder/'resume.pt').exists():
        if not resume:raise Blocked('Existing compact checkpoint requires resume')
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        if state['lock_identity']!=lock['identity'] or state['source']!=source or state['seed']!=seed or state['bank_identity']!=bank_identity:raise Blocked('Compact resume ancestry differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];last_saved=step
        rng.bit_generator.state=state['sampler'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng'])
        hard=state['hard'];mined=state['mined'];groups.visits.update(state['visits'])
    import json
    history=folder/'history.jsonl'
    if history.exists():
        rows=[json.loads(line) for line in history.read_text().splitlines() if line]
        extra=[r for r in rows if r['step']>step]
        if extra:
            write(folder/'interrupted_updates'/f'{time.time_ns()}.json',extra,immutable=True)
            history.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=step))
    def save():
        nonlocal last_saved,last_time
        value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,source=source,seed=seed,
                   lock_identity=lock['identity'],bank_identity=bank_identity,sampler=rng.bit_generator.state,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                   python_rng=random.getstate(),hard=hard,mined=mined,visits=dict(groups.visits),scheduler=dict(step=step,horizon=horizon),scaler=None)
        torch.save(value,folder/'resume.tmp.pt');(folder/'resume.tmp.pt').replace(folder/'resume.pt');last_saved=step;last_time=time.monotonic()
    try:
        if not (folder/'resume.pt').exists():save()
        while step<horizon:
            if step==horizon//2 and not mined:
                save();mining=folder/'mining'
                if (mining/'receipt.json').exists():
                    receipt=read(mining/'receipt.json')
                    if sha(mining/'frozen_checkpoint.pt')!=receipt['signature']['checkpoint_sha256']:raise Blocked('Mining snapshot integrity failure')
                    if digest(receipt['signature']['source_bank_receipts'])!=bank_identity or receipt['signature']['normalizer_sha256']!=sha(linear/'normalizer.json'):
                        raise Blocked('Mining source observation/normalizer ancestry differs from the resumed fit')
                    frozen=torch.load(mining/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
                    if not all(torch.equal(v,frozen['model'][k]) for k,v in model.state_dict().items()):raise Blocked('Mining used different model parameters')
                    hard=receipt['pool']
                elif defer_mining:
                    result=dict(status='waiting_for_mining',source=source,seed=seed,step=step,horizon=horizon,durable_step=last_saved,updated_utc=now())
                    write(folder/'progress.json',result);return result
                else:hard=mine(model.eval(),groups,normalizer,source,seed,mining,resources)
                mined=True;model.train();save()
            input_started=time.monotonic()
            samples,selection=groups.batch(rng,step+1,horizon,hard)
            input_seconds=time.monotonic()-input_started
            lr=learning_rate(step+1,horizon,3e-4,200)
            with resources.lease(2*2**30) as lease:
                model.cuda().train()
                for state in optimizer.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor) and k!='step':state[k]=v.cuda()
                row=mixed_update(model,optimizer,samples,lr);torch.cuda.synchronize()
                model.cpu()
                for state in optimizer.state.values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor):state[k]=v.cpu()
                torch.cuda.empty_cache()
            step+=1
            row.update(step=step,selection=selection,gpu_lease_seconds=lease['seconds'],unique_group_visits=len(groups.visits),
                observation_cache_bytes=groups.cache_bytes,observation_cache_clips=len(groups.loaded),io_seconds=input_seconds)
            import json
            with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(folder/'progress.json',dict(status='running',step=step,horizon=horizon,mined=mined,pid=os.getpid(),updated_utc=now()))
            if step-last_saved>=250 or time.monotonic()-last_time>=300 or step in (horizon//10,horizon//2,horizon):save()
        torch.save(dict(model=model.state_dict(),step=step,source=source,seed=seed,lock_identity=lock['identity']),folder/'final.pt')
        result=dict(status='trained',source=source,seed=seed,updates=step,event_supervised_updates=step-horizon//10,
                    weights_sha256=sha(folder/'final.pt'),guard=guard,mining_passes=int(mined),unique_group_visits=len(groups.visits),finished_utc=now(),
                    lock_identity=lock['identity'],implementation_sha256=implementation)
        write(folder/'final.json',result,immutable=True);write(folder/'progress.json',result);return result
    except Exception as exc:
        write(folder/'progress.json',dict(status='failed',step=step,durable_step=last_saved,reason=str(exc),updated_utc=now()));raise
    finally:resources.close();owner.close()
