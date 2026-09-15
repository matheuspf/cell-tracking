"""Both-embryo residual-head fits using the frozen v1 query/feature populations."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import numpy as np
import torch

from .common import REPO,WORK,OUT,GPU_LOCK,read,write,save,sha,now

OLD=REPO/'work/cellpose-refine-v1'


def prepare():
    cfg=read(REPO/'configs/cellpose-refine-v1.json')
    native=read(OUT/'data-manifest.json');old=read(OLD/'feature-lock.json')
    fitted=set()
    for source in ('44b6','6bba'):
        m=read(OLD/'inputs'/f'train-{source}.json');fitted.update(r['dataset'] for r in m['source_files'])
    assert fitted==set(native['train'])
    new=copy.deepcopy(cfg)
    new.update(study='incumbent-comparison-20260914-both-embryo-refit',scope='Both-embryo resubstitution control on the existing frozen query sample',
               target_rule='Same public training population; evaluation may overlap fitting',source_training_clips='all199 public clips; existing v1 temporal/query sample')
    plan={'created_utc':now(),'script_sha256':sha(__file__),'feature_lock_sha256':sha(OLD/'feature-lock.json'),
          'base_config_sha256':sha(REPO/'configs/cellpose-refine-v1.json'),'config':new,
          'clip_population':sorted(fitted),'eligible_clips':199,'seeds':cfg['seeds'],'fixed_updates':3000,
          'label_sampling':'Reuse v1: five fixed jitter timepoints per clip plus supported natural queries on the412-frame panel. Same clip population as the incumbent, not identical temporal/label sampling or compute.',
          'selection':'Final fixed update only; freeze both fits before evaluating either. Existing cross-embryo fits retained.',
          'source_locks':{s:old['source_summaries'][s]['manifest_sha256'] for s in ('44b6','6bba')},
          'implementation':{str(p):sha(p) for p in (REPO/'tools/cellpose_refine/model.py',REPO/'tools/cellpose_refine/train.py',REPO/'tools/cellpose_refine/common.py')}}
    path=OUT/'refit-plan.json'
    if path.exists():
        previous=read(path);assert {k:v for k,v in previous.items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
    else:write(path,plan)
    print(path,flush=True)


def run():
    from tools.cellpose_refine.common import configure,config
    from tools.cellpose_refine.model import Refiner,sparse_offset_loss
    from tools.cellpose_refine.train import load_source
    from tools.detector_screen.cellpose_adapter import gpu_lock
    configure();torch.set_num_threads(2)
    plan=read(OUT/'refit-plan.json');assert plan['script_sha256']==sha(__file__)
    assert plan['feature_lock_sha256']==sha(OLD/'feature-lock.json')
    for p,h in plan['implementation'].items():assert sha(p)==h
    populations=[load_source(source,config())[0] for source in ('44b6','6bba')]
    arrays={k:np.concatenate([x[k] for x in populations]) for k in populations[0]};del populations
    cfg=plan['config'];tc=cfg['training'];receipts=[]
    for seed in plan['seeds']:
        path=WORK/'models/refiner-all'/f'seed-{seed}.pt';sidecar=path.with_suffix('.json')
        if sidecar.exists():
            r=read(sidecar);assert sha(path)==r['sha256'] and r['plan_sha256']==sha(OUT/'refit-plan.json');receipts.append(r);continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        torch.use_deterministic_algorithms(True);history=[];began=time.perf_counter()
        with gpu_lock(GPU_LOCK) as queued:
            torch.cuda.reset_peak_memory_stats()
            data={k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
            model=Refiner(cfg).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
            natural=torch.nonzero(data['natural'],as_tuple=False).flatten();jitter=torch.nonzero(~data['natural'],as_tuple=False).flatten()
            assert len(natural) and len(jitter)
            for step in range(1,tc['updates']+1):
                if step<=tc['warmup_updates']:factor=step/tc['warmup_updates']
                else:
                    progress=(step-tc['warmup_updates'])/(tc['updates']-tc['warmup_updates'])
                    factor=tc['final_learning_rate_fraction']+(1-tc['final_learning_rate_fraction'])*(1+math.cos(math.pi*progress))/2
                for group in optimizer.param_groups:group['lr']=tc['learning_rate']*factor
                half=tc['batch_size']//2
                selected=torch.cat([natural[torch.randint(len(natural),(half,),device='cuda')],jitter[torch.randint(len(jitter),(tc['batch_size']-half,),device='cuda')]])
                optimizer.zero_grad(set_to_none=True)
                prediction=model(data['feature'][selected],data['patch'][selected],data['geometry'][selected])
                loss=sparse_offset_loss(prediction,data['target_um'][selected],data['weight'][selected],tc['loss_beta_um'])
                assert torch.isfinite(loss)
                loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if step==1 or step%100==0:
                    item={'seed':seed,'update':step,'loss':float(loss.detach()),'gradient_norm':float(gradient)};history.append(item);print(json.dumps(item),flush=True)
            save(path,{'model':{k:v.cpu() for k,v in model.state_dict().items()},'config':cfg,'source':'all','seed':seed,'update':tc['updates'],'plan_sha256':sha(OUT/'refit-plan.json')})
            peak=torch.cuda.max_memory_reserved();query_counts={'natural':len(natural),'jitter_and_center':len(jitter)}
            model.to('cpu');del data,optimizer,model;torch.cuda.empty_cache()
        receipt={'created_utc':now(),'seed':seed,'source':'all','updates':tc['updates'],'sha256':sha(path),'plan_sha256':sha(OUT/'refit-plan.json'),
                 'queries':query_counts,'peak_reserved_bytes':peak,'seconds_excluding_queue':time.perf_counter()-began-queued,'gpu_queue_seconds':queued,
                 'history':history,'scope':'Population-matched resubstitution; selected temporal/query support, not all incumbent training observations. CPDINO ancestry remains unresolved.'}
        write(sidecar,receipt);write(OUT/f'refiner-all-seed-{seed}.json',receipt);receipts.append(receipt)
    write(OUT/'refit-models-lock.json',{'created_utc':now(),'plan_sha256':sha(OUT/'refit-plan.json'),'models':receipts,'new_scores_opened':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args();{'prepare':prepare,'run':run}[a.action]()
