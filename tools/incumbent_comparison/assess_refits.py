"""Frozen resubstitution comparison against the existing400-frame head results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .common import REPO,WORK,OUT,GPU_LOCK,read,write,sha,now

OLD=REPO/'work/cellpose-refine-v1'
SCREEN=REPO/'work/detector-screen-20260914'


def prepare():
    models=read(OUT/'refit-models-lock.json')
    rows=[r for r in read(OLD/'inputs/inference.json')['rows'] if r['role']=='assessment']
    assert len(rows)==400
    plan={'created_utc':now(),'script_sha256':sha(__file__),'model_lock_sha256':sha(OUT/'refit-models-lock.json'),
          'inference_manifest_sha256':sha(OLD/'inputs/inference.json'),'frames':400,'models':models['models'],
          'rules':'Integer matching at1-7um; same400 images as v1. Preserve proposal counts/confidence/order. Freeze both models and all predictions before the new evaluation.',
          'scope':'Both-embryo resubstitution on frozen selected query support; exposure comparison, not unseen-embryo performance.'}
    path=OUT/'refit-assessment-plan.json'
    if path.exists():
        old=read(path);assert {k:v for k,v in old.items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
    else:write(path,plan)


def infer():
    from tools.cellpose_refine.model import Refiner
    from tools.cellpose_refine.infer import refine_cached,install_inference_guard
    from tools.detector_screen.cellpose_adapter import gpu_lock
    install_inference_guard()
    plan=read(OUT/'refit-assessment-plan.json');assert plan['script_sha256']==sha(__file__)
    assert plan['model_lock_sha256']==sha(OUT/'refit-models-lock.json')
    assert plan['inference_manifest_sha256']==sha(OLD/'inputs/inference.json')
    rows=[r for r in read(OLD/'inputs/inference.json')['rows'] if r['role']=='assessment']
    cfg=read(REPO/'configs/cellpose-refine-v1.json');receipts=[]
    for model_record in plan['models']:
        seed=model_record['seed'];path=WORK/'models/refiner-all'/f'seed-{seed}.pt';assert sha(path)==model_record['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);model=Refiner(state['config']);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
        for begin in range(0,len(rows),20):
            with gpu_lock(GPU_LOCK):
                model.to('cuda')
                try:
                    for row in rows[begin:begin+20]:
                        cache=Path(row['cache_path']);r=read(cache.with_suffix('.json'))
                        assert sha(cache)==r['cache_sha256'] and r['annotation_read']=='none'
                        assert sha(row['prediction_path'])==row['prediction_sha256']
                        with np.load(cache,allow_pickle=False) as f,np.load(row['prediction_path'],allow_pickle=False) as p:
                            queries=p['centers_zyx'];np.testing.assert_array_equal(queries,f['queries'])
                            arrays=refine_cached(model,queries,f['feature'],f['patch'],f['geometry'],row['shape'],cfg)
                            dest=WORK/'assessment-predictions'/f'seed-{seed}'/f"{row['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(dest,**arrays,scores=p['scores'])
                        receipts.append({'seed':seed,'key':row['key'],'sha256':sha(dest),'count':len(queries)})
                finally:model.to('cpu');torch.cuda.empty_cache()
            print(json.dumps({'seed':seed,'completed_frames':min(begin+20,len(rows)),'total_frames':len(rows)}),flush=True)
    write(OUT/'refit-assessment-predictions-lock.json',{'created_utc':now(),'plan_sha256':sha(OUT/'refit-assessment-plan.json'),'predictions':receipts})


def evaluate():
    import logging,warnings
    logging.disable(logging.WARNING);warnings.filterwarnings('ignore')
    from scipy.spatial.distance import cdist
    from tools.detector_screen.evaluate import matches
    plan=read(OUT/'refit-assessment-plan.json');lock=read(OUT/'refit-assessment-predictions-lock.json')
    assert plan['script_sha256']==sha(__file__) and lock['plan_sha256']==sha(OUT/'refit-assessment-plan.json')
    for r in lock['predictions']:assert sha(WORK/'assessment-predictions'/f"seed-{r['seed']}"/f"{r['key']}.npz")==r['sha256']
    opened=now();gt=read(SCREEN/'evaluation/ground_truth.json')
    rows=[r for r in read(OLD/'inputs/inference.json')['rows'] if r['role']=='assessment'];details=[]
    for record in plan['models']:
        seed=record['seed']
        for row in rows:
            with np.load(WORK/'assessment-predictions'/f'seed-{seed}'/f"{row['key']}.npz",allow_pickle=False) as f:centers=f['centers_zyx']
            truth=np.asarray(gt['frames'][row['key']]['gt_nodes'],np.int64).reshape(-1,5)
            mappings={radius:matches(centers,truth,row['time'],float(radius)) for radius in range(1,8)}
            d=cdist(truth[:,2:]*[1.625,.40625,.40625],truth[:,2:]*[1.625,.40625,.40625])
            close={}
            for radius in (7,14):
                pairs=list(zip(*np.where(np.triu(d<=radius,k=1))))
                close[str(radius)]={'pairs':len(pairs),'both_at3':sum(truth[i,0] in mappings[3].values() and truth[j,0] in mappings[3].values() for i,j in pairs)}
            details.append({'seed':seed,'key':row['key'],'embryo':row['embryo'],'gt':len(truth),'candidates':len(centers),'matches':{str(k):len(v) for k,v in mappings.items()},'close':close})
    groups=[]
    for seed in [r['seed'] for r in plan['models']]:
        for embryo in ('pooled','44b6','6bba'):
            rs=[r for r in details if r['seed']==seed and (embryo=='pooled' or r['embryo']==embryo)]
            total=sum(r['gt'] for r in rs);counts={str(k):sum(r['matches'][str(k)] for r in rs) for k in range(1,8)}
            groups.append({'seed':seed,'embryo':embryo,'frames':len(rs),'gt':total,'candidates':sum(r['candidates'] for r in rs),'matches':counts,
                           'recall':{k:v/total for k,v in counts.items()},'close':{k:{name:sum(r['close'][k][name] for r in rs) for name in ('pairs','both_at3')} for k in ('7','14')}})
    result={'created_utc':now(),'evaluation_opened_utc':opened,'predictions_lock_sha256':sha(OUT/'refit-assessment-predictions-lock.json'),
            'groups':groups,'per_frame':details,'previous_reference_metrics_sha256':sha(REPO/'results/cellpose-refine-v1/metrics.json'),
            'scope':plan['scope']+' Detector evidence; complete-clip competition score is a separate experiment.'}
    write(OUT/'refit-assessment.json',result);print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','infer','evaluate'));a=p.parse_args();{'prepare':prepare,'infer':infer,'evaluate':evaluate}[a.action]()
