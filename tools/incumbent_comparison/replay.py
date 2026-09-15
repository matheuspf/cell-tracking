"""Replay the shipped evaluator, preserving its 5um greedy matcher and metric."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .common import REPO,WORK,OUT,WEIGHTS,GPU_LOCK,read,write,sha,now,modules,source_hashes,verify_sources
from .data import dataset


def prepare():
    data=read(OUT/'data-manifest.json')
    plan={'created_utc':now(),'weight':str(WEIGHTS/'edge_predictor_best.pth'),
          'weight_sha256':sha(WEIGHTS/'edge_predictor_best.pth'),'data_manifest_sha256':sha(OUT/'data-manifest.json'),
          'script_sha256':sha(__file__),'native_source':source_hashes(),'batch_size':8,'shuffle':False,'workers':4,
          'monitor':data['monitor'],'monitor_embryos':['44b6'],'cap_batches':200,'also_evaluate_full_monitor':True,
          'dataset':'Original FrameWindowDataset and window eligibility; native normalization, FP16 input round trip, no augmentation.',
          'matching':'Original greedy nearest-GT matching, max distance5um, logit threshold0.3, pool5um. This is deliberately not official optimal matching.',
          'metric':'Original edge accuracy times detected-node recall; report actual edge confusion denominators and detection counts.',
          'published_reference':{'epoch':381,'accuracy':0.9998664993,'node_recall':0.9781053544,'product':0.9779747766406395},
          'limits':['Released source omits eval_max_batches; the capped prefix is implemented explicitly.','Exact historic batch ordering/runtime are not independently attested.','The 40-clip monitor is entirely in training and entirely from one embryo.'],
          'gpu_sharing':'One original batch per shared-lock acquisition; model moved to CPU between batches. Evaluation mode makes aggregation equivalent to the original full loop.'}
    path=OUT/'replay-plan.json'
    if path.exists():
        old=read(path);assert {k:v for k,v in old.items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
    else:write(path,plan)
    print(path,flush=True)


def run():
    training,prediction=modules()
    from tools.detector_screen.cellpose_adapter import gpu_lock
    plan=read(OUT/'replay-plan.json');data=read(OUT/'data-manifest.json')
    assert plan['script_sha256']==sha(__file__) and plan['data_manifest_sha256']==sha(OUT/'data-manifest.json')
    assert sha(plan['weight'])==plan['weight_sha256'];verify_sources(plan['native_source'])
    ds=dataset(plan['monitor'],max_nodes=data['max_nodes_all'])
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=4,persistent_workers=True,prefetch_factor=2)
    model,window,downsample=prediction.load_model(Path(plan['weight']),torch.device('cpu'))
    assert window==2 and tuple(downsample)==(1,4,4)
    counters={'correct':0,'total':0,'gt_matched':0,'gt_total':0,'proposals':0,'pairs':0,'loss_sum':0.}
    original_pair=training._evaluate_pair;original_match=training.detect_and_match
    def observed_pair(*args,**kwargs):
        loss,correct,total=original_pair(*args,**kwargs)
        counters['correct']+=correct;counters['total']+=total;counters['pairs']+=1;counters['loss_sum']+=loss
        return loss,correct,total
    def observed_match(*args,**kwargs):
        result=original_match(*args,**kwargs)
        counters['gt_total']+=int(args[2].sum().item())
        counters['gt_matched']+=sum(int((m>=0).sum().item()) for m in result[3])
        counters['proposals']+=int(result[2].sum().item())
        return result
    training._evaluate_pair=observed_pair;training.detect_and_match=observed_match
    started=time.perf_counter();queue_total=0.;peak=0;seen=[]
    def result(batches):
        accuracy=counters['correct']/max(counters['total'],1);recall=counters['gt_matched']/max(counters['gt_total'],1)
        return {'created_utc':now(),'plan_sha256':sha(OUT/'replay-plan.json'),'batches':batches,'window_count':len(seen),'clip_count':len({r['dataset'] for r in seen}),
                'counts':dict(counters),'accuracy':accuracy,'node_recall':recall,'product':accuracy*recall,
                'product_delta_from_published':accuracy*recall-plan['published_reference']['product'],
                'mean_pair_loss':counters['loss_sum']/max(counters['pairs'],1),
                'seconds_excluding_gpu_queue':time.perf_counter()-started-queue_total,'gpu_queue_seconds':queue_total,
                'peak_reserved_bytes':peak,'runtime_versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','tracksdata','zarr')},
                'window_order':list(seen),'scope':'Replay of training-exposed monitor; this is not the official competition score.'}
    for index,batch in enumerate(loader):
        n=len(batch['imgs']);offset=index*8
        for meta,vm in ds._data[offset:offset+n]:seen.append({'dataset':vm.zarr_path.stem,'time':meta['t_start']})
        with gpu_lock(GPU_LOCK) as waited:
            queue_total+=waited;torch.cuda.reset_peak_memory_stats();model.to('cuda').eval()
            try:training.evaluate(model,[batch],torch.device('cuda'),pool_kernel_um=5.)
            finally:
                torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_reserved());model.to('cpu');torch.cuda.empty_cache()
        batches=index+1
        if batches%10==0 or batches==1:
            r=result(batches);write(WORK/'replay-progress.json',{k:v for k,v in r.items() if k!='window_order'})
            print(json.dumps({k:r[k] for k in ('batches','window_count','accuracy','node_recall','product','seconds_excluding_gpu_queue')}),flush=True)
        if batches==200:write(OUT/'released-secondary-cap200.json',result(batches))
    r=result(len(loader));write(OUT/'released-secondary-full-monitor.json',r)
    print(json.dumps({k:r[k] for k in ('batches','window_count','accuracy','node_recall','product','product_delta_from_published')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args()
    {'prepare':prepare,'run':run}[a.action]()
