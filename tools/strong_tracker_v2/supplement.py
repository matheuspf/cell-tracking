"""Guarded risk/image inference and full official scoring for supplemental arms."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from annotation_selection.filter_graph import filtered
from annotation_selection.metric_adapter import evaluate_graph

from .common import (OUT,V1,graph_hash,inventory,load_graph,read_json,run_pool,save_arrays,sha,validate,write_json)
from .evaluate import baseline_fp_set,collect
from .infer import deny_annotations
from .policy import group_actions,group_features,ranked_mask


def risk_masks(base,native,model):
    n,e=base['nodes'],base['edges'];groups,protected=group_actions(n,e,'fork_protected')
    x=group_features(base,native,groups)
    sur=np.column_stack([h.predict(x) for h in model['surrogate_heads']])
    loss=model['actual_head'].predict(np.column_stack([x,sur]))
    order=np.lexsort((np.arange(len(groups)),loss))
    masks={}
    for r in model['config']['keep_fractions']+['threshold']:
        cap=model['config']['primary_max_removed_fraction'] if r=='threshold' else 1-r
        budget=int(np.floor(cap*len(n)));mask=np.ones(len(n),bool);removed=0
        for j in order:
            g=groups[j]
            if r=='threshold' and loss[j]>model['config']['primary_threshold']:continue
            if protected[g].any() or removed+len(g)>budget:continue
            mask[g]=False;removed+=len(g)
        key='F_risk_threshold' if r=='threshold' else f'F_risk_r{r}'
        masks[key]=mask
    return masks,loss


def risk_infer(name):
    deny_annotations();source='6bba' if name.startswith('44b6') else '44b6'
    b=load_graph(V1/'baseline/public'/f'{name}.npz');c=load_graph(OUT/'native'/f'{name}.npz')
    model_path=OUT/'risk_models'/f'{source}.joblib';lock=read_json(OUT/'risk_model_lock.json')
    assert sha(model_path)==lock['models'][source]
    model=joblib.load(model_path);masks,loss=risk_masks(b,c,model)
    # The combination is an explicit risk-model transfer to repaired graphs.
    pred=load_graph(OUT/'predictions'/f'{name}.npz');de=pred['DE_primary__edges']
    repaired=dict(b,edges=de)
    combined,_=risk_masks(repaired,c,model)
    arrays={v+'__keep':m for v,m in masks.items()}
    arrays['DEF_risk_transfer__keep']=combined['F_risk_threshold'];arrays['DEF_risk_transfer__edges']=de
    save_arrays(OUT/'risk_predictions'/f'{name}.npz',**arrays)
    write_json(OUT/'risk_predictions'/f'{name}.json',dict(source=source,annotation_access_blocked=True,variants=list(masks)+['DEF_risk_transfer'],
        risk_model_sha256=sha(model_path),risk_min=float(loss.min()),risk_max=float(loss.max()),
        repaired_graph_filter='Same learned group features, recomputed topology; model targets calibrated on original graph, transfer diagnostic'))
    return name


def image_infer(args):
    deny_annotations()
    import torch
    from .temporal_model import Encoder,make_batch,temporal_indices
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0');names=[p.stem for p in sorted((V1/'baseline/public').glob('*.npz'))]
    lock=read_json(OUT/'temporal_model_lock.json');models={}
    for source in ['44b6','6bba']:
        models[source]={}
        for fraction in [.1,.3,1.]:
            p=OUT/'temporal_models'/source/f'fraction_{fraction}.pt';assert sha(p)==lock['models'][str(p.relative_to(OUT))]
            ck=torch.load(p,map_location='cpu',weights_only=True);model=Encoder(ck['features']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            models[source][fraction]=(model,ck)
    for k,name in enumerate(names):
        dest=OUT/'image_predictions'/f'{name}.npz'
        if dest.exists() and dest.with_suffix('.json').exists():continue
        start=time.perf_counter();source='6bba' if name.startswith('44b6') else '44b6';model,ck=models[source][1.]
        b=load_graph(V1/'baseline/public'/f'{name}.npz');c=load_graph(OUT/'native'/f'{name}.npz');n,e=b['nodes'],b['edges']
        x=np.column_stack([b['features'][:,5:],c['node_extra']]).astype(np.float32)
        pp=np.load(V1/'public_patches'/f'{name}.npy',mmap_mode='r');indices,valid=temporal_indices(n,e)
        fraction_scores={f:np.empty(len(n),np.float32) for f in [.1,.3,1.]}
        bias=float(np.log(ck['source_prevalence']/(1-ck['source_prevalence'])))
        with torch.inference_mode():
            for start_i in range(0,len(n),512):
                ids=np.arange(start_i,min(start_i+512,len(n)))
                xx,ff=make_batch(pp,indices,valid,x,ids,ck['mean'].numpy(),ck['std'].numpy(),device)
                for fraction,(model,fck) in models[source].items():
                    assert torch.equal(fck['mean'],ck['mean']) and torch.equal(fck['std'],ck['std'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(xx,ff)
                    fraction_scores[fraction][ids]=torch.sigmoid(logits.float()+bias).cpu().numpy()
        score=fraction_scores[1.]
        arrays={};graphs={}
        for mode in ['tracklet','fork_protected']:
            for keep in [.995,.99,.98,.95,.9,.8,.7,.5]:
                v=f'F_temporal_{mode}_r{keep}';mask=ranked_mask(n,e,score,keep,mode)
                arrays[v+'__keep']=mask
        pred=load_graph(OUT/'predictions'/f'{name}.npz');de=pred['DE_primary__edges']
        arrays['DEF_temporal_transfer__keep']=ranked_mask(n,de,score,.99,'fork_protected')
        arrays['DEF_temporal_transfer__edges']=de
        save_arrays(dest,**arrays);save_arrays(OUT/'image_scores'/f'{name}.npz',score=score,
            fraction_0_1=fraction_scores[.1],fraction_0_3=fraction_scores[.3])
        write_json(dest.with_suffix('.json'),dict(source=source,annotation_access_blocked=True,seconds=time.perf_counter()-start,
            variants=[v.removesuffix('__keep') for v in arrays if v.endswith('__keep')],
            model_lock_sha256=sha(OUT/'temporal_model_lock.json'),calibration='Class-balanced posterior odds shifted by natural source prevalence'))
        print(f'image inference {k+1}/{len(names)} {name} {time.perf_counter()-start:.1f}s',flush=True)
    write_json(OUT/'image_inference_receipt.json',dict(samples=len(names),annotation_reads=0,
        hashes={n:sha(OUT/'image_predictions'/f'{n}.npz') for n in names},model_lock_sha256=sha(OUT/'temporal_model_lock.json')))


def score_one(task):
    row,arm=task;name=row['dataset'];start=time.perf_counter()
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
    bm={int(i):int(j) for i,j in zip(n[:,0],m['matched_gt_id']) if j>=0}
    base_tp=set(map(tuple,m['tp_edges']));base_fp=baseline_fp_set(n,e,gt,bm)
    path=OUT/f'{arm}_predictions'/f'{name}.npz';arrays=load_graph(path)
    stamp=dict(predictions_sha256=sha(path),gt_sha256=sha(V1/'evaluation/gt'/f'{name}.npz'))
    for key,keep in arrays.items():
        if not key.endswith('__keep'):continue
        v=key.removesuffix('__keep');dest=OUT/'evaluation/scores'/v/f'{name}.json'
        if dest.exists():assert read_json(dest)['inputs']==stamp;continue
        nn,ee=filtered(n,arrays.get(v+'__edges',e),keep);tic=time.perf_counter()
        validate(nn,ee,row['image_shape'],reference=n,allow_legacy_bounds=True)
        result,matches,tp=evaluate_graph(name,nn,ee,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        fp=baseline_fp_set(nn,ee,gt,matches)
        requested=float(v.rsplit('_r',1)[-1]) if '_r' in v and v.rsplit('_r',1)[-1][0].isdigit() else 1.
        source='6bba' if row['embryo']=='44b6' else '44b6'
        result.update(variant=v,embryo=row['embryo'],source=source,fold=f'{source}->{row["embryo"]}',kind=arm,
            seconds=time.perf_counter()-tic,requested_keep=requested,realized_keep=len(nn)/len(n),baseline_num_pred_nodes=len(n),
            baseline_tp=len(base_tp),baseline_tp_rematched_surviving=len(base_tp&tp),newly_recovered_tp=len(tp-base_tp),lost_tp=len(base_tp-tp),
            removed_fp=len(base_fp-fp),introduced_fp=len(fp-base_fp),baseline_matched_nodes=len(bm),
            removed_baseline_matches=sum(not k and int(i) in bm for i,k in zip(n[:,0],keep)),lane='exploratory_public_upstream_contaminated')
        write_json(dest,dict(inputs=stamp,result=result))
    return f'{name} {arm} {time.perf_counter()-start:.1f}s'


def run(args):
    if args.variant=='image-infer':image_infer(args);return
    rows=inventory();rows=rows[:args.limit] if args.limit else rows
    if args.variant=='risk-infer':
        list(run_pool(risk_infer,[r['dataset'] for r in rows],args.workers))
        write_json(OUT/'risk_inference_receipt.json',dict(samples=len(rows),annotation_reads=0,
            hashes={r['dataset']:sha(OUT/'risk_predictions'/f"{r['dataset']}.npz") for r in rows},
            model_lock_sha256=sha(OUT/'risk_model_lock.json')))
    else:
        arm=args.variant or 'risk'
        list(run_pool(score_one,[(r,arm) for r in rows],args.workers));collect()
