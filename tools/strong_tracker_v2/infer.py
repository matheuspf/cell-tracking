"""Annotation-unavailable inference for every v2 candidate pipeline."""
from __future__ import annotations

import os
import sys
import time

import joblib
import numpy as np

from annotation_selection.filter_graph import filtered

from .common import (OUT,V1,graph_hash,load_graph,now,read_json,run_pool,save_arrays,sha,stage,validate,write_json)
from .decode import associations,divisions,score_only_divisions
from .policy import ranked_mask

_MODELS={}


def deny_annotations():
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(args[0]))
            if '.geff' in p or '/evaluation/' in p or '/oracle' in p or 'census' in p or p.endswith('/inventory.json'):
                raise PermissionError(f'Annotations unavailable: {p}')
    sys.addaudithook(audit)


def variants():
    result={'identity':dict(kind='identity')}
    for model in ['D_score','D_graph','D_temporal','D_hgb']:
        for p in [.02,.05,.1]:result[f'{model}_p{p}']=dict(kind='D',model=model,threshold=p)
    for model in ['E_native','E_logistic','E_hgb']:
        for m in [.5,1.5]:result[f'{model}_m{m}']=dict(kind='E',model=model,margin=m)
    for model in ['F_logistic_quality','F_hgb_quality','F_hgb_geometry']:
        modes=['node','tracklet','segment','fork_protected'] if model=='F_hgb_quality' else ['tracklet']
        for mode in modes:
            for keep in [.995,.99,.98,.95,.9,.8,.7,.5]:
                result[f'{model}_{mode}_r{keep}']=dict(kind='F',model=model,mode=mode,keep=keep)
    result['F_v1_transfer_control']=dict(kind='transfer')
    result['DE_primary']=dict(kind='DE',D='D_temporal_p0.05',E='E_hgb_m1.5')
    # Native F is separately disclosed as a transfer when applied after repair.
    result['DEF_native_transfer']=dict(kind='DEF',D='D_temporal_p0.05',E='E_hgb_m1.5',
        F='F_hgb_quality_fork_protected_r0.99',filter_graph_version='trained on original final graph; transfer diagnostic')
    return result


def models(source):
    if source not in _MODELS:
        lock=read_json(OUT/'model_lock.json')
        paths=list((OUT/'models'/source).glob('*.joblib'))
        for p in paths:assert sha(p)==lock['models'][str(p.relative_to(OUT))]
        _MODELS[source]={p.stem:joblib.load(p) for p in paths}
    return _MODELS[source]


def one(name):
    deny_annotations();start=time.perf_counter();source='6bba' if name.startswith('44b6') else '44b6'
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    c=load_graph(OUT/'native'/f'{name}.npz');h=load_graph(OUT/'hypotheses'/f'{name}.npz')
    shape=read_json(f'/kaggle/input/competitions/biohub-cell-tracking-during-development/train/{name}.zarr/0/zarr.json')['shape']
    nx=np.column_stack([b['features'],c['node_extra']])
    scores={}
    for mid,model in models(source).items():
        x={'node':nx,'edge':c['edge_features'],'fork':h['fork_features']}[model['kind']]
        scores[mid]=model['model'].predict_proba(x[:,model['columns']])[:,1].astype(np.float32) if len(x) else np.empty(0,np.float32)
    scores['D_score']=score_only_divisions(h)
    scores['E_native']=np.where(c['edge_features'][:,1]==0,c['edge_features'][:,0],
        1/(1+np.exp(c['edge_features'][:,6]-3.)))
    save_arrays(OUT/'scores'/f'{name}.npz',**scores)
    artifacts={};metadata={};graphs={}
    for variant,config in variants().items():
        kind=config['kind'];keep=np.ones(len(n),bool);ee=e
        extra={}
        if kind=='D':ee,extra=divisions(n,e,h,scores[config['model']],config['threshold'])
        elif kind=='E':ee=associations(n,e,c,scores[config['model']],config['margin'])
        elif kind=='F':keep=ranked_mask(n,e,scores[config['model']],config['keep'],config['mode'])
        elif kind=='transfer':
            from annotation_selection.filter_graph import PreparedFilter
            # Historical frozen DoG-to-Harmonic transfer remains a negative control.
            pp=load_graph(V1/'public_predictions'/source/f'{name}.npz')
            mid='hgb_all_leaf7'
            keep=PreparedFilter(n,e).mask(pp[mid],.9,'membership_tracklets',key=mid)
        elif kind in ['DE','DEF']:
            de=graphs[config['E']]
            ee,extra=divisions(n,de,h,scores['D_temporal'],.05)
            if kind=='DEF':keep=ranked_mask(n,ee,scores['F_hgb_quality'],.99,'fork_protected')
        nn,ee=filtered(n,ee,keep)
        validate(nn,ee,shape,reference=n,allow_legacy_bounds=True)
        metadata[variant]=dict(graph_hash=graph_hash(nn,ee),nodes=len(nn),edges=len(ee),realized_keep=len(nn)/len(n),**extra)
        if kind in ['D','E','DE','DEF']:
            artifacts[variant+'__edges']=ee
            if kind=='E':graphs[variant]=ee
        artifacts[variant+'__keep']=keep
    save_arrays(OUT/'predictions'/f'{name}.npz',**artifacts)
    write_json(OUT/'predictions'/f'{name}.json',dict(dataset=name,source=source,seconds=time.perf_counter()-start,
        variants=metadata,annotation_access_blocked=True,baseline_sha256=sha(V1/'baseline/public'/f'{name}.npz'),
        model_lock_sha256=sha(OUT/'model_lock.json')))
    return f'{name} {len(metadata)} variants {time.perf_counter()-start:.1f}s'


def run(args):
    configs=variants()
    write_json(OUT/'config_locks.json',dict(created=now(),variants=configs,model_lock_sha256=sha(OUT/'model_lock.json'),
        primary=dict(D='D_temporal_p0.05',E='E_hgb_m1.5',F='F_risk_threshold'),
        scoring_has_started=False,both_directions_frozen=True,
        hypotheses='Top four physical/native alternatives; frozen retained continuation; distinct daughter persistence; bounded component ILP'),immutable=True) if not (OUT/'config_locks.json').exists() else None
    names=[p.stem for p in sorted((V1/'baseline/public').glob('*.npz'))]
    if args.limit:names=names[:args.limit]
    pending=[n for n in names if not (OUT/'predictions'/f'{n}.json').exists()]
    list(run_pool(one,pending,args.workers))
    write_json(OUT/'inference_receipt.json',dict(samples=len(names),variants=len(configs),annotation_reads=0,
        guard='Python audit hook blocks GEFF, evaluation, census, oracle and GT inventory files before model/feature loading',
        prediction_sha256={n:sha(OUT/'predictions'/f'{n}.npz') for n in names},
        configs_sha256=sha(OUT/'config_locks.json'),model_lock_sha256=sha(OUT/'model_lock.json')))
    stage('V220','inference_complete',samples=len(names));stage('V230','inference_complete',samples=len(names))
