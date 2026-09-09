"""Explicit exploratory combinations after identifying complementary arms."""
import joblib
import numpy as np

from .common import OUT,V1,inventory,load_graph,now,read_json,run_pool,save_arrays,sha,write_json
from .infer import deny_annotations
from .policy import ranked_mask
from .supplement import risk_masks

CONFIG=dict(E='E_hgb_m0.5',filters=['native_quality_0.99','native_geometry_0.98','risk_threshold','temporal_0.99'],
    selection='Exploratory complementarity check after inspecting main-arm results; not a prespecified primary comparison',
    filtering_transfer='Filters trained on original final graph; applied to the association-repaired graph with new groups/protection. Explicit transfer arms.')


def one(name):
    deny_annotations();b=load_graph(V1/'baseline/public'/f'{name}.npz');n=b['nodes']
    c=load_graph(OUT/'native'/f'{name}.npz');p=load_graph(OUT/'predictions'/f'{name}.npz');e=p['E_hgb_m0.5__edges']
    score=load_graph(OUT/'scores'/f'{name}.npz');im=load_graph(OUT/'image_scores'/f'{name}.npz')['score']
    source='6bba' if name.startswith('44b6') else '44b6';path=OUT/'risk_models'/f'{source}.joblib'
    assert sha(path)==read_json(OUT/'risk_model_lock.json')['models'][source]
    risk,_=risk_masks(dict(b,edges=e),c,joblib.load(path))
    masks={'EF_native_quality_transfer_r0.99':ranked_mask(n,e,score['F_hgb_quality'],.99,'fork_protected'),
        'EF_native_geometry_transfer_r0.98':ranked_mask(n,e,score['F_hgb_geometry'],.98,'fork_protected'),
        'EF_risk_transfer':risk['F_risk_threshold'],
        'EF_temporal_transfer_r0.99':ranked_mask(n,e,im,.99,'fork_protected')}
    arrays={}
    for v,keep in masks.items():arrays[v+'__keep']=keep;arrays[v+'__edges']=e
    save_arrays(OUT/'combo_predictions'/f'{name}.npz',**arrays)
    write_json(OUT/'combo_predictions'/f'{name}.json',dict(source=source,variants=list(masks),annotation_access_blocked=True,config=CONFIG))
    return name


def run(args):
    write_json(OUT/'combination_config_lock.json',dict(created=now(),**CONFIG,
        model_lock_sha256=sha(OUT/'model_lock.json'),risk_model_lock_sha256=sha(OUT/'risk_model_lock.json'),
        image_model_lock_sha256=sha(OUT/'temporal_model_lock.json')),immutable=True) if not (OUT/'combination_config_lock.json').exists() else None
    names=[r['dataset'] for r in inventory()]
    list(run_pool(one,names,args.workers))
    write_json(OUT/'combo_inference_receipt.json',dict(samples=len(names),annotation_reads=0,
        hashes={n:sha(OUT/'combo_predictions'/f'{n}.npz') for n in names}))
