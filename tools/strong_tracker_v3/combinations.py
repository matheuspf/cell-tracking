"""Frozen A+D and A+D+R; graph-dependent features refreshed after association."""
import time
import numpy as np
from .common import *

CONFIG=dict(association='A_residual_m1.5',division_model='image_20260909',division_threshold=.2,
    division_replace=False,rescue='R_image_persistent',variants=['AD_primary','ADR_primary'],
    policy='Fixed primary source-trained A/D; bounded corrected image rescue. No embryo-specific thresholds.',
    graph_features='Rebuild adjacency, velocity, ranks, votes, continuity and event candidates after A; unchanged image samples reused at identical coordinates')

def one(task):
    from .features import build
    from .event_pipeline import apply as apply_division
    from .rescue_fixed import apply as apply_rescue
    from .inference import deny_annotations
    ctx,sample=task;deny_annotations();name=sample['dataset']
    ap=ctx.out/'candidate_graphs'/CONFIG['association']/f'{name}.npz'
    g=load_graph(ap);b=load_graph(ctx.incumbent(name));native=load_graph(ctx.out/'features'/f'{name}.npz')
    stamp=dict(association_graph_sha256=sha(ap),incumbent_sha256=sha(ctx.incumbent(name)),
        event_models_sha256=sha(ctx.out/'event_inference_config.json'),config=digest(CONFIG),code=sha(__file__),
        feature_input_sha256=sha(ctx.out/'features'/f'{name}.npz'),
        image_content_sha256=read_json(ctx.out/'input_hashes'/f'{name}.json')['image_content_sha256'],
        code_files={f:sha(ctx.repo/'tools/strong_tracker_v3'/f) for f in
            ['features.py','disagreements.py','event_pipeline.py','event_proposals.py','event_model.py',
             'decode.py','rescue_fixed.py','rescue.py','common.py']})
    receipt=ctx.out/'combinations'/f'{name}.json'
    if receipt.exists():
        old=read_json(receipt)
        if old['inputs']!=stamp:raise ValueError('Combination fingerprint drift')
        for v,h in old['hashes'].items():
            if sha(ctx.out/'candidate_graphs'/v/f'{name}.npz')!=h:raise ValueError('Combination graph changed')
        return name+' exact cache'
    start=time.perf_counter()
    if not np.array_equal(g['nodes'],b['nodes']):raise ValueError('Association unexpectedly changed node coordinates')
    refreshed,meta=build(ctx,sample,g['nodes'],g['edges'],node_features=native['node_features'])
    n,e,div=apply_division(ctx,sample,g['nodes'],g['edges'],model=CONFIG['division_model'],
        threshold=CONFIG['division_threshold'],replace=False,native=refreshed)
    validate(n,e,sample['image_shape']);p=ctx.out/'candidate_graphs/AD_primary'/f'{name}.npz';save_graph(p,n,e)
    hashes={'AD_primary':sha(p)}
    nn,ee,ledger,stats=apply_rescue(ctx,sample,n,e)
    validate(nn,ee,sample['image_shape']);p=ctx.out/'candidate_graphs/ADR_primary'/f'{name}.npz';save_graph(p,nn,ee)
    hashes['ADR_primary']=sha(p)
    write_json(receipt,dict(dataset=name,inputs=stamp,hashes=hashes,seconds=time.perf_counter()-start,
        A_graph_hash=graph_hash(g['nodes'],g['edges']),AD_graph_hash=graph_hash(n,e),ADR_graph_hash=graph_hash(nn,ee),
        graph_features_recomputed=True,features=meta,division=div,rescue_ledger=ledger,rescue_stats=stats))
    return f'{name} AD/ADR {time.perf_counter()-start:.1f}s'

def run(ctx,args):
    for name in ['association_model_lock.json','event_model_lock.json','event_inference_config.json']:
        if not (ctx.out/name).exists():raise RuntimeError('Freeze both source-trained component directions before combinations')
    path=ctx.out/'combination_config_lock.json';old=read_json(path) if path.exists() else {}
    write_json(path,dict(created=old.get('created',now()),config=CONFIG,
        association_model_lock_sha256=sha(ctx.out/'association_model_lock.json'),
        event_model_lock_sha256=sha(ctx.out/'event_model_lock.json'),
        event_inference_config_sha256=sha(ctx.out/'event_inference_config.json'),
        decoder_sha256=sha(ctx.repo/'tools/strong_tracker_v3/decode.py'),
        exposure='Stage outcomes already examined; adaptive exploratory combination round, no probability recalibration',
        code=sha(__file__)),immutable=True)
    list(run_pool(one,[(ctx,s) for s in ctx.samples()],args.workers))
    write_json(ctx.out/'combination_inference_receipt.json',dict(samples=199,variants=CONFIG['variants'],
        config=CONFIG,both_source_directions_frozen=True,graph_features_recomputed=True,
        hashes={v:{s['dataset']:sha(ctx.out/'candidate_graphs'/v/f'{s["dataset"]}.npz') for s in ctx.samples()} for v in CONFIG['variants']}))
