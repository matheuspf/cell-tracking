"""V330/V340 execution: fixed proposals, source fitting, frozen opposite inference."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import time

import numpy as np

from .common import digest, graph_hash, load_graph, now, read_json, run_pool, save_arrays, sha, write_json
from .context import RunContext
from .event_proposals import EVENT_FEATURES, ProposalConfig, build, extract_crops

CONFIGS={'base':ProposalConfig(),'fallback':ProposalConfig(neighbors=8,parent_gate_um=20.,sister_gate_um=24.)}
SEEDS=[20260909,314159]


def stable_lock(path,payload,keys):
    """Preserve creation time/hash when a scientifically identical run resumes."""
    if Path(path).exists():
        previous=read_json(path)
        if any(previous.get(k)!=payload.get(k) for k in keys):raise ValueError(f'Locked event artifact changed: {path}')
        return previous
    write_json(path,payload);return payload


def prepare_one(task):
    # Independent preparation queues can overlap safely after source coverage
    # triggers the one predeclared fallback; only exact fingerprints resume.
    import fcntl
    ctx,sample,pool=task
    lock=ctx.out/'events'/pool/f".{sample['dataset']}.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _prepare_one_unlocked(task)


def _prepare_one_unlocked(task):
    ctx,sample,pool=task;name=sample['dataset'];start=time.perf_counter()
    destination=ctx.out/'events'/pool/f'{name}.npz'
    feature=ctx.out/'features'/f'{name}.npz'
    deadline=time.monotonic()+7200
    while not feature.with_suffix('.json').exists():
        if time.monotonic()>deadline:raise TimeoutError(f'Incumbent features unavailable: {name}')
        time.sleep(2)
    stamp=dict(graph_sha256=sha(ctx.incumbent(name)),feature_sha256=sha(feature),config=asdict(CONFIGS[pool]),
        proposal_code_sha256=sha(Path(__file__).with_name('event_proposals.py')),
        label_code_sha256=sha(Path(__file__).with_name('source_labels.py')),
        gt_sha256=sha(ctx.v1/'evaluation/gt'/f'{name}.npz'),metric_revision=ctx.metric_revision)
    receipt_path=destination.with_suffix('.json')
    if receipt_path.exists():
        meta=read_json(receipt_path)
        if meta['inputs']!=stamp or meta['sha256']!=sha(destination):raise ValueError(f'Stale event cache: {name}')
        return dict(dataset=name,pool=pool,resumed=True,**{k:meta[k] for k in ['alternatives','crop_parents']})
    b=load_graph(ctx.incumbent(name));native=load_graph(feature)
    feature_meta=read_json(feature.with_suffix('.json'))
    if feature_meta['inputs']['graph_hash']!=graph_hash(b['nodes'],b['edges']):
        raise ValueError('Event features were built on a different graph')
    h,meta=build(b['nodes'],b['edges'],native,sample['physical_scale'],CONFIGS[pool])
    proposal_seconds=time.perf_counter()-start
    # Proposals and crop centers have already been fixed before this source-only
    # label adapter is imported. All target comparative scores stay unexposed.
    from .source_labels import labels
    gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
    lab,lmeta=labels(b['nodes'],b['edges'],h['events'],gt['nodes'],gt['edges'],sample['physical_scale'])
    label_seconds=time.perf_counter()-start-proposal_seconds
    import zarr
    image=zarr.open_group(sample['image_path'],mode='r')['0']
    crops=extract_crops(image,b['nodes'],h['crop_parents'],sample['physical_scale'])
    save_arrays(destination,**h,crops=crops)
    lp=ctx.out/'evaluation/event_labels'/pool/f'{name}.npz';save_arrays(lp,**lab)
    write_json(lp.with_suffix('.json'),dict(**lmeta,inputs=stamp,sha256=sha(lp)))
    write_json(receipt_path,dict(**meta,inputs=stamp,sha256=sha(destination),created=now(),
        proposal_seconds=proposal_seconds,label_seconds=label_seconds,seconds=time.perf_counter()-start,
        uncompressed_bytes=sum(v.nbytes for v in h.values())+crops.nbytes))
    return dict(dataset=name,pool=pool,alternatives=meta['alternatives'],events_covered=lmeta['positive_event_groups'],
        total_events=len(lmeta['coverage']),seconds=time.perf_counter()-start)


def coverage(ctx,pool):
    result={}
    for embryo in ['44b6','6bba']:
        rows=[read_json(ctx.out/'evaluation/event_labels'/pool/f"{s['dataset']}.json") for s in ctx.samples() if s['embryo']==embryo]
        events=[event for row in rows for event in row['coverage']]
        totals={key:sum(row[key] for row in rows) for key in ['proposals','positive_alternatives','negative_population','unknown','sampled_negative','positive_event_groups','negative_groups']}
        result[embryo]=dict(**totals,total_events=len(events),covered_events=sum(e['covered'] for e in events),
            coverage=sum(e['covered'] for e in events)/max(1,len(events)),
            rows_with_local_roles=sum(e['local_roles_available'] for e in events))
    return result


def freeze_source_pools(ctx):
    base=coverage(ctx,'base')
    selected={e:'fallback' if row['coverage']<.85 else 'base' for e,row in base.items()}
    return stable_lock(ctx.out/'event_source_pool_lock.json',dict(created=now(),source_pools=selected,base_coverage=base,
        criterion='source-only base coverage <85%; one predeclared fallback; no target performance used',outer_event_scores_exposed=False),
        ['source_pools','base_coverage'])


def prepare(ctx,workers=4):
    samples=ctx.samples();list(run_pool(prepare_one,[(ctx,s,'base') for s in samples],workers))
    base=coverage(ctx,'base')
    selected={e:'fallback' if row['coverage']<.85 else 'base' for e,row in base.items()}
    # A fallback is selected exclusively from source coverage, never target
    # performance. Both source/target pools are prepared if either source needs it.
    result=dict(base=base,source_selected_pool=selected,fallback_trigger='source-only event coverage <85%',outer_scores_exposed=False)
    if 'fallback' in selected.values():
        list(run_pool(prepare_one,[(ctx,s,'fallback') for s in samples],workers))
        result['fallback']=coverage(ctx,'fallback')
    write_json(ctx.out/'event_candidate_coverage.json',result)
    return result


def source_corpus(ctx,source,pool):
    xx=[];yy=[];ww=[];bb=[];cc=[];ci=[];offset=0;inputs={}
    for sample in ctx.samples():
        if sample['embryo']!=source:continue
        name=sample['dataset'];ep=ctx.out/'events'/pool/f'{name}.npz';lp=ctx.out/'evaluation/event_labels'/pool/f'{name}.npz'
        with np.load(lp,allow_pickle=False) as archive:l={key:archive[key] for key in ['index','y','weight','bag']}
        ids=l['index'];inputs[name]=dict(events_sha256=sha(ep),labels_sha256=sha(lp))
        if not len(ids):continue
        # All supported positive alternatives are retained. Training images are
        # a subset of the exact label-free crop cache used during inference.
        with np.load(ep,allow_pickle=False) as archive:
            use_crop,local_index=np.unique(archive['crop_index'][ids],return_inverse=True)
            xx.append(archive['event_features'][ids]);cc.append(archive['crops'][use_crop])
        yy.append(l['y']);ww.append(l['weight'])
        bb.extend([f'{name}:{int(b)}' for b in l['bag']]);ci.append(local_index+offset)
        offset+=len(use_crop)
    result=dict(x=np.concatenate(xx),y=np.concatenate(yy),weights=np.concatenate(ww),bags=np.asarray(bb),
        crops=np.concatenate(cc),crop_index=np.concatenate(ci),inputs=inputs)
    import hashlib
    fingerprint=hashlib.sha256()
    for key in ['x','y','weights','bags','crops','crop_index']:
        array=np.ascontiguousarray(result[key]);fingerprint.update(str((key,array.shape,array.dtype.str)).encode());fingerprint.update(memoryview(array))
    result['corpus_sha256']=fingerprint.hexdigest()
    return result


def fit(ctx,sources=None):
    from .event_model import fit_image,fit_logistic
    freeze_grid(ctx)
    source_lock=freeze_source_pools(ctx);cov=dict(source_selected_pool=source_lock['source_pools'])
    if (ctx.out/'event_candidate_coverage.json').exists():
        if read_json(ctx.out/'event_candidate_coverage.json')['source_selected_pool']!=cov['source_selected_pool']:
            raise ValueError('Final event pool selection differs from source freeze')
    requested=list(sources or ['44b6','6bba']);models={};started=time.perf_counter()
    for source in requested:
        pool=cov['source_selected_pool'][source];d=source_corpus(ctx,source,pool)
        meta=dict(source_embryo=source,target_embryo='6bba' if source=='44b6' else '44b6',pool=pool,
            feature_schema_sha256=digest(EVENT_FEATURES),source_inputs_sha256=digest(d['inputs']),
            actual_training_arrays_sha256=d['corpus_sha256'],
            source_inputs=d['inputs'],model_code_sha256=sha(Path(__file__).with_name('event_model.py')),
            label_code_sha256=sha(Path(__file__).with_name('source_labels.py')),metric_revision=ctx.metric_revision,
            exploratory_reused_embryos=True,outer_scores_exposed=False)
        path=ctx.out/'event_models'/f'{source}_logistic.joblib'
        if path.with_suffix('.json').exists():
            lr=read_json(path.with_suffix('.json'))
            if any(lr.get(k)!=v for k,v in meta.items()) or lr['model_sha256']!=sha(path):raise ValueError('Stale event logistic model')
        else:lr=fit_logistic(d['x'],d['y'],d['weights'],path,meta)
        models[f'{source}_logistic']=lr
        for seed in SEEDS:
            path=ctx.out/'event_models'/f'{source}_image_{seed}.pt'
            if path.with_suffix('.json').exists():
                receipt=read_json(path.with_suffix('.json'))
                if any(receipt.get(k)!=v for k,v in meta.items()) or receipt['model_sha256']!=sha(path):raise ValueError('Stale event image model')
            else:receipt=fit_image(d['x'],d['y'],d['bags'],d['crops'],d['crop_index'],path,meta,seed=seed,steps=10000)
            models[f'{source}_image_{seed}']=receipt
            if sum(x.get('seconds',0.) for x in models.values())>8*3600:raise RuntimeError('Event GPU budget exhausted')
        # Source-only event-group learning diagnostics, after valid main fits.
        # Fractions refer to unique observed division bags, never crop count.
        positive_bags=np.unique(d['bags'][d['y']==1]);order=np.random.default_rng(20260909).permutation(positive_bags)
        diagnostics=[]
        for fraction in [.25,.5,1.]:
            keep=set(order[:max(1,int(np.ceil(fraction*len(order))))])
            mask=(d['y']==0)|np.array([b in keep for b in d['bags']])
            if fraction==1.:diagnostic=lr
            else:
                path=ctx.out/'event_models/diagnostics'/f'{source}_logistic_fraction{int(fraction*100)}.joblib'
                diagnostic_meta=dict(**meta,positive_group_fraction=fraction,unique_selected_event_groups=len(keep),
                    interpretation='source diagnostic fit, not independent validation or biological event deduplication')
                if path.with_suffix('.json').exists():
                    diagnostic=read_json(path.with_suffix('.json'))
                    if any(diagnostic.get(k)!=v for k,v in diagnostic_meta.items()) or diagnostic['model_sha256']!=sha(path):
                        raise ValueError('Stale event learning diagnostic')
                else:diagnostic=fit_logistic(d['x'][mask],d['y'][mask],d['weights'][mask],path,diagnostic_meta)
            diagnostics.append(dict(fraction=fraction,positive_groups=len(keep),training_rows=int(mask.sum()),
                source_weighted_bce=diagnostic['source_weighted_bce'],model_sha256=diagnostic['model_sha256']))
        write_json(ctx.out/'event_models/diagnostics'/f'{source}_learning_curve.json',dict(source=source,rows=diagnostics,
            method='25/50/100% source positive event groups; all supported source negatives; no held-out probability calibration'))
        del d
    if requested!=['44b6','6bba']:
        return stable_lock(ctx.out/'event_source_model_locks'/f"{'_'.join(requested)}.json",
            dict(created=now(),models=models,source_pools=cov['source_selected_pool'],outer_event_scores_exposed=False,
                 note='Partial source fit; comparative prediction remains blocked until both source directions are frozen'),['models','source_pools'])
    lock=dict(created=now(),models=models,source_pools=cov['source_selected_pool'],outer_scores_exposed=False,
              total_elapsed_seconds=time.perf_counter()-started)
    lock=stable_lock(ctx.out/'event_model_lock.json',lock,['models','source_pools'])
    write_json(ctx.out/'event_inference_config.json',dict(source_pools=cov['source_selected_pool'],seeds=SEEDS,
        feature_names=EVENT_FEATURES,model_sha256={k:v['model_sha256'] for k,v in models.items()},
        proposal_configs={k:asdict(v) for k,v in CONFIGS.items()},source_only=True))
    return lock


def probabilities(ctx):
    from .event_model import load_image,predict_image,predict_logistic
    lock=read_json(ctx.out/'event_model_lock.json');receipts=[]
    for source in ['44b6','6bba']:
        pool=lock['source_pools'][source]
        models={seed:load_image(ctx.out/'event_models'/f'{source}_image_{seed}.pt') for seed in SEEDS}
        for sample in ctx.samples():
            if sample['embryo']==source:continue
            name=sample['dataset'];path=ctx.out/'event_probabilities'/f'{name}.npz';ep=ctx.out/'events'/pool/f'{name}.npz'
            stamp=dict(source=source,pool=pool,event_sha256=sha(ep),model_lock_sha256=sha(ctx.out/'event_model_lock.json'),
                       prediction_code_sha256=sha(Path(__file__).with_name('event_model.py')))
            if path.with_suffix('.json').exists():
                receipt=read_json(path.with_suffix('.json'))
                if receipt['inputs']!=stamp or receipt['sha256']!=sha(path):raise ValueError('Stale event probabilities')
            else:
                tic=time.perf_counter();h=load_graph(ep)
                ps=dict(logistic=predict_logistic(ctx.out/'event_models'/f'{source}_logistic.joblib',h['event_features']))
                for seed,m in models.items():ps[f'image_{seed}']=predict_image(m,h['event_features'],h['crops'],h['crop_index'])
                save_arrays(path,**ps);receipt=dict(dataset=name,inputs=stamp,sha256=sha(path),seconds=time.perf_counter()-tic,
                    candidates=len(h['events']),mean_absolute_seed_difference=float(np.mean(abs(ps['image_20260909']-ps['image_314159']))),
                    threshold_counts={model:{str(p):int(np.sum(scores>=p)) for p in [.05,.1,.2,.4]} for model,scores in ps.items()},
                    seed_threshold_disagreements={str(p):int(np.sum((ps['image_20260909']>=p)!=(ps['image_314159']>=p))) for p in [.1,.2,.4]},
                    interpretation='label-free candidate score disagreement; not calibrated posteriors or independent event counts')
                write_json(path.with_suffix('.json'),receipt)
            receipts.append(receipt);print('event_probabilities',name,receipt.get('seconds'),flush=True)
        del models
    stable_lock(ctx.out/'event_probability_lock.json',dict(created=now(),predictions=receipts,outer_scores_exposed=False),['predictions'])


def variants():
    out=[dict(name='D_existing_p020',model='logistic',threshold=.2,existing_only=True,replace=False),
         dict(name='D_existing_p005_control',model='logistic',threshold=.05,existing_only=True,replace=False)]
    for prefix,model,replacement in [('D_tabular','logistic',False),('D_image','image_20260909',False),('D_replace','image_20260909',True)]:
        for p in ([.2,.4] if replacement else [.1,.2,.4]):
            out.append(dict(name=f'{prefix}_p{int(p*100):03d}',model=model,threshold=p,existing_only=False,replace=replacement))
    # Both secondary source fits and predictions remain measured artifacts. A
    # standalone secondary scored graph is omitted to count historical v1 within
    # the total32 comparison cap; this was fixed before any event scoring.
    assert len(out)==10
    return out


def freeze_grid(ctx):
    path=ctx.out/'event_grid_lock.json'
    config=dict(variants=variants(),seeds=SEEDS,steps_per_source_seed=10000,source_coverage_fallback_threshold=.85,
        common_edit_margin_logodds=1.5,maximum_changed_edge_fraction=.02,max_component_alternatives=256,
        max_buffered_actions=50000,outer_event_scores_exposed=False,
        cap_adjustments=['Secondary-seed graph omitted to count historical v1 inside total32; both fits retained',
                        'Replacement p.10 omitted before event fitting to reserve V350 correctness repair inside total32'])
    if path.exists():
        previous=read_json(path)
        if previous['config']!=config:raise ValueError('Frozen event grid changed')
    else:write_json(path,dict(created=now(),config=config))
    return config


def decode_one(task):
    from .decode import divisions
    ctx,sample,configs=task;name=sample['dataset'];start=time.perf_counter()
    pr=read_json(ctx.out/'event_probabilities'/f'{name}.json');pool=pr['inputs']['pool']
    b=load_graph(ctx.incumbent(name));h=load_graph(ctx.out/'events'/pool/f'{name}.npz')
    native=load_graph(ctx.out/'features'/f'{name}.npz');ps=load_graph(ctx.out/'event_probabilities'/f'{name}.npz');receipts=[]
    for config in configs:
        variant=config['name'];path=ctx.out/'candidate_graphs'/variant/f'{name}.npz'
        stamp=dict(incumbent_sha256=sha(ctx.incumbent(name)),event_sha256=pr['inputs']['event_sha256'],
                   probabilities_sha256=pr['sha256'],config=config,decode_code_sha256=sha(Path(__file__).with_name('decode.py')))
        if path.with_suffix('.json').exists():
            receipt=read_json(path.with_suffix('.json'))
            if receipt['inputs']!=stamp or receipt['sha256']!=sha(path):raise ValueError('Stale decoded event graph')
        else:
            e,r,ledger=divisions(b['nodes'],b['edges'],h,ps[config['model']],native,sample['physical_scale'],
                threshold=config['threshold'],existing_only=config['existing_only'],replace=config['replace'])
            save_arrays(path,nodes=b['nodes'],edges=e)
            receipt=dict(dataset=name,variant=variant,inputs=stamp,sha256=sha(path),graph_hash=graph_hash(b['nodes'],e),
                decode=r,seconds=time.perf_counter()-start)
            write_json(path.with_suffix('.json'),receipt)
            write_json(ctx.out/'edit_ledgers'/variant/f'{name}.json',dict(dataset=name,actions=ledger))
        receipts.append(receipt)
    return dict(dataset=name,seconds=time.perf_counter()-start,variants=len(configs))


def decode(ctx,workers=4):
    configs=variants();samples=ctx.samples()
    stable_lock(ctx.out/'event_round_config.json',dict(created=now(),variants=configs,outer_scores_exposed=False,
        model_lock_sha256=sha(ctx.out/'event_model_lock.json'),probability_lock_sha256=sha(ctx.out/'event_probability_lock.json'),
        secondary_seed_scored_graph_omitted='Count historical v1 inside total32 complete scored variants; decided before event scoring; both secondary source fits and probabilities retained',
        replacement_p010_omitted='Reserve one of total32 comparisons for the V350 persistent-maxima correctness repair; fixed before event fitting/scoring, not event-score tuning'),
        ['variants','model_lock_sha256','probability_lock_sha256'])
    list(run_pool(decode_one,[(ctx,s,configs) for s in samples],workers))
    predictions={v['name']:{s['dataset']:sha(ctx.out/'candidate_graphs'/v['name']/f"{s['dataset']}.npz") for s in samples} for v in configs}
    stable_lock(ctx.out/'event_round_lock.json',dict(created=now(),variants=configs,predictions=predictions,
        samples=len(samples),both_source_directions_frozen=True,outer_scores_exposed=False),['variants','predictions','samples'])
    import json
    import pandas as pd
    records=[]
    for v in configs:
        for sample in samples:
            ledger=read_json(ctx.out/'edit_ledgers'/v['name']/f"{sample['dataset']}.json")
            for action in ledger['actions']:
                records.append(dict(dataset=sample['dataset'],variant=v['name'],candidate=action['candidate'],
                    kind=action['kind'],value=action['value'],event_probability=action['event_probability'],
                    removed_edges=json.dumps(action['removed_edges']),added_edges=json.dumps(action['added_edges']),
                    canonical_nodes=json.dumps(action['canonical_nodes']),native_support=json.dumps(action['native_support']),
                    owner_alternatives=json.dumps(action['owner_alternatives']),solver_status=action['solver_status']))
    pd.DataFrame(records,columns=['dataset','variant','candidate','kind','value','event_probability','removed_edges','added_edges',
        'canonical_nodes','native_support','owner_alternatives','solver_status']).to_parquet(ctx.out/'event_edit_ledger.parquet',index=False)


def profile_decode(ctx):
    """Measure the largest fixed candidate pool before the complete graph round."""
    import resource
    lock=read_json(ctx.out/'event_model_lock.json')
    sizes={s['dataset']:read_json(ctx.out/'events'/lock['source_pools']['6bba' if s['embryo']=='44b6' else '44b6']/f"{s['dataset']}.json")['alternatives'] for s in ctx.samples()}
    sample=max(ctx.samples(),key=lambda s:sizes[s['dataset']]);rows=[]
    probability_receipt=ctx.out/'event_probabilities'/f"{sample['dataset']}.json"
    while not probability_receipt.exists():time.sleep(10)
    prediction=read_json(probability_receipt)
    if prediction['inputs']['model_lock_sha256']!=sha(ctx.out/'event_model_lock.json') or prediction['sha256']!=sha(probability_receipt.with_suffix('.npz')):
        raise ValueError('Dense profile requires fixed, verified opposite-source probabilities')
    # Reuse the ordinary fingerprinted decoder outputs in the subsequent round.
    # This is a label-free resource profile, not a threshold selection round.
    for config in variants():
        start=time.perf_counter();result=decode_one((ctx,sample,[config]))
        receipt=read_json(ctx.out/'candidate_graphs'/config['name']/f"{sample['dataset']}.json")
        row=dict(variant=config['name'],seconds=time.perf_counter()-start,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            decode=receipt['decode'],graph_sha256=receipt['sha256'])
        rows.append(row);print('event_dense_decode_profile',sample['dataset'],row,flush=True)
        write_json(ctx.out/'event_dense_decode_profile.json',dict(dataset=sample['dataset'],
            alternatives=sizes[sample['dataset']],model_lock_sha256=sha(ctx.out/'event_model_lock.json'),
            profiled_clip_probability_sha256=prediction['sha256'],outer_scores_exposed=False,
            selection='largest label-free fixed candidate pool; no threshold adjustments',rows=rows))
    return rows


def tables(ctx):
    """Stream literal feature/label parquet contracts without duplicating crops."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    lock=read_json(ctx.out/'event_model_lock.json');samples=ctx.samples()
    feature_path=ctx.out/'event_features.parquet';label_path=ctx.out/'evaluation/training_labels.parquet'
    stamp=dict(model_lock_sha256=sha(ctx.out/'event_model_lock.json'),feature_names=EVENT_FEATURES,
               schema_version='prediction_event_paths_v3_2')
    stamp['event_inputs']={s['dataset']:sha(ctx.out/'events'/lock['source_pools']['6bba' if s['embryo']=='44b6' else '44b6']/f"{s['dataset']}.npz") for s in samples}
    manifest_path=ctx.out/'event_table_manifest.json'
    if manifest_path.exists():
        previous=read_json(manifest_path)
        if previous['inputs']!=stamp or previous['feature_sha256']!=sha(feature_path) or previous['label_sha256']!=sha(label_path):
            raise ValueError('Stale event parquet tables')
        return previous
    feature_writer=None;label_writer=None;features=labels_count=0
    feature_tmp=feature_path.with_suffix('.tmp.parquet');label_tmp=label_path.with_suffix('.tmp.parquet')
    for sample in samples:
        name=sample['dataset'];source='6bba' if sample['embryo']=='44b6' else '44b6'
        pool=lock['source_pools'][source];h=load_graph(ctx.out/'events'/pool/f'{name}.npz')
        b=load_graph(ctx.incumbent(name));nodes=b['nodes'];gh=graph_hash(nodes,b['edges'])
        for start in range(0,len(h['events']),100000):
            end=min(len(h['events']),start+100000);ev=h['events'][start:end];size=len(ev)
            columns=dict(dataset=pa.array([name]*size),proposal_pool=pa.array([pool]*size),graph_hash=pa.array([gh]*size),
                candidate_index=pa.array(np.arange(start,end,dtype=np.int64)),split_time=pa.array(nodes[ev[:,0],1].astype(np.int16)),
                crop_index=pa.array(h['crop_index'][start:end]),
                uniform_anchor=pa.array(nodes[ev[:,0],0].astype(np.int64)%CONFIGS[pool].uniform_modulus==0),
                proposal_inclusion_given_trigger=pa.array(np.ones(size,np.float32)))
            for j,key in enumerate(['parent_id','daughter1_id','daughter2_id','daughter1_next_id','daughter2_next_id']):
                columns[key]=pa.array(np.where(ev[:,j]>=0,nodes[ev[:,j],0],-1).astype(np.int64))
            for j,key in enumerate(EVENT_FEATURES):columns[key]=pa.array(h['event_features'][start:end,j])
            table=pa.table(columns)
            if feature_writer is None:feature_writer=pq.ParquetWriter(feature_tmp,table.schema,compression='zstd',use_dictionary=['dataset','proposal_pool','graph_hash'])
            feature_writer.write_table(table);features+=size
        # This separate training table uses each sample only as the source in
        # its opposite-embryo direction, with its source-selected pool.
        source=sample['embryo'];pool=lock['source_pools'][source]
        lp=ctx.out/'evaluation/event_labels'/pool/f'{name}.npz';l=load_graph(lp);size=len(l['index'])
        columns=dict(dataset=pa.array([name]*size,type=pa.string()),source_embryo=pa.array([source]*size,type=pa.string()),
            proposal_pool=pa.array([pool]*size,type=pa.string()),candidate_index=pa.array(l['index']),
            event_bag=pa.array(l['bag']),source_label=pa.array(l['y']),positive_bag_weight=pa.array(l['weight']),
            parent_group_sampling_fraction=pa.array(l['inclusion']),matcher_revision=pa.array([ctx.metric_revision]*size,type=pa.string()))
        table=pa.table(columns)
        if label_writer is None:label_writer=pq.ParquetWriter(label_tmp,table.schema,compression='zstd')
        label_writer.write_table(table);labels_count+=size
        print('event_tables',name,features,labels_count,flush=True)
    feature_writer.close();label_writer.close();feature_tmp.replace(feature_path);label_tmp.replace(label_path)
    receipt=dict(created=now(),inputs=stamp,feature_rows=features,supported_training_rows=labels_count,
        feature_sha256=sha(feature_path),label_sha256=sha(label_path),feature_bytes=feature_path.stat().st_size,
        label_bytes=label_path.stat().st_size,label_path=str(label_path),feature_path=str(feature_path),
        features_have_no_gt_fields=True,labels_separate_from_inference=True,
        sampling_metadata_limitation='Frozen label NPZ inclusion is the parent-group sampled fraction, not exact row propensity when hard/ordinary strata mix. Export names it parent_group_sampling_fraction. Fits use event/group-balanced weights, never inverse propensity weighting; labels and fitted inputs remain unchanged.',
        proposal_sampling='Every enumerated pair/path within a selected trigger has conditional inclusion1; trigger selection and ID-modulus anchors are deterministic, so no unconditional random-sampling propensity is claimed. Dataset plus parent_id is the prediction event grouping key; source event bags are separate label fields.',
        note='Dataset/candidate IDs are keys, not model inputs; one prediction-selected pool per clip; crops remain per-clip uint8 NPZ only')
    write_json(manifest_path,receipt);return receipt


def apply(ctx,sample,nodes,edges,model='image_20260909',threshold=.2,replace=False,existing_only=False,source=None,native=None,image=None):
    """Annotation-free V360/V370 API; rebuild graph-dependent proposals/inputs."""
    from .decode import divisions
    from .event_model import load_image,predict_image,predict_logistic
    config=read_json(ctx.out/'event_inference_config.json')
    if source is None:
        if sample.get('embryo') not in ['44b6','6bba']:raise ValueError('Specify a frozen source model for an unseen embryo')
        source='6bba' if sample['embryo']=='44b6' else '44b6'
    pool=config['source_pools'][source]
    if native is None:
        from .features import build as build_features
        native,_=build_features(ctx,sample,nodes,edges)
    h,meta=build(nodes,edges,native,sample['physical_scale'],CONFIGS[pool])
    if model=='logistic':
        path=ctx.out/'event_models'/f'{source}_logistic.joblib'
        if sha(path)!=config['model_sha256'][f'{source}_logistic']:raise ValueError('Event model hash mismatch')
        p=predict_logistic(path,h['event_features'])
    else:
        seed=int(model.rsplit('_',1)[1]);path=ctx.out/'event_models'/f'{source}_image_{seed}.pt'
        if sha(path)!=config['model_sha256'][f'{source}_image_{seed}']:raise ValueError('Event model hash mismatch')
        if image is None:
            import zarr
            image=zarr.open_group(sample['image_path'],mode='r')['0']
        crops=extract_crops(image,nodes,h['crop_parents'],sample['physical_scale'])
        p=predict_image(load_image(path),h['event_features'],crops,h['crop_index'])
    out,receipt,ledger=divisions(nodes,edges,h,p,native,sample['physical_scale'],threshold=threshold,existing_only=existing_only,replace=replace)
    return nodes,out,dict(proposals=meta,decode=receipt,ledger=ledger,source=source,pool=pool,
                          input_graph_hash=graph_hash(nodes,edges),output_graph_hash=graph_hash(nodes,out),features_recomputed=True)


def run(ctx,args):
    stage=getattr(args,'variant',None) or 'all'
    if stage not in ['prepare','fit','fit-44b6','fit-6bba','tables','probabilities','profile-decode','decode','oracles','all']:raise ValueError(f'Unknown event stage: {stage}')
    if stage in ['fit-44b6','fit-6bba']:return fit(ctx,[stage.split('-')[1]])
    if stage=='profile-decode':return profile_decode(ctx)
    if stage=='oracles':return source_oracles(ctx,getattr(args,'workers',4))
    for name in ['prepare','fit','tables','probabilities','decode']:
        if stage in [name,'all']:
            if name in ['prepare','decode']:globals()[name](ctx,getattr(args,'workers',4))
            else:globals()[name](ctx)


def oracle_one(task):
    """Retrospective source-only feasible-edit diagnostic; never promoted."""
    from .decode import divisions
    from annotation_selection.metric_adapter import evaluate_graph
    ctx,sample,pool=task;name=sample['dataset'];b=load_graph(ctx.incumbent(name))
    h=load_graph(ctx.out/'events'/pool/f'{name}.npz');l=load_graph(ctx.out/'evaluation/event_labels'/pool/f'{name}.npz')
    native=load_graph(ctx.out/'features'/f'{name}.npz');gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
    row=next(r for r in ctx.eval_rows() if r['dataset']==name)
    probability=np.where(l['all_y']==1,.999999,.000001);results=[]
    for existing in [True,False]:
        arm='source_existing_pool' if existing else 'source_expanded_pool'
        e,receipt,ledger=divisions(b['nodes'],b['edges'],h,probability,native,sample['physical_scale'],
                                  threshold=.2,existing_only=existing,replace=False,margin=1.5)
        path=ctx.out/'evaluation/source_event_oracles'/arm/f'{name}.npz';save_arrays(path,nodes=b['nodes'],edges=e)
        score,_,_=evaluate_graph(name,b['nodes'],e,gt['nodes'],gt['edges'],sample['physical_scale'],row['estimated_total'])
        score.update(embryo=sample['embryo'],variant=arm,oracle_source=sample['embryo'],pool=pool,
                     inference_eligible=False,heuristic_not_global_bound=True,decode=receipt)
        write_json(path.with_suffix('.json'),score);results.append(score)
    return results


def source_oracles(ctx,workers=4):
    if not (ctx.out/'event_round_lock.json').exists():raise RuntimeError('Freeze all non-oracle predictions before retrospective source diagnostics')
    config=read_json(ctx.out/'event_candidate_coverage.json');rows=[]
    for result in run_pool(oracle_one,[(ctx,s,config['source_selected_pool'][s['embryo']]) for s in ctx.samples()],workers):rows.extend(result)
    from annotation_selection.metric_adapter import aggregate
    summary={}
    for arm in ['source_existing_pool','source_expanded_pool']:
        summary[arm]={}
        for embryo in ['44b6','6bba','pooled']:
            selected=[r for r in rows if r['variant']==arm and (embryo=='pooled' or r['embryo']==embryo)]
            summary[arm][embryo]=aggregate(selected,[r['dataset'] for r in selected])
    write_json(ctx.out/'source_event_oracle_summary.json',dict(created=now(),summary=summary,source_only=True,
        interpretation='legal incumbent-fixed-fork add-only heuristic; not a global bound; no target-oracle tuning or promotion',
        frozen_prediction_lock_sha256=sha(ctx.out/'event_round_lock.json')))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','prepare-fallback','fit','fit-44b6','fit-6bba','tables','probabilities','profile-decode','decode','oracles','all']);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--reverse',action='store_true',help='Process fallback samples in reverse order to share a locked queue efficiently')
    args=parser.parse_args();ctx=RunContext.default().check_outputs()
    if args.stage in ['fit-44b6','fit-6bba']:fit(ctx,[args.stage.split('-')[1]]);return
    if args.stage=='profile-decode':profile_decode(ctx);return
    if args.stage=='oracles':source_oracles(ctx,args.workers);return
    if args.stage=='prepare-fallback':
        samples=ctx.samples();trigger=None
        for embryo in ['44b6','6bba']:
            paths=[ctx.out/'evaluation/event_labels/base'/f"{s['dataset']}.json" for s in samples if s['embryo']==embryo]
            if not all(p.exists() for p in paths):continue
            events=[e for p in paths for e in read_json(p)['coverage']]
            covered=sum(e['covered'] for e in events)
            if covered/max(1,len(events))<.85:trigger=dict(source_embryo=embryo,covered=covered,total=len(events));break
        if trigger is None:raise RuntimeError('Fallback requires complete source coverage <85%')
        stable_lock(ctx.out/'event_fallback_trigger.json',dict(created=now(),**trigger,outer_scores_exposed=False),['source_embryo','covered','total'])
        ordered=samples[::-1] if args.reverse else samples
        list(run_pool(prepare_one,[(ctx,s,'fallback') for s in ordered],args.workers));return
    for stage in ['prepare','fit','tables','probabilities','decode']:
        if args.stage in [stage,'all']:
            if stage in ['prepare','decode']:globals()[stage](ctx,args.workers)
            else:globals()[stage](ctx)


if __name__=='__main__':main()
