"""Full expected-sample, fresh-rematching outer graph evaluation."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np
import pandas as pd

from .common import OUT,SEED,digest,graph_hash,load_graph,now,read_json,sha,stage,write_json
from .filter_graph import PreparedFilter,filtered
from .metric_adapter import aggregate,evaluate_graph

FRACTIONS=[.9,.8,.7,.5,.3,.1]


def variants(model_ids):
    out=[dict(model_id='identity',policy='identity',seed=SEED,requested_keep=1.)]
    for seed in range(20):
        for policy in ['random_nodes','random_tracklets']:
            for r in FRACTIONS:out.append(dict(model_id='random',policy=policy,seed=seed,requested_keep=r))
    for model in model_ids:
        prefix='quality' if model=='quality' else 'membership'
        for policy in [prefix+'_nodes',prefix+'_tracklets',prefix+'_tracklets_fork_protected']:
            for r in FRACTIONS:out.append(dict(model_id=model,policy=policy,seed=SEED,requested_keep=r))
    out.append(dict(model_id='quality',policy='absolute_response_0.05',seed=SEED,requested_keep=-1.))
    out.append(dict(model_id='identity',policy='isolated_node_cleanup',seed=SEED,requested_keep=-1.))
    # Hindsight headroom; never eligible for model/rule selection.
    for policy in ['oracle_nodes','oracle_tracklets']:
        for r in FRACTIONS:out.append(dict(model_id='oracle',policy=policy,seed=SEED,requested_keep=r))
    for v in out:v['variant_id']=f'{v["model_id"]}__{v["policy"]}__s{v["seed"]}__r{v["requested_keep"]:g}'
    return out


def confusion(labels,keep):
    n=len(labels);a=int(labels.sum());k=int(keep.sum());tp=int(labels[keep].sum());fn=a-tp;fp=k-tp;tn=n-a-fp
    denominator=float(k)*a*(n-a)*(n-k)
    return dict(candidate_nodes=n,baseline_matched_nodes=a,kept_matched_nodes=tp,
                natural_membership_prevalence=a/n if n else None,realized_keep=k/n if n else None,
                annotation_recall=tp/a if a else None,specificity=tn/(n-a) if n>a else None,
                precision=tp/k if k else None,phi=(tp*tn-fp*fn)/np.sqrt(denominator) if denominator>0 else None,
                deleted_annotation_rate=fn/(n-k) if n>k else None,deleted_candidates=n-k,deleted_annotations=fn,
                membership_tp=tp,membership_fp=fp,membership_fn=fn,membership_tn=tn)


def evaluate_one(task):
    row,source=task;name=row['dataset']
    path=OUT/'evaluation/score_samples'/f'{name}.json'
    pp=OUT/'predictions'/source/f'{name}.npz';bp=OUT/'baseline/clean'/f'{name}.npz';gp=OUT/'evaluation/gt'/f'{name}.npz'
    stamp=dict(prediction_sha=sha(pp),baseline_sha=sha(bp),gt_sha=sha(gp),estimate=row['estimated_total'],
               prediction_lock_sha=sha(OUT/'prediction_lock.json'))
    if path.exists():
        cached=read_json(path)
        if cached['inputs']!=stamp:raise ValueError('Scored sample drift')
        return name,cached['seconds'],True
    start=time.perf_counter()
    b=load_graph(bp);gt=load_graph(gp)
    with np.load(pp) as f:scores={k:f[k] for k in f.files}
    n,e=b['nodes'],b['edges'];prepare=PreparedFilter(n,e)
    base,matches,base_tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    labels=np.array([int(i in matches) for i in n[:,0]],np.uint8)
    overlap_group=read_json(OUT/'fold_manifest.json')['overlap_groups'][name]
    with np.load(OUT/'evaluation/membership'/f'{name}.npz') as f:
        if not np.array_equal(labels,f['annotation_label']):raise ValueError('Baseline node matching changed')
    results=[];index={int(i):j for j,i in enumerate(n[:,0])}
    for v in variants(list(scores)):
        policy=v['policy'];model=v['model_id'];requested=v['requested_keep']
        if policy=='identity':keep=np.ones(len(n),bool)
        elif policy=='absolute_response_0.05':keep=b['features'][:,5]>=.05
        elif policy=='isolated_node_cleanup':keep=np.isin(n[:,0],e.ravel())
        else:
            s=labels.astype(float) if model=='oracle' else np.zeros(len(n)) if model=='random' else scores[model]
            keep=prepare.mask(s,requested,policy.replace('oracle','membership'),v['seed'],key=model)
        nn,ee=filtered(n,e,keep)
        if policy=='identity':
            er=base.copy();tp=base_tp
        else:
            er,_,tp=evaluate_graph(name,nn,ee,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        old_kept=sum(keep[index[a]] and keep[index[c]] for a,c in base_tp)
        stats=confusion(labels,keep)
        er.update(v);er.update(stats)
        er.update(fold=f'{source}->{row["embryo"]}',embryo=row['embryo'],source=source,lane='clean',
                  overlap_group=overlap_group,
                  baseline_tp=len(base_tp),baseline_tp_endpoints_kept=int(old_kept),baseline_tp_rematched_surviving=len(tp&base_tp),
                  newly_recovered_tp=len(tp-base_tp),baseline_edge_fp=base['edge_fp'],baseline_num_pred_nodes=len(n),
                  count_ratio=len(nn)/row['estimated_total'],baseline_count_ratio=len(n)/row['estimated_total'])
        results.append(er)
    write_json(path,dict(dataset=name,inputs=stamp,seconds=time.perf_counter()-start,results=results))
    return name,time.perf_counter()-start,False


def run(args):
    if not (OUT/'prediction_lock.json').exists():raise ValueError('Freeze and reproduce BOTH directions first')
    lock=read_json(OUT/'prediction_lock.json')
    if not lock['gt_unavailable_reproduction_passed']:raise ValueError('GT-unavailable inference did not pass')
    inventory=read_json(OUT/'inventory.json');manifest=read_json(OUT/'fold_manifest.json')
    names=[r['dataset'] for r in inventory]
    if set(names)!=set(manifest['expected_samples']):raise ValueError('Expected sample set changed')
    tasks=[]
    for r in inventory:
        source=next(d['source'] for d in manifest['directions'] if d['outer']==r['embryo'])
        pp=OUT/'predictions'/source/f'{r["dataset"]}.npz'
        if not pp.exists() or sha(pp)!=lock['prediction_hashes'][str(pp.relative_to(OUT/'predictions'))]:
            raise ValueError('Missing or changed prediction')
        tasks.append((r,source))
    if args.limit:tasks=tasks[:args.limit]
    stage('S090','running',evaluation_start=now(),expected_samples=len(tasks),metric='pinned official evaluate -> per_sample_metrics -> summarise')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate_one,t):t[0]['dataset'] for t in tasks}
        for i,future in enumerate(as_completed(futures),1):
            name,seconds,resumed=future.result()
            print(f'Official evaluation {i}/{len(tasks)} {name}: {seconds:.1f}s {"resumed" if resumed else ""}',flush=True)
    rows=[]
    for r,source in tasks:rows.extend(read_json(OUT/'evaluation/score_samples'/f'{r["dataset"]}.json')['results'])
    df=pd.DataFrame(rows);df.to_csv(OUT/'score_rows.csv',index=False)
    df.to_parquet(OUT/'score_rows.parquet',index=False)
    write_json(OUT/'score_manifest.json',dict(samples=[t[0]['dataset'] for t in tasks],rows=len(rows),variants=df.variant_id.nunique(),
                                           complete=not bool(args.limit),created=now()))
    stage('S090','pilot_complete' if args.limit else 'complete',samples=len(tasks),score_rows=len(rows),variants=df.variant_id.nunique())
