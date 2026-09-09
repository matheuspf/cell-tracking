"""Fresh official whole-graph evaluation, locked complete rounds, exact weighting."""
import time
import numpy as np
import pandas as pd
from .common import *

COUNTS=['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes']

def graph_path(ctx,variant,name):
    if variant=='incumbent':return ctx.incumbent(name)
    if variant=='historical_v1':return ctx.v1/'baseline/public'/f'{name}.npz'
    return ctx.out/'candidate_graphs'/variant/f'{name}.npz'

def freeze(ctx,variants,round_name):
    if len(variants)!=len(set(variants)):raise ValueError('Duplicate variants')
    names=[r['dataset'] for r in ctx.samples()]
    roots={}
    for variant in variants:
        if variant not in ['incumbent','historical_v1']:
            actual={p.stem for p in (ctx.out/'candidate_graphs'/variant).glob('*.npz')}
            if actual!=set(names):raise ValueError(f'Incomplete variant {variant}: {len(actual)} / {len(names)}')
        roots[variant]={name:sha(graph_path(ctx,variant,name)) for name in names}
    p=ctx.out/'rounds'/f'{round_name}.json'
    existing=read_json(p) if p.exists() else {}
    data=dict(created=existing.get('created',now()),variants=roots,
        inputs_sha256=sha(ctx.out/'inputs.json'),scorer_sha256=scorer_hash(ctx),
        code=code_hashes(ctx),exposure='Both direction predictions frozen before this round comparative scoring',
        validation='operational exploratory; repeated embryos and upstream public checkpoint contamination')
    # Later unrelated module additions cannot invalidate already completed graphs.
    if existing:
        for k in ['variants','inputs_sha256','scorer_sha256']:
            if existing[k]!=data[k]:raise RuntimeError(f'Round input drift: {round_name} {k}')
        return existing
    other=set()
    for p0 in (ctx.out/'rounds').glob('*.json'):other.update(read_json(p0)['variants'])
    if len(other|set(variants))-int('historical_v1' in other|set(variants))>32:raise ValueError('32 variant cap exceeded')
    write_json(p,data,immutable=True);return data

def scorer_hash(ctx):
    paths=[ctx.repo/'tools/annotation_selection/metric_adapter.py',ctx.repo/'handover/annotation-selection-v1/analysis.py',
        ctx.repo/'tools/strong_tracker_v3/evaluate.py',*sorted((ctx.official/'src/tracking_cellmot').glob('*.py'))]
    return digest({str(p.relative_to(ctx.repo)):sha(p) for p in paths})

def one(task):
    ctx,row,variant,locked,scorer=task;name=row['dataset'];path=graph_path(ctx,variant,name)
    stamp=dict(graph_file_sha256=sha(path),gt_sha256=sha(ctx.v1/'evaluation/gt'/f'{name}.npz'),
        estimated_total=row['estimated_total'],physical_scale=row['physical_scale'],scorer_hash=scorer,
        input_manifest_sha256=sha(ctx.out/'inputs.json'))
    if stamp['graph_file_sha256']!=locked:raise RuntimeError('Frozen prediction changed')
    dest=ctx.out/'evaluation/scores'/variant/f'{name}.json'
    if dest.exists():
        if read_json(dest)['inputs']!=stamp:raise RuntimeError(f'Score cache drift {variant}/{name}')
        return name+' '+variant+' exact cache'
    from annotation_selection.metric_adapter import evaluate_graph
    start=time.perf_counter();g=load_graph(path);gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
    n,e=g['nodes'],g['edges'];valid=validate(n,e,row['image_shape'],allow_legacy_bounds=variant=='historical_v1')
    result,matches,tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    recovered={(matches[int(a)],matches[int(b)]) for a,b in tp}
    gout=set(gt['edges'][:,0]);gin=set(gt['edges'][:,1]);gs=set(map(tuple,gt['edges']))
    fp={(int(a),int(b)) for a,b in e if (matches.get(int(a)),matches.get(int(b))) not in gs and
        (matches.get(int(a)) in gout or matches.get(int(b)) in gin)}
    result.update(variant=variant,embryo=row['embryo'],seconds=time.perf_counter()-start,
        graph_file_sha256=stamp['graph_file_sha256'],scorer_hash=scorer,gt_sha256=stamp['gt_sha256'],
        input_manifest_sha256=stamp['input_manifest_sha256'],**{f'validation_{k}':v for k,v in valid.items()})
    if len(fp)!=result['edge_fp'] or len(recovered)!=result['edge_tp']:raise ValueError('Sparse edge audit count mismatch')
    save_arrays(ctx.out/'evaluation/matches'/variant/f'{name}.npz',
        matched_ids=np.array(sorted(matches.items()),np.int64).reshape(-1,2),
        tp_pairs=np.array(sorted(tp),np.int64).reshape(-1,2),fp_pairs=np.array(sorted(fp),np.int64).reshape(-1,2),
        recovered_gt_edges=np.array(sorted(recovered),np.int64).reshape(-1,2))
    write_json(dest,dict(inputs=stamp,result=result));return f'{name} {variant} {result["seconds"]:.1f}s'

def collect(ctx):
    from annotation_selection.metric_adapter import aggregate
    rows=[];summary=[];expected=[r['dataset'] for r in ctx.samples()]
    for d in sorted((ctx.out/'evaluation/scores').iterdir()):
        if not d.is_dir():continue
        rr=[read_json(p)['result'] for p in sorted(d.glob('*.json'))]
        if {r['dataset'] for r in rr}!=set(expected):continue
        rows.extend(rr)
        for embryo in ['44b6','6bba','pooled']:
            subset=[r for r in rr if embryo=='pooled' or r['embryo']==embryo]
            ids=[r['dataset'] for r in ctx.samples() if embryo=='pooled' or r['embryo']==embryo]
            a=aggregate(subset,ids)
            summary.append(dict(variant=d.name,embryo=embryo,samples=len(subset),
                **{k:v for k,v in a.items() if k!='counts'},**{k:v for k,v in a['counts'].items() if k not in a},seconds=sum(r['seconds'] for r in subset)))
    baseline={r['embryo']:r['score'] for r in summary if r['variant']=='incumbent'}
    for r in summary:r['delta_v2']=r['score']-baseline[r['embryo']]
    pd.DataFrame(rows).to_csv(ctx.out/'score_rows.csv',index=False)
    pd.DataFrame(summary).to_csv(ctx.out/'operating_points.csv',index=False)
    write_json(ctx.out/'score_completeness.json',dict(expected_samples=199,variants=sorted({r['variant'] for r in rows}),
        score_rows=len(rows),fresh_whole_graph_matching=True))
    return summary

def run(ctx,args):
    variants=args.variant.split(',') if args.variant else ['incumbent']
    lock=freeze(ctx,variants,args.round or 'round_'+variants[0]);rows=ctx.eval_rows()
    if args.limit:raise ValueError('Comparative scoring requires all 199 expected clips')
    tasks=[(ctx,r,v,lock['variants'][v][r['dataset']],lock['scorer_sha256']) for r in rows for v in variants]
    list(run_pool(one,tasks,args.workers));summary=collect(ctx)
    if 'incumbent' in variants:
        base=next(r for r in summary if r['variant']=='incumbent' and r['embryo']=='pooled')
        expected=dict(edge_tp=123023,edge_fp=4930,edge_fn=5860,division_tp=29,division_fp=92,division_fn=122,num_pred_nodes=4108943)
        assert all(base[k]==v for k,v in expected.items()),base
        assert abs(base['score']-0.9342063149703403)<1e-10,base
        write_json(ctx.out/'baseline_verification.json',dict(**base,parity=True,fresh_official_scoring=True))
    for r in summary:
        if r['embryo']=='pooled':print(r['variant'],r['score'],r['delta_v2'],flush=True)
