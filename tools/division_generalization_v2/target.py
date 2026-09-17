"""Predict both directions before starting a separate annotation-owning evaluator."""
import numpy as np
from .common import (WORK,RESULTS,PRIOR_WORK,DATA,inputs,read_json,write_json,write_csv,
                     load_graph,sha,verified_graph)
from .screen import prediction_job,launch


def predict_all():
    freeze=read_json(RESULTS/'target_freeze.json')
    for arm in freeze['qualified_exports']:
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            root=WORK/'target'/arm/str(seed)
            for i,row in enumerate(inputs(),1):
                source='44b6' if row['embryo']=='6bba' else '6bba'
                selected=freeze['selected'][f'{arm}/{source}/{seed}']['selected']
                checkpoint=WORK/'training'/arm/source/str(seed)/f'checkpoint-{selected["step"]}.pt'
                job=prediction_job(row,source,checkpoint,root/'predictions'/row['dataset'],selected['calibration'])
                launch(job,root/'jobs'/(row['dataset']+'.json'))
                print(f'Frozen target prediction {arm}/{seed}: {i}/199',flush=True)
            write_json(root/'both_directions_predicted.json',dict(clips=199,seed=seed,arm=arm,
                freeze_sha256=sha(RESULTS/'target_freeze.json'),target_labels_opened=False),immutable=True)
    write_json(RESULTS/'all_target_predictions_complete.json',dict(arms=freeze['qualified_exports'],
        freeze_sha256=sha(RESULTS/'target_freeze.json'),both_directions_before_scoring=True),immutable=True)


def evaluate_all():
    from annotation_selection.metric_adapter import evaluate_graph,aggregate
    from center_comparison.pipeline import read_gt
    from center_comparison.tracking_index import division_flags
    from tracking_cellmot import division_metrics as dm
    from annotation_selection.metric_adapter import make_graph
    freeze=read_json(RESULTS/'target_freeze.json')
    if not (RESULTS/'all_target_predictions_complete.json').exists():
        raise ValueError('Complete both target directions before label access')
    summaries=[];per_embryo=[];transitions=[]
    for arm in freeze['qualified_exports']:
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            root=WORK/'target'/arm/str(seed);scores=[]
            for row in inputs():
                source='44b6' if row['embryo']=='6bba' else '6bba'
                app=freeze['selected'][f'{arm}/{source}/{seed}']['selected']['application']
                path=root/'predictions'/row['dataset']/app/(row['dataset']+'.npz')
                graph=load_graph(path);old=verified_graph(row)
                np.testing.assert_array_equal(graph['nodes'],old['nodes'])
                gn,ge=read_gt(DATA,row['dataset'],row['physical_scale'])
                score,matches,tp=evaluate_graph(row['dataset'],graph['nodes'],graph['edges'],gn,ge,
                    row['physical_scale'],row['estimated_total'])
                score.update(arm=arm,seed=seed,embryo=row['embryo'],application=app,
                    prediction_sha256=sha(path),unchanged_node_hash=sha(row['baselines']['P0']['path']))
                write_json(root/'evaluation'/(row['dataset']+'.json'),score);scores.append(score)
                with np.load(PRIOR_WORK/'evaluation/P0'/(row['dataset']+'.npz')) as f:
                    old_tp=set(map(tuple,f['tp_edges']))
                gtset=set(map(tuple,ge));gtout={int(a) for a,_ in ge};gtin={int(b) for _,b in ge}
                def false_edges(edges):
                    return {tuple(map(int,e)) for e in edges if (matches.get(int(e[0])) in gtout or
                        matches.get(int(e[1])) in gtin) and
                        (matches.get(int(e[0])),matches.get(int(e[1]))) not in gtset}
                fp,oldfp=false_edges(graph['edges']),false_edges(old['edges'])
                oldg,_=make_graph(old['nodes'],old['edges']);newg,_=make_graph(graph['nodes'],graph['edges']);gt,_=make_graph(gn,ge)
                oldd=dm.score_divisions(oldg,gt,tuple(row['physical_scale']),7.)
                newd=dm.score_divisions(newg,gt,tuple(row['physical_scale']),7.)
                recovered={k for k,v in newd.scores.items() if v and not oldd.scores[k]}
                lost={k for k,v in oldd.scores.items() if v and not newd.scores[k]}
                detail=dict(dataset=row['dataset'],source=source,arm=arm,seed=seed,embryo=row['embryo'],
                    recovered_edges=sorted(tp-old_tp),lost_edges=sorted(old_tp-tp),fp_added=sorted(fp-oldfp),fp_removed=sorted(oldfp-fp),
                    recovered_divisions=sorted(recovered),lost_divisions=sorted(lost),
                    division_fp_added=sorted(newd.fp_forks-oldd.fp_forks),division_fp_removed=sorted(oldd.fp_forks-newd.fp_forks))
                write_json(root/'errors'/(row['dataset']+'.json'),detail)
                transitions.append({k:(len(v) if isinstance(v,list) else v) for k,v in detail.items()})
            for embryo in ('pooled','44b6','6bba'):
                rs=[r for r in scores if embryo=='pooled' or r['embryo']==embryo]
                a=aggregate(rs,[r['dataset'] for r in rs]);a.update(a.pop('counts'))
                a.update(arm=arm,seed=seed,embryo=embryo,clips=len(rs),status='measured',
                    target_met=a['score']>=.95,matched_nodes=sum(r['matched_nodes'] for r in rs))
                (summaries if embryo=='pooled' else per_embryo).append(a)
    baseline=read_json(RESULTS/'baseline_validation.json')
    write_csv(RESULTS/'scores.csv',[*baseline['pooled'],*summaries])
    write_csv(RESULTS/'per_embryo_scores.csv',[*baseline['embryos'],*per_embryo])
    write_csv(RESULTS/'error_transitions.csv',transitions)
    write_json(RESULTS/'target_evaluation.json',dict(status='complete',pooled=summaries,embryos=per_embryo,
        exported_arms=freeze['qualified_exports'],full_graphs=True,independent_aggregation=True))
    return summaries
