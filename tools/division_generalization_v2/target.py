"""Predict both directions before starting a separate annotation-owning evaluator."""
import numpy as np
import hashlib
from .common import (WORK,RESULTS,PRIOR_WORK,DATA,inputs,read_json,write_json,write_csv,
                     load_graph,sha,verified_graph,digest)
from .screen import prediction_job,launch,run_matrix


def predict_all():
    freeze=read_json(RESULTS/'target_freeze.json')
    arms=[(arm,seed) for arm in freeze['qualified_exports']
          for seed in ((20260916,) if arm=='G30' else (20260916,314159))]
    for i,row in enumerate(inputs() if arms else [],1):
        source='44b6' if row['embryo']=='6bba' else '6bba'
        specs={};destinations={}
        for arm,seed in arms:
                selected=freeze['selected'][f'{arm}/{source}/{seed}']['selected']
                checkpoint=WORK/'training'/arm/source/str(seed)/f'checkpoint-{selected["step"]}.pt'
                key=f'{arm}-{seed}'
                specs[key]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),calibration=selected['calibration'])
                destinations[key]=WORK/'target'/arm/str(seed)/'predictions'/row['dataset']
        output=WORK/'target_matrix'/row['dataset']
        job=dict(row={k:row[k] for k in ('dataset','image_path','image_shape','physical_scale','metadata_sha256')},
            source=source,output=str(output),packages=specs,graph_path=row['baselines']['P0']['path'],
            graph_sha256=row['baselines']['P0']['sha256'],native_path=row['evidence']['path'],native_sha256=row['evidence']['sha256'])
        run_matrix(job,WORK/'target_matrix/jobs'/(row['dataset']+'.json'))
        for key,dest in destinations.items():
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to((output/key).resolve(),target_is_directory=True)
        print(f'Frozen target matrix: {i}/199, {len(arms)} packages',flush=True)
    for arm,seed in arms:
            root=WORK/'target'/arm/str(seed)
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
                    prediction_sha256=sha(path),unchanged_node_hash=hashlib.sha256(np.ascontiguousarray(graph['nodes']).tobytes()).hexdigest())
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
                    recovered_gt_edges=sorted((matches[a],matches[b]) for a,b in tp-old_tp),
                    lost_gt_edges=sorted((matches[a],matches[b]) for a,b in old_tp-tp),
                    recovered_gt_divisions=[int(gn[k,0]) for k in sorted(recovered)],
                    lost_gt_divisions=[int(gn[k,0]) for k in sorted(lost)],
                    recovered_divisions=sorted(recovered),lost_divisions=sorted(lost),
                    division_fp_added=sorted(newd.fp_forks-oldd.fp_forks),division_fp_removed=sorted(oldd.fp_forks-newd.fp_forks))
                write_json(root/'errors'/(row['dataset']+'.json'),detail)
                transitions.append({k:(len(v) if isinstance(v,list) else v) for k,v in detail.items()})
            for embryo in ('pooled','44b6','6bba'):
                rs=[r for r in scores if embryo=='pooled' or r['embryo']==embryo]
                a=aggregate(rs,[r['dataset'] for r in rs]);a.update(a.pop('counts'))
                a.update(arm=arm,seed=seed,embryo=embryo,clips=len(rs),status='measured',
                    target_met=a['score']>=.95,matched_nodes=sum(r['matched_nodes'] for r in rs),
                    node_identity_sha256=digest([(r['dataset'],r['unchanged_node_hash'],int(r['num_pred_nodes'])) for r in rs]))
                if a['node_identity_sha256']!=read_json(RESULTS/'node_identity.json')['corpora'][embryo]:
                    raise ValueError('Full target node corpus hash drift')
                (summaries if embryo=='pooled' else per_embryo).append(a)
    baseline=read_json(RESULTS/'baseline_validation.json')
    write_csv(RESULTS/'scores.csv',[*baseline['pooled'],*summaries])
    write_csv(RESULTS/'per_embryo_scores.csv',[*baseline['embryos'],*per_embryo])
    write_csv(RESULTS/'error_transitions.csv',transitions)
    write_json(RESULTS/'target_evaluation.json',dict(status='complete',pooled=summaries,embryos=per_embryo,
        exported_arms=freeze['qualified_exports'],full_graphs=True,independent_aggregation=True))
    return summaries
