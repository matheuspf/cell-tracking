"""Exact scoring of the full motion-bypass pipeline and coordinate repairs."""
import time

import numpy as np

from annotation_selection.metric_adapter import evaluate_graph

from .common import OUT,V1,export_nodes,inventory,load_graph,read_json,run_pool,save_arrays,validate,write_json
from .evaluate import baseline_fp_set,collect


def one(row):
    name=row['dataset'];b=load_graph(V1/'baseline/public'/f'{name}.npz')
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
    baseline_tp=set(map(tuple,m['tp_edges']));bm={int(i):int(j) for i,j in zip(b['nodes'][:,0],m['matched_gt_id']) if j>=0}
    baseline_fp=baseline_fp_set(b['nodes'],b['edges'],gt,bm)
    raw=load_graph(OUT/'replay/no_motion'/name/'final_export.npz')
    assert read_json(OUT/'replay/no_motion'/name/'complete.json')['annotation_access_blocked']
    normalized=export_nodes(raw['nodes']);normalized[:,2:]=np.clip(normalized[:,2:],0,np.array(row['image_shape'][1:])-1)
    pred=load_graph(OUT/'predictions'/f'{name}.npz')
    corrected=b['nodes'].copy();corrected[:,2:]=np.clip(corrected[:,2:],0,np.array(row['image_shape'][1:])-1)
    graphs={'bypass_motion':(raw['nodes'],raw['edges']),
        'bypass_motion_bounds':(normalized,raw['edges']),
        'E_hgb_m0.5_bounds':(corrected,pred['E_hgb_m0.5__edges'])}
    for v,(n,e) in graphs.items():
        dest=OUT/'evaluation/scores'/v/f'{name}.json'
        if dest.exists():continue
        tic=time.perf_counter();allow=v=='bypass_motion'
        valid=validate(n,e,row['image_shape'],allow_legacy_bounds=allow)
        save_arrays(OUT/'candidate_graphs'/v/f'{name}.npz',nodes=n,edges=e)
        r,matches,tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        fp=baseline_fp_set(n,e,gt,matches)
        r.update(variant=v,embryo=row['embryo'],source='public_fixed_pipeline' if 'bypass' in v else ('6bba' if row['embryo']=='44b6' else '44b6'),
            kind='bypass' if 'bypass' in v else 'bounds',seconds=time.perf_counter()-tic,requested_keep=1.,
            realized_keep=len(n)/len(b['nodes']),baseline_num_pred_nodes=len(b['nodes']),baseline_tp=len(baseline_tp),
            baseline_tp_rematched_surviving=len(baseline_tp&tp),newly_recovered_tp=len(tp-baseline_tp),lost_tp=len(baseline_tp-tp),
            removed_fp=len(baseline_fp-fp),introduced_fp=len(fp-baseline_fp),out_of_bounds=valid['out_of_bounds'],
            lane='exploratory_public_upstream_contaminated',baseline_matched_nodes=len(bm))
        write_json(dest,dict(result=r,annotation_access_blocked_at_inference=True))
    return name


def run(args):
    rows=inventory();rows=rows[:args.limit] if args.limit else rows
    list(run_pool(one,rows,args.workers));collect()
