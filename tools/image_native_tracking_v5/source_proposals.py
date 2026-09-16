"""Source-only full-field proposal diagnostics under the final frozen J objective."""
import time
from .common import *


def run():
    from .predict_batch import transform,VARIANTS
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v3.common import validate
    pilots=read(OUT/'density_selection.json')['pilots']
    missing=[r for r in pilots if not (OUT/'banks'/r['embryo']/'P1'/f"{r['dataset']}.npz").exists()]
    if missing:
        from . import native_adapter as native
        from .banks import cached
        with gpu_aux():
            model=native.load()
            for r in missing:cached(model,r['dataset'],r['embryo'],True)
            del model
    records=[]
    with cpu_batch():
        for pilot in pilots:
            name=pilot['dataset'];source=pilot['embryo'];row=next(r for r in inventory() if r['dataset']==name)
            base=graph(name);c=arrays(OUT/'observations'/f'{name}.npz');gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
            joint=read(OUT/'calibration'/f'{source}_J.json');dc=arrays(OUT/'deepcenter'/f'{name}.npz')['confidence']
            for variant in ['J_native_frozen','P_union_native_J','P_DC_native_J']:
                dest=OUT/'source_pilots'/f'{name}_{variant}_final_objective.json'
                if dest.exists():records.append(read(dest));continue
                cfg=VARIANTS[variant];common=arrays(OUT/'banks'/source/('P0' if cfg['pop']=='P0' else 'P1')/f'{name}.npz')
                start=time.monotonic();n,e,decode=transform(c,base,common,{},cfg,joint,None,dc)
                validate(n,e,row['image_shape'])
                result,matches,_=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
                baseline=read(OUT/'evaluation/C0'/f'{name}.json');bm=arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches']
                oldids=set(map(int,base['nodes'][:,0]));ids=set(map(int,n[:,0]));oldgt=set(map(int,bm[:,1]));newgt=set(matches.values())
                r=dict(dataset=name,source=source,source_only=True,variant=variant,score=result,C0=baseline,
                    added_nodes=len(ids-oldids),removed_nodes=len(oldids-ids),new_GT_matches=len(newgt-oldgt),lost_GT_matches=len(oldgt-newgt),
                    matched_new_nodes=sum(int(i) not in oldids for i in matches),seconds=time.monotonic()-start,
                    decode={k:v for k,v in decode.items() if k!='windows'},
                    source_calibration_sha256=sha(OUT/'calibration'/f'{source}_J.json'),
                    interpretation='complete source-embryo pilot with final objective; sparse unmatched proposals are unknown, not confirmed false cells')
                write(dest,r);records.append(r);print('Source proposals',source,variant,result['edge_tp'],result['edge_fp'],result['division_tp'],result['division_fp'],flush=True)
    # Record the conditional decision explicitly; this is a compute-routing judgment,
    # not a negative conclusion about dense detection from unlabeled voxels.
    write(OUT/'P2_decision.json',dict(at=now(),run=False,status='conditional dense-detector extension not scheduled',
        based_on='source-only final-objective full-field proposal pilots; no opposite-direction score tuning',
        source_pilots=[{k:r[k] for k in ['dataset','source','variant','added_nodes','removed_nodes','new_GT_matches','lost_GT_matches','matched_new_nodes','source_calibration_sha256']} for r in records],
        reasoning='P1 discovery and the P0/P1 representation factorial are running in full. Source coverage alone does not establish reliable dense teacher labels; fixed-node association headroom has not been exhausted. Complete the registered H/N/P evidence before spending a new dense-detector training arm.',
        real_dense_voxels_supervised=False,sparse_unknowns_remain_unknown=True,
        optional_FOCUS='no already-authorized checkpoint found; no gated terms accepted or microscopy uploaded'))


if __name__=='__main__':run()
