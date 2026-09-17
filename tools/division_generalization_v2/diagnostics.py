"""Post-freeze raw-scene panels, exact error identities and timing attribution."""
from collections import Counter
from pathlib import Path
import html
import gzip
import json
import numpy as np

from .common import (DATA,WORK,RESULTS,PRIOR_WORK,inputs,read_json,write_json,write_csv,load_graph,verified_graph,verified_evidence,digest,sha)


def render_scene(row,t,position,destination,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pipeline_error_training.crops import Images
    from .scenes import crop
    image=Images(row['image_path'])
    scene=crop(image,int(t),np.asarray(position))
    fig,axes=plt.subplots(2,7,figsize=(14,4.4),layout='constrained')
    for scale in range(2):
        for k,dt in enumerate(range(-3,4)):
            radius=13*(2**scale)
            ax=axes[scale,k]
            ax.imshow(scene[k,scale,0].max(0),origin='lower',cmap='gray',vmin=0,vmax=255,
                      extent=(-radius,radius,-radius,radius))
            ax.plot(0,0,'+',color='#e79b38',markersize=5)
            ax.set_title(f't={int(t)+dt}',fontsize=8)
            ax.set_xlabel('X offset (µm)',fontsize=7)
            if k==0:ax.set_ylabel(f'{"Fine" if scale==0 else "Coarse"} Y (µm)',fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(title+'\nNative Z maximum projections; fixed per-frame 1/99% normalization; marker is the displayed center',fontsize=10)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=140);plt.close(fig)


def lock_strata():
    from .dataset import SourceDataset
    values={}
    for source in ('44b6','6bba'):
        data=SourceDataset(source,image=False)
        rs={r['dataset']:r for r in data.rows}
        samples=[]
        for key,a in data.anchors.items():
            if not a['random_included']:continue
            f=data.arrays(key)['features'][0]
            samples.append([a['position'][0]/rs[a['dataset']]['image_shape'][1],float(f[23]*30),float(f[18]),float(f[122])])
        x=np.asarray(samples)
        values[source]=dict(depth=np.quantile(x[:,0],[.25,.75]).tolist(),
            predicted_density10=np.quantile(x[:,1],[.25,.75]).tolist(),
            frozen_native_response=np.quantile(x[:,2],[.25,.75]).tolist(),
            frozen_native_intensity=np.quantile(x[:,3],[.25,.75]).tolist(),
            boundary_definition='within 3 frames of a temporal edge or 16 um of a spatial edge',
            count=len(x),labels_used=False,partition='fit')
    write_json(RESULTS/'diagnostic_strata_lock.json',dict(sources=values,target_results_read=False),immutable=True)


def candidate_witnesses(row):
    """Current-study label archives are diagnostic only after the target freeze."""
    path=WORK/'diagnostics/candidate_witnesses'/(row['dataset']+'.json')
    if path.exists():return read_json(path)
    source=WORK/'source'/row['embryo']/row['dataset']/'anchors.json.gz'
    with gzip.open(source,'rt') as f:anchors=json.load(f)
    events={}
    for a in anchors:
        for d,label in zip(a['decisions'],a['labels']):
            if d['kind']=='keep':continue
            for event in label['compatible_events']:
                events.setdefault(str(event),set()).add(digest([sorted(d['remove']),sorted(d['add'])]))
    result=dict(events={k:sorted(v) for k,v in events.items()},archive_sha256=sha(source),
        scope='A witnessed complete bank action satisfies the official local division window; absence is unestablished because the archive samples anchors',
        prediction_shortlisting=False,post_freeze_only=True)
    write_json(path,result);return result


def case_strata(case,row,graph,native,lock):
    p=np.asarray(case['position']);t=case['t'];source='44b6' if row['embryo']=='6bba' else '6bba'
    same=np.flatnonzero(graph['nodes'][:,1]==t)
    nearest=None
    if len(same):
        distance=np.linalg.norm((graph['nodes'][same,2:]-p)*np.asarray(row['physical_scale']),axis=1)
        j=int(np.argmin(distance))
        if distance[j]<=7:nearest=int(same[j])
    nf=native.get('node_features',native.get('features'))
    values=dict(depth=float(p[0]/row['image_shape'][1]),
        predicted_density10=float(nf[nearest,8]) if nearest is not None and nf is not None else None,
        frozen_native_response=float(nf[nearest,5]) if nearest is not None and nf is not None else None,
        frozen_native_intensity=float(nf[nearest,6]) if nearest is not None and nf is not None else None)
    strata={}
    for name,value in values.items():
        lo,hi=lock['sources'][source][name]
        strata[name]='unmatched' if value is None else 'low' if value<=lo else 'high' if value>=hi else 'middle'
    spatial=float(np.min(np.minimum(p,np.asarray(row['image_shape'][1:])-1-p)*np.asarray(row['physical_scale'])))
    strata['boundary']=bool(t<3 or t>=row['image_shape'][0]-3 or spatial<16)
    return dict(source=source,strata=strata,values=values,
        nearest_predicted_id=int(graph['nodes'][nearest,0]) if nearest is not None else None,
        native_assignment='Nearest same-frame predicted point within 7 um; diagnosis only')


def run():
    from center_comparison.pipeline import read_gt
    from .labels import Counterfactual
    freeze=read_json(RESULTS/'target_freeze.json')
    rows={r['dataset']:r for r in inputs()}
    cases=[];timings=[];stages=[];owner_errors=[];action_details=[]
    lock=read_json(RESULTS/'diagnostic_strata_lock.json')
    for arm in freeze['qualified_exports']:
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            root=WORK/'target'/arm/str(seed)
            false=[]
            for path in sorted((root/'errors').glob('*.json')):
                error=read_json(path);row=rows[error['dataset']]
                source=error['source'];selected=freeze['selected'][f'{arm}/{source}/{seed}']['selected']
                predicted=root/'predictions'/row['dataset']/selected['application']/(row['dataset']+'.npz')
                receipt=read_json(predicted.with_suffix('.json'))
                score=read_json(root/'evaluation'/(row['dataset']+'.json'))
                trace=read_json(predicted.parent.parent/'trace.json')
                maxima={m['key']:m['gain'] for m in trace['maxima'] if m['kind']!='keep'}
                witness=candidate_witnesses(row)
                witness_stages=[dict(gt_event_id=int(event),local_compatible_candidate=True,
                    positive_max_selected_compatible=any(maxima.get(key,0)>1e-9 for key in keys))
                    for event,keys in witness['events'].items()]
                stages.append(dict(arm=arm,seed=seed,dataset=row['dataset'],source=source,
                    counts=receipt['counts'],solver={k:v for k,v in receipt['ledger'].items() if k!='edits'},
                    score_summary=receipt.get('score_summary'),
                    calibration=selected['calibration'],official_counts={k:score[k] for k in
                        ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')},
                    candidate_witnesses=witness_stages,unwitnessed_availability=None,
                    unavailable_reason='No absence claim from sampled training archives',
                    new_target_metrics_never_feed_training=True))
                changed=error['recovered_divisions'] or error['lost_divisions']
                graph=load_graph(predicted)
                lost_edges=set(map(tuple,error['lost_edges']));recovered_edges=set(map(tuple,error['recovered_edges']))
                for edit in receipt['ledger']['edits']:
                    removed={(int(graph['nodes'][a,0]),int(graph['nodes'][b,0])) for a,b in edit['remove']}
                    added={(int(graph['nodes'][a,0]),int(graph['nodes'][b,0])) for a,b in edit['add']}
                    owner_count=max(0,len(edit['owners'])-1)
                    lost=sorted(removed&lost_edges);recovered=sorted(added&recovered_edges)
                    record=dict(arm=arm,seed=seed,dataset=row['dataset'],key=edit['key'],kind=edit['kind'],
                        calibrated_gain=edit['value'],trained_gain_reconstructed=(edit['value']-selected['calibration']['intercept'])*selected['calibration']['temperature'],
                        reconstruction_rounding='Inverse affine calibration; original float32 rounding remains',
                        was_anchor_max=edit['key'] in maxima,displaced_owners=owner_count,
                        application=selected['application'],selected_by_solver=True,
                        lost_supported_edges=lost,recovered_supported_edges=recovered)
                    action_details.append(record)
                    if owner_count and lost:owner_errors.append(record)
                if changed:
                    gn,ge=read_gt(DATA,row['dataset'],row['physical_scale'])
                    counter=Counterfactual(graph['nodes'],graph['edges'],gn,ge,row['physical_scale'],score)
                    old_counter=None
                    if error['lost_divisions']:
                        old=verified_graph(row)
                        old_counter=Counterfactual(old['nodes'],old['edges'],gn,ge,row['physical_scale'],
                            read_json(PRIOR_WORK/'evaluation/P0'/(row['dataset']+'.json')))
                    for kind in ('recovered_divisions','lost_divisions'):
                        for event in error[kind]:
                            p=counter.base_div['pairing'].get(event)
                            before=old_counter.base_div['pairing'].get(event) if old_counter else None
                            offset=int(graph['nodes'][p,1]-gn[event,1]) if p is not None else None
                            old_offset=int(graph['nodes'][before,1]-gn[event,1]) if before is not None else None
                            shown_offset=offset if offset is not None else old_offset
                            timings.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind=kind,
                                gt_event_id=int(gn[event,0]),predicted_parent_id=int(graph['nodes'][p,0]) if p is not None else None,
                                previous_parent_id=int(graph['nodes'][before,0]) if before is not None else None,
                                split_frame_offset=offset,previous_split_frame_offset=old_offset,
                                timing='unmatched' if shown_offset is None else 'early' if shown_offset<0 else 'late' if shown_offset>0 else 'on_time'))
                            cases.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind=kind,index=event,
                                gt_event_id=int(gn[event,0]),t=int(gn[event,1]),position=gn[event,2:].tolist(),annotation_centered_diagnostic=True))
                for p in error['division_fp_added']:
                    false.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind='false_fork_sample',index=p,
                        predicted_parent_id=int(graph['nodes'][p,0]),
                        t=int(graph['nodes'][p,1]),position=graph['nodes'][p,2:].tolist(),annotation_centered_diagnostic=False))
            cases.extend(sorted(false,key=lambda r:digest([r['dataset'],r['index']]))[:16])
    gallery=WORK/'diagnostics/gallery';entries=[]
    case_rows=[]
    for case in cases:
        row=rows[case['dataset']];graph=verified_graph(row);native=verified_evidence(row)
        case['source_fixed_strata']=case_strata(case,row,graph,native,lock)
        case_rows.append(dict(arm=case['arm'],seed=case['seed'],dataset=case['dataset'],kind=case['kind'],
            index=case['index'],**case['source_fixed_strata']['strata']))
        key=digest(case)[:20];path=gallery/(key+'.png')
        identity=case.get('gt_event_id',case.get('predicted_parent_id'))
        label=f'{case["arm"]} seed {case["seed"]} · {case["dataset"]} · {case["kind"]} ID {identity}'
        render_scene(rows[case['dataset']],case['t'],case['position'],path,label)
        entries.append(f'<figure><figcaption>{html.escape(label)}</figcaption><img loading="lazy" width="1200" src="{key}.png"></figure>')
    gallery.mkdir(parents=True,exist_ok=True)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen division error review</title>'
        '<style>body{font:16px sans-serif;background:#fafafa;color:#222;max-width:1240px;margin:30px auto}figure{margin:30px 0}img{max-width:100%}</style>'
        '<h1>Post-freeze division errors</h1><p>Every recovered and lost division; a fixed hash sample of added false forks. Diagnostic panels never feed fitting.</p>'+''.join(entries))
    write_json(WORK/'diagnostics/cases.json',cases)
    write_json(WORK/'diagnostics/selected_action_stages.json',action_details)
    write_json(WORK/'diagnostics/competing_owner_errors.json',owner_errors)
    write_csv(RESULTS/'diagnostic_case_strata.csv',case_rows)
    write_csv(RESULTS/'split_timing.csv',timings)
    write_json(RESULTS/'stage_attribution.json',dict(status='measured',stages=stages,
        raw_scene_cases=len(cases),gallery=str(gallery/'index.html'),timing_cases=len(timings),
        competing_owner_supported_loss_actions=len(owner_errors),selected_action_records=len(action_details),
        selected_action_records_sha256=sha(WORK/'diagnostics/selected_action_stages.json'),
        competing_owner_errors_sha256=sha(WORK/'diagnostics/competing_owner_errors.json'),
        source_fixed_strata_sha256=sha(RESULTS/'diagnostic_strata_lock.json'),
        categories_overlap=True,visual_review_post_freeze=True,
        excluded_claim='Whole-clip matches are not substituted for official local division matches'))
    return cases
