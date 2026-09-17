"""Post-freeze raw-scene panels, exact error identities and timing attribution."""
from collections import Counter
from pathlib import Path
import html
import numpy as np

from .common import (DATA,WORK,RESULTS,inputs,read_json,write_json,write_csv,load_graph,verified_graph,digest)


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
            samples.append([a['position'][0]/rs[a['dataset']]['image_shape'][1],float(f[23]*30),float(f[18])])
        x=np.asarray(samples)
        values[source]=dict(depth=np.quantile(x[:,0],[.25,.75]).tolist(),
            predicted_density10=np.quantile(x[:,1],[.25,.75]).tolist(),
            frozen_native_response=np.quantile(x[:,2],[.25,.75]).tolist(),
            boundary_definition='within 3 frames of a temporal edge or 16 um of a spatial edge',
            count=len(x),labels_used=False,partition='fit')
    write_json(RESULTS/'diagnostic_strata_lock.json',dict(sources=values,target_results_read=False),immutable=True)


def run():
    from center_comparison.pipeline import read_gt
    from .labels import Counterfactual
    freeze=read_json(RESULTS/'target_freeze.json')
    rows={r['dataset']:r for r in inputs()}
    cases=[];timings=[];stages=[]
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
                stages.append(dict(arm=arm,seed=seed,dataset=row['dataset'],source=source,
                    counts=receipt['counts'],solver={k:v for k,v in receipt['ledger'].items() if k!='edits'},
                    calibration=selected['calibration'],official_counts={k:score[k] for k in
                        ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')}))
                changed=error['recovered_divisions'] or error['lost_divisions']
                graph=load_graph(predicted)
                if changed:
                    gn,ge=read_gt(DATA,row['dataset'],row['physical_scale'])
                    counter=Counterfactual(graph['nodes'],graph['edges'],gn,ge,row['physical_scale'],score)
                    for kind in ('recovered_divisions','lost_divisions'):
                        for event in error[kind]:
                            p=counter.base_div['pairing'].get(event)
                            offset=int(graph['nodes'][p,1]-gn[event,1]) if p is not None else None
                            timings.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind=kind,
                                gt_event_id=int(gn[event,0]),predicted_parent_id=int(graph['nodes'][p,0]) if p is not None else None,
                                split_frame_offset=offset,timing='lost' if offset is None else 'early' if offset<0 else 'late' if offset>0 else 'on_time'))
                            cases.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind=kind,index=event,
                                t=int(gn[event,1]),position=gn[event,2:].tolist(),annotation_centered_diagnostic=True))
                for p in error['division_fp_added']:
                    false.append(dict(arm=arm,seed=seed,dataset=row['dataset'],kind='false_fork_sample',index=p,
                        t=int(graph['nodes'][p,1]),position=graph['nodes'][p,2:].tolist(),annotation_centered_diagnostic=False))
            cases.extend(sorted(false,key=lambda r:digest([r['dataset'],r['index']]))[:16])
    gallery=WORK/'diagnostics/gallery';entries=[]
    for case in cases:
        key=digest(case)[:20];path=gallery/(key+'.png')
        label=f'{case["arm"]} seed {case["seed"]} · {case["dataset"]} · {case["kind"]} {case["index"]}'
        render_scene(rows[case['dataset']],case['t'],case['position'],path,label)
        entries.append(f'<figure><figcaption>{html.escape(label)}</figcaption><img loading="lazy" width="1200" src="{key}.png"></figure>')
    gallery.mkdir(parents=True,exist_ok=True)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen division error review</title>'
        '<style>body{font:16px sans-serif;background:#fafafa;color:#222;max-width:1240px;margin:30px auto}figure{margin:30px 0}img{max-width:100%}</style>'
        '<h1>Post-freeze division errors</h1><p>Every recovered and lost division; a fixed hash sample of added false forks. Diagnostic panels never feed fitting.</p>'+''.join(entries))
    write_json(WORK/'diagnostics/cases.json',cases)
    write_csv(RESULTS/'split_timing.csv',timings)
    write_json(RESULTS/'stage_attribution.json',dict(status='measured',stages=stages,
        raw_scene_cases=len(cases),gallery=str(gallery/'index.html'),timing_cases=len(timings),
        categories_overlap=True,visual_review_post_freeze=True,
        excluded_claim='Whole-clip matches are not substituted for official local division matches'))
    return cases
