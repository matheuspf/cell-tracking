"""All source calibration clips, both complete-edit applications, exact scores."""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .common import (RESULTS,WORK,PRIOR_WORK,DATA,inputs,read_json,write_json,write_csv,
                     sha,load_graph,verified_graph)


def prediction_job(row,source,checkpoint,output,calibration):
    return dict(row={k:row[k] for k in ('dataset','image_path','image_shape','physical_scale','metadata_sha256')},
        source=source,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),output=str(output),
        graph_path=row['baselines']['P0']['path'],graph_sha256=row['baselines']['P0']['sha256'],
        native_path=row['evidence']['path'],native_sha256=row['evidence']['sha256'],
        calibration={k:calibration[k] for k in ('status','temperature','intercept')})


def launch(job,path):
    output=Path(job['output'])
    complete=output/'trace.json'
    if complete.exists():
        guard=read_json(output/'guard.json')
        if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
            raise RuntimeError('Inference guard failed')
        return
    write_json(path,job,immutable=True)
    log=path.with_suffix('.log')
    with log.open('a') as f:
        result=subprocess.run([sys.executable,'-m','division_generalization_v2.prediction_entry',str(path)],
                               stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Prediction failed ({result.returncode}); preserved {log}')


def screen(source,arm,seed,step):
    from annotation_selection.metric_adapter import evaluate_graph,aggregate
    from center_comparison.pipeline import read_gt
    from .infer import load_checkpoint
    from .dataset import SourceDataset
    from .calibration import run as calibrate
    folder=WORK/'training'/arm/source/str(seed)
    checkpoint=folder/f'checkpoint-{step}.pt'
    root=WORK/'screens'/arm/source/str(seed)/str(step)
    final=root/'summary.json'
    if final.exists():return read_json(final)
    model,recipe=load_checkpoint(checkpoint)
    dataset=SourceDataset(source,'calibration',image=model.image)
    calibration=calibrate(model,dataset,root/'calibration.json',recipe['amp'])
    scores={a:[] for a in ('protected','replacement')};baselines=[];lost={a:0 for a in scores}
    rows=inputs(source,'calibration')
    for i,row in enumerate(rows,1):
        output=root/'predictions'/row['dataset']
        job=prediction_job(row,source,checkpoint,output,calibration)
        launch(job,root/'jobs'/(row['dataset']+'.json'))
        gn,ge=read_gt(DATA,row['dataset'],row['physical_scale'])
        baseline=read_json(PRIOR_WORK/'evaluation/P0'/(row['dataset']+'.json'))
        with np.load(PRIOR_WORK/'evaluation/P0'/(row['dataset']+'.npz'),allow_pickle=False) as f:
            old_tp=set(map(tuple,f['tp_edges']))
        baselines.append(baseline)
        for app in scores:
            path=output/app/(row['dataset']+'.npz')
            graph=load_graph(path)
            np.testing.assert_array_equal(graph['nodes'],verified_graph(row)['nodes'])
            value,_,tp=evaluate_graph(row['dataset'],graph['nodes'],graph['edges'],gn,ge,
                row['physical_scale'],row['estimated_total'])
            value.update(arm=arm,seed=seed,step=step,source=source,application=app,
                         lost_supported_edges=len(old_tp-tp),prediction_sha256=sha(path))
            lost[app]+=len(old_tp-tp);scores[app].append(value)
            write_json(output/app/(row['dataset']+'.score.json'),value)
        print(f'Full source screen {arm}/{source}/{seed}/{step}: {i}/{len(rows)}',flush=True)
    names=[r['dataset'] for r in rows]
    baseline=aggregate(baselines,names)
    summaries={app:dict(aggregate(rs,names),lost_supported_edges=lost[app],clips=len(rs)) for app,rs in scores.items()}
    for app,a in summaries.items():a['delta']=a['score']-baseline['score']
    result=dict(source=source,arm=arm,seed=seed,step=step,status='measured',baseline=baseline,
        applications=summaries,per_clip=scores,calibration=calibration,
        checkpoint_sha256=sha(checkpoint),full_source_screen=True,target_used=False)
    write_json(final,result)
    return result


def run(args):
    return screen(args.source,args.arm,args.seed,args.stop_at or 4096)


def prepare_source_matrix(source):
    """Evaluate initial frozen checkpoints together to share raw I/O and bank work."""
    from .infer import load_checkpoint
    from .dataset import SourceDataset
    from .calibration import run as calibrate
    specs={}
    for arm in ('G30','J_uniform','J_mined'):
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            for step in ((4096,) if arm=='G30' else (3072,4096)):
                key=f'{arm}-{seed}-{step}'
                root=WORK/'screens'/arm/source/str(seed)/str(step)
                checkpoint=WORK/'training'/arm/source/str(seed)/f'checkpoint-{step}.pt'
                model,recipe=load_checkpoint(checkpoint)
                data=SourceDataset(source,'calibration',image=model.image)
                cal=calibrate(model,data,root/'calibration.json',recipe['amp'])
                specs[key]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                    calibration={k:cal[k] for k in ('status','temperature','intercept')},root=str(root))
                del model,data
    for row in inputs(source,'calibration'):
        output=WORK/'frozen_matrix'/source/'source_initial'/row['dataset']
        job=dict(row={k:row[k] for k in ('dataset','image_path','image_shape','physical_scale','metadata_sha256')},
            source=source,output=str(output),packages=specs,graph_path=row['baselines']['P0']['path'],
            graph_sha256=row['baselines']['P0']['sha256'],native_path=row['evidence']['path'],native_sha256=row['evidence']['sha256'])
        run_matrix(job,WORK/'frozen_matrix'/source/'jobs'/(row['dataset']+'.json'))
        for key,spec in specs.items():
            destination=Path(spec['root'])/'predictions'/row['dataset']
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():destination.symlink_to((output/key).resolve(),target_is_directory=True)


def run_matrix(job,path):
    output=Path(job['output'])
    if (output/'complete.json').exists():return
    write_json(path,job,immutable=True)
    with path.with_suffix('.log').open('a') as f:
        p=subprocess.run([sys.executable,'-m','division_generalization_v2.matrix_entry',str(path)],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Frozen matrix failed: '+str(path.with_suffix('.log')))
