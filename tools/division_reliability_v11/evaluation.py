"""Official full-graph evaluation in annotation-enabled processes only."""
from pathlib import Path
from .common import DATA,REPO,WORK,RESULTS,Blocked,read,write,sha


def finite(value):
    import math
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def score(clip,nodes,edges,label_path):
    from .data import labels
    from annotation_selection.metric_adapter import evaluate_graph,make_graph
    from tracking_cellmot import division_metrics as dm
    gt,ge=labels(label_path)
    estimate=read(label_path/'zarr.json')['attributes']['geff']['extra']['estimated_number_of_nodes']
    row,matches,_=evaluate_graph(clip,nodes,edges,gt,ge,[1.625,.40625,.40625],estimate)
    graph,reverse=make_graph(nodes,edges);truth,gt_reverse=make_graph(gt,ge)
    detail=dm.score_divisions(graph,truth,(1.625,.40625,.40625),7.)
    events=dict(recovered_gt_events=sorted(int(gt_reverse[k]) for k,v in detail.scores.items() if v),
                missed_gt_events=sorted(int(gt_reverse[k]) for k,v in detail.scores.items() if not v),
                fp_predicted_forks=sorted(int(reverse[k]) for k in detail.fp_forks),
                tp_predicted_forks=sorted(int(reverse[k]) for k in detail.tp_forks))
    if len(events['recovered_gt_events'])!=row['division_tp'] or len(events['fp_predicted_forks'])!=row['division_fp']:
        raise Blocked('Official detailed division matching disagrees with aggregate counts')
    return row,events


def run(source,seed,arm,clip):
    """Target evaluation requires a complete immutable global prediction freeze."""
    import sys,tempfile
    from .freeze import verify
    key=f'{source}/{seed}/{arm}/{clip}'
    frozen=verify(key)
    verify(f'{source}/{seed}/C00/{clip}')
    if key not in frozen['predictions']:raise Blocked('Prediction absent from global freeze')
    prediction=WORK/'predictions'/arm/source/str(seed)/clip
    folder=WORK/'evaluation'/source/str(seed)/arm/clip;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():return read(folder/'receipt.json')
    target='6bba' if source=='44b6' else '44b6'
    if not clip.startswith(target+'_'):raise Blocked('Not the registered excluded target')
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,WORK/'predictions/C00'/source/str(seed)/clip,
        WORK/'packages'/source/str(seed)/arm/'calibration.json',DATA/'train'/f'{clip}.geff'],outputs=[folder],code_roots=[REPO/'tools',REPO/'handover',
        REPO/'work/annotation-selection-v1/official',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    from .graphs import read_csv
    nodes,edges=read_csv(prediction/'submission.csv',clip)
    row,events=score(clip,nodes,edges,DATA/'train'/f'{clip}.geff')
    from .diagnostics import target_details
    detail=target_details(source,seed,arm,clip,nodes,edges,events)
    result=dict(status='scored',source=source,target=target,seed=seed,arm=arm,dataset=clip,metrics=finite(row),events=events,
                guard=guard,prediction_sha256=sha(prediction/'graph.npz'),freeze_identity=frozen['identity'],frames=100,
                diagnostics=detail)
    write(folder/'receipt.json',result,immutable=True);return result
