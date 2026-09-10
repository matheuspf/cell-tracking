"""Source pilot of the true joint objective, before comparative full-batch outcomes."""
import time
from .common import *
from . import native_adapter as native
from .banks import cached
from .temporal_decode import decode

def run(name,expanded=False,calibrated=False,deepcenter=False,suffix=''):
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v3.common import validate
    source=name.split('_')[0];row=next(r for r in inventory() if r['dataset']==name);c=arrays(OUT/'observations'/f'{name}.npz')
    model=native.load();common=cached(model,name,source,expanded);m=np.ones(len(c['nodes']),bool) if expanded else c['oldmask'];base=graph(name)
    scores=common['native_logits'];config=None
    if calibrated:
        from .joint_calibration import fit
        cal=fit(source);scores=cal['native_scale']*scores+cal['native_offset'];config=cal['config']
    confidence=c['confidence'][m]
    if deepcenter:
        dc=arrays(OUT/'deepcenter'/f'{name}.npz')['confidence'][m];confidence=np.sqrt(np.clip(confidence*dc,1e-8,1-1e-6))
    start=time.monotonic();n,e,receipt=decode(c['nodes'][m],common['pairs'],scores,base['nodes'],base['edges'],
        confidence,c['split_owner'][m] if expanded else None,config=config)
    validate(n,e,row['image_shape']);gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    score,match,tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    label=f'{name}_N0_P{int(expanded)}'+('_calibrated' if calibrated else '')+('_DC' if deepcenter else '')+suffix
    write(OUT/'source_pilots'/f'{label}.json',
        dict(source=source,dataset=name,expanded=expanded,seconds=time.monotonic()-start,score=score,decode=receipt,
        source_only=True,interpretation='operational source pilot; no claim of a complete score'))
    print(dict(dataset=name,expanded=expanded,score=score,changed_edges=receipt['changed_edges'],added_nodes=receipt['added_nodes'],
        fallback_windows=receipt['fallback_windows'],seconds=time.monotonic()-start),flush=True)

if __name__=='__main__':
    import sys
    run(sys.argv[1],'P1' in sys.argv[2:],'calibrated' in sys.argv[2:],'DC' in sys.argv[2:], '_final_event_J' if 'final' in sys.argv[2:] else '')
