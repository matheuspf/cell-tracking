"""GT-free all-clip parity between primary batched and portable head inference."""
from . import offline_guard
from .infer import transform,GUARD,EXTERNAL_GUARD
from .common import *
import subprocess
import sys
import time

def run(source=None):
    import torch
    import zarr
    torch.set_num_threads(2)
    if source is None:
        config=dict(created=now(),variant='C4',samples=199,labels_available=False,
            reason='Check the individual-graph deployment path against primary shared-patch batching',
            target_outcomes_used_to_change_policy=False,calibration_sha256=sha(OUT/'calibration.json'),
            inference_code_sha256={n:sha(Path(__file__).parent/n) for n in ['infer.py','models.py','adapters.py','proposals.py','decode.py']})
        lock=OUT/'head_parity_lock.json'
        if lock.exists():
            assert {k:v for k,v in read(lock).items() if k!='created'}=={k:v for k,v in config.items() if k!='created'}
        else:write(lock,config)
        processes=[subprocess.Popen([sys.executable,'-m','multidata_training_v4.head_parity',s]) for s in ['44b6','6bba']]
        codes=[p.wait() for p in processes]
        receipts=[read(p) for p in sorted((OUT/'head_parity_rows').glob('*.json'))]
        complete=not any(codes) and len(receipts)==199
        passed=complete and all(r['nodes_equal'] and r['edges_equal'] for r in receipts)
        write(OUT/'head_parity_receipt.json',dict(completed=complete,passed=passed,variant='C4',samples=len(receipts),
            process_returncodes=codes,all_graphs_exact=passed,labels_read=False,thresholds_or_weights_changed=False,
            summed_clip_seconds=sum(r['seconds'] for r in receipts),
            mismatches=[r for r in receipts if not(r['nodes_equal'] and r['edges_equal'])],
            workers=[read(OUT/f'head_parity_worker_{s}.json') for s in ['44b6','6bba']]))
        assert passed,'Individual-graph inference differs from the scored primary path'
        print('C4 individual-graph parity passed on all 199 clips',flush=True);return
    assert source in ['44b6','6bba'];cal=read(OUT/'calibration.json')
    for row in inputs():
        if row['embryo']==source:continue
        name=row['dataset'];proof=OUT/'head_parity_rows'/f'{name}.json'
        if proof.exists():
            assert sha(OUT/'predictions/C4'/f'{name}.npz')==read(proof)['scored_prediction_sha256'];continue
        start=time.monotonic();base=arrays(V3/'selected_predictions'/f'{name}.npz')
        image=zarr.open_group(row['image_path'],mode='r')['0'][:]
        n,e,r=transform(base['nodes'],base['edges'],image,row['physical_scale'],source,'C4',OUT/'models',cal)
        expected=arrays(OUT/'predictions/C4'/f'{name}.npz')
        equal_n=bool(np.array_equal(n,expected['nodes']));equal_e=bool(np.array_equal(e,expected['edges']))
        if not(equal_n and equal_e):save(OUT/'head_parity_mismatches'/f'{name}.npz',nodes=n,edges=e)
        write(proof,dict(dataset=name,source=source,nodes_equal=equal_n,edges_equal=equal_e,
            scored_prediction_sha256=sha(OUT/'predictions/C4'/f'{name}.npz'),seconds=time.monotonic()-start,
            weights_loaded=r['weights_loaded']))
        print('head parity',name,equal_n and equal_e,flush=True)
    write(OUT/f'head_parity_worker_{source}.json',dict(completed=True,annotation_guard=GUARD,external_guard=EXTERNAL_GUARD,
        offline_guard=dict(installed_before_dependencies=True,blocked_network_attempts=len(offline_guard.BLOCKED_NETWORK_ATTEMPTS),
            blocked_dataset_reads=len(offline_guard.BLOCKED_EXTERNAL_DATA_READS))))

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else None)
