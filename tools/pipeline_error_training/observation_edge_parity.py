"""CPU parity of original and indexed observation actions on real source clips."""
import os
from pathlib import Path
import time


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from .guard import install
    guard = install(source='44b6')
    if not guard['installed_before_numerical']:
        raise RuntimeError('Source parity guard must precede numerical imports')
    import numpy as np
    import torch
    from .common import RESULTS,WORK,digest,graph_hash,inputs,read_json,sha,verified_graph,write_json
    from .observation_infer import apply_observations,construct_actions
    from .observations import ObservationBank,raw_graph
    from .resources import Monitor,cpu_budget
    from .scoring import load_model
    cpu_budget()
    torch.set_num_threads(2)
    package = WORK/'training/O10_swap/44b6/20260915'
    model,spec = load_model(package)
    root = WORK/'observation_edge_parity'
    records = []
    with Monitor(root/'resources.json') as monitor:
        for name in ['44b6_d754aa59','44b6_e31261b4']:
            row = next(r for r in inputs() if r['dataset']==name)
            graph,raw = verified_graph(row),raw_graph(row)
            bank = ObservationBank(row,graph,raw)
            cache = WORK/'source_screen/O10_swap/44b6/20260915/predictions'/name/'current_image_cache'
            values = []
            for basename in ['P0.npz','raw.npz']:
                path = cache/basename
                receipt = read_json(path.with_suffix('.json'))
                if sha(path)!=receipt['sha256'] or receipt['inputs']['model_sha256']!=spec['weights_sha256'] \
                        or receipt['inputs']['image_metadata_sha256']!=row['metadata_sha256']:
                    raise ValueError('Source embedding provenance changed')
                with np.load(path,allow_pickle=False) as arrays:
                    values.append(arrays['embedding'])
            z,rz = values
            out = []
            with torch.inference_mode():
                for start in range(0,len(bank.pairs),2048):
                    pairs = bank.pairs[start:start+2048]
                    context = np.stack([z[sorted({int(i),*bank.pred[i],*bank.succ[i]})].mean(0) for i,_ in pairs])
                    evidence = np.stack([bank.evidence(int(i),int(j)) for i,j in pairs])
                    logits = model.selection_scores(torch.as_tensor(z[pairs[:,0]]),torch.as_tensor(rz[pairs[:,1]]),
                        torch.as_tensor(context),torch.as_tensor(evidence)).numpy()
                    calibrated = logits/spec['calibration']['temperature']+spec['calibration']['state_intercepts']
                    out.append(calibrated[:,1:]-calibrated[:,:1])
            gain = np.concatenate(out)
            for restore in [False,True]:
                actions,timing,streaming = [],[],[]
                for indexed in [False,True]:
                    began = time.monotonic()
                    a,s = construct_actions(bank,gain,restore,indexed_edges=indexed)
                    timing.append(time.monotonic()-began)
                    actions.append(a);streaming.append(s)
                    print(f'Observation action parity {name} restore={restore} indexed={indexed}: {timing[-1]:.3f}s, {len(a)} actions',flush=True)
                    monitor.check()
                if actions[0]!=actions[1] or streaming[0]!=streaming[1]:
                    raise ValueError('Indexed observation actions changed')
                old,_ = apply_observations(bank,actions[0],streaming[0],restore)
                new,_ = apply_observations(bank,actions[1],streaming[1],restore)
                for field in ['nodes','edges']:
                    np.testing.assert_array_equal(old[field],new[field])
                records.append(dict(dataset=name,restore=restore,pairs=len(bank.pairs),actions=len(actions[0]),
                    original_seconds=timing[0],indexed_seconds=timing[1],all_action_fields_exact=True,
                    streaming_counters_exact=True,selected_graph_exact=True,
                    selected_graph_hash=graph_hash(new['nodes'],new['edges']),gain_sha256=digest(gain),
                    bank_sha256=bank.hash,streaming=streaming[0]))
                write_json(root/'progress.json',dict(records=records))
    if guard['blocked_reads'] or guard['blocked_network'] or torch.cuda.is_initialized():
        raise RuntimeError('Observation CPU parity guard failed')
    result = dict(status='measured',source='44b6',model_sha256=spec['weights_sha256'],
        records=records,guard=guard,CUDA_initialized=False,CPU_head_scores_shared_by_both_paths=True,
        full_native_refresh_rerun=False,new_target_metrics_read=False,
        training_recipe_or_decision_policy_changed=False,
        implementation_sha256={n:sha(Path(__file__).with_name(n+'.py'))
            for n in ['observation_infer','observation_edges']})
    write_json(RESULTS/'observation_edge_parity.json',result,immutable=True)
    return result


if __name__=='__main__':
    print(run(),flush=True)
