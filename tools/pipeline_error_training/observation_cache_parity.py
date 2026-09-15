"""Cold source-image replay against observation-policy embedding reuse."""
from pathlib import Path
import subprocess
import sys
import time


def run():
    import numpy as np
    from .common import RESULTS, WORK, graph_hash, inputs, load_graph, read_json, sha, write_json
    from .resources import Monitor
    from .source_screen import job
    source, arm, seed = '44b6', 'O10_restore', '20260915'
    summary_path = WORK/'source_screen'/arm/source/seed/'summary.json'
    if not summary_path.exists():
        result = dict(status='not run',reason='Source observation restoration replay is unavailable',source=source)
        write_json(RESULTS/'observation_cache_parity.json',result)
        return result
    summary = read_json(summary_path)
    selected = min(summary['per_clip'],key=lambda r:r['num_pred_nodes'])
    name = selected['dataset']
    row = next(r for r in inputs() if r['dataset']==name)
    original = WORK/'source_screen'/arm/source/seed/'predictions'/name
    package = WORK/'training/O10_swap'/source/seed
    root = WORK/'observation_cache_parity'/name
    root.mkdir(parents=True,exist_ok=True)
    destination = root/f'{name}.npz'
    path = root/'job.json'
    write_json(path,job(row,source,package,destination,arm=arm),immutable=True)
    began = time.monotonic()
    with Monitor(root/'resources.json'):
        if not destination.exists():
            if (root/'current_image_cache').exists():
                raise ValueError('Cold policy replay has a partial cache; preserve the attempt before retrying')
            with (root/'prediction.log').open('a') as log:
                subprocess.run([sys.executable,'-m','pipeline_error_training.prediction_entry',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        guard = read_json(destination.with_suffix('.guard.json'))
        if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
            raise ValueError('Cold observation policy replay access guard failed')
        records = []
        for basename in ['P0.npz','raw.npz']:
            a = original/'current_image_cache'/basename
            b = root/'current_image_cache'/basename
            a_receipt, b_receipt = read_json(a.with_suffix('.json')),read_json(b.with_suffix('.json'))
            if a_receipt['inputs']!=b_receipt['inputs'] or a_receipt['image_frame_hashes']!=b_receipt['image_frame_hashes']:
                raise ValueError('Reused and cold policy image provenance differs')
            for p,r in [(a,a_receipt),(b,b_receipt)]:
                if sha(p)!=r['sha256']:
                    raise ValueError('Policy image embedding bytes changed')
            with np.load(a,allow_pickle=False) as aa,np.load(b,allow_pickle=False) as bb:
                for field in ['embedding','valid']:
                    np.testing.assert_array_equal(aa[field],bb[field])
            records.append(dict(name=basename,reference_sha256=sha(a),cold_sha256=sha(b),exact=True))
        expected, actual = load_graph(original/f'{name}.npz'), load_graph(destination)
        for key in ['nodes','edges']:
            np.testing.assert_array_equal(expected[key],actual[key])
        native_reused = (original/'fresh_native/query_reuse.json').exists()
        if native_reused:
            for basename in ['query.npz','current_features.npz']:
                with np.load(original/'fresh_native'/basename,allow_pickle=False) as a, \
                        np.load(root/'fresh_native'/basename,allow_pickle=False) as b:
                    if set(a.files)!=set(b.files):
                        raise ValueError('Cold and reused native query feature schemas differ')
                    for field in a.files:
                        np.testing.assert_array_equal(a[field],b[field])
    result = dict(status='measured',source=source,dataset=name,arm=arm,
        source_model_sha256=sha(package/'model.pt'),seconds=time.monotonic()-began,
        graph_hash=graph_hash(actual['nodes'],actual['edges']),complete_graph_exact=True,
        model_and_image_specific_embedding_parity=records,cold_reference_replay=True,
        guard=guard,new_target_metrics_read=False,fresh_upstream_pipeline_claim=False)
    result.update(native_query_reused_in_reference=native_reused,
        cold_full_native_arrays_exact_if_reused=native_reused)
    write_json(RESULTS/'observation_cache_parity.json',result,immutable=True)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    print(run(),flush=True)
