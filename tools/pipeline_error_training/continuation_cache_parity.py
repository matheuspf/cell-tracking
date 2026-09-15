"""Prove shared division/continuation crops and both complete source graphs."""
from pathlib import Path
import subprocess
import sys
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .cache_reuse import continuation
    from .common import RESULTS, WORK, graph_hash, inputs, load_graph, read_json, sha, verified_evidence, verified_graph, write_json
    from .resources import Monitor
    from .source_screen import job
    from .staged_infer import predict_many
    source = '44b6'
    summary_path = WORK/'source_screen/A10'/source/'20260915/summary.json'
    if not summary_path.exists():
        result = dict(status='not run',reason='Complete source continuation control unavailable; cold inference remains enabled')
        write_json(RESULTS/'continuation_cache_parity.json',result)
        return result
    selected = min(read_json(summary_path)['per_clip'],key=lambda r:r['num_pred_nodes'])
    row = next(r for r in inputs() if r['dataset']==selected['dataset'])
    name = row['dataset']
    root = WORK/'continuation_cache_parity'/name
    root.mkdir(parents=True,exist_ok=True)
    packages = {arm:WORK/'training'/arm/source/'20260915' for arm in ['D20_temporal','A10']}
    hashes = {arm:read_json(p/'frozen_package.json')['weights_sha256'] for arm,p in packages.items()}
    references = {}
    for arm in packages:
        reference = WORK/'source_screen'/arm/source/'20260915/predictions'/name
        cache = reference/'current_image_cache'
        if arm.startswith('D'):
            cache = cache/'embeddings'
        references[arm] = (reference/f'{name}.npz',cache/hashes[arm]/f'{name}.npz')
        for path in references[arm]:
            receipt = read_json(path.with_suffix('.json'))
            if sha(path)!=receipt.get('sha256',receipt.get('prediction_sha256')):
                raise ValueError('Independent source graph or embedding bytes changed')
    torch.set_num_threads(2)
    began = time.monotonic()
    destination = root/'reused'/f'{name}.npz'
    with Monitor(root/'resources.json'):
        predict_many(row,verified_graph(row),verified_evidence(row),{'D20_temporal':packages['D20_temporal']},
            {'D20_temporal':root/'division.npz'},root/'shared',wait=True,
            embedding_packages={'A10':packages['A10']})
        comparisons = []
        for arm,(_,expected) in references.items():
            actual = root/'shared/embeddings'/hashes[arm]/f'{name}.npz'
            ar,er = read_json(actual.with_suffix('.json')),read_json(expected.with_suffix('.json'))
            if ar['inputs']!=er['inputs'] or ar['image_frame_hashes']!=er['image_frame_hashes']:
                raise ValueError('Shared and independent source embedding provenance differs')
            with np.load(actual,allow_pickle=False) as a,np.load(expected,allow_pickle=False) as e:
                for field in ['embedding','valid']:
                    np.testing.assert_array_equal(a[field],e[field])
            comparisons.append(dict(arm=arm,shared_sha256=sha(actual),independent_sha256=sha(expected),exact=True))
        continuation(root/'shared/embeddings'/hashes['A10']/f'{name}.npz',
            destination.parent/'current_image_cache'/hashes['A10']/f'{name}.npz',hashes['A10'],row['metadata_sha256'])
        path = root/'continuation-job.json'
        spec = job(row,source,packages['A10'],destination)
        spec['wait_for_lease'] = True
        write_json(path,spec,immutable=True)
        with (root/'continuation.log').open('a') as log:
            subprocess.run([sys.executable,'-m','pipeline_error_training.prediction_entry',str(path)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        guard = read_json(destination.with_suffix('.guard.json'))
        if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
            raise ValueError('Reused continuation prediction access guard failed')
        graphs = {}
        for arm,path in [('D20_temporal',root/'division.npz'),('A10',destination)]:
            actual,expected = load_graph(path),load_graph(references[arm][0])
            for field in ['nodes','edges']:
                np.testing.assert_array_equal(actual[field],expected[field])
            graphs[arm] = dict(exact=True,graph_hash=graph_hash(actual['nodes'],actual['edges']))
    result = dict(status='measured',source=source,dataset=name,frames=row['image_shape'][0],
        models=hashes,node_count_per_model=len(actual['nodes']),seconds=time.monotonic()-began,
        embeddings_and_masks=comparisons,complete_graphs=graphs,guard=guard,
        independent_heads_and_decoders=True,no_embedding_averaging=True,
        no_target_metrics_read=True,training_recipe_changed=False,
        implementation_sha256={n:sha(Path(__file__).with_name(n+'.py'))
            for n in ['staged_infer','shared_embeddings','cache_reuse','matrix_entry','final_predictions','point_predictions']})
    write_json(RESULTS/'continuation_cache_parity.json',result,immutable=True)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    print(run(),flush=True)
