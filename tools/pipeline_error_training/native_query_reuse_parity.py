"""CPU proof of exact query reuse and rebuilt features at a real changed point."""
import json
import os
from pathlib import Path
import time


def run():
    # The reference already ran both neural models on this changed observation.
    # Hide CUDA here to make any accidental new neural execution fail explicitly.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    repo = Path(__file__).resolve().parents[2]
    work = repo/'work/pipeline-error-training-20260915'
    job = json.loads((work/'native_validation/job.json').read_text())
    reference = Path(job['root'])
    root = work/'native_query_reuse_parity'
    from .guard import install
    guard = install(fresh_root=root,allowed_models=[d['path'] for d in job['dependencies']]+
        [str(reference/'changed_query'),str(reference/'explicit_changed_observation.npz')],images=[job['row']['image_path']])
    if not guard['installed_before_numerical']:
        raise RuntimeError('Native reuse proof guard must precede numerical dependencies')
    from .resources import Monitor,cpu_budget
    cpu_budget()
    import numpy as np
    import torch
    from .cache_reuse import native_query,native_reference
    from .common import RESULTS,load_graph,sha,write_json
    from .native_refresh import refresh
    row = job['row']
    for dependency in job['dependencies']:
        if sha(dependency['path'])!=dependency['sha256']:
            raise ValueError('Native reuse proof input changed')
    graph = load_graph(reference/'explicit_changed_observation.npz')
    old = load_graph(row['baselines']['P0']['path'])
    native = load_graph(row['evidence']['path'])
    source = root/'reference'
    destination = root/'rebuilt'
    began = time.monotonic()
    with Monitor(root/'resources.json'):
        native_reference(reference/'changed_query',source,row['metadata_sha256'])
        changed_id = graph['nodes'].copy()
        changed_id[0,0] += int(graph['nodes'][:,0].max())+1
        changed_coordinate = graph['nodes'].copy()
        changed_coordinate[0,4] += 1
        for changed in [changed_id,changed_coordinate]:
            if native_query(source,root/'different_query',changed,row['metadata_sha256']) is not None:
                raise ValueError('Changed query was incorrectly reused')
        reused = native_query(source,destination,graph['nodes'],row['metadata_sha256'])
        if reused is None:
            raise ValueError('Identical actual changed-coordinate query was not reused')
        current = refresh(row,graph,old,native,destination)
        expected = load_graph(reference/'changed_query/current_features.npz')
        if set(current)!=set(expected):
            raise ValueError('Rebuilt native feature schema changed')
        for field in current:
            np.testing.assert_array_equal(current[field],expected[field])
        if torch.cuda.is_initialized():
            raise RuntimeError('The query-reuse CPU proof unexpectedly initialized CUDA')
    if guard['blocked_reads'] or guard['blocked_network']:
        raise ValueError('Query reuse proof access guard failed')
    result = dict(status='measured',dataset=row['dataset'],frames=row['image_shape'][0],
        seconds=time.monotonic()-began,source_query_sha256=sha(reference/'changed_query/query.npz'),
        reused_query_sha256=sha(destination/'query.npz'),all_rebuilt_feature_arrays_exact=True,
        fields=sorted(current),actual_changed_observation=True,changed_ids_and_coordinates_rejected=True,
        original_query_loader_verified_weights_code_and_image_chunks=True,CUDA_hidden=True,
        CUDA_initialized=False,new_target_metrics_read=False,annotation_reads=0,guard=guard,
        implementation_sha256={n:sha(Path(__file__).with_name(n+'.py'))
            for n in ['cache_reuse','native_refresh','observation_infer']},
        scope='Previously executed full-ensemble query reused only for identical observations; graph-dependent features rebuilt')
    write_json(root/'guard.json',guard)
    write_json(RESULTS/'native_query_reuse_parity.json',result,immutable=True)
    return result


if __name__=='__main__':
    print(run(),flush=True)
