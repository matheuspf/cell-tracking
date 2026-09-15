"""Whole-source embedding parity before sharing crops across frozen models."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .bank import EventBank
    from .common import RESULTS, WORK, inputs, read_json, sha, verified_evidence, verified_graph, write_json
    from .infer import embeddings
    from .resources import Lease, Monitor
    from .scoring import load_model
    from .shared_embeddings import embeddings_many
    torch.set_num_threads(2)
    row = next(r for r in inputs() if r['dataset']=='44b6_5f15d135')
    graph, native = verified_graph(row), verified_evidence(row)
    bank = EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    root = WORK/'shared_embedding_parity'
    packages = {arm:WORK/'training'/arm/'44b6/20260915'
                for arm in ['D10_frozen','D10_adapted','D20_compact','D20_temporal']}
    packages['temporal_pilot'] = WORK/'training_pilot/D20_temporal/44b6/20260915'
    loaded = {name:load_model(path) for name,path in packages.items()}
    models = {name:value[0] for name,value in loaded.items()}
    hashes = {name:value[1]['weights_sha256'] for name,value in loaded.items()}
    paths = {name:root/'shared'/f'{name}.npz' for name in models}
    if any(path.exists() for path in paths.values()):
        raise ValueError('Whole-source shared-crop proof requires a cold output directory')
    expected, references = {}, {}
    for name in ['D10_frozen','D10_adapted','D20_compact','temporal_pilot']:
        path = (WORK/'inference_profile/temporal'/f'{row["dataset"]}.npz' if name=='temporal_pilot'
            else WORK/'source_screen'/name/'44b6/20260915/predictions'/row['dataset']/
                 ('current_image_cache/embeddings' if name=='D20_compact' else 'current_image_cache')/
                 hashes[name]/f'{row["dataset"]}.npz')
        receipt = read_json(path.with_suffix('.json'))
        if sha(path)!=receipt['sha256'] or receipt['inputs']['model_sha256']!=hashes[name]:
            raise ValueError('Preserved independent source embedding changed')
        for t,value in receipt['image_frame_hashes'].items():
            if sha(Path(row['image_path'])/f'0/c/{t}/0/0/0')!=value:
                raise ValueError('Source image no longer matches independent embedding')
        with np.load(path,allow_pickle=False) as data:
            expected[name] = (data['embedding'],data['valid'])
        references[name] = dict(embedding_sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')))
    timings = []
    with Monitor(root/'resources.json'),Lease(required_gib=8.):
        for model in models.values():
            model.cuda()
        path = root/'independent/D20_temporal.npz'
        begin = time.monotonic()
        expected['D20_temporal'] = embeddings(models['D20_temporal'],row,graph,bank,path,hashes['D20_temporal'])
        references['D20_temporal'] = dict(embedding_sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
            cold_wall_seconds=time.monotonic()-begin)
        for family in ['organoid','temporal','compact']:
            names = [name for name,model in models.items() if model.family==family]
            if len({hashes[name] for name in names}) != len(names):
                raise ValueError('Paired source proof requires distinct fixed weights')
            begin = time.monotonic()
            actual = embeddings_many({n:models[n] for n in names},row,graph,bank,
                {n:paths[n] for n in names},{n:hashes[n] for n in names})
            torch.cuda.synchronize()
            seconds = time.monotonic()-begin
            for name in names:
                for value,reference in zip(actual[name],expected[name]):
                    if not np.isfinite(value).all():
                        raise ValueError('A complete source embedding or mask is nonfinite')
                    np.testing.assert_array_equal(value,reference)
            timings.append(dict(family=family,models=names,node_count_per_model=len(graph['nodes']),
                seconds=seconds,model_node_embeddings_per_second=len(names)*len(graph['nodes'])/seconds,
                exact_all_embeddings=True,exact_all_masks=True,cached_embeddings_used=False))
        cached = embeddings_many(models,row,graph,bank,paths,hashes)
        for name in models:
            for value,reference in zip(cached[name],expected[name]):
                np.testing.assert_array_equal(value,reference)
        for model in models.values():
            model.cpu()
        torch.cuda.empty_cache()
    receipt = dict(status='measured',source='44b6',dataset=row['dataset'],frames=row['image_shape'][0],
        node_count_per_model=len(graph['nodes']),models=hashes,references=references,timings=timings,
        exact_full_clip_embeddings_and_masks=True,ordinary_cache_loader_verified=True,
        implementation_sha256=sha(Path(__file__).with_name('shared_embeddings.py')),
        reference_embedding_function_sha256=sha(Path(__file__).with_name('infer.py')),
        mixed_family_cached_reload=True,distinct_weights_for_each_paired_family=True,
        existing_final_178_update_and_pilot_weights_for_parity_only=True,
        new_target_metrics_read=False,training_recipe_changed=False,embedding_averaging=False)
    write_json(RESULTS/'shared_embedding_parity.json',receipt,immutable=True)
    print(receipt,flush=True)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    run()
    import sys
    if '--await-adoption' in sys.argv:
        from .common import WORK, read_json, sha
        marker = WORK/'maintenance/shared_embedding_adoption.json'
        print('Shared-crop parity passed; GPU released; waiting for inference adoption.',flush=True)
        while not marker.exists():
            time.sleep(5)
        accepted = read_json(marker)
        if accepted['shared_embedding_sha256']!=sha(Path(__file__).with_name('shared_embeddings.py')) or \
                accepted['staged_inference_sha256']!=sha(Path(__file__).with_name('staged_infer.py')):
            raise ValueError('Shared-crop adoption does not identify the verified implementation')
