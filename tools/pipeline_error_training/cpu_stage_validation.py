"""Exact source parity after moving geometry enumeration outside the GPU lease."""
import argparse
from pathlib import Path
import shutil
import time


def run(case):
    source = '44b6' if case=='small' else '6bba'
    from .guard import install
    install(source=source)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from .common import RESULTS, WORK, inputs, load_graph, read_json, sha, verified_evidence, verified_graph, write_json
    from .resources import Monitor
    from .staged_infer import prepare, solve
    dataset = '44b6_d754aa59' if case=='small' else '6bba_767a1e17'
    row = next(r for r in inputs() if r['dataset']==dataset)
    previous = WORK/'multi_model_validation/neural' if case=='small' else WORK/'staged_crowded_parity'
    root = WORK/'cpu_stage_validation'/case
    old = read_json(previous/'prepared.json')
    packages, destinations = {}, {}
    for name in old['names']:
        arm = old['arms'][name]
        packages[name] = WORK/'training'/arm/source/'20260915'
        destinations[name] = root/name/f'{dataset}.npz'
        embedding = previous/'embeddings'/old['inputs']['model_sha256'][name]/f'{dataset}.npz'
        receipt = read_json(embedding.with_suffix('.json'))
        if sha(embedding)!=receipt['sha256']:
            raise ValueError('Prior source embedding changed')
        dest = root/'neural/embeddings'/old['inputs']['model_sha256'][name]/embedding.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.npz','.json']:
            shutil.copyfile(embedding.with_suffix(suffix),dest.with_suffix(suffix))
    start = time.monotonic()
    with Monitor(root/'resources.json') as monitor:
        graph, native = verified_graph(row), verified_evidence(row)
        bank, measured = prepare(row, graph, native, packages, root/'neural', monitor, device='cpu')
        if sha(root/'neural/events.bin')!=sha(previous/'events.bin'):
            raise ValueError('CPU staging changed the exact per-event neural values or event order')
        old_scalars, new_scalars = load_graph(previous/'scalars.npz'), load_graph(root/'neural/scalars.npz')
        for key in old_scalars:
            np.testing.assert_array_equal(old_scalars[key],new_scalars[key])
        solve(row,graph,bank,root/'neural',destinations,monitor)
        records = []
        for name, destination in destinations.items():
            arm = old['arms'][name]
            expected = WORK/'source_screen'/arm/source/'20260915/predictions'/dataset/f'{dataset}.npz'
            result, control = load_graph(destination), load_graph(expected)
            for key in ['nodes','edges']:
                np.testing.assert_array_equal(result[key],control[key])
            records.append(dict(arm=arm,prediction_sha256=sha(destination),reference_sha256=sha(expected),
                exact_nodes=True,exact_edges=True))
    receipt = dict(status='measured',case=case,dataset=dataset,source=source,
        event_count=measured['records'],model_count=len(packages),models=records,
        exact_neural_event_and_scalar_values=True,geometry_seconds=measured['cpu_geometry_seconds'],
        seconds=time.monotonic()-start,verified_source_embeddings_reused=True,
        cold_image_runtime=False,measured_gpu_lease_seconds=0.,new_target_metrics_read=False,
        implementation_sha256=sha(Path(__file__).with_name('staged_infer.py')))
    write_json(RESULTS/f'cpu_geometry_parity_{case}.json',receipt)
    print(receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['small','crowded'],required=True)
    args=parser.parse_args()
    from .resources import cpu_budget
    cpu_budget(4)
    run(args.case)
