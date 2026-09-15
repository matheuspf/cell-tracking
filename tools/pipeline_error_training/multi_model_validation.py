"""Compare the shared model bank with two completed separate source replays."""
from pathlib import Path
import shutil
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from .common import RESULTS, WORK, inputs, load_graph, read_json, sha, verified_evidence, verified_graph, write_json
    from .resources import Monitor
    from .staged_infer import prepare, solve
    row = next(r for r in inputs() if r['dataset']=='44b6_d754aa59')
    root = WORK/'multi_model_validation'
    packages, destinations, expected = {}, {}, {}
    for arm in ['D10_frozen', 'D10_adapted']:
        package = WORK/'training'/arm/'44b6/20260915'
        spec = read_json(package/'frozen_package.json')
        source = WORK/'source_screen'/arm/'44b6/20260915/predictions'/row['dataset']
        embedding = source/'current_image_cache'/spec['weights_sha256']/f'{row["dataset"]}.npz'
        receipt = read_json(embedding.with_suffix('.json'))
        if sha(embedding) != receipt['sha256']:
            raise ValueError('Source image-derived embedding changed')
        destination = root/'neural/embeddings'/spec['weights_sha256']/embedding.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ['.npz', '.json']:
            if destination.with_suffix(suffix).exists():
                if sha(destination.with_suffix(suffix)) != sha(embedding.with_suffix(suffix)):
                    raise ValueError('Completed validation embedding changed')
            else:
                shutil.copyfile(embedding.with_suffix(suffix), destination.with_suffix(suffix))
        packages[arm] = package
        destinations[arm] = root/arm/embedding.name
        expected[arm] = source/embedding.name
    start = time.monotonic()
    records = []
    with Monitor(root/'resources.json') as monitor:
        graph, native = verified_graph(row), verified_evidence(row)
        bank, receipt = prepare(row, graph, native, packages, root/'neural', monitor=monitor, device='cpu')
        solve(row, graph, bank, root/'neural', destinations, monitor)
        for arm in packages:
            observed, reference = load_graph(destinations[arm]), load_graph(expected[arm])
            for key in ['nodes', 'edges']:
                np.testing.assert_array_equal(observed[key], reference[key])
            records.append(dict(arm=arm, exact_nodes=True, exact_edges=True,
                separate_prediction_sha256=sha(expected[arm]), shared_prediction_sha256=sha(destinations[arm])))
    result = dict(status='measured', dataset=row['dataset'], source='44b6',
        model_count=len(packages), event_count=receipt['records'], models=records,
        seconds=time.monotonic()-start, verified_source_image_embeddings_reused=True,
        cold_image_runtime=False, new_target_metrics_read=False)
    write_json(RESULTS/'multi_model_parity.json', result)
    return result


if __name__ == '__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    print(run())
