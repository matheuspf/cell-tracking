"""Source-only graph parity for the bounded additive inference implementation."""
import numpy as np
import torch

from .common import RESULTS, WORK, graph_hash, inputs, load_graph, read_json, sha, verified_evidence, verified_graph, write_json


def run():
    from .guard import install
    install(source='44b6')
    from .fast_infer import predict_many
    torch.set_num_threads(2)
    name = '44b6_d754aa59'
    row = next(r for r in inputs() if r['dataset'] == name)
    package = WORK / 'training/D10_frozen/44b6/20260915'
    reference = WORK / 'source_screen/D10_frozen/44b6/20260915/predictions' / name
    output = WORK / 'inference_parity' / f'{name}.npz'
    result = predict_many(row, verified_graph(row), verified_evidence(row), {'D10_frozen': package},
        {'D10_frozen': output}, reference / 'current_image_cache', device='cpu')
    measured, expected = load_graph(output), load_graph(reference / f'{name}.npz')
    equal = (np.array_equal(measured['nodes'], expected['nodes']) and
             np.array_equal(measured['edges'], expected['edges']))
    receipt = dict(status='measured' if equal else 'failed', dataset=name, source_only=True, exact_nodes=bool(np.array_equal(measured['nodes'], expected['nodes'])),
        exact_edges=bool(np.array_equal(measured['edges'], expected['edges'])),
        reference_graph_hash=graph_hash(expected['nodes'], expected['edges']),
        bounded_graph_hash=graph_hash(measured['nodes'], measured['edges']),
        baseline_method='Original per-parent CUDA complete-decision scoring',
        optimized_method='One-time pair scoring, event batches, optimistic complete-assignment upper bounds; CPU audit',
        target_scores_read=False, measured=result,
        files={name: sha(__import__('pathlib').Path(__file__).with_name(name+'.py')) for name in ['actions', 'fast_infer']})
    write_json(WORK / 'inference_parity/receipt.json', receipt)
    if not equal:
        raise ValueError('Bounded inference changed the actual source graph')
    write_json(RESULTS / 'inference_optimization.json', {k: v for k, v in receipt.items() if k != 'measured'}, immutable=True)
    print('Bounded additive inference exactly reproduces the complete source graph', flush=True)
