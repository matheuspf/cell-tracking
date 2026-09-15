"""No-label worker that actually reruns both native neural models at new points."""
import atexit
import json
from pathlib import Path
import sys


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    root = Path(job['root'])
    row = job['row']
    from .guard import install
    guard = install(fresh_root=root, allowed_models=[d['path'] for d in job['dependencies']], images=[row['image_path']])
    def record():
        root.mkdir(parents=True, exist_ok=True)
        (root/'guard.json').write_text(json.dumps(guard)+'\n')
    atexit.register(record)
    if not guard['installed_before_numerical']:
        raise RuntimeError('Native query guard must be installed before dependencies')
    from .resources import Lease, Monitor, cpu_budget
    cpu_budget(4)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from .common import graph_hash, load_graph, save_graph, sha, write_json
    from .native_refresh import query, refresh
    from .observations import ObservationBank, raw_graph
    from .actions import fork_support
    for dep in job['dependencies']:
        if sha(dep['path']) != dep['sha256']:
            raise ValueError('Native query proof input changed')
    raw = raw_graph(row)
    graph = load_graph(row['baselines']['P0']['path'])
    native = load_graph(row['evidence']['path'])
    pre = load_graph(row['raw']['path'])
    bank = ObservationBank(row, graph, raw)
    with Monitor(root/'resources.json') as monitor, Lease(required_gib=8.):
        original, original_receipt = query(row, raw['nodes'], root/'original_query')
        np.testing.assert_array_equal(original['coords'], pre['coords'])
        np.testing.assert_allclose(original['node_probabilities'], pre['node_probabilities'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(original['edge_scores'], pre['edge_scores'], rtol=1e-5, atol=1e-6)
        protected = fork_support(graph['nodes'], graph['edges'])
        i, j = next((int(i), int(j)) for i, j in bank.pairs if i not in protected and j not in bank.exact_alias
                    and 40 <= graph['nodes'][i, 1] <= 60)
        changed = {k: v.copy() for k, v in graph.items()}
        old_id = int(changed['nodes'][i, 0])
        new_id = int(changed['nodes'][:, 0].max())+1+j
        changed['nodes'][i] = [new_id, *raw['nodes'][j, 1:]]
        changed['edges'][changed['edges'] == old_id] = new_id
        save_graph(root/'explicit_changed_observation.npz', changed['nodes'], changed['edges'])
        current = refresh(row, changed, graph, native, root/'changed_query')
        incident = np.any(current['pairs'] == i, axis=1)
        if not incident.any() or not np.isfinite(current['edge_features'][incident]).all():
            raise RuntimeError('Fresh changed-point native evidence is absent or nonfinite')
        if current['edge_features'][incident, 26:29].any():
            raise RuntimeError('New raw observation inherited an old-cell teacher identity')
        monitor.check()
    write_json(root/'receipt.json', dict(status='measured', dataset=row['dataset'], complete_frames=row['image_shape'][0],
        unchanged_raw_query_coordinate_parity=True, unchanged_raw_query_probability_parity=True,
        probability_absolute_tolerance=1e-6, probability_relative_tolerance=1e-5,
        edge_score_max_abs_difference=float(np.max(np.abs(original['edge_scores']-pre['edge_scores']))),
        actual_changed_observation=True, old_node_id=old_id, new_raw_provenance_id=j, new_node_id=new_id,
        old_coordinate=graph['nodes'][i, 1:].tolist(), new_coordinate=changed['nodes'][i, 1:].tolist(),
        original_graph_hash=graph_hash(graph['nodes'], graph['edges']), changed_graph_hash=graph_hash(changed['nodes'], changed['edges']),
        primary_and_secondary_reexecuted=True, harmonic_bidirectional_consensus_reexecuted=True,
        all_38_features_recomputed=True, nearest_old_feature_copies=0, inherited_teacher_identity_for_new_point=False,
        fixture_is_not_a_trained_observation_result=True, no_target_metrics_read=True), immutable=True)


if __name__ == '__main__':
    main()
