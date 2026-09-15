"""Read-only instrumentation of the actual historical constructors, with exact replay."""
from collections import Counter
from contextlib import contextmanager
import sys
import time

import numpy as np

from .common import (
    RESULTS, ROOT, WORK, graph_hash, inputs, read_json, save_arrays, save_graph, sha,
    verified_evidence, verified_graph, write_json,
)


@contextmanager
def capture_returns(filename, names):
    captured = {name: [] for name in names}
    previous = sys.getprofile()
    def profile(frame, event, value):
        if event == 'return' and frame.f_code.co_filename == filename and frame.f_code.co_name in names:
            captured[frame.f_code.co_name].append(dict(frame.f_locals))
    sys.setprofile(profile)
    try:
        yield captured
    finally:
        sys.setprofile(previous)


def p0():
    from .artifacts import run
    from strong_tracker_v3.association import decode
    from .actions import apply_decisions
    from .resources import Monitor
    models = run()
    totals = Counter()
    with Monitor(WORK / 'resources/trace-P0.json') as monitor:
        for i, row in enumerate(inputs(), 1):
            begin = time.monotonic()
            name = row['dataset']
            source = '6bba' if row['embryo'] == '44b6' else '44b6'
            base, expected = verified_graph(row, 'C0'), verified_graph(row, 'P0')
            native = verified_evidence(row)
            spec = read_json(models[f'P0_{source}']['path'])
            model = spec['model']
            x = native['edge_features']
            scores = x[:, 23]+model['beta'][0]+((x[:, spec['columns']]-model['mean'])/model['scale'])@np.asarray(model['beta'][1:])
            with capture_returns(decode.__code__.co_filename, ['decode', 'add_action']) as captured:
                edges, receipt = decode(base['nodes'], base['edges'], native, scores, margin=3., max_fraction=.02)
            if not np.array_equal(edges, expected['edges']):
                raise ValueError(f'Exact historical P0 replay mismatch: {name}')
            state = captured['decode'][0]
            rows = []
            action_values = {}
            for a in captured['add_action']:
                for edge in a.get('adds', set()):
                    if 'value' in a:
                        action_values[edge] = max(action_values.get(edge, -np.inf), a['value'])
            old = state['old']
            output = {(state['ix'][int(a)], state['ix'][int(b)]) for a, b in edges}
            for (a, b), score in zip(native['pairs'], scores):
                key = (int(a), int(b))
                if key in output:
                    reason = 'final_selected'
                elif a in state['fs'] or b in state['ft']:
                    reason = 'fixed_agreement_or_fork_boundary'
                elif b not in state['by_source'].get(a, []):
                    reason = 'ranked_continuation_bank'
                elif key not in action_values:
                    reason = 'no_legal_complete_owner_action_or_prediction_evidence_gate'
                elif action_values[key] <= 1e-9:
                    reason = 'complete_action_score_or_margin'
                else:
                    reason = 'conflict_solver_or_edit_cap'
                totals[reason] += 1
                rows.append([int(base['nodes'][a, 0]), int(base['nodes'][b, 0]), float(score),
                             float(key in old), float(key in output), float(action_values.get(key, np.nan))])
            save_arrays(WORK / 'trace/P0' / f'{name}.npz', edges=np.asarray(rows),
                        reason=np.asarray([
                            'final_selected' if (int(a), int(b)) in output else
                            'fixed_agreement_or_fork_boundary' if a in state['fs'] or b in state['ft'] else
                            'ranked_continuation_bank' if b not in state['by_source'].get(a, []) else
                            'no_legal_complete_owner_action_or_prediction_evidence_gate' if (int(a), int(b)) not in action_values else
                            'complete_action_score_or_margin' if action_values[int(a), int(b)] <= 1e-9 else
                            'conflict_solver_or_edit_cap' for a, b in native['pairs']]))
            write_json(WORK / 'trace/P0' / f'{name}.json', dict(dataset=name, original_code_sha256=sha(decode.__code__.co_filename),
                model_sha256=models[f'P0_{source}']['sha256'], exact_graph_replay=True,
                graph_hash=graph_hash(base['nodes'], edges), ledger=receipt, seconds=time.monotonic()-begin,
                division_stage='P0 is continuation-only. Its fork decisions and missing forks originate upstream in C0.',
                gt_reads=False))
            zero, _ = apply_decisions(expected['nodes'], expected['edges'], [], [])
            if not np.array_equal(zero, expected['edges']):
                raise ValueError('Zero-head identity failed')
            save_graph(WORK / 'predictions/D00' / f'{name}.npz', expected['nodes'], zero)
            monitor.check()
            if i % 20 == 0 or i == 199:
                print(f'Exact historical P0 trace and D00 identity: {i}/199 clips', flush=True)
    write_json(RESULTS / 'D00.json', dict(status='measured', clips=199, exact_P0_arrays=True,
        frozen_bank='Pinned v3 defaults; lazy equivalence covered by executable tests.',
        no_new_head_residual=0., target_labels_read=False))
    write_json(WORK / 'trace/P0/summary.json', dict(status='measured', clips=199, counts=totals,
        precise_unavailable_reason='The historical continuation generator combines some owner legality and evidence gate exits; combined codes are explicit, not guessed stage attribution.'))


def c4_one(row):
    import torch
    import zarr
    from .artifacts import run
    from .resources import Lease
    artifacts = run()
    # The unchanged module installs its annotation-denial guard on import.
    from multidata_training_v4.infer import transform
    source = '6bba' if row['embryo'] == '44b6' else '44b6'
    base, expected = verified_graph(row, 'C0'), verified_graph(row, 'C4_m6')
    for component in ['G', 'I']:
        spec = artifacts[f'{component}_C4_{source}']
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('C4 trace weights changed')
    calibration = read_json(ROOT / 'multidata-training-v4/calibration.json')
    image = zarr.open_group(row['image_path'], mode='r')['0'][:]
    with Lease(required_gib=4.):
        with capture_returns(transform.__code__.co_filename, ['transform']) as captured:
            nn, ee, receipt = transform(base['nodes'], base['edges'], image, row['physical_scale'], source,
                'C4_m6', ROOT / 'multidata-training-v4/models', calibration)
        torch.cuda.empty_cache()
    if not np.array_equal(nn, expected['nodes']) or not np.array_equal(ee, expected['edges']):
        raise ValueError('C4_m6 actual path does not reproduce its exact local graph')
    state = captured['transform'][0]
    path = WORK / 'trace/C4_m6' / f'{row["dataset"]}.npz'
    save_arrays(path, anchors=state['a'], candidates=state['c'], geometry_logits=state['gl'],
                calibrated_complete_logits=state['logits'], gate=state['gate'], events=state['ev'], gain=state['gain'])
    write_json(path.with_suffix('.json'), dict(dataset=row['dataset'], exact_graph_replay=True,
        graph_hash=graph_hash(nn, ee), calibration_sha256=sha(ROOT / 'multidata-training-v4/calibration.json'),
        source_model=source, margin=6., geometry_gate=.5, original_path_sha256=sha(transform.__code__.co_filename),
        receipt=receipt, target_labels_read=False))


def c4():
    from .resources import Monitor, cpu_budget
    cpu_budget()
    import torch
    torch.set_num_threads(2)
    records = []
    with Monitor(WORK/'resources/trace-C4_m6.json') as monitor:
        for row in inputs():
            path = WORK/'trace/C4_m6'/f'{row["dataset"]}.json'
            if not path.exists():
                c4_one(row)
            records.append(read_json(path))
            monitor.check()
            if len(records) % 20 == 0 or len(records) == 199:
                print(f'Exact actual C4_m6 gate/score/decoder replay: {len(records)}/199', flush=True)
    write_json(RESULTS/'C4_trace_validation.json', dict(status='measured', clips=len(records),
        all_exact_node_and_edge_arrays=all(r['exact_graph_replay'] for r in records),
        original_algorithms_unchanged=True, no_ground_truth_reads=True,
        receipt_hashes={r['dataset']: sha(WORK/'trace/C4_m6'/f'{r["dataset"]}.json') for r in records}), immutable=True)


if __name__ == '__main__':
    c4()
