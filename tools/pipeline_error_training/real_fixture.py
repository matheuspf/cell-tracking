"""The actual 44b6_12dfb391 t=66 event, with full-graph official replay."""
from collections import Counter

import numpy as np

from .common import DATA, RESULTS, WORK, graph_hash, inputs, save_graph, sha, verified_graph, write_json


def run():
    from .guard import install
    install(source='44b6')
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    from .actions import apply_decisions
    from .feasibility import from_record, records
    from .labels import SourceLabels
    name, event_id = '44b6_12dfb391', 172000000050
    row = next(r for r in inputs() if r['dataset'] == name)
    graph = verified_graph(row)
    supported = [r for r in records('44b6', name) if event_id in r['labels']['compatible_events']]
    if not supported or len({r['group'] for r in supported}) != 1:
        raise ValueError('Real event alternatives/time anchors lost their biological group')
    gn, ge = read_gt(DATA, name, row['physical_scale'])
    labels = SourceLabels(graph['nodes'], graph['edges'], gn, ge, row['physical_scale'])
    for record in supported:
        actual = labels.decision(from_record(record))
        if actual != record['labels']:
            raise ValueError('Prepared real-event supervision differs from fresh official local matching')
    selected = min(supported, key=lambda r: (abs(int(graph['nodes'][r['anchor'], 1])-66), r['anchor'], from_record(r).key))
    out, ledger = apply_decisions(graph['nodes'], graph['edges'], [from_record(selected)], [1.])
    path = WORK/'real_fixture'/f'{name}.npz'
    save_graph(path, graph['nodes'], out)
    measured, _, _ = evaluate_graph(name, graph['nodes'], out, gn, ge, row['physical_scale'], row['estimated_total'])
    baseline, _, _ = evaluate_graph(name, graph['nodes'], graph['edges'], gn, ge, row['physical_scale'], row['estimated_total'])
    if measured['division_tp'] <= baseline['division_tp']:
        raise ValueError('Real event witness did not recover the official division')
    result = dict(status='measured', source='44b6', dataset=name, annotated_parent_time=66,
        alternatives=len(supported), anchor_times=sorted({int(graph['nodes'][r['anchor'], 1]) for r in supported}),
        grouped_across_all_alternatives_and_time_anchors=True,
        source_label_guided_fixture_not_model_result=True, baseline=baseline, measured=measured,
        prediction_sha256=sha(path), graph_hash=graph_hash(graph['nodes'], out), changed_edges=ledger['changed_edges'],
        no_unmatched_branch_negative=True, new_target_model_scores_read=False)
    write_json(RESULTS/'real_event_fixture.json', result, immutable=True)
    print({k: result[k] for k in ['status', 'alternatives', 'anchor_times', 'changed_edges']}, flush=True)


if __name__ == '__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
