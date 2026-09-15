"""Post-freeze error partitions using source-frozen image/geometry thresholds."""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .common import DATA, RESULTS, WORK, inputs, load_graph, read_json, verified_evidence, verified_graph, write_json
from .evaluate import write_csv


def bins(row, node, features, near, thresholds):
    spatial = np.asarray(node[2:], float)
    scale = np.asarray(row['physical_scale'])
    distance = np.minimum(spatial, np.asarray(row['image_shape'][1:])-1-spatial)*scale
    value = dict(depth=['low','middle','high'][np.searchsorted(thresholds['normalized_depth'], spatial[0]/max(1,row['image_shape'][1]-1))],
        spatial_boundary='within_7um' if distance.min()<thresholds['spatial_boundary_um'] else 'interior',
        temporal_boundary='boundary' if node[1]<2 or node[1]>=row['image_shape'][0]-4 else 'interior',
        close_cells='within_7um_competitor' if near else 'isolated')
    for name, column in [('intensity',6), ('local_contrast',7), ('density10',8)]:
        value[name] = 'no_assigned_observation' if features is None else ['low','middle','high'][np.searchsorted(thresholds[name], features[column])]
    return value


def one(row, arm, audit):
    from center_comparison.pipeline import read_gt
    source = '6bba' if row['embryo']=='44b6' else '44b6'
    thresholds = read_json(RESULTS/'diagnostic_lock.json')['thresholds'][source]
    graph = verified_graph(row, arm) if arm in ['P0','C4_m6','C0'] else load_graph(WORK/'predictions'/arm/f'{row["dataset"]}.npz')
    node_index = {int(n[0]): i for i,n in enumerate(graph['nodes'])}
    gt, _ = read_gt(DATA, row['dataset'], row['physical_scale'])
    gt_index = {int(n[0]): n for n in gt}
    if 'matches' in audit:
        matches = audit['matches']
    else:
        matches = load_graph(WORK/'evaluation'/arm/f'{row["dataset"]}.npz')['matches']
    reverse = {int(g): int(p) for p,g in matches}
    native = verified_evidence(row)
    p0 = verified_graph(row)
    if np.array_equal(graph['nodes'], p0['nodes']):
        features = native['node_features']
    else:
        path = (WORK/'composition'/arm/row['dataset']/'current_evidence.npz'
                if arm.startswith('C10') else
                WORK/'final_point_inference'/arm/row['dataset']/'fresh_native/current_features.npz')
        if not path.exists():
            raise FileNotFoundError('Changed-point image strata require the refreshed native evidence')
        features = load_graph(path)['node_features']
    neighbors = {}
    for t in np.unique(graph['nodes'][:,1]):
        ids = np.flatnonzero(graph['nodes'][:,1]==t)
        pos = graph['nodes'][ids,2:]*row['physical_scale']
        count = cKDTree(pos).query_ball_point(pos, 7., return_length=True)-1
        neighbors.update({int(graph['nodes'][i,0]): int(n) for i,n in zip(ids,count)})
    result = Counter()
    def add(metric, node, ground_truth):
        if ground_truth:
            position = gt_index[int(node)]
            pid = reverse.get(int(node))
        else:
            pid = int(node)
            position = graph['nodes'][node_index[pid]]
        f = features[node_index[pid]] if pid in node_index else None
        for dimension, level in bins(row, position, f, neighbors.get(pid,0)>0, thresholds).items():
            result[dimension,level,metric] += 1
    for a,b in audit['tp_edges']:
        add('edge_tp',b,True)
    for p in audit['tp_divisions']:
        add('division_tp',p,True)
    for flag in audit['flags']:
        category = flag['category']
        metric = 'edge_fn' if category in ['edge_endpoint','edge_link'] else category
        ground_truth = metric in ['edge_fn','division_fn']
        node = flag['other'] if metric.startswith('edge') else flag['node']
        add(metric,node,ground_truth)
        if category.startswith('division'):
            result['event_support',flag['stage'],metric] += 1
    return result


def collect(monitor):
    if not (RESULTS/'target_freeze.json').exists():
        raise PermissionError('New case diagnostics follow the directional target freeze')
    counts = Counter()
    arms = [p.name for p in (WORK/'full_evaluation').iterdir() if (p/'summary.json').exists()]
    for arm in sorted(arms):
        for row in inputs():
            monitor.check()
            path = WORK/'full_evaluation'/arm/'audit'/f'{row["dataset"]}.json'
            for (dimension,level,metric),count in one(row,arm,read_json(path)).items():
                counts[arm,row['embryo'],dimension,level,metric] += count
    rows = [dict(arm=a, embryo=e, dimension=d, stratum=l, metric=m, count=c, status='measured')
            for (a,e,d,l,m),c in sorted(counts.items())]
    write_csv(RESULTS/'diagnostic_strata.csv',rows)
    write_json(RESULTS/'strata_validation.json',dict(status='measured',arms=arms,source_only_thresholds=True,
        image_features_at='Actual matched or evaluated predicted observation; unavailable centers have an explicit missing-observation bucket.',
        strata_are_descriptive_not_independent_tests=True, no_target_threshold_selection=True))


def run():
    from .resources import Monitor
    with Monitor(WORK/'strata/resources.json') as monitor:
        collect(monitor)


if __name__ == '__main__':
    from .resources import cpu_budget
    cpu_budget()
    run()
