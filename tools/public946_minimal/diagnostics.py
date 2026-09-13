"""Independent integrity checks and descriptive, nonselecting diagnostics."""
from __future__ import annotations

from pathlib import Path

from .common import array_hash, load_arrays, read_json, write_json


def stage_check(out, arm, rows, scope='full'):
    """Reject differences before the declared mechanism's first permitted stage."""
    import numpy as np
    from .common import ARMS
    modules = ARMS[arm]['modules']
    if arm.startswith('X'):
        modules = read_json(out/'finalist_lock.json')['recipes'][arm]
    if arm in ('B0', 'B1'):
        return None
    checked = []
    missing = []
    for row in rows:
        name = row['dataset']
        base = out/'predictions'/scope/('B1' if arm.startswith('X') else 'B0')/name
        current = out/'predictions'/scope/arm/name
        if not (base/'complete.json').exists():
            missing.append(name)
            continue
        if any(m in modules for m in ('E04', 'E08')):
            continue
        a = load_arrays(Path(read_json(base/'complete.json')['neural_evidence']))
        b = load_arrays(Path(read_json(current/'complete.json')['neural_evidence']))
        if not np.array_equal(a['coords'], b['coords']):
            raise AssertionError(f'{arm}/{name} unexpectedly changed detections')
        if set(modules).issubset({'E01', 'E02', 'E07'}):
            for phase in (['raw'] if 'E01' in modules else ['raw', 'motion', 'gap1', 'gap2', 'divisions', 'presmooth']):
                x, y = load_arrays(base/f'{phase}.npz'), load_arrays(current/f'{phase}.npz')
                if array_hash(x['nodes'], x['edges']) != array_hash(y['nodes'], y['edges']):
                    raise AssertionError(f'Undeclared first-stage difference: {arm}/{name}/{phase}')
        checked.append(name)
    result = dict(arm=arm, verified=checked, missing_baseline_clips=missing, no_undeclared_difference=not missing)
    write_json(out/'stage_checks'/scope/(arm+'.json'), result)
    return result


def prediction_disagreement(original, transformed, shape, transform):
    """Label-free matching at the unchanged official 7 um convention."""
    import numpy as np
    from annotation_selection.metric_adapter import match_nodes
    a, b = load_arrays(original), load_arrays(transformed)
    n = b['nodes'].copy()
    if transform in ('reflect_x', 'reflect_y'):
        axis = 4 if transform == 'reflect_x' else 3
        n[:, axis] = shape[axis - 1] - 1 - n[:, axis]
    elif transform == 'shift_x1':
        n[:, 4] -= 1
        # Common valid interior only; no wrapping and no altered match tolerance.
        keep = (n[:, 4] >= 1) & (n[:, 4] < shape[-1]-1)
        n = n[keep]
        keep_a = (a['nodes'][:, 4] >= 1) & (a['nodes'][:, 4] < shape[-1]-1)
        a['nodes'] = a['nodes'][keep_a]
        valid_ids = set(a['nodes'][:, 0])
        a['edges'] = np.asarray([e for e in a['edges'] if e[0] in valid_ids and e[1] in valid_ids], np.int64).reshape(-1, 2)
    valid_ids = set(n[:, 0])
    edges = np.asarray([e for e in b['edges'] if e[0] in valid_ids and e[1] in valid_ids], np.int64).reshape(-1, 2)
    matches = match_nodes(n, edges, a['nodes'], a['edges'], [1.625, .40625, .40625])
    original_edges = set(map(tuple, a['edges']))
    mapped = {(matches[int(u)], matches[int(v)]) for u, v in edges if int(u) in matches and int(v) in matches}
    common = len(mapped & original_edges)
    return dict(original_nodes=len(a['nodes']), transformed_nodes=len(n), matched_nodes=len(matches),
        original_edges=len(original_edges), transformed_edges=len(edges), common_edges=common,
        node_disagreement=1 - len(matches)/max(len(a['nodes']), len(n), 1),
        edge_disagreement=1 - common/max(len(original_edges | mapped), 1),
        matching_radius_um=7., scope='common_valid_interior' if transform == 'shift_x1' else 'full_pilot')


def descriptive_strata(args):
    """Fixed quartiles, image occupancy, depth and temporal boundary summaries."""
    import numpy as np
    import zarr
    from .evaluation import gt_arrays
    from annotation_selection.metric_adapter import aggregate
    rows = read_json(args.out/'image_inventory.json')
    occupancy_path = args.out/'diagnostics/raw_image_crowding.json'
    if occupancy_path.exists():
        crowding = read_json(occupancy_path)
    else:
        crowding = {}
        for row in rows:
            # Signal occupancy above q90 in the complete input volume; metadata threshold,
            # no annotations or model scores. This is an intensity-occupancy proxy for crowding.
            arr = zarr.open_group(row['path'], mode='r')['0']
            count = 0
            for t in range(arr.shape[0]):
                count += int((np.asarray(arr[t]) > row['quantiles']['0.9']).sum())
            crowding[row['dataset']] = dict(signal_occupancy_q90=count / np.prod(arr.shape).item(),
                                             measured_voxels=np.prod(arr.shape).item())
        write_json(occupancy_path, crowding)
    contrasts = np.percentile([r['contrast_q99_q10'] for r in rows], [25, 50, 75])
    occupancies = np.percentile([crowding[r['dataset']]['signal_occupancy_q90'] for r in rows], [25, 50, 75])
    outputs = {}
    for arm in ['B0', 'B1', 'E01', 'E02', 'E04', 'E05', 'E07', 'E08', 'C01', 'C02', 'C03', 'X01', 'X02']:
        root = args.out/'scores/full'/arm
        if not (root/'summary.json').exists():
            continue
        records = [read_json(root/(r['dataset']+'.json')) for r in rows]
        groups = {}
        for key, values, boundaries in (
            ('metadata_contrast_quartile', {r['dataset']: r['contrast_q99_q10'] for r in rows}, contrasts),
            ('raw_image_signal_occupancy_quartile', {n: c['signal_occupancy_q90'] for n, c in crowding.items()}, occupancies)):
            for quartile in range(4):
                sub = [r for r in records if int(np.searchsorted(boundaries, values[r['dataset']], side='right')) == quartile]
                if sub:
                    groups[f'{key}_{quartile+1}'] = aggregate(sub, [r['dataset'] for r in sub])
        temporal, depth = {}, {}
        for row, rec in zip(rows, records):
            n, edges, _ = gt_arrays(Path(row['path']).with_suffix('.geff'))
            nodes = {int(r[0]): r for r in n}
            recovered = set(map(tuple, rec['tp_gt_edges']))
            for a, b in edges:
                boundary = min(nodes[a][1], row['image_shape'][0]-1-nodes[b][1])
                time_group = 'first_last_two_frames' if boundary < 2 else 'interior'
                depth_group = min(3, int(4*nodes[a][2]/row['image_shape'][1]))
                for target, group in ((temporal, time_group), (depth, str(depth_group))):
                    record = target.setdefault(group, dict(gt_edges=0, recovered_gt_edges=0))
                    record['gt_edges'] += 1
                    record['recovered_gt_edges'] += int((int(a), int(b)) in recovered)
        outputs[arm] = dict(clip_groups=groups, temporal_edge_recovery=temporal, depth_edge_recovery=depth)
    write_json(args.out/'diagnostics/strata.json', dict(definitions={
        'contrast':'metadata q99-q10 quartiles pooled across the 199 clips',
        'crowding':'fraction of complete raw-image voxels above metadata q90; intensity occupancy proxy, not a count of individual cells',
        'depth':'GT source-node z divided by true Z extent into four fixed bins; evaluator only',
        'temporal':'GT consecutive edges within two frames of either temporal endpoint versus interior; evaluator only',
        'limitations':'Descriptive correlated groups, unknown crop overlaps, no subgroup deployment routing'}, arms=outputs))
