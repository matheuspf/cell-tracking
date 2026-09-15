"""Fresh full-ensemble native evidence at the actual selected observation coordinates.

The original predictor's neural, bidirectional, consensus, and detector-retention
bodies are unchanged. Only the final coordinate query and diagnostic destinations
are adapted in a private source copy. Original detector calls still measure the
released retention guard; selected coordinates are queried after that guard.
"""
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .common import REPO, ROOT, WORK, digest, load_graph, read_json, save_arrays, sha, write_json


class QueryAdapter(ast.NodeTransformer):
    def __init__(self):
        self.coordinates = 0
        self.probabilities = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'arr'
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == '_detect_cells_pooled'):
            node.value = ast.Call(func=ast.Name(id='_study_query', ctx=ast.Load()),
                                  args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='ds_arr', ctx=ast.Load())], keywords=[])
            self.coordinates += 1
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == '_study_node_probabilities'):
            node.args = [ast.Call(func=ast.Name(id='_study_probability_at', ctx=ast.Load()),
                args=[ast.Name(id='_study_prob', ctx=ast.Load()), ast.Name(id='arr', ctx=ast.Load())], keywords=[])]
            self.probabilities += 1
        return node


def query(row, nodes, destination):
    import torch
    import tracksdata as td
    from strong_tracker_v3.replay import notebook_environment
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    package = ROOT / 'image-native-tracking-v5/inference_package_validation/base'
    manifest = read_json(package / 'manifest.json')
    full = ROOT / 'annotation-selection-v1/public_harmonic_full'
    origin = full / 'tracking_repo'
    for relative, expected in manifest['tracking_source_files'].items():
        if sha(origin / relative) != expected:
            raise ValueError('Native predictor source differs from the exact incumbent package')
    weights = manifest['external_checkpoint_paths']
    for name in ['primary', 'secondary']:
        if sha(weights[name]) != manifest[name+'_weights_sha256']:
            raise ValueError('Native query checkpoint drift')
    stamp = dict(image_metadata_sha256=sha(Path(row['image_path'])/'zarr.json'),
        observations_sha256=digest(nodes), primary_sha256=manifest['primary_weights_sha256'],
        secondary_sha256=manifest['secondary_weights_sha256'], code_sha256=sha(Path(__file__)))
    output = destination / 'query.npz'
    if output.exists():
        receipt = read_json(destination / 'query.json')
        if receipt['inputs'] != stamp or sha(output) != receipt['sha256']:
            raise ValueError('Fresh coordinate-query cache mismatch')
        for relative, expected in receipt['image_files'].items():
            if sha(Path(row['image_path'])/relative) != expected:
                raise ValueError('Fresh coordinate-query image content changed')
        return load_graph(output), receipt
    private = destination / 'tracking_source'
    for relative in manifest['tracking_source_files']:
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin / relative, target)
    script = private / 'scripts/predict_unet_transformer.py'
    source = script.read_text()
    old = '/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full'
    if source.count(old) != 2:
        raise ValueError('Unexpected predictor diagnostic destinations')
    source = source.replace(old, str(destination))
    tree = ast.parse(source)
    adapter = QueryAdapter()
    tree = ast.fix_missing_locations(adapter.visit(tree))
    if (adapter.coordinates, adapter.probabilities) != (1, 1):
        raise ValueError('Native coordinate adapter did not match its two exact call sites')
    source = ast.unparse(tree)+'\n'
    script.write_text(source)
    sys.path[:0] = [str(private / 'src'), str(private / 'scripts')]
    name = 'pipeline_error_native_query_'+digest(str(destination))[:12]
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    order = np.concatenate([np.flatnonzero(nodes[:, 1] == t) for t in range(row['image_shape'][0])])
    expected_coords = nodes[order, 1:].astype(np.int64)
    def coordinate_query(t, downsample):
        points = nodes[nodes[:, 1] == t, 1:].astype(np.float32).copy()
        points[:, 1:] /= downsample
        return points
    def probability_query(probability, arr):
        xyz = np.clip(arr[:, 1:].astype(int), 0, np.asarray(probability.shape)-1)
        return probability[tuple(xyz.T)]
    module._study_query = coordinate_query
    module._study_probability_at = probability_query
    settings = notebook_environment((full / 'harmonic_isolated.py').read_text())
    settings.update(BIOHUB_SECONDARY_WEIGHTS=weights['secondary'], BIOHUB_DIAGNOSTIC_ARM='pipeline_error_native_query')
    previous = {k: os.environ.get(k) for k in settings}
    os.environ.update(settings)
    images = destination / 'images/test'
    images.mkdir(parents=True, exist_ok=True)
    image = images / 'explicit_observations.zarr'
    if not image.exists():
        image.symlink_to(Path(row['image_path']).resolve(), target_is_directory=True)
    begin = time.monotonic()
    try:
        primary, window, downsample = module.load_model(Path(weights['primary']), torch.device('cuda'))
        secondary, other_window, other_downsample = module.load_model(Path(weights['secondary']), torch.device('cuda'))
        if (window, downsample) != (other_window, other_downsample):
            raise ValueError('Incumbent dual model contracts differ')
        cfg = module.PredictConfig(det_threshold=float(settings['BIOHUB_DET_THRESHOLD']),
            threshold=float(settings['BIOHUB_DUAL_SEED_EDGE_THRESHOLD']), use_ilp=True,
            ilp_edge_weight=-1., ilp_appearance_weight=float(settings['BIOHUB_ILP_APPEARANCE_WEIGHT']),
            ilp_disappearance_weight=float(settings['BIOHUB_ILP_DISAPPEARANCE_WEIGHT']),
            ilp_division_weight=float(settings['BIOHUB_ILP_DIVISION_WEIGHT']))
        coords, edges = module.predict_video(primary, image, torch.device('cuda'), cfg,
            window_size=window, downsample=downsample, unet_batch_size=4, secondary_model=secondary,
            secondary_edge_weight=float(settings['BIOHUB_SECONDARY_EDGE_WEIGHT']),
            secondary_detection_weight=float(settings['BIOHUB_SECONDARY_DETECTION_WEIGHT']),
            secondary_link_mode=settings['BIOHUB_SECONDARY_LINK_MODE'],
            secondary_mix_temperature=float(settings['BIOHUB_SECONDARY_MIX_TEMPERATURE']),
            secondary_low_margin_max=float(settings['BIOHUB_SECONDARY_LOW_MARGIN_MAX']))
        np.testing.assert_array_equal(coords, expected_coords)
        g = module.build_graph(coords, edges)
        if g.num_edges():
            solver = td.solvers.ILPSolver(edge_weight=-1.*td.EdgeAttr('edge_prob'),
                appearance_weight=cfg.ilp_appearance_weight, disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight)
            with module.suppress_output():
                g = solver.solve(g)
        ids = g.node_attrs().select('node_id').to_numpy().flatten()
        # The installed graph library's edge attributes include an extra edge ID.
        ee = g.edge_attrs().select('source_id', 'target_id').to_numpy().astype(np.int64).reshape(-1, 2)
        pre = load_graph(destination / 'images/pre_ilp_explicit_observations.npz')
        result = dict(coords=coords, node_probabilities=pre['node_probabilities'], edge_scores=pre['edge_scores'],
                      raw_edges=ee, raw_selected_ids=ids, original_row_order=order)
        save_arrays(output, **result)
        del primary, secondary
        torch.cuda.empty_cache()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    image_files = {str(p.relative_to(Path(row['image_path']).resolve())): sha(p)
                   for p in Path(row['image_path']).resolve().rglob('*') if p.is_file()}
    receipt = dict(status='measured', inputs=stamp, sha256=sha(output), seconds=time.monotonic()-begin,
        observations=len(nodes), exact_coordinate_queries=True, primary_and_secondary=True,
        full_harmonic_bidirectional_and_consensus=True, detector_retention_guard_unchanged=True,
        source_algorithm_sha256=manifest['tracking_source_files']['scripts/predict_unet_transformer.py'],
        adapted_source_sha256=sha(script), settings=settings, image_files=image_files,
        transformations=['explicit selected coordinate query', 'bounded probability sampling at inherited border points',
                         'private diagnostic destinations'], target_labels_read=False)
    write_json(destination / 'query.json', receipt, immutable=True)
    return result, receipt


def refresh(row, graph, old_graph, old_native, destination):
    """Rebuild all 38 native features; transfer teacher IDs only for unchanged observations."""
    from strong_tracker_v3.association_fresh import build_from_inputs
    from strong_tracker_v3.context import RunContext
    from strong_tracker_v3.inference import joint_image_features
    q, receipt = query(row, graph['nodes'], destination)
    order = q['original_row_order']
    nodes = np.column_stack([np.arange(len(order)), graph['nodes'][order, 1:]])
    mapping = {int(graph['nodes'][i, 0]): j for j, i in enumerate(order)}
    convert = lambda edges: np.asarray([(mapping[int(a)], mapping[int(b)]) for a, b in edges
                                       if int(a) in mapping and int(b) in mapping], np.int64).reshape(-1, 2)
    edges = convert(graph['edges'])
    original_ids = {int(n[0]) for i, n in enumerate(old_graph['nodes']) if old_native['native_index'][i] >= 0}
    donor = old_native['old_nodes']
    eligible = {int(n[0]) for n in donor if int(n[0]) in mapping and int(n[0]) in original_ids}
    donor_nodes = np.asarray([[mapping[int(n[0])], *n[1:]] for n in donor if int(n[0]) in eligible], np.int64).reshape(-1, 5)
    donor_convert = lambda es: convert([(a, b) for a, b in es if int(a) in eligible and int(b) in eligible])
    old = dict(nodes=donor_nodes, edges=donor_convert(old_native['old_edges']))
    raw = dict(nodes=nodes, edges=q['raw_edges'])
    current_features = joint_image_features(row, [(nodes, edges)])[0]
    ctx = RunContext.default(repo=REPO, out=Path(destination), work=Path(destination)/'features')
    f, provenance = build_from_inputs(ctx, row, nodes, edges, raw, q, old,
        donor_convert(old_native['e_native']), donor_convert(old_native['e_hgb']), node_features=current_features)
    f['pairs'] = order[f['pairs']]
    pair_order = np.lexsort((f['pairs'][:, 1], f['pairs'][:, 0]))
    f['pairs'], f['edge_features'] = f['pairs'][pair_order], f['edge_features'][pair_order]
    inverse = np.argsort(order)
    for key in ['node_features', 'tracklet', 'native_index', 'native_confidence', 'node_relocation_um']:
        f[key] = f[key][inverse]
    save_arrays(Path(destination)/'current_features.npz', **f)
    write_json(Path(destination)/'feature_receipt.json', dict(status='measured', query_sha256=receipt['sha256'],
        all_features_recomputed=True, prior_nearest_node_feature_copies=0, new_observations_receive_legacy_teacher_votes=False,
        features_sha256=sha(Path(destination)/'current_features.npz'), provenance=provenance))
    return f
