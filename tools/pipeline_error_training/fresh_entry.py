"""Startup-guarded full P0 reconstruction with explicit image/model arguments.

This entry point intentionally imports only the standard library until the
annotation/cache/network guard is installed. It is not a cached-graph replay.
"""
import argparse
import atexit
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['baseline', 'point-head', 'module'], required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--source', choices=['44b6', '6bba'], required=True)
    parser.add_argument('--p0-model', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--identity-model', type=Path)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.package / 'manifest.json').read_text())
    explicit = [*manifest['external_checkpoint_paths'].values(), *manifest['external_teacher_paths'].values(),
                args.package / 'association_models', args.package / 'event_models', args.p0_model]
    specifications = {}
    for kind, model_path in [('event', args.model), ('identity', args.identity_model)]:
        if model_path is None:
            continue
        explicit.append(model_path)
        specification = json.loads((model_path / 'frozen_package.json').read_text())
        specifications[kind] = specification
        if specification['recipe']['source'] != args.source:
            raise PermissionError('Fresh model source must match the explicit source argument')
        if 'architecture_dependency' in specification:
            explicit.append(specification['architecture_dependency']['path'])
    from .guard import install
    guard = install(fresh_root=args.output, allowed_models=explicit, images=list(args.images.glob('*.zarr')))
    if not guard['installed_before_numerical']:
        raise RuntimeError('Fresh inference guard was installed after numerical imports')
    def report():
        path = args.output / 'startup_audits' / f'{args.stage}-{os.getpid()}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(guard, stage=args.stage, source_model=args.source,
            scope='Python audit hooks, including cache and socket denial; not a Linux security namespace'))+'\n')
    atexit.register(report)
    # The exact baseline package imports its own pinned algorithm modules.
    sys.path.insert(0, str(args.package / 'tools'))
    sys.dont_write_bytecode = True
    from strong_tracker_v3.inference import deny_annotations, predict_directory
    deny_annotations(allowed_geff_root=args.output / 'fresh')
    from .resources import cpu_budget
    cpu_budget()
    os.environ.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                      V5_V1=str(args.v1), V5_V2=str(args.v2))
    if args.stage == 'baseline':
        # Call the exact pinned function whose startup guard was installed above.
        # run_module would create a second module and install its guard too late.
        args.source_model = args.source
        predict_directory(args)
        return
    # P0's original continuation residual is reconstructed from fresh full
    # harmonic/secondary/DeepCenter evidence using its exact recovered weights.
    from strong_tracker_v3.common import load_graph, save_arrays, sha, graph_hash, write_json, read_json
    row = read_json(args.output / 'inputs.json')[0]
    name = row['dataset']
    if args.stage == 'point-head':
        import numpy as np
        from strong_tracker_v3.context import RunContext
        from strong_tracker_v3.inference import rebuild_teachers
        from strong_tracker_v3.association_fresh import build_from_inputs
        from strong_tracker_v3.association import decode
        ctx = RunContext.default(repo=args.package, v1=args.v1, v2=args.v2,
                                 out=args.output, work=args.output / 'point_scratch')
        baseline = load_graph(args.output / 'fresh/incumbent' / f'{name}.npz')
        c0 = load_graph(args.output / 'predictions' / f'{name}.npz')
        raw = load_graph(args.output / 'fresh/raw' / f'{name}.npz')
        pre = load_graph(args.output / 'fresh/inputs' / f'pre_ilp_{name}.npz')
        begin = time.monotonic()
        old, en, eh, provenance = rebuild_teachers(ctx, row, raw, pre, source_model=args.source,
            image_dir=Path(row['image_path']).parent, current_graph=(baseline['nodes'], baseline['edges']))
        f, _ = build_from_inputs(ctx, row, baseline['nodes'], baseline['edges'], raw, pre, old, en, eh,
                                 node_features=provenance.pop('_joint_current_features'))
        spec = read_json(args.p0_model)
        model, x = spec['model'], f['edge_features']
        value = x[:, 23]+model['beta'][0]+((x[:, spec['columns']]-model['mean'])/model['scale'])@np.asarray(model['beta'][1:])
        edges, ledger = decode(c0['nodes'], c0['edges'], f, value, margin=3., max_fraction=.02)
        save_arrays(args.output / 'P0.npz', nodes=c0['nodes'], edges=edges)
        save_arrays(args.output / 'current_evidence.npz', **f, old_nodes=old['nodes'], old_edges=old['edges'],
                    e_native=en, e_hgb=eh)
        write_json(args.output / 'P0_receipt.json', dict(source_model=args.source, model_sha256=sha(args.p0_model),
            seconds=time.monotonic()-begin, graph_hash=graph_hash(c0['nodes'], edges),
            complete_pipeline='Primary + secondary detection, harmonic bidirectional links, low-margin secondary consensus, DeepCenter repairs, selected C0 and exact P0 residual',
            native_recomputed=True, ledger=ledger, fresh_only=True))
        from .serialization import export_csv
        export_csv(args.output / 'P0.csv', name, c0['nodes'], edges)
        return
    if not args.model:
        parser.error('module stage requires an explicit frozen --model package')
    from .serialization import export_csv, export_geff
    graph, native = load_graph(args.output / 'P0.npz'), load_graph(args.output / 'current_evidence.npz')
    row.update(metadata_sha256=sha(Path(row['image_path']) / 'zarr.json'))
    spec = specifications['event']
    arm = specifications.get('identity', spec)['recipe']['arm']
    component_model = args.identity_model or args.model
    component_destination = args.output/('identity.npz' if args.identity_model else 'module.npz')
    if arm == 'A10':
        from .infer import predict
        predict(row, graph, native, component_model, component_destination, cache_root=args.output/'current_module_cache')
    elif arm == 'O10_swap':
        import numpy as np
        from .observation_infer import predict
        raw_path = args.output/'fresh/raw'/f'{name}.npz'
        pre_path = args.output/'fresh/inputs'/f'pre_ilp_{name}.npz'
        raw_graph, pre = load_graph(raw_path), load_graph(pre_path)
        raw = dict(nodes=np.column_stack([np.arange(len(pre['coords'])), pre['coords']]).astype(np.int64),
            edges=raw_graph['edges'], node_probabilities=pre['node_probabilities'], edge_scores=pre['edge_scores'],
            original_graph_sha256=sha(raw_path))
        row.update(baselines={'P0': {'sha256': sha(args.output/'P0.npz')}}, raw={'sha256': sha(pre_path)})
        predict(row, graph, native, component_model, {'O10_swap': component_destination}, args.output/'current_module_cache',
                args.p0_model, raw_override=raw)
    else:
        if args.identity_model:
            raise ValueError('Fresh composition requires the frozen A10 or O10 identity component')
        from .fast_infer import predict_many
        predict_many(row, graph, native, {'model': args.model}, {'model': args.output/'module.npz'}, args.output/'current_module_cache')
    if args.identity_model:
        import numpy as np
        from .fast_infer import predict_many
        current_graph = load_graph(component_destination)
        if all(np.array_equal(current_graph[k], graph[k]) for k in ['nodes', 'edges']):
            current_native = native
            refresh_kind = 'Identity component exactly preserves the freshly rebuilt P0 graph'
        elif np.array_equal(current_graph['nodes'], graph['nodes']):
            from strong_tracker_v3.association_fresh import build_from_inputs
            from strong_tracker_v3.context import RunContext
            ctx = RunContext.default(repo=args.package, v1=args.v1, v2=args.v2,
                out=args.output, work=args.output/'composition_structure')
            current_native, _ = build_from_inputs(ctx, row, current_graph['nodes'], current_graph['edges'],
                load_graph(args.output/'fresh/raw'/f'{name}.npz'),
                load_graph(args.output/'fresh/inputs'/f'pre_ilp_{name}.npz'),
                dict(nodes=native['old_nodes'], edges=native['old_edges']), native['e_native'], native['e_hgb'],
                node_features=native['node_features'])
            refresh_kind = 'Same fresh points; current association and teacher context rebuilt'
        else:
            from .native_refresh import refresh
            # Observation inference already queried these exact new points in
            # this fresh output root. Rebuild context after its association step.
            current_native = refresh(row, current_graph, graph, native, args.output/'fresh_native')
            refresh_kind = 'Fresh selected coordinates with recomputed native and graph features'
        save_arrays(args.output/'composition_evidence.npz', **current_native)
        receipt = predict_many(row, current_graph, current_native, {'model': args.model},
            {'model': args.output/'module.npz'}, args.output/'composition_event_cache')['model']
        receipt.update(arm='C10', composition_order=['identity_or_observation', 'event'],
            identity_model_sha256=specifications['identity']['weights_sha256'],
            event_model_sha256=spec['weights_sha256'], native_context=refresh_kind,
            no_new_fitted_blend=True, cached_prediction_inputs=False)
        write_json(args.output/'module.json', receipt)
    result = load_graph(args.output / 'module.npz')
    export_csv(args.output / 'module.csv', name, result['nodes'], result['edges'])
    export_geff(args.output/'fresh/module.geff', result['nodes'], result['edges'])


if __name__ == '__main__':
    main()
