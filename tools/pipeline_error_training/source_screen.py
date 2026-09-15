"""Predeclared complete-clip source replays, separate from target comparisons."""
from pathlib import Path
import subprocess
import sys
import time

from .common import DATA, RESULTS, ROOT, WORK, inputs, load_graph, read_json, sha, write_json


def lock():
    split = read_json(RESULTS / 'split_manifest.json')['directions']
    selected = {}
    for source in ['44b6', '6bba']:
        rows = [r for r in inputs() if r['embryo'] == source and split[source][r['dataset']]['partition'] == 'calibration']
        ordered = sorted(rows, key=lambda r: (r['baselines']['P0']['nodes'], r['dataset']))
        chosen = {ordered[len(ordered)//2]['dataset']: 'ordinary: median source calibration P0 count',
                  ordered[-1]['dataset']: 'crowded: maximum source calibration P0 count'}
        def source_events(row):
            return read_json(WORK / 'source' / source / 'receipts' / f'{row["dataset"]}.json')['covered_division_groups']
        event = max(rows, key=lambda r: (source_events(r), r['dataset']))
        chosen[event['dataset']] = 'event support: maximum available source calibration division groups'
        selected[source] = chosen
    result = dict(status='source_only_frozen', selected=selected,
        source_replay_scope='Complete 100-frame ordinary, crowded and event-supported source calibration clips; duplicate selections collapse.',
        qualification=dict(max_source_score_regression=.001, max_extra_division_fp=2,
            division_nominees=['D10_adapted', 'D20_temporal'], identity_nominees=['A10', 'O10_swap'],
            matched_loss='Raw supported held-group loss before calibration; useful distinct full-graph tradeoff may also qualify.',
            both_directions_required=True, maximum_division_nominees=1, maximum_identity_nominees=1),
        split_manifest_sha256=sha(RESULTS / 'split_manifest.json'),
        independently_certified=False, new_target_scores_read=False)
    write_json(RESULTS / 'source_screen_lock.json', result, immutable=True)
    return result


def job(row, source, package, destination, arm=None):
    # Do not provide annotations, estimates, match flags, or GT-derived fields to
    # the prediction process. Model direction is always an explicit argument.
    safe = {k: row[k] for k in ['dataset', 'image_path', 'image_shape', 'physical_scale', 'metadata_sha256']}
    spec = read_json(package / 'frozen_package.json')
    dependencies = [spec['architecture_dependency']] if 'architecture_dependency' in spec else []
    result = dict(row=safe, source=source, package=str(package), package_sha256=sha(package / 'frozen_package.json'), dependencies=dependencies,
        operator='bounded_additive' if spec['recipe']['arm'].startswith('D') else 'continuation',
        graph_path=row['baselines']['P0']['path'], graph_sha256=row['baselines']['P0']['sha256'],
        evidence_path=row['evidence']['path'], evidence_sha256=row['evidence']['sha256'], destination=str(destination))
    if spec['recipe']['arm'] == 'O10_swap':
        from .artifacts import P0_MODELS
        safe.update(raw=row['raw'], evidence=row['evidence'], baselines={'P0': row['baselines']['P0']})
        p0 = read_json(RESULTS / 'model_input_manifest.json')[f'P0_{source}']
        original = ROOT / 'strong-tracker-v2/raw' / f'{row["dataset"]}.npz'
        receipt_path = Path(row['evidence']['path']).with_suffix('.json')
        original_hash = read_json(receipt_path)['inputs']['raw_sha256']
        dependencies.extend([p0, dict(path=str(original), sha256=original_hash),
            dict(path=row['raw']['path'], sha256=row['raw']['sha256']),
            dict(path=str(receipt_path), sha256=row['evidence']['receipt_sha256'])])
        manifest = read_json(ROOT / 'image-native-tracking-v5/inference_package_validation/base/manifest.json')
        dependencies.extend(dict(path=manifest['external_checkpoint_paths'][key], sha256=manifest[key+'_weights_sha256'])
                            for key in ['primary', 'secondary'])
        result.update(operator='observation', arm=arm or 'O10_swap', p0_model=p0['path'])
    if arm and arm.endswith('_replacement'):
        result['operator'] = 'complete_replacement'
    return result


def run(source, arm, seed=20260915):
    from .guard import install
    install(source=source)
    from .evaluate import summary
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    package_arm = 'O10_swap' if arm == 'O10_restore' else arm.removesuffix('_replacement')
    package = WORK / 'training' / package_arm / source / str(seed)
    selected = read_json(RESULTS / 'source_screen_lock.json')['selected'][source]
    rows = [r for r in inputs() if r['dataset'] in selected]
    root = WORK / 'source_screen' / arm / source / str(seed)
    final = root / 'summary.json'
    if final.exists():
        return read_json(final)
    scores, baselines, receipts = [], [], []
    for row in rows:
        name = row['dataset']
        destination = root / 'predictions' / name / f'{name}.npz'
        if arm=='O10_restore':
            from .cache_reuse import observation as reuse
            source_cache = WORK/'source_screen/O10_swap'/source/str(seed)/'predictions'/name/'current_image_cache'
            reuse(source_cache,destination.parent/'current_image_cache',sha(package/'model.pt'),row['metadata_sha256'])
        path = root / 'jobs' / f'{name}.json'
        write_json(path, job(row, source, package, destination, arm=arm), immutable=True)
        log_path = root / f'{name}.log'
        begin = time.monotonic()
        if not destination.with_suffix('.guard.json').exists() or not destination.exists():
            with log_path.open('a') as log:
                child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.prediction_entry', str(path)],
                                       stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                write_json(root / 'failure.json', dict(status='failed', dataset=name, returncode=child.returncode,
                    log_sha256=sha(log_path), independent_lanes_continue=True))
                raise RuntimeError('Source graph replay failed; see preserved prediction log')
        graph = load_graph(destination)
        guard = read_json(destination.with_suffix('.guard.json'))
        if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
            raise RuntimeError('Source prediction violated its label-denial contract')
        gn, ge = read_gt(DATA, name, row['physical_scale'])
        score, _, _ = evaluate_graph(name, graph['nodes'], graph['edges'], gn, ge, row['physical_scale'], row['estimated_total'])
        score.update(arm=arm, embryo=source)
        scores.append(score)
        baseline = read_json(WORK / 'evaluation/P0' / f'{name}.json')
        baseline['arm'] = 'P0'
        baselines.append(baseline)
        receipt = read_json(destination.with_suffix('.json'))
        receipts.append(dict(dataset=name, prediction_sha256=sha(destination), seconds=time.monotonic()-begin,
            model_inference_seconds=receipt['seconds'], ledger=receipt['ledger'], guard=guard))
        print(f'Complete source replay {arm}/{source}: {name}', flush=True)
    measured, baseline = summary(scores, arm), summary(baselines, 'P0')
    result = dict(status='measured', source=source, arm=arm, seed=seed,
        measured=measured, baseline=baseline, delta=measured['score']-baseline['score'],
        predictions=receipts, per_clip=scores, source_only=True, target_model_scores_read=False,
        model_sha256=sha(package / 'model.pt'), package_sha256=sha(package / 'frozen_package.json'),
        source_screen_lock_sha256=sha(RESULTS / 'source_screen_lock.json'))
    write_json(final, result, immutable=True)
    return result
