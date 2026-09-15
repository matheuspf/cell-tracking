"""Seal all source-selected directions and seeds before any new target comparison."""
from pathlib import Path

from .common import RESULTS, WORK, inputs, now, read_json, sha, write_json

PRIMARY = ['D10_frozen', 'D10_adapted', 'D20_compact', 'D20_temporal', 'D20_no_pretrain', 'A10', 'O10_swap']


def run():
    nomination = read_json(RESULTS/'nomination.json')
    replication = read_json(RESULTS/'replication.json')
    fits = [(arm, 20260915, arm) for arm in PRIMARY]
    if replication['conditional_random_control']:
        fits.append(('D10_random', 20260915, 'D10_random'))
    for arm in [nomination['division_nominee'], nomination['identity_nominee']]:
        if arm:
            fits.append((arm, 314159, arm+'_replication'))
    packages, experiments, excluded = [], ['D00'], []
    for arm, seed, experiment in fits:
        directional = []
        for source in ['44b6', '6bba']:
            folder = WORK/'training'/arm/source/str(seed)
            manifest = folder/'frozen_package.json'
            if not manifest.exists():
                excluded.append(dict(arm=arm, seed=seed, source=source, status='failed', reason='Frozen directional model is unavailable'))
                continue
            spec = read_json(manifest)
            if spec['recipe']['source'] != source or spec['recipe']['seed'] != seed or sha(folder/'model.pt') != spec['weights_sha256']:
                raise ValueError('Directional checkpoint or source recipe mismatch')
            directional.append(dict(experiment=experiment, arm=arm, seed=seed, source=source,
                manifest_path=str(manifest), manifest_sha256=sha(manifest), weights_path=str(folder/'model.pt'),
                weights_sha256=spec['weights_sha256'], recipe_sha256=sha(folder/'recipe.json'),
                calibration_sha256=sha(folder/'calibration.json'), completed_updates=spec['completed_updates']))
        # Partial fits remain reported; do not manufacture an all-199 arm by
        # substituting P0 or another seed for a missing direction.
        if len(directional) == 2:
            packages.extend(directional); experiments.append(experiment)
        else:
            excluded.extend(dict(arm=arm, seed=seed, source=p['source'], status='not run', reason='Opposite directional model failed; no complete target arm') for p in directional)
    if 'O10_swap' in experiments:
        experiments.append('O10_restore')
    replacements = [a for a in ['D10_adapted', 'D20_temporal'] if a in experiments]
    experiments.extend(a+'_replacement' for a in replacements)
    composition = bool(nomination['division_nominee'] and nomination['identity_nominee'])
    if composition:
        experiments.extend(['C10', 'C10_replication'])
    result = dict(status='frozen_before_new_target_scores', created=now(), packages=packages, experiments=experiments,
        excluded=excluded, replacement_arms=replacements, replacement_nomination=False,
        composition=dict(enabled=composition, order=['identity_or_observation', 'event'],
            division=nomination['division_nominee'], identity=nomination['identity_nominee'],
            native_refresh_after_changed_observations=True),
        nomination_sha256=sha(RESULTS/'nomination.json'), replication_sha256=sha(RESULTS/'replication.json'),
        input_manifest_sha256=sha(RESULTS/'input_manifest.json'),
        execution_lock_sha256=sha(RESULTS/'execution_repair_lock.json'),
        diagnostic_lock_sha256=sha(RESULTS/'diagnostic_lock.json'),
        files={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        expected_target_clips=[r['dataset'] for r in inputs()],
        direct_new_head_target_fitting=False, inherited_upstream_exposure=True,
        clean_end_to_end_out_of_fold=False, production_default_unchanged=True)
    write_json(RESULTS/'target_freeze.json', result, immutable=True)
    return result


if __name__ == '__main__':
    print(run())
