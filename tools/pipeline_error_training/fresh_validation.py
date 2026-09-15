"""Two renamed full-image reconstructions; no new target metric is read here."""
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import RESULTS, ROOT, WORK, graph_hash, inputs, load_graph, now, read_json, sha, verified_graph, write_json
from .resources import Lease, Monitor, cpu_budget

BASE = ROOT/'image-native-tracking-v5/inference_package_validation/base'


def verify_base():
    manifest = read_json(BASE/'manifest.json')
    for collection in ['code', 'package_files']:
        for relative, expected in manifest[collection].items():
            if sha(BASE/relative) != expected:
                raise ValueError('Exact full baseline package changed')
    for name, path in manifest['external_checkpoint_paths'].items():
        if sha(path) != manifest[name+'_weights_sha256']:
            raise ValueError('Full baseline checkpoint changed')
    for source, path in manifest['external_teacher_paths'].items():
        if sha(path) != manifest['legacy_teacher_model_hashes'][source]:
            raise ValueError('Full baseline E teacher changed')
    for relative, expected in manifest['tracking_source_files'].items():
        if sha(Path(manifest['upstream_tracking_source_root'])/relative) != expected:
            raise ValueError('Pinned upstream neural code changed')
    write_json(RESULTS/'fresh_input_manifest.json', dict(base_manifest_sha256=sha(BASE/'manifest.json'),
        source_code=manifest['code'], inherited_checkpoint_hashes={k: manifest[k] for k in manifest if k.endswith('_weights_sha256')},
        teacher_hashes=manifest['legacy_teacher_model_hashes'], all_internal_and_external_hashes_verified=True), immutable=True)


def run(arm='D10_frozen', *, candidate=False):
    verify_base()
    composition = read_json(RESULTS/'target_freeze.json')['composition'] if arm=='C10' else None
    if composition and not composition['enabled']:
        raise ValueError('Fresh composition was not authorized by source-only nomination')
    # Selected source-supported representatives are already frozen; no new target
    # outcome is used to choose an easier image or a different model.
    choices = [('44b6_d754aa59', 'specimen_alder'), ('6bba_bb9f20c3', 'specimen_birch')]
    results = []
    for dataset, unfamiliar in choices:
        row = next(r for r in inputs() if r['dataset'] == dataset)
        source = '6bba' if row['embryo'] == '44b6' else '44b6'
        model = WORK/'training'/(composition['division'] if composition else arm)/source/'20260915'
        identity_model = WORK/'training'/composition['identity']/source/'20260915' if composition else None
        if not (model/'frozen_package.json').exists():
            raise FileNotFoundError('Both directional frozen model packages are required')
        spec = read_json(model/'frozen_package.json')
        identity_spec = read_json(identity_model/'frozen_package.json') if identity_model else None
        study_root = WORK/'fresh_candidates'/arm if candidate else WORK/'fresh_validation'
        images = study_root/unfamiliar/'images'
        output = study_root/unfamiliar/'output'
        images.mkdir(parents=True, exist_ok=True)
        renamed = images/f'{unfamiliar}.zarr'
        if not renamed.exists():
            renamed.symlink_to(Path(row['image_path']).resolve(), target_is_directory=True)
        final = output/'validation.json'
        if final.exists():
            result = read_json(final)
            if sha(output/'module.npz') != result['module_sha256']:
                raise ValueError('Completed fresh output changed')
            results.append(result)
            continue
        p0 = read_json(RESULTS/'model_input_manifest.json')[f'P0_{source}']
        command = [sys.executable, '-m', 'pipeline_error_training.fresh_entry', '--images', str(images),
            '--output', str(output), '--package', str(BASE), '--source', source, '--p0-model', p0['path'],
            '--model', str(model), '--v1', str(ROOT/'annotation-selection-v1'), '--v2', str(ROOT/'strong-tracker-v2')]
        if identity_model:
            command += ['--identity-model', str(identity_model)]
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stages = []
        for stage, artifact in [('baseline', 'inference_receipt.json'), ('point-head', 'P0_receipt.json'), ('module', 'module.json')]:
            if (output/artifact).exists():
                stages.append(dict(stage=stage, status='measured', resumed=True))
                continue
            begin = time.monotonic()
            with Monitor(output/f'{stage}.resources.json'), Lease(required_gib=12. if stage == 'baseline' else 8.):
                with (output/f'{stage}.log').open('a') as log:
                    child = subprocess.run([*command, '--stage', stage], stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                failure = dict(status='failed', dataset=dataset, unfamiliar_name=unfamiliar, stage=stage,
                    returncode=child.returncode, log_sha256=sha(output/f'{stage}.log'), new_target_metrics_read=False)
                write_json(output/'failure.json', failure)
                results.append(failure)
                break
            stages.append(dict(stage=stage, status='measured', seconds=time.monotonic()-begin))
        else:
            audits = [read_json(p) for p in (output/'startup_audits').glob('*.json')]
            if any(r['blocked_reads'] or r['blocked_network'] or not r['installed_before_numerical'] for r in audits):
                write_json(RESULTS/(f'fresh_candidate_{arm}.json' if candidate else 'fresh_image_validation.json'),
                    dict(status='failed',stage='access_guard',dataset=dataset,guard_audits=audits,
                         new_target_scores_read=False))
                raise RuntimeError('Fresh startup guard contract failed')
            p0_graph, expected = load_graph(output/'P0.npz'), verified_graph(row)
            for key in ['nodes', 'edges']:
                np.testing.assert_array_equal(p0_graph[key], expected[key])
            module = load_graph(output/'module.npz')
            result = dict(status='measured', dataset=dataset, unfamiliar_name=unfamiliar, frames=row['image_shape'][0],
                explicit_source=source, baseline_P0_exact_nodes_and_edges=True, module_sha256=sha(output/'module.npz'),
                module_graph_hash=graph_hash(module['nodes'], module['edges']), source_checkpoint_sha256=spec['weights_sha256'],
                stages=stages, seconds=time.monotonic()-started, startup_guards=audits,
                exact_CSV_and_GEFF_roundtrip=True, complete_primary_secondary_harmonic_DeepCenter_P0=True,
                cached_prediction_inputs=False, new_target_metrics_read=False)
            if identity_spec:
                result.update(identity_checkpoint_sha256=identity_spec['weights_sha256'],
                    composition_order=['identity_or_observation', 'event'], no_new_fitted_blend=True)
            write_json(final, result, immutable=True)
            results.append(result)
    result = dict(status='measured' if len(results)==2 and all(r['status']=='measured' for r in results) else 'failed',
        created=now(), clips=results, new_target_scores_read=False, Kaggle_12_hour_runtime_proven=False)
    result.update(arm=arm, candidate_proof=candidate)
    write_json(RESULTS/(f'fresh_candidate_{arm}.json' if candidate else 'fresh_image_validation.json'), result)
    return result


if __name__ == '__main__':
    cpu_budget()
    raise SystemExit(0 if run()['status']=='measured' else 1)
