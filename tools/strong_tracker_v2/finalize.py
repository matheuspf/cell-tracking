"""Verify completed measurements and seal local artifacts and aggregate evidence."""
from __future__ import annotations

import ast
import importlib.metadata
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch

from .common import (DATA, METRIC_REV, OUT, REPO, V1, WORK, graph_hash, inventory,
    load_graph, now, preservation_check, read_json, sha, write_json)


def check_measurements():
    names = {r['dataset'] for r in inventory()}
    assert len(names) == 199
    complete = read_json(OUT / 'score_completeness.json')
    assert complete['score_rows'] == 20696 and len(complete['complete_variants']) == 104
    frame = pd.read_csv(OUT / 'score_rows.csv')
    assert len(frame) == 20696 and not frame.duplicated(['variant', 'dataset']).any()
    assert set(frame.metric_revision) == {METRIC_REV}
    for _, group in frame.groupby('variant'):
        assert set(group.dataset) == names
    stages = pd.read_csv(OUT / 'stage_scores.csv')
    for _, group in stages.groupby('variant'):
        assert set(group.dataset) == names
    assert len(stages) == 1990
    assert sum(pd.read_csv(OUT / 'division_failure_census.csv').kind == 'gt_division') == 151
    assert len(pd.read_csv(OUT / 'edge_failure_census.csv')) == 6682 + 6885
    baseline_hashes = {n: sha(V1 / 'baseline/public' / f'{n}.npz') for n in names}
    gt_hashes = {n: sha(V1 / 'evaluation/gt' / f'{n}.npz') for n in names}
    prediction_hashes = {}
    annotation_receipts = {}
    for receipt, directory, key in [
        ('inference_receipt.json', 'predictions', 'prediction_sha256'),
        ('risk_inference_receipt.json', 'risk_predictions', 'hashes'),
        ('image_inference_receipt.json', 'image_predictions', 'hashes'),
        ('combo_inference_receipt.json', 'combo_predictions', 'hashes'),
    ]:
        obj = read_json(OUT / receipt)
        assert obj['samples'] == 199 and obj['annotation_reads'] == 0
        assert set(obj[key]) == names
        for name, expected in obj[key].items():
            assert sha(OUT / directory / f'{name}.npz') == expected
            metadata = read_json(OUT / directory / f'{name}.json')
            assert metadata['annotation_access_blocked']
            assert metadata['source'] != name.split('_')[0]
        prediction_hashes[directory] = obj[key]
        annotation_receipts[receipt] = {'samples': 199, 'hashes_verified': True, 'annotation_reads': 0}
    main = read_json(OUT / 'config_locks.json')['variants']
    graph_rows = 0
    for variant in complete['complete_variants']:
        paths = list((OUT / 'evaluation/scores' / variant).glob('*.json'))
        assert {p.stem for p in paths} == names
        for path in paths:
            obj = read_json(path)
            name = path.stem
            if 'inputs' in obj:
                inputs = obj['inputs']
                if variant in main:
                    directory = 'predictions'
                    assert inputs['baseline_sha256'] == baseline_hashes[name]
                    assert inputs['metric_revision'] == METRIC_REV
                elif variant.startswith('EF_'):
                    directory = 'combo_predictions'
                elif 'temporal' in variant:
                    directory = 'image_predictions'
                else:
                    assert 'risk' in variant
                    directory = 'risk_predictions'
                assert inputs['predictions_sha256'] == prediction_hashes[directory][name]
                assert inputs['gt_sha256'] == gt_hashes[name]
            else:
                assert obj['annotation_access_blocked_at_inference']
                graph = load_graph(OUT / 'candidate_graphs' / variant / f'{name}.npz')
                assert graph_hash(graph['nodes'], graph['edges']) == obj['result']['graph_hash']
                graph_rows += 1
    selected = read_json(OUT / 'selected_prediction_lock.json')
    assert set(selected['hashes']) == names and len(selected['graphs']) == 199
    assert selected['strict_coordinates'] and selected['byte_identical_to_scored_graphs']
    for name, expected in selected['hashes'].items():
        assert sha(OUT / 'selected_predictions' / f'{name}.npz') == expected
        assert sha(OUT / 'candidate_graphs/bypass_motion_bounds' / f'{name}.npz') == expected
        assert read_json(OUT / 'replay/no_motion' / name / 'complete.json')['annotation_access_blocked']
    assert all(r['valid'] and r['out_of_bounds'] == 0 for r in selected['graphs'])
    return dict(samples=199, complete_variants=104, score_rows=20696, stage_rows=1990,
        oracle_variants=4, oracle_samples=read_json(OUT / 'oracle_diagnostics.json')['samples'],
        source_label_directions_checked=True, input_hashes_verified=True,
        independently_checked_graph_hash_rows=graph_rows, selected_graphs_verified=199,
        annotation_inference_receipts=annotation_receipts)


def environment():
    official = REPO / 'work/annotation-selection-v1/official'
    rev = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    assert rev == METRIC_REV
    packages = {}
    for package in ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'polars',
                    'zarr', 'tracksdata', 'highspy', 'joblib', 'pytest']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    props = torch.cuda.get_device_properties(0)
    return dict(created=now(), python=sys.version, executable=sys.executable, packages=packages,
        gpu=props.name, gpu_bytes=props.total_memory, cuda=torch.version.cuda,
        metric_revision=rev, metric_sources={str(p.relative_to(official)): sha(p)
            for p in sorted((official / 'src').rglob('*.py'))},
        study_metadata_exception='Pinned evaluator validated on existing torch 2.8.0+cu128; upstream package metadata requests a later torch. Shared stack unchanged.',
        root_remote_environment_sha256=sha(REPO / 'scripts/root_remote_env.sh'),
        caps=dict(gpu_gib=20, aggregate_ram_gib=128, new_v2_disk_gib=60,
            minimum_free_disk_gib=20, nonfocus_gpu_hours=16, cpu_quota_cores=40.96))


def snapshot_sources():
    paths = list((REPO / 'tools/strong_tracker_v2').glob('*.py'))
    paths += list((REPO / 'tests/strong_tracker_v2').glob('*.py'))
    paths += [REPO / 'tools/strong_tracker_v2/dashboard_template.html',
              REPO / 'scripts/run_strong_tracker_v2.sh', REPO / 'docs/strong-tracker-v2.md']
    hashes = {}
    for path in sorted(paths):
        if path.suffix == '.py':
            ast.parse(path.read_text())
        relative = path.relative_to(REPO)
        dest = OUT / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[str(relative)] = sha(path)
    write_json(OUT / 'source_snapshot.json', dict(created=now(), files=hashes,
        reused_pure_v1_sources={str(p.relative_to(REPO)): sha(p)
            for p in sorted((REPO / 'tools/annotation_selection').glob('*.py'))}))
    doc = (REPO / 'docs/strong-tracker-v2.md').read_text()
    commands = re.search(r'```bash\n(cd /root/code/kaggle/cell-tracking.*?)\n```', doc, re.S)[1]
    (OUT / 'reproduce.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + commands + '\n')
    (OUT / 'reproduce.sh').chmod(0o755)
    return len(hashes)


def portable_bundle():
    dest = REPO / 'results/strong-tracker-v2'
    dest.mkdir(parents=True, exist_ok=True)
    names = ['v2_report.md', 'dashboard.html', 'summary.json', 'operating_points.csv',
        'native_classifier_metrics.csv', 'learning_curves.csv', 'stage_summary.json',
        'oracle_diagnostics.json', 'candidate_coverage.json', 'association_attribution.json',
        'filter_quality_summary.json', 'config_locks.json', 'fit_config.json', 'risk_config.json',
        'temporal_config.json', 'combination_config_lock.json', 'winning_config.json',
        'validation_receipt.json', 'environment.json', 'runtime_summary.json', 'status.json']
    for name in names:
        shutil.copyfile(OUT / name, dest / name)
    (dest / 'README.md').write_text('''# Strong tracker v2: completed measured study

V200–V270 are complete. Decision: **significant_local_gain**. Bypassing the original motion relinker, with strict image bounds, raises pooled local score from **0.911774 to 0.934206 (+0.022432)**. Both embryo directions improve: **44b6 +0.019091; 6bba +0.022928**.

This is an exploratory result on reused embryos with contaminated public upstream checkpoints. The winner was selected after inspection; it is not a hidden-test forecast or an increment to the historical 0.946 leaderboard score.

- [Measured report](v2_report.md)
- [Self-contained offline dashboard](dashboard.html) — download/open in a browser; all 104 settings, both embryos, failure census and training curves
- [All aggregate operating points](operating_points.csv)
- [Validation receipt](validation_receipt.json)
- [Implementation and reproduction](../../docs/strong-tracker-v2.md)

The full local artifact root is `/kaggle/working/cell-tracking/strong-tracker-v2/`. It includes all 20,696 per-sample score rows, detailed failure censuses, 199 selected prediction graphs, frozen models, resumable checkpoints, logs, source snapshots and the complete artifact manifest. `seconds` in operating points is summed evaluator time; cached repair timings are separately labelled in `runtime_summary.json`. Neural inference was reused, so these are not fresh full-notebook or hidden-set runtime measurements.

Only code/configuration and aggregate evidence are committed here. Detailed GT identifiers, coordinates, microscopy, model weights and predictions remain local. The v1 store and unrelated work are preserved. Copy the full v2 output root, ignored `work/strong-tracker-v2/`, and this Git commit out before destroying the Vast instance; reproduction also requires the existing v1 artifacts, official evaluator and competition inputs.
''')
    write_json(dest / 'bundle_manifest.json', dict(created=now(), scope='sanitized aggregate evidence',
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(dest.iterdir()) if p.is_file() and p.name != 'bundle_manifest.json'}))
    return str(dest)


def seal():
    paths = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p != OUT / 'artifact_manifest.json']

    def entry(path):
        return str(path.relative_to(OUT)), dict(bytes=path.stat().st_size, sha256=sha(path))

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = dict(pool.map(entry, paths))
    total = sum(v['bytes'] for v in entries.values())
    assert total < 60 * 1024**3 and shutil.disk_usage(OUT).free > 20 * 1024**3
    write_json(OUT / 'artifact_manifest.json', dict(created=now(), study_id='strong-tracker-v2',
        manifest_excludes_itself=True, files=entries, file_count=len(entries), total_bytes=total,
        free_disk_bytes=shutil.disk_usage(OUT).free))
    print(f'Sealed {len(entries):,} files, {total / 1024**3:.3f} GiB', flush=True)


def run(args):
    if args.variant == 'seal':
        seal()
        return
    assert (OUT / 'resource_monitor.stop').exists(), 'Stop the v2 resource monitor before finalization'
    before_size = (OUT / 'resource_samples.csv').stat().st_size
    measurements = check_measurements()
    models = 0
    for file in ['model_lock.json', 'temporal_model_lock.json']:
        for path, expected in read_json(OUT / file)['models'].items():
            assert sha(OUT / path) == expected
            models += 1
    for source, expected in read_json(OUT / 'risk_model_lock.json')['models'].items():
        assert sha(OUT / 'risk_models' / f'{source}.joblib') == expected
        models += 1
    temporal = read_json(OUT / 'temporal_model_lock.json')
    assert len(temporal['full_fits']) == 6
    assert all(f['steps'] == 5000 and f['all_selected_positives_used'] for f in temporal['full_fits'])
    assert sum(f['selected_positives'] for f in temporal['full_fits'] if f['fraction'] == 1.) == 130959
    browser = read_json(OUT / 'dashboard_validation.json')
    assert browser['passed'] and browser['dashboard_sha256'] == sha(OUT / 'dashboard.html')
    tests = (OUT / 'logs/final-tests.log').read_text()
    assert re.search(r'129 passed, 8 warnings', tests)
    preservation = preservation_check()
    env = environment()
    write_json(OUT / 'environment.json', env)
    resource = pd.read_csv(OUT / 'resource_samples.csv')
    assert resource.rss_bytes.max() < 128 * 1024**3
    assert resource.gpu_used_mib.max() < 20 * 1024
    assert resource.free_disk_bytes.min() > 20 * 1024**3
    assert temporal['gpu_synchronized_hours'] < 16
    assert not (OUT / 'resource_limit_alert.json').exists()
    runtime = {}
    for mode in ['identity', 'no_motion']:
        records = [read_json(p) for p in (OUT / 'replay' / mode).glob('*/complete.json')]
        assert len(records) == 199
        runtime[mode] = {}
        for embryo in ['44b6', '6bba', 'pooled']:
            group = [r for r in records if embryo == 'pooled' or r['dataset'].startswith(embryo)]
            timed = [r for r in group if r['seconds'] is not None]
            runtime[mode][embryo] = dict(summed_worker_seconds=sum(r['seconds'] for r in timed),
                clips_timed=len(timed), clips_total=len(group), clips_missing_timing=len(group)-len(timed))
    write_json(OUT / 'runtime_summary.json', dict(cached_repair_timings=runtime,
        interpretation='Sum of per-clip worker wall time, excluding cached neural inference; parallel execution means this is not elapsed study duration.',
        official_evaluation_seconds='Summed per-variant per-sample elapsed scorer time in operating_points.csv',
        temporal_gpu_synchronized_training_hours=temporal['gpu_synchronized_hours'],
        peak_aggregate_rss_gib=float(resource.rss_bytes.max() / 1024**3),
        peak_gpu_used_gib=float(resource.gpu_used_mib.max() / 1024),
        minimum_free_disk_gib=float(resource.free_disk_bytes.min() / 1024**3)))
    sources = snapshot_sources()
    receipt = dict(created=now(), passed=True, measurements=measurements,
        v1_preservation=preservation, frozen_models_sha256_verified=models,
        current_source_files_snapshotted=sources, metric_revision=METRIC_REV,
        tests=dict(passed=129, warnings=8, log_sha256=sha(OUT / 'logs/final-tests.log'),
            scope='Project, study contract and pinned official metric/division tests'),
        dashboard=dict(passed=True, population_family_states=len(browser['states']),
            desktop_mobile_checked=True, no_network_requests=True,
            dashboard_sha256=browser['dashboard_sha256']),
        selected_strict_annotation_unavailable_graphs=199,
        numerical_replay_qualification='Two documented half-integer ULP export ties restored to sealed baseline; native rounding separately rescored with unchanged score.',
        scientific_qualification='Exploratory reused embryos and contaminated public upstream checkpoints; no clean OOF or hidden-test claim.',
        resource_caps_satisfied=True, package_stack_unchanged=True)
    write_json(OUT / 'validation_receipt.json', receipt)
    summary = read_json(OUT / 'summary.json')
    status = dict(study_id='strong-tracker-v2', updated=now(), status='complete',
        decision=summary['decision'], selected_variant=summary['selected_variant'],
        provenance=summary['provenance'], validation=summary['validation'],
        stages={f'V{number}': dict(status='complete', updated=now()) for number in range(200, 271, 10)},
        samples=199, complete_variants=104, score_rows=20696,
        optional_focus='Not run; conditional arm not selected',
        historical_failures_preserved=True, v1_preserved=True)
    write_json(OUT / 'status.json', status)
    portable = portable_bundle()
    assert (OUT / 'resource_samples.csv').stat().st_size == before_size
    seal()
    print('V200–V270 complete:', summary['decision'], portable, flush=True)
