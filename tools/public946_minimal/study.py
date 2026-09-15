"""Resumable local scheduler for the immutable revision-2 arm matrix."""
from __future__ import annotations

import copy
import os
import subprocess
import time
from pathlib import Path

from .common import (ARMS, HANDOVER, REPO, check_resources, code_hashes, digest, now,
                     read_json, sha, write_json)

SCIENTIFIC_FILES = ('common.py', 'source.py', 'neural.py', 'modules.py', 'worker.py', 'isolation.py')
ORDER = ['E01', 'E07', 'E02', 'E03', 'E08', 'E05', 'E06', 'E04']


def verify_selection_lock(args):
    for relative, expected in read_json(args.out/'selection_code_lock.json').items():
        if sha(REPO/relative) != expected:
            raise ValueError('Frozen selection rule drift: '+relative)


def execution_lock(args):
    path = args.out / 'execution_lock.json'
    payload = dict(plan_sha256=sha(args.out / 'plan_freeze.json'),
        image_inventory_sha256=sha(args.out / 'image_inventory.json'),
        artifact_hashes_sha256=sha(args.out / 'artifact_hashes.json'),
        source=read_json(args.out / 'public_source/materialization.json'),
        applicability=read_json(args.out / 'applicability.json'),
        pilots=read_json(args.out / 'pilots.json'),
        limits=read_json(args.out / 'preflight.json')['limits'],
        metric=read_json(args.out / 'metric_identity.json'),
        scientific_code={name: sha(REPO / 'tools/public946_minimal' / name) for name in SCIENTIFIC_FILES},
        recipes=ARMS, selection=sha(HANDOVER / 'VALIDATION.md'),
        sanitation='All arms: round naturally with ties-to-even, then clip spatial coordinates to true bounds; original serialized coordinates retained and freshly scored separately',
        run_envelope='96 total worker GPU device-hours, 22 GiB allocated GPU, 48 GiB new scratch, 10 GiB reserve; Kaggle GPU <=12h; actual hidden-population feasibility unverified until Kaggle run')
    write_json(path, payload, immutable=True)
    return payload


def job_for(args, arm, row, scope='full', modules=None, transform=None, reuse=None):
    execution = read_json(args.out / 'execution_lock.json')
    for relative, expected in execution['source']['source_hashes'].items():
        if sha(args.out/'public_source'/relative) != expected:
            raise ValueError(f'Public materialized source drift: {relative}')
    for name, expected in execution['scientific_code'].items():
        if sha(REPO / 'tools/public946_minimal' / name) != expected:
            raise ValueError(f'Scientific code drift: {name}; retain failure and amend correctness receipt before rerun')
    mods = list(ARMS[arm]['modules'] if modules is None else modules)
    if reuse is not None and 'E01' in mods:
        cached=Path(reuse)/'neural.json'
        if cached.exists() and read_json(cached).get('evidence_format'):
            raise ValueError('E01 requires full native probability evidence; use the preserved B0 cache')
    source_root = args.out / 'public_source/tracking_repo'
    primary = source_root / 'weights/unet_transformer/split_0/edge_predictor_best.pth'
    secondary = args.out / 'public_source/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth'
    deepcenter = args.artifacts / 'biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt'
    artifact_hashes = read_json(args.out/'artifact_hashes.json')
    for slug in ('biohub-tracking-support-pack-50ep-v1','biohub-temporal-unet3d-seed314159-v1'):
        relative='weights/unet_transformer/split_0/config.json'
        if sha(args.artifacts/slug/relative) != artifact_hashes[slug]['files'][relative]:
            raise ValueError('Public checkpoint configuration drift')
    model_hashes = {str(p): sha(p) for p in (primary, secondary, deepcenter)}
    settings = execution['source']['settings']
    neural_modules = sorted(set(mods) & {'E03', 'E04', 'E05', 'E08'})
    neural_fingerprint = digest(dict(source=execution['source']['source_hashes'],
        model_hashes=model_hashes, settings=settings, modules=neural_modules, transform=transform,
        image_sha256=row['image_sha256'], code={k: execution['scientific_code'][k] for k in ('neural.py', 'modules.py', 'worker.py')}))
    directory = args.out / 'predictions' / scope / arm / row['dataset']
    job = dict(arm=arm, modules=mods, row=row, out=str(args.out), directory=str(directory),
        source_root=str(source_root), archive_source=str(args.archive / 'biohub-harmonic-fusion.py'),
        original_source_sha256=execution['source']['original_sha256'],
        primary_weights=str(primary), secondary_weights=str(secondary), deepcenter_weights=str(deepcenter),
        model_hashes=model_hashes, settings=settings, transform=transform,
        limits=execution['limits'], neural_fingerprint=neural_fingerprint,
        reuse_neural=str(reuse) if reuse is not None else None, execution_lock_sha256=sha(args.out / 'execution_lock.json'))
    if scope == 'fresh_finalists':
        # Standalone confirmation must recompute DeepCenter as well as both
        # tracking models. Its input-hash inventory is linked by delivery.py.
        job['out'] = str(args.out / 'fresh_finalist_cache' / arm / row['dataset'])
    slots = getattr(args, '_worker_slots', 1)
    # Resource-only scheduler changes do not invalidate completed scientific work.
    # Reconstruct its actual recorded allocation; every other job field is still
    # recomputed and checked by the complete-job fingerprint comparison below.
    if (directory / 'complete.json').exists() and (directory / 'job.json').exists():
        previous_job = read_json(directory / 'job.json')
        slots = previous_job.get('concurrent_worker_slots', 1)
        if slots not in (1, 2, 3, 4):
            raise ValueError('Invalid recorded resource allocation')
    if slots > 1:
        job['limits'] = dict(execution['limits'])
        job['limits']['gpu_allocated_gib_max'] /= slots
        job['concurrent_worker_slots'] = slots
    job['fingerprint'] = digest(job)
    return job


def used_seconds(out):
    costs = list((out / 'cost_receipts').rglob('*.json'))
    receipts = list((out / 'predictions').rglob('complete.json')) if not costs else []
    failed = list(out.rglob('failure-*.json'))
    smoke = list((out/'unscored_engineering_smoke').rglob('complete.json'))
    packaged = list((out/'package_validation').glob('*/package_test_receipt.json'))
    return (sum(read_json(p)['seconds'] for p in receipts+costs+smoke+packaged)
            + sum(read_json(p).get('elapsed_seconds', 0) for p in failed))


def execute(args, arm, rows, scope='full', modules=None, transform=None, reuse_arm=None):
    if scope in ('full', 'fresh_finalists') and len(rows) > 1 and getattr(args, 'workers', 2) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        concurrency = min(4, args.workers)
        child_args = copy.copy(args)
        child_args._worker_slots = concurrency
        child_args._parallel_child = True
        # Preserve aggregate budget accounting for prior serial pilot receipts.
        for p in (args.out/'predictions').rglob('complete.json'):
            relative = p.parent.relative_to(args.out/'predictions')
            cost_path = args.out/'cost_receipts'/relative/'cost.json'
            if not cost_path.exists():
                value = read_json(p)
                write_json(cost_path,dict(seconds=value['seconds'],gpu_peak_bytes=value['gpu_peak_bytes'],rss_peak_bytes=value['rss_peak_bytes']))
        records, failures = [], []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(execute, child_args, arm, [row], scope, modules, transform, reuse_arm) for row in rows]
            for future in as_completed(futures):
                r, f = future.result()
                records.extend(r); failures.extend(f)
                print(f'{scope}/{arm} total {len(records)}/{len(rows)}; failed={len(failures)}',flush=True)
                write_json(args.out/'progress.json',dict(scope=scope,arm=arm,completed=len(records),failed=len(failures),expected=len(rows),updated=now()))
        write_json(args.out/'execution'/scope/(arm+'.json'),dict(expected=[r['dataset'] for r in rows],
            completed=sorted(r['dataset'] for r in records),failed=failures,
            seconds=sum(r['seconds'] for r in sorted(records,key=lambda r:r['dataset'])),workers=concurrency))
        return records, failures
    records = []
    failures = []
    for i, row in enumerate(rows):
        reuse = args.out / 'predictions' / scope / reuse_arm / row['dataset'] if reuse_arm else None
        if reuse is not None and (reuse/'complete.json').exists():
            reuse = Path(read_json(reuse/'complete.json')['neural_evidence']).parent
        job = job_for(args, arm, row, scope, modules, transform, reuse)
        directory = Path(job['directory'])
        complete = directory / 'complete.json'
        if complete.exists():
            record = read_json(complete)
            if record['fingerprint'] != job['fingerprint'] or record['final_sha256'] != sha(directory / 'final.npz'):
                raise ValueError(f'Completed prediction drift: {complete}')
            from .retention import finish
            finish(job,scope)
            records.append(record)
            continue
        if used_seconds(args.out) / 3600 >= job['limits']['gpu_device_hours_max']:
            raise RuntimeError('Registered total GPU device-hour budget exhausted')
        check_resources(args.out, job['limits'])
        directory.mkdir(parents=True, exist_ok=True)
        job_path = directory / 'job.json'
        write_json(job_path, job, immutable=True)
        env = os.environ.copy()
        env.update(PUBLIC946_INFERENCE='1', PUBLIC946_OUT=str(args.out), PYTHONNOUSERSITE='1',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            PYTHONPATH=str(REPO / 'tools/public946_minimal/guard') + ':' + str(REPO / 'tools'))
        start = time.perf_counter()
        with (directory / 'worker.log').open('a') as log:
            result = subprocess.run([str(args.runtime), '-m', 'public946_minimal', 'worker', '--job', str(job_path)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failure = dict(dataset=row['dataset'], arm=arm, scope=scope, returncode=result.returncode,
                           elapsed_seconds=time.perf_counter()-start, log=str(directory / 'worker.log'),
                           job_sha256=sha(job_path), at=now())
            failure_path = directory / ('failure-' + str(time.time_ns()) + '.json')
            write_json(failure_path, failure)
            failures.append(failure)
            print(f'FAILED {scope}/{arm}/{row["dataset"]}; independent clips continue', flush=True)
        else:
            records.append(read_json(complete))
            record = records[-1]
            from .retention import finish
            finish(job,scope)
            write_json(args.out/'cost_receipts'/scope/arm/row['dataset']/'cost.json',
                       dict(seconds=record['seconds'],gpu_peak_bytes=record['gpu_peak_bytes'],rss_peak_bytes=record['rss_peak_bytes']))
            print(f'{scope}/{arm} {i+1}/{len(rows)} {row["dataset"]} {records[-1]["seconds"]:.1f}s', flush=True)
        if not getattr(args, '_parallel_child', False):
            write_json(args.out / 'progress.json', dict(scope=scope, arm=arm, completed=len(records),
                       failed=len(failures), expected=len(rows), gpu_device_seconds=used_seconds(args.out), updated=now()))
    if not getattr(args, '_parallel_child', False):
        write_json(args.out / 'execution' / scope / (arm + '.json'), dict(expected=[r['dataset'] for r in rows],
            completed=[r['dataset'] for r in records], failed=failures, seconds=sum(r['seconds'] for r in records)))
    return records, failures


def score(args, arm, rows, scope='full', transform=None, baseline=None):
    if arm not in ('B0', 'B1') and not transform:
        from .diagnostics import stage_check
        stage_check(args.out, arm, rows, scope)
    score_root = args.out / 'scores' / scope / arm
    job = dict(arm=arm, rows=rows, prediction_root=str(args.out / 'predictions' / scope / arm),
        score_root=str(score_root), transform=transform,
        baseline_score_root=str(args.out / 'scores' / scope / baseline)
            if baseline and (args.out/'scores'/scope/baseline/'summary.json').exists() else None)
    path = args.out / 'evaluation_jobs' / scope / (arm + '.json')
    write_json(path, job, immutable=True)
    env = os.environ.copy()
    env.pop('PUBLIC946_INFERENCE', None)
    env.update(PYTHONPATH=str(REPO / 'tools'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    subprocess.run([str(args.runtime), '-m', 'public946_minimal', 'evaluate', '--job', str(path)], env=env, check=True)
    return read_json(score_root / 'summary.json')


def pilot(args):
    execution_lock(args)
    inventory = read_json(args.out / 'image_inventory.json')
    names = {p['dataset'] for p in read_json(args.out / 'pilots.json')}
    rows = [r for r in inventory if r['dataset'] in names]
    # Exactly two independent fresh B0 runs for the engineering noise screen.
    for scope in ('aa1', 'aa2'):
        _, failed = execute(args, 'B0', rows, scope)
        if not failed:
            score(args, 'B0', rows, scope)
    if all((args.out / 'scores' / s / 'B0/summary.json').exists() for s in ('aa1', 'aa2')):
        a, b = [read_json(args.out / 'scores' / s / 'B0/summary.json') for s in ('aa1', 'aa2')]
        noise = {}
        for key in ('score', 'division_jaccard', 'adj_edge_jaccard'):
            differences = [abs(a['pooled'][key] - b['pooled'][key])]
            differences += [abs(a['per_embryo'][e][key] - b['per_embryo'][e][key]) for e in a['per_embryo']]
            for row in rows:
                ra, rb = [read_json(args.out / 'scores' / s / 'B0' / (row['dataset'] + '.json')) for s in ('aa1', 'aa2')]
                for value in (ra, rb):
                    denominator = sum(value[k] for k in ('division_tp', 'division_fp', 'division_fn'))
                    value['division_jaccard'] = value['division_tp'] / denominator if denominator else 0.
                    value['score'] = value['adj_edge_jaccard'] + .1 * value['division_jaccard']
                if key in ra and key in rb:
                    differences.append(abs(ra[key] - rb[key]))
            noise[key] = max(differences)
        write_json(args.out / 'numerical_noise.json', noise, immutable=True)
    for arm in ([args.arm] if args.arm else ['B1', *ORDER]):
        if arm in ('E03', 'E06'):
            continue
        # Graph-only arms reuse exact pilot neural evidence; their entire recipes run on full clips.
        _, failed = execute(args, arm, rows, 'aa1', reuse_arm='B0' if arm in ('B1', 'E01', 'E02', 'E07') else None)
        if not failed:
            score(args, arm, rows, 'aa1', baseline='B0')


def controls(args):
    execution_lock(args)
    rows = read_json(args.out / 'image_inventory.json')
    for arm in ('B0', 'B1'):
        _, failed = execute(args, arm, rows, reuse_arm='B0' if arm == 'B1' else None)
        if not failed:
            score(args, arm, rows, baseline='B0' if arm == 'B1' else None)


def singles(args):
    from .selection import eligible
    verify_selection_lock(args)
    rows = read_json(args.out / 'image_inventory.json')
    for arm in ([args.arm] if args.arm else ORDER):
        if arm in ('E03', 'E06'):
            receipts = [read_json(p) for r in rows
                        if (p := args.out/'predictions/full/B0'/r['dataset']/'neural.json').exists()]
            if arm == 'E03':
                count = sum(r['summaries']['feature_samples'] for r in receipts)
                fractional = sum(r['summaries']['fractional_samples'] for r in receipts)
                delta = max((r['summaries']['integer_identity_max_abs'] for r in receipts), default=0.)
                assert fractional == 0 and delta == 0
                proof = dict(sampled_features=count, fractional_coordinates=fractional, max_abs_feature_difference=delta,
                             actual_baseline_clips_inspected=len(receipts), source_proof_scope='all source-generated integer peaks')
            else:
                assert all(r['actual_window'] == 2 for r in receipts)
                proof = dict(window_size=2, transitions=sum(r['image_shape'][0] - 1 for r in rows),
                             contexts_per_transition=1, additional_contexts=0,actual_baseline_clips_inspected=len(receipts))
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='not_applicable', eligible=False,
                       reason='Source and full-cohort tensor identity; no substituted mechanism', proof=proof))
            continue
        _, failed = execute(args, arm, rows, reuse_arm='B0' if arm in ('E01', 'E02', 'E07') else None)
        if failed:
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='engineering_failures', eligible=False, failed=failed))
            continue
        result = score(args, arm, rows, baseline='B0')
        if not (args.out/'scores/full/B0/summary.json').exists() or not (args.out/'numerical_noise.json').exists():
            write_json(args.out/'decisions'/(arm+'.json'),dict(status='measured_with_comparison_blocked',eligible=False,
                reasons=['Full arm independently scored; baseline/noise screen unavailable']))
            continue
        baseline = read_json(args.out / 'scores/full/B0/summary.json')
        noise = read_json(args.out / 'numerical_noise.json')
        gate = eligible(result, baseline, noise)
        if arm == 'E01':
            identical = (args.out/'scores/full/B1/summary.json').exists() and all(read_json(args.out / 'predictions/full/E01' / r['dataset'] / 'complete.json')['graph_hash'] ==
                            read_json(args.out / 'predictions/full/B1' / r['dataset'] / 'complete.json')['graph_hash'] for r in rows)
            if identical:
                gate = dict(eligible=False, reasons=['Exact full-cohort graph identity to known B1; not a novel candidate'],
                            equivalent_to='B1', identical_clips=len(rows))
        write_json(args.out / 'decisions' / (arm + '.json'), dict(status='measured', **gate))


def combinations(args):
    from .selection import eligible
    verify_selection_lock(args)
    rows = read_json(args.out / 'image_inventory.json')
    for arm in ('C01', 'C02', 'C03'):
        constituent = [read_json(args.out / 'decisions' / (m + '.json')) for m in ARMS[arm]['modules']]
        if not all(r['eligible'] for r in constituent):
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='not_eligible', eligible=False,
                reason='At least one constituent did not independently pass the registered full-cohort gate',
                constituent_decisions=dict(zip(ARMS[arm]['modules'],constituent))))
            continue
        _, failed = execute(args, arm, rows)
        if failed:
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='engineering_failures', eligible=False, failed=failed))
            continue
        result = score(args, arm, rows, baseline='B0')
        gate = eligible(result, read_json(args.out / 'scores/full/B0/summary.json'), read_json(args.out / 'numerical_noise.json'))
        write_json(args.out / 'decisions' / (arm + '.json'), dict(status='measured', **gate))


def transfers(args):
    from .selection import eligible, ranking
    verify_selection_lock(args)
    rows = read_json(args.out / 'image_inventory.json')
    baseline = read_json(args.out / 'scores/full/B0/summary.json')
    candidate_ids = [a for a in [*ORDER, 'C01', 'C02', 'C03'] if a != 'E01'
                     and read_json(args.out / 'decisions' / (a + '.json'))['eligible']]
    ranked = sorted(candidate_ids, key=lambda a: ranking(a, read_json(args.out / 'scores/full' / a / 'summary.json'),
                    baseline, read_json(args.out / 'execution/full' / (a + '.json'))['seconds']))
    finalists = {x: ranked[i] if i < len(ranked) else None for i, x in enumerate(('X01', 'X02'))}
    import importlib.util
    spec=importlib.util.spec_from_file_location('public946_registered_contract',HANDOVER/'study_contract.py')
    contract=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    contract.validate_resolution(read_json(HANDOVER/'experiment_lock.json'),finalists)
    write_json(args.out / 'finalist_lock.json', dict(resolution=finalists, recipes={x: ARMS[a]['modules'] if a else None for x, a in finalists.items()},
        eligibility_hashes={a: sha(args.out / 'decisions' / (a + '.json')) for a in ranked},
        prediction_hashes={a: digest([read_json(args.out / 'predictions/full' / a / r['dataset'] / 'complete.json')['final_sha256'] for r in rows]) for a in ranked},
        execution_lock_sha256=sha(args.out / 'execution_lock.json')), immutable=True)
    for arm, original in finalists.items():
        if original is None:
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='no_eligible_finalist', eligible=False))
            continue
        modules = ARMS[original]['modules'] + ['no_motion']
        _, failed = execute(args, arm, rows, modules=modules, reuse_arm=original)
        if failed:
            write_json(args.out / 'decisions' / (arm + '.json'), dict(status='engineering_failures', eligible=False, failed=failed))
            continue
        result = score(args, arm, rows, baseline='B1')
        gate = eligible(result, read_json(args.out / 'scores/full/B1/summary.json'), read_json(args.out / 'numerical_noise.json'))
        write_json(args.out / 'decisions' / (arm + '.json'), dict(status='measured', parent_recipe=original, **gate))


def run(args):
    if args.command in ('pilot', 'controls', 'singles', 'combinations', 'transfers'):
        return globals()[args.command](args)
    if args.command == 'run':
        from .preflight import run as preflight, audit
        if not (args.out / 'preflight.json').exists():
            preflight(args)
        if not (args.out / 'applicability.json').exists():
            audit(args)
        for step in (pilot, controls, singles, combinations, transfers):
            step(args)
        from .delivery import robustness, package, report
        from .diagnostics import descriptive_strata
        for step in (robustness, package, descriptive_strata, report):
            step(args)
        return
    from . import delivery
    return getattr(delivery, args.command)(args)
