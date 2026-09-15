"""Finite source-selected conditional control and seed replication; no target scorer."""
from pathlib import Path
import subprocess
import sys
import time

from .common import RESULTS, WORK, now, read_json, sha, write_json
from .selection import directional, nominate


def execute(arm, source, seed, label):
    folder = WORK/'training'/arm/source/str(seed)
    root = WORK/'replication'; root.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, artifact in [('train', 'package.json'), ('calibrate', 'frozen_package.json')]:
        complete = (folder/artifact).exists()
        if stage == 'calibrate':
            complete = complete and (WORK/'source_screen'/arm/source/str(seed)/'summary.json').exists()
        if complete:
            spec = read_json(folder/artifact)
            if sha(folder/'model.pt') != spec['weights_sha256']:
                raise ValueError('Replicated model hash changed')
            records.append(dict(stage=stage, status='measured', resumed_verified=True))
            continue
        if stage == 'calibrate' and not (folder/'package.json').exists():
            records.append(dict(stage=stage, status='not run', reason='Directional fit failed'))
            continue
        key = f'{label}-{stage}-{arm}-{source}-{seed}'
        command = [sys.executable, '-m', 'pipeline_error_training', stage, '--source', source, '--arm', arm, '--seed', str(seed)]
        begin = time.monotonic()
        with (root/f'{key}.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            write_json(root/'active.json', dict(job=key, pid=child.pid, command=command, started=now()))
            code = child.wait()
        records.append(dict(stage=stage, status='measured' if code==0 else 'failed', returncode=code,
            seconds=time.monotonic()-begin, log_sha256=sha(root/f'{key}.log')))
    result = dict(arm=arm, source=source, seed=seed, label=label, jobs=records)
    write_json(root/f'{label}-{arm}-{source}-{seed}.json', result)
    return result


def run():
    for name in ['queue', 'observation_queue']:
        if read_json(WORK/name/'progress.json')['status'] != 'complete':
            raise RuntimeError('Independent primary lanes must finish before source-only selection')
    if not (RESULTS/'observation_cache_parity.json').exists():
        subprocess.run([sys.executable,'-m','pipeline_error_training.observation_cache_parity'],check=True)
    if not (RESULTS/'continuation_cache_parity.json').exists():
        subprocess.run([sys.executable,'-m','pipeline_error_training.continuation_cache_parity'],check=True)
    # Compute the independent source identity diagnostics in separate guarded
    # processes so one source's annotation capability never leaks into the other.
    for source in ['44b6', '6bba']:
        package = WORK/'training/A10'/source/'20260915'
        if (package/'frozen_package.json').exists() and not (package/'identity_diagnostic.json').exists():
            subprocess.run([sys.executable, '-m', 'pipeline_error_training.selection', '--identity-source', source], check=True)
    records = []
    qualified_organoid = all(directional('D10_adapted', s)['qualified'] for s in ['44b6','6bba'])
    if qualified_organoid:
        for source in ['44b6','6bba']:
            records.append(execute('D10_random', source, 20260915, 'conditional_control'))
    # A fixed decoder ablation uses the same heads; it never changes the default
    # nominee or authorizes a new margin or model selection loop.
    for arm in ['D10_adapted', 'D20_temporal']:
        for source in ['44b6', '6bba']:
            if (WORK/'training'/arm/source/'20260915/frozen_package.json').exists():
                root = WORK/'replication'; root.mkdir(parents=True, exist_ok=True)
                with (root/f'source-{arm}-replacement-{source}.log').open('a') as log:
                    child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.source_replacement', '--arm', arm, '--source', source],
                                           stdout=log, stderr=subprocess.STDOUT)
                records.append(dict(label='fixed_decoder_ablation', arm=arm+'_replacement', source=source, seed=20260915,
                    status='measured' if child.returncode==0 else 'failed', returncode=child.returncode))
    nomination = nominate()
    for arm in [nomination['division_nominee'], nomination['identity_nominee']]:
        if arm is not None:
            for source in ['44b6', '6bba']:
                records.append(execute(arm, source, 314159, 'replication'))
    replicated = [r for r in records if r['label']=='replication']
    failed = any(j['status'] in ['failed', 'blocked', 'not run'] for r in replicated for j in r['jobs'])
    result = dict(status=('failed' if failed else 'measured') if replicated else 'not run',
        reason=('One or more nominated directional replication jobs failed; independent jobs continued' if failed else None)
            if replicated else 'No family qualified in both source directions under the frozen screen',
        nomination_sha256=sha(RESULTS/'nomination.json'), conditional_random_control=qualified_organoid,
        jobs=records, primary_seed_retained=20260915, replication_seed=314159, target_scores_used=False,
        model_ensembling=False, matched_controls_replicated=False,
        learning_mechanism_claim='A second-seed advantage over a newly trained matched control is not established by nominee-only replication.')
    write_json(RESULTS/'replication.json', result)
    return result


if __name__ == '__main__':
    from .resources import cpu_budget
    cpu_budget()
    run()
