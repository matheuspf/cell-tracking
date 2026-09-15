"""Serial, resumable execution of independent primary GPU lanes."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import WORK, now, read_json, sha, write_json
from .resources import Monitor

PRIMARY = ['D10_frozen', 'D10_adapted', 'D20_compact', 'D20_temporal', 'D20_no_pretrain', 'A10']


def run():
    root = WORK / 'queue'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        with Monitor(root / 'resources.json') as monitor:
            for arm in PRIMARY:
                for source in ['44b6', '6bba']:
                    for stage, receipt_name in [('train', 'package.json'), ('calibrate', 'frozen_package.json')]:
                        model = WORK / 'training' / arm / source / '20260915'
                        receipt = model / receipt_name
                        key = f'{stage}-{arm}-{source}'
                        complete = receipt.exists() and (stage == 'train' or (WORK/'source_screen'/arm/source/'20260915/summary.json').exists())
                        if complete:
                            spec = read_json(receipt)
                            if sha(model / 'model.pt') != spec['weights_sha256']:
                                raise ValueError('Completed queue artifact hash changed')
                            jobs.append(dict(job=key, status='measured', resumed_verified=True))
                            continue
                        if stage == 'calibrate' and not (model / 'package.json').exists():
                            jobs.append(dict(job=key, status='not run', reason='Directional training failed; independent lanes continue'))
                            continue
                        command = [sys.executable, '-m', 'pipeline_error_training', stage, '--source', source, '--arm', arm]
                        log_path = root / f'{key}.log'
                        begin = time.monotonic()
                        write_json(root / 'progress.json', dict(status='running', current_job=key, started=now(), jobs=jobs))
                        with log_path.open('a') as log:
                            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                            write_json(root / 'active.json', dict(pid=child.pid, job=key, command=command, started=now()))
                            code = child.wait()
                        text = log_path.read_text(errors='replace')[-4000:]
                        status = 'measured' if code == 0 and receipt.exists() else (
                            'blocked' if any(word in text for word in ['BlockingIOError', 'Resource envelope', 'Other GPU use', 'free-space floor']) else 'failed')
                        jobs.append(dict(job=key, status=status, returncode=code, seconds=time.monotonic()-begin,
                                         log_sha256=sha(log_path), failure_tail=text[-1500:] if status != 'measured' else None))
                        write_json(root / 'progress.json', dict(status='running', current_job=None, jobs=jobs))
                        print(f'Queue {key}: {status}', flush=True)
                        # A failed lane never stops an independent primary lane.
                        monitor.check()
        write_json(root / 'progress.json', dict(status='complete', jobs=jobs, completed=now()))
