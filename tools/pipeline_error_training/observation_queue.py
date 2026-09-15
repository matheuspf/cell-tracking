"""Run both independent observation fits after the current primary GPU queue."""
import fcntl
import subprocess
import sys
import time

from .common import WORK, now, read_json, sha, write_json
from .resources import Monitor, cpu_budget


def run():
    root = WORK/'observation_queue'
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'progress.json', dict(status='waiting', reason='Existing study primary queue owns its runner lock', created=now()))
    with (WORK/'queue/runner.lock').open('a+') as primary:
        # This waits only in the study runner; the assistant remains available.
        fcntl.flock(primary, fcntl.LOCK_EX)
        jobs = []
        with Monitor(root/'resources.json') as monitor:
            for source in ['44b6', '6bba']:
                folder = WORK/'training/O10_swap'/source/'20260915'
                for stage in ['train', 'calibrate']:
                    key = f'{stage}-O10_swap-{source}'
                    package = folder/('package.json' if stage == 'train' else 'frozen_package.json')
                    complete = package.exists()
                    if stage == 'calibrate':
                        complete = complete and all((WORK/'source_screen'/a/source/'20260915/summary.json').exists()
                            for a in ['O10_swap', 'O10_restore'])
                    if complete:
                        if sha(folder/'model.pt') != read_json(package)['weights_sha256']:
                            raise ValueError('Completed observation model hash mismatch')
                        jobs.append(dict(job=key, status='measured', resumed_verified=True))
                        continue
                    if stage == 'calibrate' and not (folder/'package.json').exists():
                        jobs.append(dict(job=key, status='not run', reason='Directional selector fit failed'))
                        continue
                    command = [sys.executable, '-m', 'pipeline_error_training', stage, '--source', source, '--arm', 'O10_swap']
                    log_path = root/f'{key}.log'
                    start = time.monotonic()
                    with log_path.open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        write_json(root/'active.json', dict(job=key, command=command, pid=child.pid, started=now()))
                        code = child.wait()
                    tail = log_path.read_text(errors='replace')[-1600:]
                    status = 'measured' if code == 0 else 'blocked' if any(s in tail for s in ['BlockingIOError', 'Resource envelope', 'Other GPU use']) else 'failed'
                    jobs.append(dict(job=key, status=status, returncode=code, seconds=time.monotonic()-start,
                        log_sha256=sha(log_path), failure_tail=tail if code else None))
                    write_json(root/'progress.json', dict(status='running', jobs=jobs))
                    monitor.check()
        write_json(root/'progress.json', dict(status='complete', jobs=jobs, completed=now()))


if __name__ == '__main__':
    cpu_budget()
    run()
