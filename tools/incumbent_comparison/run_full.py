"""Run the frozen full-clip stages in their appropriate isolated runtimes."""
from __future__ import annotations

import fcntl
import os
import subprocess

from .common import REPO, WORK, OUT, now, read, write, sha

NATIVE_PY = '/kaggle/envs/cell-tracking-notebooks/bin/python'
CELLPOSE_PY = '/kaggle/envs/detector-screen-cellpose/bin/python'


def main():
    lock_path = WORK / 'locks/full-clips.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        stages = [('banks', CELLPOSE_PY, OUT / 'full-clips-banks-lock.json'),
                  ('link', NATIVE_PY, OUT / 'full-clips-graphs-lock.json'),
                  ('evaluate', NATIVE_PY, OUT / 'full-clips-evaluation.json')]
        receipt = dict(started_utc=now(), pid=os.getpid(), plan_sha256=sha(OUT / 'full-clips-plan.json'), stages=[])
        write(WORK / 'full-clips-run.json', receipt)
        for action, python, completion in stages:
            if completion.exists():
                receipt['stages'].append(dict(stage=action, status='existing; inputs rechecked by next stage'))
                continue
            entry = dict(stage=action, started_utc=now(), status='running')
            receipt['stages'].append(entry)
            write(WORK / 'full-clips-run.json', receipt)
            with (WORK / f'full-{action}.log').open('a') as log:
                result = subprocess.run([python, '-m', 'tools.incumbent_comparison.full_clips', action],
                                        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry.update(ended_utc=now(), exit_code=result.returncode, status='complete' if result.returncode == 0 else 'failed')
            write(WORK / 'full-clips-run.json', receipt)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        receipt['completed_utc'] = now()
        write(WORK / 'full-clips-run.json', receipt)


if __name__ == '__main__':
    main()
