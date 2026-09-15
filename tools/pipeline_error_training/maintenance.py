"""Consume explicit study repair jobs at serial worker boundaries."""
import fcntl
import os
import subprocess
import sys
import time

from .common import WORK, now, read_json, sha, write_json


def drain():
    if os.environ.get('PIPELINE_STUDY_MAINTENANCE_CHILD') == '1':
        return
    root = WORK/'maintenance'
    path = root/'requests.json'
    if not path.exists():
        return
    with (root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for job in read_json(path)['jobs']:
            receipt = root/f'{job["id"]}.json'
            if receipt.exists():
                continue
            if not job['module'].startswith('pipeline_error_training'):
                raise ValueError('Maintenance jobs are restricted to this study')
            command = [sys.executable, '-m', job['module'], *job['arguments']]
            start = time.monotonic()
            env = dict(os.environ, PIPELINE_STUDY_MAINTENANCE_CHILD='1')
            with (root/f'{job["id"]}.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(root/'active.json', dict(job=job['id'], pid=child.pid, command=command, started=now()))
                code = child.wait()
            write_json(receipt, dict(status='measured' if code == 0 else 'failed', returncode=code,
                seconds=time.monotonic()-start, log_sha256=sha(root/f'{job["id"]}.log'), independent_lanes_continue=True))


if __name__ == '__main__':
    drain()
