"""Conservative measured lease wall-time accounting across every study worker.

Other programs' GPU utilization is not charged. CPU time spent holding our GPU
lease is charged, so this is an upper bound on our GPU kernel time.
"""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .common import Blocked, WORK

CAPS = dict(preflight_and_pilots=4., first_seed_training=24., replication=12., final_inference_and_package=8.)
ROOT = WORK/'gpu_budget'


def category(arguments=None):
    args = ' '.join(sys.argv if arguments is None else arguments)
    if any(s in args for s in ['matrix_entry', 'composition_entry', 'stress', 'final_point_inference', 'final_inference', 'fresh_candidates']):
        return 'final_inference_and_package'
    if any(s in args for s in ['train-pilot', 'profile', 'parity', 'fresh_validation', 'fresh_entry', 'native_validation', 'resume_validation']):
        return 'preflight_and_pilots'
    if '314159' in args:
        return 'replication'
    return 'first_seed_training'


@contextlib.contextmanager
def state():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT/'ledger.json'
        data = json.loads(path.read_text()) if path.exists() else dict(completed=[], active={})
        yield data
        temp = path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(data, separators=(',', ':'))+'\n')
        temp.replace(path)


def hours(data, timestamp=None):
    now = time.time() if timestamp is None else timestamp
    total = {key: 0. for key in CAPS}
    for row in data['completed']:
        total[row['category']] += row['seconds']/3600
    for row in data['active'].values():
        total[row['category']] += max(0., now-row['started_epoch'])/3600
    return total


def assert_room(total, lane):
    if sum(total.values()) >= 48 or total[lane] >= CAPS[lane]:
        raise Blocked(f'Measured GPU lease budget reached: {lane} {total[lane]:.6f}/{CAPS[lane]:g} hours; total {sum(total.values()):.6f}/48')


def begin():
    lane, token = category(), uuid.uuid4().hex
    with state() as data:
        assert_room(hours(data), lane)
        data['active'][token] = dict(pid=os.getpid(), category=lane, started_epoch=time.time(), arguments=sys.argv)
    return token


def finish(token):
    with state() as data:
        row = data['active'].pop(token)
        row.update(seconds=max(0., time.time()-row['started_epoch']), token=token, status='measured')
        data['completed'].append(row)


def check():
    path = ROOT/'ledger.json'
    if not path.exists():
        return
    # Atomic replacement makes this read consistent without stalling samplers.
    data = json.loads(path.read_text())
    total = hours(data)
    for row in data['active'].values():
        assert_room(total, row['category'])


def report():
    with state() as data:
        total = hours(data)
        return dict(measured_gpu_lease_hours=total, total_hours=sum(total.values()), limits_hours=CAPS,
            active_leases=list(data['active'].values()), completed_intervals=len(data['completed']),
            accounting='GPU lease wall time, including CPU work under lease; conservative pre-ledger resource-wall charges are identified separately.',
            charge_other_programs=False)


def backfill():
    """One immutable pre-ledger import of nonoverlapping completed measurements."""
    candidates = []
    for path in WORK.rglob('history.jsonl'):
        if 'training' not in path.parts and 'training_pilot' not in path.parts:
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        candidates.append(dict(path=str(path.relative_to(WORK)), seconds=sum(r['seconds'] for r in records),
            category='preflight_and_pilots' if 'training_pilot' in path.parts else ('replication' if '314159' in path.parts else 'first_seed_training'),
            accounting='Measured completed optimizer-step lease intervals'))
    paths = [*WORK.glob('calibration/*/*/*/resources.json'), *WORK.glob('source_screen/*/*/*/predictions/*/*.resources.json'),
        *WORK.glob('invalid/startup_validation_attempt1/**/resources.json'),
        *WORK.glob('invalid/startup_validation_attempt1/**/*.resources.json'), WORK/'resources/profile.json']
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        candidates.append(dict(path=str(path.relative_to(WORK)), seconds=record['wall_seconds'],
            category='first_seed_training' if path.parts[len(WORK.parts)] in ['calibration', 'source_screen'] else 'preflight_and_pilots',
            accounting='Conservative full worker resource-wall charge; may include CPU work outside the lease'))
    from .common import sha, write_json, RESULTS
    for row in candidates:
        row.update(status='measured', measurement_sha256=sha(WORK/row['path']), pre_ledger=True)
    with state() as data:
        if any(r.get('pre_ledger') for r in data['completed']):
            raise ValueError('Pre-ledger measurements have already been imported')
        data['completed'].extend(candidates)
    write_json(RESULTS/'gpu_budget_backfill.json', dict(status='measured', records=candidates,
        incomplete_existing_worker='The D10_adapted/44b6 calibration process started before ledger installation; its complete resource-wall receipt will be charged separately.',
        scientific_recipe_unchanged=True), immutable=True)
    return report()


if __name__ == '__main__':
    print(backfill() if '--backfill' in sys.argv else report())
