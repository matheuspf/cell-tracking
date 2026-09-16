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


def transfer_settled():
    path = ROOT/'reservation_settlement.json'
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt['original_reservations_hours']!=CAPS or receipt['combined_first_seed_and_final_hours']!=32. \
            or receipt['total_cap_hours']!=48. or receipt['replication_reservation_hours']!=12.:
        raise ValueError('Frozen GPU reservation settlement changed')
    return True


def effective_caps(total, transfer=False):
    caps = dict(CAPS)
    if transfer:
        # Only unused first-seed time can move. Preflight and all 12 replication
        # hours remain reserved, and first-seed work itself still cannot exceed 24.
        caps['final_inference_and_package'] = 32.-total['first_seed_training']
        caps['first_seed_training'] = min(24.,32.-total['final_inference_and_package'])
    return caps


def category(arguments=None):
    args = ' '.join(sys.argv if arguments is None else arguments)
    if any(s in args for s in ['matrix_entry', 'composition_entry', 'stress', 'final_point_inference', 'final_inference', 'fresh_candidate']):
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


def assert_room(total, lane, transfer=False):
    caps = effective_caps(total,transfer)
    if sum(total.values()) >= 48 or total[lane] >= caps[lane]:
        raise Blocked(f'Measured GPU lease budget reached: {lane} {total[lane]:.6f}/{caps[lane]:g} hours; total {sum(total.values()):.6f}/48')


def begin():
    lane, token = category(), uuid.uuid4().hex
    with state() as data:
        assert_room(hours(data), lane,transfer_settled())
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
        assert_room(total, row['category'],transfer_settled())


def report():
    with state() as data:
        total = hours(data)
        transferred = transfer_settled()
        interrupted = [r for r in data['completed'] if r.get('status')=='interrupted_upper_bound']
        return dict(measured_gpu_lease_hours=total, total_hours=sum(total.values()),
            original_reservations_hours=CAPS,limits_hours=effective_caps(total,transferred),
            unused_first_seed_reservation_transfer_enabled=transferred,
            combined_first_seed_and_final_hours=32. if transferred else None,
            dependent_phase_limits_not_additive=transferred,
            active_leases=list(data['active'].values()), completed_intervals=len(data['completed']),
            interrupted_upper_bound_hours=sum(r['seconds'] for r in interrupted)/3600,
            interrupted_upper_bound_intervals=len(interrupted),
            accounting='Charged GPU lease wall time, including CPU work under lease. Conservative pre-ledger resource-wall charges and interrupted upper bounds are identified separately. An interrupted lease charged through the next host boot can include downtime; its exact end was not observed.',
            charge_other_programs=False)


def settle_reservations():
    """Seal source-complete resource allocation before target access, not recipes."""
    from .common import RESULTS,STUDY,now,read_json,sha,write_json
    path = ROOT/'reservation_settlement.json'
    if path.exists():
        transfer_settled()
        if sha(path)!=sha(RESULTS/'gpu_reservation_settlement.json'):
            raise ValueError('Sanitized and runtime GPU reservation settlements differ')
        return read_json(path)
    study = read_json(STUDY)
    if study['limits']['measured_gpu_hours']!=48 or study['gpu_hour_reservation']!=CAPS:
        raise ValueError('Registered total budget or phase reservations changed')
    if (RESULTS/'target_freeze.json').exists():
        raise ValueError('Initial reservation settlement must precede target freezing')
    for name in ['queue','observation_queue']:
        if read_json(WORK/name/'progress.json')['status']!='complete':
            raise ValueError('Primary source queues must complete before settlement')
    with state() as data:
        if data['active']:
            raise ValueError('A study GPU lease is still active during reservation settlement')
        total = hours(data)
    result = dict(status='measured',created=now(),study_sha256=sha(STUDY),
        original_reservations_hours=CAPS,measured_hours_at_settlement=total,
        first_seed_unused_hours_at_settlement=max(0.,24.-total['first_seed_training']),
        combined_first_seed_and_final_hours=32.,total_cap_hours=48.,replication_reservation_hours=12.,
        preflight_reservation_hours=4.,first_seed_individual_cap_hours=24.,
        rule='After all source jobs, unused first-seed reservation may cover final inference; the combined first-seed/final pool is 32 hours. Any later same-recipe repair consumes that same pool.',
        allocation_basis='study.json distinguishes limits.measured_gpu_hours from gpu_hour_reservation; PLAN.md retains the replication reservation.',
        training_updates_changed=False,model_or_threshold_selection_changed=False,new_target_metrics_read=False)
    write_json(path,result,immutable=True)
    write_json(RESULTS/'gpu_reservation_settlement.json',result,immutable=True)
    return result


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
