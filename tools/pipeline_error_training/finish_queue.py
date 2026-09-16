"""Finite post-primary execution; failures do not cancel independent lanes."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from .common import RESULTS,WORK,inputs,now,read_json,sha,write_json


def execute(name,module,args=(),*,update_active=True):
    root=WORK/'finish_queue';root.mkdir(parents=True,exist_ok=True)
    previous = root/f'{name}.json'
    if previous.exists() and name not in ['validation', 'report']:
        receipt = read_json(previous)
        if receipt['status']=='measured':
            return dict(receipt, resumed_completed_stage=True)
    command=[sys.executable,'-m','pipeline_error_training.'+module,*args]
    start=time.monotonic()
    with (root/f'{name}.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        if update_active:
            write_json(root/'active.json',dict(job=name,pid=process.pid,command=command,started=now()))
        code=process.wait()
    result=dict(job=name,status='measured' if code==0 else 'failed',returncode=code,
        seconds=time.monotonic()-start,log_sha256=sha(root/f'{name}.log'))
    write_json(root/f'{name}.json',result)
    print(f'Post-primary {name}: {result["status"]}',flush=True)
    return result


def primary_ready():
    return all((WORK/name/'progress.json').exists() and
               read_json(WORK/name/'progress.json')['status']=='complete'
               for name in ['queue','observation_queue'])


def verify_primary_availability():
    from .freeze import PRIMARY
    decisions = read_json(RESULTS/'lane_blockers.json')['lanes'] if (RESULTS/'lane_blockers.json').exists() else []
    missing = []
    for arm in PRIMARY:
        for source in ['44b6','6bba']:
            complete = (WORK/'training'/arm/source/'20260915/frozen_package.json').exists() and \
                (WORK/'source_screen'/arm/source/'20260915/summary.json').exists()
            accounted = any(d['arm']==arm and d['source']==source and d['status'] in ['failed','blocked']
                            and d.get('reason') for d in decisions)
            if not complete and not accounted:
                missing.append(f'{arm}/{source}')
    if missing:
        raise RuntimeError('Repair incomplete primary lanes or record their concrete blocker before freezing: '+', '.join(missing))


def verify_existing_freeze():
    frozen = read_json(RESULTS/'target_freeze.json')
    settlement = frozen.get('gpu_reservation_settlement_sha256')
    if settlement is not None:
        from .budget import ROOT as budget_root,transfer_settled
        if not transfer_settled() or sha(RESULTS/'gpu_reservation_settlement.json')!=settlement \
                or sha(budget_root/'reservation_settlement.json')!=settlement:
            raise ValueError('Frozen GPU reservation settlement changed before queue resumption')
    for package in frozen['packages']:
        for kind in ['manifest','weights']:
            if sha(package[kind+'_path'])!=package[kind+'_sha256']:
                raise ValueError('Frozen directional model changed before queue resumption')
    for name in ['nomination','replication']:
        if sha(RESULTS/f'{name}.json')!=frozen[name+'_sha256']:
            raise ValueError('Frozen source selection changed before queue resumption')


def run(wait_for_primary=False):
    if wait_for_primary:
        root=WORK/'finish_queue';root.mkdir(parents=True,exist_ok=True)
        while not primary_ready():
            write_json(root/'progress.json',dict(status='waiting',reason='Independent primary queues are still executing'))
            time.sleep(10)
    for name in ['queue','observation_queue']:
        if read_json(WORK/name/'progress.json')['status']!='complete':
            raise RuntimeError('Primary independent queues must complete before post-primary execution')
    root=WORK/'finish_queue';root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        records=[]
        if (RESULTS/'target_freeze.json').exists():
            verify_existing_freeze()
            records.append(dict(job='existing-freeze',status='measured',resumed_verified=True))
        else:
            verify_primary_availability()
            for name in ['replication','freeze']:
                records.append(execute(name,name))
                if records[-1]['status']!='measured':
                    write_json(root/'progress.json',dict(status='failed',jobs=records,reason='Source-only nomination/freeze needs implementation repair'))
                    raise RuntimeError('Source-only selection/freeze requires repair before new target comparisons')
        for module in ['final_predictions','point_predictions','composition','stress_suite','fresh_candidates']:
            records.append(execute(module,module))
            write_json(root/'progress.json',dict(status='running',jobs=records))
        for arm in read_json(RESULTS/'target_freeze.json')['experiments']:
            if arm=='D00':continue
            complete=all((WORK/'predictions'/arm/f'{r["dataset"]}.npz').exists() for r in inputs())
            if complete:
                records.append(execute('score-'+arm,'accounting',[arm]))
            else:
                records.append(dict(job='score-'+arm,status='not run',reason='Complete all-199 predictions are unavailable; no P0 substitution'))
        records.append(execute('strata','strata'))
        records.append(execute('validation','validation',['--tests']))
        records.append(execute('report','report'))
        write_json(root/'progress.json',dict(status='complete',jobs=records,completed=now()))


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    run('--wait-for-primary' in sys.argv)
