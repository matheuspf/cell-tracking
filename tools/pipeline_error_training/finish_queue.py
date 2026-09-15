"""Finite post-primary execution; failures do not cancel independent lanes."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from .common import RESULTS,WORK,inputs,now,read_json,sha,write_json


def execute(name,module,args=()):
    root=WORK/'finish_queue';root.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pipeline_error_training.'+module,*args]
    start=time.monotonic()
    with (root/f'{name}.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write_json(root/'active.json',dict(job=name,pid=process.pid,command=command,started=now()))
        code=process.wait()
    result=dict(job=name,status='measured' if code==0 else 'failed',returncode=code,
        seconds=time.monotonic()-start,log_sha256=sha(root/f'{name}.log'))
    write_json(root/f'{name}.json',result)
    print(f'Post-primary {name}: {result["status"]}',flush=True)
    return result


def run():
    for name in ['queue','observation_queue']:
        if read_json(WORK/name/'progress.json')['status']!='complete':
            raise RuntimeError('Primary independent queues must complete before post-primary execution')
    root=WORK/'finish_queue';root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        records=[]
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
    run()
