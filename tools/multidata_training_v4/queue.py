"""Local deadline-aware continuation of already authorized experiment stages."""
import os
import sys
import time
import signal
import subprocess
from datetime import datetime,timezone
from .common import *

VARIANTS='C0,decoder_only,original_objective_C4,C1short,C1,C2,C3,C4,C5,C6,C4_Gonly,C4_m2,C4_m6,C1_relinked,C4_relinked,D_real,D_synthetic,D_synthetic_C4'

def run():
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    path=OUT/'scheduled_tests.json';jobs=[dict(stage=s,state='waiting') for s in ['calibrate','infer','evaluate']]
    write(path,dict(deadline_utc=read(OUT/'authorized_run_window.json')['deadline_utc'],jobs=jobs,waiting_for='checkpoint_manifest.json'))
    while not (OUT/'checkpoint_manifest.json').exists():
        if time.time()>deadline:write(path,dict(state='deadline_before_training_finished',jobs=jobs));return
        time.sleep(30)
    for job in jobs:
        stage=job['stage'];remaining=deadline-time.time()-600
        if remaining<=0:job['state']='not_started_deadline_reserve';break
        job.update(state='running',started=now());write(path,dict(jobs=jobs,deadline=deadline))
        args=[sys.executable,'-m','multidata_training_v4',stage]
        if stage=='evaluate':args+=['--variants',VARIANTS,'--workers','8']
        log=OUT/'logs'/f'queued_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);code=124
        job.update(state='complete' if code==0 else 'failed',returncode=code,ended=now(),log=str(log))
        write(path,dict(jobs=jobs,deadline=deadline))
        if code:break
    write(OUT/'scheduled_tests_receipt.json',dict(jobs=jobs,deadline=deadline,finished=now()))
