"""Local continuation through measured reports and verified inference delivery."""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from .common import *

def run():
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    state=OUT/'finish_queue.json';jobs=[]
    write(state,dict(state='waiting_for_second_seed',queued_at=now(),deadline_utc=read(OUT/'authorized_run_window.json')['deadline_utc']))
    while not (OUT/'secondary_verification.json').exists():
        if time.time()>deadline-1800:
            write(state,dict(state='delivery_reserve_reached',jobs=jobs));return
        time.sleep(30)
    if not read(OUT/'secondary_verification.json')['completed']:
        write(state,dict(state='second_seed_requires_attention',jobs=jobs));return
    for stage in ['rendering_trial','coverage','report','delivery','head_parity','unknown_name_pilot','rendered_holdout','additional_pilots','finalize']:
        remaining=deadline-time.time()-600
        if remaining<=0:break
        log=OUT/'logs'/f'finish_{stage}.log';start=time.monotonic()
        write(state,dict(state='running',stage=stage,jobs=jobs))
        with log.open('w') as f:
            proc=subprocess.Popen([sys.executable,'-m','multidata_training_v4',stage],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);code=124
        jobs.append(dict(stage=stage,returncode=code,seconds=time.monotonic()-start,log=str(log)))
        if code:break
    write(state,dict(state='complete' if jobs and jobs[-1].get('stage')=='finalize' and jobs[-1].get('returncode')==0 else 'requires_attention',
        jobs=jobs,finished=now(),git_review_and_push_pending=True))
