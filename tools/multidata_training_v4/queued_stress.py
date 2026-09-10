"""Run the fixed external stress checks when their pretraining weights exist."""
import subprocess
import sys
import time
from datetime import datetime
from .common import *

def run():
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    write(OUT/'queued_stress.json',dict(state='waiting_for_pretraining',queued_at=now(),deadline_utc=read(OUT/'authorized_run_window.json')['deadline_utc']))
    while not (OUT/'models/I_randomized_6bba.json').exists():
        if time.time()>deadline-1800:
            write(OUT/'queued_stress.json',dict(state='skipped',reason='Reserve delivery before deadline'));return
        time.sleep(30)
    start=time.monotonic();log=OUT/'logs/external_stress.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    write(OUT/'queued_stress.json',dict(state='running',started=now(),log=str(log)))
    with log.open('w') as f:
        p=subprocess.run([sys.executable,'-m','multidata_training_v4','stress'],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-600))
    write(OUT/'queued_stress.json',dict(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.monotonic()-start,log=str(log)))
