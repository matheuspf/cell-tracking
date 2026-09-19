"""One source-fit GPU owner at a time; restart only from durable owned state."""
import fcntl
import os
import subprocess
import sys
from .common import WORK,RESULTS,REPO,read,write,now
from .readiness import require_production


def run():
    require_production('upstream-queue')
    folder=WORK/'controller';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'upstream-owner.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cells=read(RESULTS/'execution_lock.json')['schedule']
        for cell in cells:
            source,seed=cell['source'],cell['seed']
            command=[sys.executable,'-m','division_reliability_v11','train-upstream','--source',source,'--seed',str(seed),'--resume']
            log=folder/f'upstream-{source}-{seed}.log'
            with log.open('a') as output:
                process=subprocess.Popen(command,cwd=REPO,stdout=output,stderr=subprocess.STDOUT)
                write(folder/'upstream.json',dict(status='running',cell=cell,controller_pid=os.getpid(),worker_pid=process.pid,started_utc=now(),command=command))
                code=process.wait()
            if code:
                write(folder/'upstream.json',dict(status='failed',cell=cell,exit_code=code,finished_utc=now()))
                return code
        write(folder/'upstream.json',dict(status='complete',cells=cells,finished_utc=now()))
    return 0


if __name__=='__main__':sys.exit(run())
