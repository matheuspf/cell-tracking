"""Detached, resumable v5 supervisor with two bounded GPU lanes."""
import os,sys,time,subprocess,fcntl
from .common import *


def prepare():
    from .queue_predictions import call
    for module,args,label in [('cache',[],'cache_all'),('train_hoct',['extract'],'hoct_extract_common_bank'),
            ('deepcenter',[],'deepcenter'),('queue_predictions',['banks'],'bank_queue')]:
        with gpu_aux():call(module,args,label)
    write(OUT/'preparation_complete.json',dict(at=now(),complete=True))


def supervise():
    lock=(OUT/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A v5 supervisor is already running')
    jobs=[('preparation','resume',['prepare']),('native_training','queue_native',[]),
        ('hoct_training','queue_hoct',[]),('controls','queue_predictions',['controls']),
        ('hoct_predictions','queue_predictions',['hoct']),('N1_predictions','queue_predictions',['native','--stage','N1']),
        ('N2_predictions','queue_predictions',['native','--stage','N2']),
        ('N2_replication','queue_predictions',['native','--stage','N2','--seed','314159']),('headroom','headroom',[])]
    handles=[];children={};started=now()
    for name,module,args in jobs:
        log=(OUT/'logs'/f'resumed_{name}.log').open('a');handles.append(log)
        log.write(f'\nSupervisor start {started}\n');log.flush()
        children[name]=subprocess.Popen([sys.executable,'-m',f'image_native_tracking_v5.{module}',*args],stdout=log,stderr=subprocess.STDOUT)
    monitor_log=(OUT/'logs/resource_monitor.log').open('a');handles.append(monitor_log)
    monitor=subprocess.Popen([sys.executable,'-m','image_native_tracking_v5.monitor'],stdout=monitor_log,stderr=subprocess.STDOUT)
    while True:
        state={name:dict(pid=p.pid,returncode=p.poll(),state='running' if p.poll() is None else 'complete' if p.returncode==0 else 'failed') for name,p in children.items()}
        write(OUT/'supervisor_state.json',dict(at=now(),started=started,pid=os.getpid(),jobs=state,
            gpu_policy='one native training lane plus one auxiliary lane; process pools serialized'))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(15)
    write(OUT/'compute_queue_finished.json',dict(at=now(),jobs=state,all_success=all(p.returncode==0 for p in children.values())))
    monitor.terminate();monitor.wait()
    for h in handles:h.close()
    if any(p.returncode for p in children.values()):raise RuntimeError('One or more v5 jobs failed; inspect supervisor_state.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['supervise','prepare']);a=p.parse_args()
    prepare() if a.action=='prepare' else supervise()
