"""Resumable complete study execution; never lowers optimizer-update floors."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil

from .common import WORK,RESULTS,read_json,write_json,sha,now


def call(command,*arguments):
    key='-'.join([command,*map(str,arguments)]).replace('/','_')
    log=WORK/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    argv=[sys.executable,'-m','division_generalization_v2',command,*map(str,arguments)]
    print('RUN '+' '.join(argv),flush=True)
    with log.open('a') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
        write_json(WORK/'queue/active.json',dict(pid=child.pid,command=argv,log=str(log),started=now()))
        result=child.wait()
    write_json(WORK/'queue/last.json',dict(command=argv,returncode=result,log_sha256=sha(log),seconds=time.monotonic()-started))
    if result:raise RuntimeError(f'Preserved failed stage: {log}')


def train(source,arm,seed,stop=None):
    folder=WORK/'training'/arm/source/str(seed)
    expected=stop or (2048 if arm=='prefix' else 4096)
    if (folder/'progress.json').exists() and read_json(folder/'progress.json').get('joint_optimizer_updates',0)>=expected:
        return
    args=['--source',source,'--arm',arm,'--seed',seed]
    if stop:args+=['--stop-at',stop]
    call('train',*args)


def source_screens(source,arm,seed,steps):
    for step in steps:
        path=WORK/'screens'/arm/source/str(seed)/str(step)/'summary.json'
        if not path.exists():call('screen','--source',source,'--arm',arm,'--seed',seed,'--stop-at',step)


def run(args=None):
    WORK.mkdir(parents=True,exist_ok=True);(WORK/'queue').mkdir(exist_ok=True)
    handle=(WORK/'queue/worker.lock').open('a+')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(WORK/'queue/owner.json',dict(pid=os.getpid(),started=now()))
    try:
        if not (RESULTS/'baseline_validation.json').exists():call('baselines')
        # Existing independent preparers may finish before this queue starts.
        for source in ('44b6','6bba'):
            while True:
                preparers=[]
                for p in psutil.process_iter(['cmdline']):
                    cmd=p.info['cmdline'] or []
                    if 'division_generalization_v2' in cmd and 'prepare' in cmd and source in cmd:
                        preparers.append(p.pid)
                if not preparers:break
                write_json(WORK/'queue/waiting.json',dict(stage='existing_source_preparer',source=source,pids=preparers,time=now()))
                time.sleep(5)
            if not (WORK/'source'/source/'manifest.json').exists():call('prepare','--source',source)
        if not (RESULTS/'image_manifest.json').exists():
            from .image_audit import run as image_audit
            image_audit()
        from .dataset import sampling_audit
        sampling_audit()
        from .diagnostics import lock_strata
        lock_strata()
        from .validation import run as validate
        validate()
        if not (RESULTS/'profile.json').exists():call('profile','--source','44b6')
        # Both main arms receive an adequate attempt regardless of cheap control.
        for source in ('44b6','6bba'):train(source,'G30',20260916)
        profile=WORK/'crowded_profile/active.json'
        if profile.exists():
            pid=read_json(profile)['pid']
            while psutil.pid_exists(pid):
                write_json(WORK/'queue/waiting.json',dict(stage='crowded_inference_profile',pid=pid,time=now()))
                time.sleep(5)
        for seed in (20260916,314159):
            for source in ('44b6','6bba'):
                train(source,'prefix',seed)
                for arm in ('J_uniform','J_mined'):train(source,arm,seed)
        # No target predictions are generated before all source decisions and
        # both seed packages are frozen. Every calibration clip is replayed.
        for source in ('44b6','6bba'):call('source-matrix','--source',source)
        for source in ('44b6','6bba'):source_screens(source,'G30',20260916,(4096,))
        from .selection import extension,freeze
        for seed in (20260916,314159):
            for source in ('44b6','6bba'):
                for arm in ('J_uniform','J_mined'):source_screens(source,arm,seed,(3072,4096))
                extended=extension(source,seed)
                if extended['extend']:
                    for arm in ('J_uniform','J_mined'):
                        train(source,arm,seed,8192);source_screens(source,arm,seed,(8192,))
        freeze(None,None)
        from .target import predict_all,evaluate_all
        predict_all();evaluate_all()
        from .diagnostics import run as diagnostics
        diagnostics()
        from .fresh import run as fresh
        fresh()
        write_json(WORK/'queue/complete.json',dict(status='complete',finished=now()))
    except BaseException as exc:
        write_json(WORK/'queue/failure.json',dict(status='incomplete_resumable',exception=repr(exc),
            traceback=traceback.format_exc(),time=now(),no_failed_architecture_claim=True))
        raise
    finally:
        from .report import run as report
        report()
        fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
