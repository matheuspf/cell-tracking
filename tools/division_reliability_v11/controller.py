"""Resumable study dependency queue. Every data/model worker is a fresh process."""
from pathlib import Path
import concurrent.futures as futures
from collections import deque
import fcntl,os,subprocess,sys,time,threading
from .common import REPO,WORK,RESULTS,Blocked,read,write,sha,now
from .populations import clips

FOLDER=WORK/'controller'


def live(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False


def worker(stage,source=None,seed=None,arm=None,clip=None,part=None):
    args=[sys.executable,'-m','division_reliability_v11',stage]
    for key,value in [('source',source),('seed',seed),('arm',arm),('clip',clip),('part',part)]:
        if value is not None:args += ['--'+key,str(value)]
    if stage.startswith('train-'):args.append('--resume')
    if stage=='train-compact':args.append('--defer-mining')
    name='-'.join(str(x) for x in (stage,source,seed,arm,part,clip) if x is not None)
    folder=FOLDER/'jobs';folder.mkdir(parents=True,exist_ok=True);log=folder/(name+'.log')
    previous=folder/(name+'.json')
    if previous.exists():
        old=read(previous)
        if old.get('status') in ('failed','blocked','running'):
            write(folder/'history'/f'{name}-{time.time_ns()}.json',old,immutable=True)
    tick=time.monotonic()
    with log.open('a') as stream:
        process=subprocess.Popen(args,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        write(folder/(name+'.json'),dict(status='running',stage=stage,source=source,seed=seed,arm=arm,clip=clip,part=part,
            pid=process.pid,started_utc=now(),command=args))
        import psutil
        violation=None;peak_rss=0;worker_peak_rss=0;minimum_available=None;minimum_free=None
        while process.poll() is None:
            members=[]
            for p in psutil.process_iter(['pid','cmdline','memory_info']):
                try:
                    if 'division_reliability_v11' in ' '.join(p.info['cmdline'] or []):members.append(p.info['memory_info'].rss)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            available=psutil.virtual_memory().available
            free=__import__('shutil').disk_usage(WORK).free
            peak_rss=max(peak_rss,sum(members))
            try:worker_peak_rss=max(worker_peak_rss,psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:pass
            minimum_available=available if minimum_available is None else min(minimum_available,available)
            minimum_free=free if minimum_free is None else min(minimum_free,free)
            sample=dict(study_rss_bytes=sum(members),host_available_bytes=available,durable_free_bytes=free,
                study_rss_peak_bytes=peak_rss,worker_rss_peak_bytes=worker_peak_rss,
                host_available_min_bytes=minimum_available,durable_free_min_bytes=minimum_free,utc=now())
            write(folder/(name+'.resources.json'),sample)
            if sum(members)>44*2**30 or available<10*2**30 or free<12*2**30:
                violation=sample;process.terminate();break
            time.sleep(1)
        code=process.wait()
    result=dict(status='complete' if code==0 else 'blocked' if code==2 else 'failed',stage=stage,source=source,seed=seed,arm=arm,
        clip=clip,part=part,exit_code=code,wall_seconds=time.monotonic()-tick,finished_utc=now(),log_sha256=sha(log))
    if code:result['reason']=log.read_text()[-2500:]
    if violation:result['resource_violation']=violation
    write(folder/(name+'.json'),result)
    if code:raise Blocked(f'{name}: {result["status"]}; private log {log.relative_to(REPO)}')
    return result


def batch(jobs,workers):
    queue=deque(jobs);active=set();errors=[]
    if not queue:return
    stage=queue[0]['stage'];arm=queue[0].get('arm')
    def concurrency():
        options=read(FOLDER/'options.json') if (FOLDER/'options.json').exists() else {}
        overrides=options.get('workers_by_stage',{})
        limit=overrides.get(stage+':'+str(arm),overrides.get(stage,options.get('workers',workers)))
        if not isinstance(limit,int) or not 1<=limit<=8:raise Blocked('CPU worker limit must be in [1,8] and supported by measured memory headroom')
        return limit
    # Start with three workers. Operator changes use measured per-worker RAM;
    # the registered aggregate memory/disk floors remain enforced per process.
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        while queue or active:
            limit=concurrency()
            while queue and len(active)<limit:active.add(pool.submit(worker,**queue.popleft()))
            done,active=futures.wait(active,timeout=1,return_when=futures.FIRST_COMPLETED)
            for job in done:
                try:job.result()
                except Exception as exc:errors.append(str(exc))
            if errors:queue.clear()  # Finish owned workers; leave unstarted clips for resume.
        if errors:raise Blocked('; '.join(errors))


def await_upstream(source,seed):
    final=WORK/'fits'/source/str(seed)/'upstream/final.json'
    while not final.exists():
        state=read(FOLDER/'upstream.json') if (FOLDER/'upstream.json').exists() else {}
        if state.get('status')=='running' and live(state.get('controller_pid')) and live(state.get('worker_pid')):
            write(FOLDER/'pipeline.json',dict(status='waiting_for_upstream',source=source,seed=seed,
                controller_pid=os.getpid(),existing_upstream_worker=state['worker_pid'],updated_utc=now()))
            time.sleep(10);continue
        # Never duplicate a live cell owner. The worker also enforces its flock.
        worker('train-upstream',source,seed);break


def run(*,workers=2):
    from .readiness import require_production
    require_production('run')
    if not 1<=workers<=8:raise Blocked('Use one to eight bounded CPU workers; increases require measured memory headroom')
    FOLDER.mkdir(parents=True,exist_ok=True)
    owner=(FOLDER/'pipeline-owner.lock').open('a+')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise Blocked('An existing pipeline owns this study; inspect live PIDs instead of duplicating it') from exc
    try:
        for cell in read(RESULTS/'execution_lock.json')['schedule']:
            source,seed=cell['source'],cell['seed'];await_upstream(source,seed)
            write(FOLDER/'pipeline.json',dict(status='running_source_stages',cell=cell,controller_pid=os.getpid(),updated_utc=now()))
            worker('package',source,seed,'C00')
            for part in ('fit','calibration'):
                batch([dict(stage='predict',source=source,seed=seed,arm='C00',clip=n,part=part) for n in clips(source,part)],workers)
                batch([dict(stage='prepare-actions',source=source,seed=seed,clip=n,part=part) for n in clips(source,part)],workers)
            worker('source-diagnostics',source,seed)
            source_evidence=WORK/'source_diagnostics'/source/str(seed)/'summary.json'
            support=read(source_evidence)['census']
            from .retention import block
            if not support.get('positive') or not support.get('negative'):
                block(source,seed,['C01','C11'],'source_support',
                    'Complete source-fit deployment census lacks a reachable positive or completely supported negative occurrence group',source_evidence)
                continue
            try:worker('fit-linear',source,seed)
            except Blocked:
                linear=WORK/'fits'/source/str(seed)/'linear/model.json'
                if not linear.exists() or read(linear)['status']!='nonconverged':raise
                block(source,seed,['C01','C11'],'linear_convergence','Registered 1000-iteration C01 fit did not converge',linear)
                continue
            if not support.get('identity_groups'):
                block(source,seed,['C11'],'identity_support','No supported incoming identity groups for the compact prefix',source_evidence)
                batch([dict(stage='calibration-predict',source=source,seed=seed,arm='C01',clip=n) for n in clips(source,'calibration')],workers)
                worker('calibrate',source,seed,'C01');worker('package',source,seed,'C01')
                continue
            worker('train-compact',source,seed)
            if not (WORK/'fits'/source/str(seed)/'compact/final.json').exists():
                worker('mining-prepare',source,seed)
                batch([dict(stage='mine',source=source,seed=seed,clip=n) for n in clips(source,'fit')],workers)
                worker('mining-finish',source,seed)
                worker('train-compact',source,seed)
            for arm in ('C01','C11'):
                batch([dict(stage='calibration-predict',source=source,seed=seed,arm=arm,clip=n) for n in clips(source,'calibration')],workers)
                worker('calibrate',source,seed,arm);worker('package',source,seed,arm)
        # Every retained fit and safety calibration is finished before target prediction.
        from .retention import resolve
        retention=resolve()
        for cell in read(RESULTS/'execution_lock.json')['schedule']:
            source,seed=cell['source'],cell['seed']
            for arm in ('C00','C01','C11'):
                if retention['matrix'][f'{source}/{seed}/{arm}']['status']!='retained':continue
                batch([dict(stage='predict',source=source,seed=seed,arm=arm,clip=n,part='target') for n in clips(source,'target')],workers)
        worker('freeze')
        worker('cold')
        for cell in read(RESULTS/'execution_lock.json')['schedule']:
            source,seed=cell['source'],cell['seed']
            for arm in ('C00','C01','C11'):
                if retention['matrix'][f'{source}/{seed}/{arm}']['status']!='retained':continue
                batch([dict(stage='evaluate',source=source,seed=seed,arm=arm,clip=n) for n in clips(source,'target')],workers)
        worker('report')
        result=dict(status='complete',finished_utc=now());write(FOLDER/'pipeline.json',result);return result
    except Exception as exc:
        result=dict(status='blocked',reason=str(exc),finished_utc=now());write(FOLDER/'pipeline.json',result)
        try:worker('report')
        except Exception:pass
        raise
    finally:owner.close()
