"""Explicit, fail-closed v11 stage entry points."""
import argparse
import json
import sys
from pathlib import Path
from .common import CONFIG, DATA, WORK, RESULTS, Blocked, read, write, now


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['reconcile','preflight','audit-splits','pilot','overfit','source-inference',
                                  'source-evaluate','lock','run','train-upstream','prepare-actions',
                                  'fit-linear','train-compact','calibrate','predict','freeze','evaluate',
                                  'validate','package','infer','report','cold','cold-select','source-diagnostics','upstream-queue','calibration-predict',
                                  'mining-prepare','mine','mining-finish'])
    p.add_argument('--source',choices=['44b6','6bba'])
    p.add_argument('--seed',type=int,choices=[20260918,314159])
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--clip')
    p.add_argument('--arm',choices=['C00','C01','C11'],default='C00')
    p.add_argument('--part',choices=['fit','calibration','target'],default='target')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--defer-mining',action='store_true')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--overfit',action='store_true')
    p.add_argument('--updates',type=int,default=250,help='Diagnostic overfit updates only; production horizon comes from the lock')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--package',type=Path)
    p.add_argument('--images',type=Path)
    p.add_argument('--output',type=Path)
    return p


def main():
    p=parser();a=p.parse_args()
    if a.config.resolve()!=CONFIG.resolve():p.error('This executor accepts only the registered v11 config')
    if a.stage in ('pilot','overfit','source-inference','source-evaluate','train-upstream','prepare-actions',
                   'fit-linear','train-compact','calibrate','predict','evaluate','package','source-diagnostics','calibration-predict',
                   'mining-prepare','mine','mining-finish'):
        if a.source is None or a.seed is None:p.error('--source and --seed are required')
    try:
        from .execution_precision import configure
        configure(a.stage,a.arm,a.package,source=a.source,seed=a.seed,clip=a.clip)
        if a.stage=='reconcile':
            from .reconcile import run
            result=run()
        elif a.stage=='preflight':
            from .preflight import run
            result=run(a.data_root)
        elif a.stage=='audit-splits':
            from .splits import audit
            result=audit(a.data_root)
        elif a.stage in ('pilot','overfit'):
            from .pilot import run
            result=run(a.source,a.seed,data=a.data_root,overfit_updates=a.updates if a.stage=='overfit' else 0)
        elif a.stage=='source-inference':
            if not a.clip:p.error('--clip is required')
            from .source_inference import run
            result=run(a.source,a.seed,a.clip,a.data_root,overfit=a.overfit)
        elif a.stage=='source-evaluate':
            if not a.clip:p.error('--clip is required')
            from .source_evaluation import run
            result=run(a.source,a.seed,a.clip,overfit=a.overfit,data=a.data_root)
        elif a.stage=='train-upstream':
            from .readiness import require_production
            require_production('train-upstream')
            from .train_upstream import run
            result=run(a.source,a.seed,resume=a.resume)
        elif a.stage=='upstream-queue':
            from .upstream_queue import run
            return run()
        elif a.stage=='prepare-actions':
            if not a.clip or a.part=='target':p.error('A source --clip and --part fit/calibration are required')
            from .dataset import run
            result=run(a.source,a.seed,a.clip,part=a.part)
        elif a.stage=='fit-linear':
            from .fit_linear import run
            result=run(a.source,a.seed)
        elif a.stage=='train-compact':
            from .train_compact import run
            result=run(a.source,a.seed,resume=a.resume,defer_mining=a.defer_mining)
        elif a.stage in ('mining-prepare','mine','mining-finish'):
            from . import mining
            if a.stage=='mine':
                if not a.clip:p.error('--clip is required')
                result=mining.run(a.source,a.seed,a.clip)
            elif a.stage=='mining-prepare':result=mining.prepare(a.source,a.seed)
            else:result=mining.finish(a.source,a.seed)
        elif a.stage=='calibrate':
            from .calibrate import run
            result=run(a.source,a.seed,a.arm)
        elif a.stage=='calibration-predict':
            if not a.clip:p.error('--clip is required')
            from .calibration_cache import run
            result=run(a.source,a.seed,a.arm,a.clip)
        elif a.stage=='package':
            from .packaging import run
            result=run(a.source,a.seed,a.arm)
        elif a.stage=='predict':
            if not a.clip:p.error('--clip is required')
            from .populations import predict
            result=predict(a.source,a.seed,a.arm,a.clip,a.part)
        elif a.stage=='infer':
            if not all((a.package,a.images,a.output)):p.error('--package, --images and --output are required')
            from .inference import run
            result=run(a.package,a.images,a.output,cold=a.cold,baseline=a.baseline)
        elif a.stage=='freeze':
            from .freeze import run
            result=run()
        elif a.stage=='evaluate':
            if not a.clip:p.error('--clip is required')
            from .evaluation import run
            result=run(a.source,a.seed,a.arm,a.clip)
        elif a.stage=='source-diagnostics':
            from .diagnostics import source_summary
            result=source_summary(a.source,a.seed)
        elif a.stage=='cold':
            from .cold import run
            result=run()
        elif a.stage=='cold-select':
            from .cold import select
            result=select()
        elif a.stage=='run':
            from .controller import run
            result=run(workers=a.workers)
        elif a.stage=='lock':
            from .readiness import lock
            result=lock()
        elif a.stage=='report':
            from .report import run
            result=run()
        elif a.stage=='validate':
            import subprocess
            import re
            command=[sys.executable,'-m','unittest','division_reliability_v11.test_execution','-v']
            folder=WORK/'checks';folder.mkdir(parents=True,exist_ok=True)
            r=subprocess.run(command,capture_output=True,text=True)
            log=r.stdout+r.stderr;(folder/'runtime.log').write_text(log)
            print(log,file=sys.stderr)
            count=re.search(r'Ran (\d+) tests?',log)
            result=dict(status='unit_tests_passed' if r.returncode==0 else 'unit_tests_failed',
                        count=int(count[1]) if count else None,exit_code=r.returncode,finished_utc=now(),
                        source_resume_proof='see pilot receipts',tests_are_not_model_results=True)
            write(folder/'runtime.json',result)
            if r.returncode:return r.returncode
        print(json.dumps({k:v for k,v in result.items() if k not in ('clips','steps','architecture','v10_cells','predictions','ancestry')},indent=2))
        return 0
    except Blocked as e:
        receipt=dict(status='blocked',stage=a.stage,source=a.source,seed=a.seed,reason=str(e),utc=now())
        try:write(WORK/'command_receipts'/f'{__import__("time").time_ns()}.json',receipt)
        except PermissionError:receipt['command_receipt_write_denied']=True
        print(json.dumps(receipt),file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
