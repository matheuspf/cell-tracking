"""Repeatable final checks and a receipt bound to the delivered code."""
import os
import subprocess
import sys
import tempfile
import time

from .common import code_hashes, now, sha, write_json


def run(ctx, args=None):
    tests=['tests/strong_tracker_v3','tests/strong_tracker_v2','tests/annotation_selection',
        'work/annotation-selection-v1/official/tests/test_metrics.py',
        'work/annotation-selection-v1/official/tests/test_division_metrics.py',
        'work/annotation-selection-v1/official/tests/test_division_sandbox_examples.py']
    commands=[('final_tests',[sys.executable,'-m','pytest','-q',*tests],ctx.repo),
        ('final_handover_tests',[sys.executable,'-m','unittest','discover','-s',
            'handover/strong-tracker-v3','-p','test_*.py'],ctx.repo),
        ('foreign_cwd_wrapper',[str(ctx.repo/'scripts/run_strong_tracker_v3.sh'),
            '--python',sys.executable,'--help'],tempfile.gettempdir())]
    records=[]
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    for name,command,cwd in commands:
        path=ctx.out/'logs'/f'{name}.log';path.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        with path.open('w') as output:
            result=subprocess.run(command,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
        record=dict(name=name,command=command,cwd=str(cwd),exit_code=result.returncode,
            seconds=time.perf_counter()-start,log_sha256=sha(path),log=str(path.relative_to(ctx.out)),
            summary=path.read_text().strip().splitlines()[-1])
        records.append(record)
        print(name,record['exit_code'],record['summary'],flush=True)
        if result.returncode:
            raise RuntimeError(f'Final validation failed: {name}; inspect {path}')
    hashes=code_hashes(ctx)
    for p in sorted((ctx.repo/'tests/strong_tracker_v3').glob('*.py')):
        hashes[str(p.relative_to(ctx.repo))]=sha(p)
    for relative in ['scripts/run_strong_tracker_v3.sh','tools/strong_tracker_v3/dashboard_template.html',
                     'docs/strong-tracker-v3.md']:
        hashes[relative]=sha(ctx.repo/relative)
    receipt=dict(created=now(),passed=True,checks=records,code_sha256=hashes,
        note='Unit/metric/contract checks and foreign-working-directory CLI check. Full graph scoring, image parity, copied-package smoke, preservation and offline browser checks have separate exact-hash receipts.')
    write_json(ctx.out/'validation_summary.json',receipt)
    return receipt


if __name__=='__main__':
    from .context import RunContext
    run(RunContext.default())
