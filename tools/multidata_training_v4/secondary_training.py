"""Same prescribed seed2 fits, with independent jobs sharing the single GPU."""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from .common import *

def specifications():
    plan=read(OUT/'secondary_plan.json');arm=plan['arm'];seed=314159
    domains={'C2':['synthetic'],'C3':['zebrafish'],'C4':['synthetic','zebrafish'],
        'C5':['synthetic','zebrafish','synthetic','zebrafish','synthetic','zebrafish','ascidian','ascidian'],
        'C6':['synthetic','zebrafish']}[arm]
    pre=[dict(name='G_external_seed2',component='G',domains=domains,steps=20000,seed=seed)];adapt=[]
    if arm not in ['C3','C6']:pre.append(dict(name='I_external_seed2',component='I',domains=['synthetic'],steps=20000,init='D_synthetic_seed2',seed=seed))
    for source in ['44b6','6bba']:
        if arm=='C6':pre.append(dict(name=f'I_external_{source}_seed2',component='I',domains=['synthetic'],steps=20000,
            init='D_synthetic_seed2',randomized=True,calibration_source=source,seed=seed))
        adapt += [dict(name=f'G_C1_seed2_{source}',component='G',domains=[source],steps=8000,init=f'G_real_{source}_seed2',source=source,seed=seed),
            dict(name=f'I_C1_seed2_{source}',component='I',domains=[source],steps=8000,init=f'I_real_{source}_seed2',source=source,seed=seed),
            dict(name=f'G_{arm}_seed2_{source}',component='G',domains=[source],steps=8000,init='G_external_seed2',source=source,replay=None if arm=='C3' else domains,seed=seed)]
        iinit=f'I_real_{source}_seed2' if arm=='C3' else f'I_external_{source}_seed2' if arm=='C6' else 'I_external_seed2'
        adapt.append(dict(name=f'I_{arm}_seed2_{source}',component='I',domains=[source],steps=8000,init=iinit,source=source,
            replay=None if arm=='C3' else ['synthetic'],randomized=arm=='C6',seed=seed))
        if plan.get('repeat_short_control'):
            for component in ['G','I']:adapt.append(dict(name=f'{component}_C1short_seed2_{source}',component=component,domains=[source],steps=8000,source=source,seed=seed))
    return pre,adapt

def worker(name):
    from .train import fit
    fit(**next(j for j in sum(specifications(),[]) if j['name']==name))

def run():
    from .train import fit
    plan=read(OUT/'secondary_plan.json');seed=314159
    if plan['arm']!='C3':fit('D_synthetic_seed2','D',['synthetic'],12000,seed=seed)
    for source in ['44b6','6bba']:
        for comp,steps in [('D',12000),('G',20000),('I',20000)]:
            fit(f'{comp}_real_{source}_seed2',comp,[source],steps,init=f'D_real_{source}_seed2' if comp=='I' else None,seed=seed)
    records=[]
    def launch(job):
        started=time.monotonic()
        with (OUT/'logs'/f'seed2_fit_{job["name"]}.log').open('a') as f:
            r=subprocess.run([sys.executable,'-m','multidata_training_v4.secondary_training',job['name']],stdout=f,stderr=subprocess.STDOUT)
        record=dict(model=job['name'],returncode=r.returncode,seconds=time.monotonic()-started);print(record,flush=True);return record
    for phase,batch in zip(['pretraining','adaptation'],specifications()):
        status('W440',state='second-seed matched training',phase=phase,concurrent_gpu_processes=4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for record in pool.map(launch,batch):records.append(record);write(OUT/'secondary_training_progress.json',records)
        assert all(r['returncode']==0 for r in records),records
    write(OUT/'secondary_training_receipt.json',dict(completed=True,jobs=records,same_prescribed_configs_and_budgets=True))
    write(OUT/'checkpoint_manifest_secondary.json',{p.stem:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted((OUT/'models').glob('*seed2*.pt'))})

if __name__=='__main__':worker(sys.argv[1])
