"""Resumable v5-only sequential training queue. Historical v4 queues are never called."""
import time,subprocess,sys
from .common import *

def run():
    planned=[(seed,stage,source) for seed in [20260910,314159] for stage in ['N1','N2'] for source in ['44b6','6bba']]
    write(OUT/'native_queue_plan.json',dict(created=now(),runs=planned,head_updates=8000,backbone_updates=12000,
        context='full native two-frame volume and all predicted C0 competitors; five-frame graph decoder',
        source_only=True,seed2='preregistered N2 representation and P1_N2 replication, same recipe'))
    for seed,stage,source in planned:
        expected=[r['dataset'] for r in inventory() if r['embryo']==source]
        while any(not (OUT/'observations'/f'{n}.npz').exists() for n in expected):time.sleep(15)
        hours=sum(read(p)['seconds'] for p in (OUT/'models/native').glob('*.json') if not read(p).get('tiny'))/3600
        if hours>=48:raise RuntimeError('48 summed training-hour cap reached')
        key=f'{source}_{stage}_{seed}';dest=OUT/'models/native'/f'{key}.json'
        with (OUT/'logs'/f'{key}.log').open('a') as log:
            log.write(f'\nV5 queue invocation {now()}; resume supported\n');log.flush()
            if not dest.exists():
                subprocess.run([sys.executable,'-m','image_native_tracking_v5.train_native',source,stage,'--seed',str(seed)],check=True,stdout=log,stderr=subprocess.STDOUT)
            subprocess.run([sys.executable,'-m','image_native_tracking_v5.calibrate_native',source,stage,str(seed)],check=True,stdout=log,stderr=subprocess.STDOUT)
        print('finished',key,flush=True)
    write(OUT/'native_queue_complete.json',dict(created=now(),runs=planned,complete=True))

if __name__=='__main__':run()
