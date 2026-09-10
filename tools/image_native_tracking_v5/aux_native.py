"""Run one already registered native fit on the bounded auxiliary GPU lane."""
import subprocess,sys
from .common import *

def run(source,stage,seed):
    key=f'{source}_{stage}_{seed}'
    assert [seed,stage,source] in read(OUT/'native_queue_plan.json')['runs']
    with gpu_aux(),(OUT/'logs'/f'{key}_auxiliary.log').open('a') as log:
        log.write(f'Already registered fit on auxiliary lane: {now()}\n');log.flush()
        subprocess.run([sys.executable,'-m','image_native_tracking_v5.train_native',source,stage,'--seed',str(seed)],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    write(OUT/f'auxiliary_{key}_complete.json',dict(at=now(),complete=True,registered_fit=True,
        checkpoint_sha256=sha(OUT/'models/native'/f'{key}.pt'),source_calibration='main queue completes the existing calibration stage'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',choices=['44b6','6bba']);p.add_argument('stage',choices=['N1','N2']);p.add_argument('--seed',type=int,default=20260910);a=p.parse_args();run(a.source,a.stage,a.seed)
