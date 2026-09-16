"""Use each source's vacated training lane for its already registered replica.

Per-fit locks coordinate with the original queue. Calibration remains in that
queue and shares the single auxiliary GPU lane with inference. No recipes change.
"""
import fcntl,subprocess,sys,time
from .common import *

def run(source):
    handle=(OUT/f'source_lane_{source}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (OUT/'models/native'/f'{source}_N2_20260910.json').exists():time.sleep(15)
    planned=read(OUT/'native_queue_plan.json')['runs'];finished=[]
    for stage in ['N1','N2']:
        seed=314159;assert [seed,stage,source] in planned
        key=f'{source}_{stage}_{seed}'
        with (OUT/'logs'/f'{key}_source_lane.log').open('a') as log:
            log.write(f'Registered replica, source training lane: {now()}\n');log.flush()
            subprocess.run([sys.executable,'-m','image_native_tracking_v5.train_native',source,stage,'--seed',str(seed)],
                check=True,stdout=log,stderr=subprocess.STDOUT)
        finished.append(key)
    write(OUT/f'source_lane_{source}_complete.json',dict(at=now(),complete=True,registered_fits=finished,
        per_fit_exclusion=True,calibration='original native queue, serialized auxiliary GPU lane'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',choices=['44b6','6bba']);a=p.parse_args();run(a.source)
