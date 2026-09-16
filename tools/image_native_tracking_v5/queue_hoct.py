import subprocess,sys,time
from .common import *

def run():
    for seed in [20260910,314159]:
        for source in ['44b6','6bba']:
            names=[r['dataset'] for r in inventory() if r['embryo']==source]
            while any(not (OUT/'hoct_training_features'/f'{n}.npz').exists() for n in names):time.sleep(15)
            with gpu_aux(),(OUT/'logs'/f'{source}_H_probe_{seed}.log').open('a') as log:
                subprocess.run([sys.executable,'-m','image_native_tracking_v5.train_hoct','fit','--source',source,'--seed',str(seed)],check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run([sys.executable,'-m','image_native_tracking_v5.calibrate',source,str(seed)],check=True,stdout=log,stderr=subprocess.STDOUT)
    write(OUT/'hoct_queue_complete.json',dict(created=now(),complete=True))

if __name__=='__main__':run()
