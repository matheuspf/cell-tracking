"""Advance fixed second-seed controls before primary comparative outcomes."""
import subprocess
import sys
import time
from .common import *

def one(source):
    from .train import fit
    for component,steps in [('D',12000),('G',20000),('I',20000)]:
        fit(f'{component}_real_{source}_seed2',component,[source],steps,
            init=f'D_real_{source}_seed2' if component=='I' else None,seed=314159)

def run():
    from .train import fit
    started=time.monotonic()
    write(OUT/'seed2_bootstrap_lock.json',dict(created=now(),primary_comparative_outcomes_opened=False,
        seed=314159,settings_predeclared=True,reason='Use free GPU capacity after source44 primary adaptation',
        sources=['44b6','6bba'],pretrain_updates={'D':12000,'G':20000,'I':20000},
        synthetic_D_note='Shared static seed2 checkpoint; unused if the later eligible external arm is geometry-only C3'))
    fit('D_synthetic_seed2','D',['synthetic'],12000,seed=314159)
    processes=[];streams=[]
    for source in ['44b6','6bba']:
        stream=(OUT/'logs'/f'seed2_bootstrap_{source}.log').open('a');streams.append(stream)
        processes.append(subprocess.Popen([sys.executable,'-m','multidata_training_v4.seed2_bootstrap',source],stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for stream in streams:stream.close()
    write(OUT/'seed2_bootstrap_receipt.json',dict(completed=not any(codes),returncodes=codes,seconds=time.monotonic()-started,
        future_second_seed_stage_will_verify_and_reuse=True))
    assert not any(codes),codes

if __name__=='__main__':one(sys.argv[1])
