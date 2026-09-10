"""Score complete frozen variants on one bounded CPU lane while decoding continues."""
import fcntl,time
from .common import *


def run():
    from .evaluate import run as evaluate,aggregate_complete
    handle=(OUT/'auto_evaluate.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=list(read(OUT/'execution_protocol.json')['variants']);rows=inventory();completed=[]
    while True:
        pending=[v for v in expected if any(not (OUT/'evaluation'/v/f"{r['dataset']}.json").exists() for r in rows)]
        if not pending:break
        ready=[v for v in pending if v!='C0' and not v.startswith('Oracle_') and
            all((OUT/'prediction_receipts'/v/f"{r['dataset']}.json").exists() for r in rows)]
        if not ready:time.sleep(30);continue
        for variant in ready:
            try:evaluate([variant],serial=True,nonblocking=True)
            except BlockingIOError:continue
            completed.append(variant)
            print('Scored complete registered variant',variant,now(),flush=True)
            write(OUT/'auto_evaluation_progress.json',dict(at=now(),completed=completed,
                per_variant_exclusion=True,extra_serial_workers=1))
            break
        else:time.sleep(30)
    aggregate_complete()
    write(OUT/'auto_evaluation_complete.json',dict(at=now(),complete=True,registered_variants=expected,
        extra_serial_workers=1,completed_in_this_process=completed,
        numerical_scorer_unchanged=True,original_batch_queues_preserved=True))


if __name__=='__main__':run()
