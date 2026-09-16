"""One spare CPU lane for the existing frozen evaluation-only diagnostics.

Separate output roots avoid races with the already queued four-worker census.
Completed files are linked without replacing any existing receipt or graph.
"""
import fcntl,time
import pandas as pd
import psutil
from .common import *

def run():
    from .headroom import one
    from .evaluate import run as evaluate
    handle=(OUT/'early_headroom.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shadow=OUT/'early_headroom_work';results=[];start=now()
    for i,row in enumerate(inventory()):
        name=row['dataset'];dest=OUT/'headroom'/f'{name}.json'
        if dest.exists():result=read(dest)
        else:
            # The initial six-worker control batch leaves this one CPU lane free.
            # Wait if a larger pool has since started; no new pool is created here.
            while True:
                worker_ids=set()
                for p in psutil.process_iter(['cmdline']):
                    try:
                        if any(x in (p.info['cmdline'] or []) for x in ['image_native_tracking_v5.predict_batch','image_native_tracking_v5.headroom','image_native_tracking_v5.evaluate']):
                            worker_ids.update(c.pid for c in p.children() if c.status()!=psutil.STATUS_ZOMBIE)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                if len(worker_ids)<=6:break
                time.sleep(15)
            result=one(row,shadow)
            names=[f'deltas/{v}/{name}.npz' for v in ['Oracle_fixed','Oracle_augmented']]
            names += [f'oracle_events/{name}.json',f'headroom/{name}.json']
            for relative in names:
                source=shadow/relative;target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
                try:os.link(source,target)
                except FileExistsError:
                    if source.suffix=='.npz':
                        a,b=arrays(source),arrays(target);assert a.keys()==b.keys()
                        assert all(np.array_equal(a[k],b[k]) for k in a),relative
                    else:
                        a,b=read(source),read(target)
                        for key in ['seconds']:a.pop(key,None);b.pop(key,None)
                        assert a==b,relative
        results.append(result)
        if i%10==0:print('Single-worker coverage',i+1,199,flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(OUT/'coverage_rows.csv',index=False)
    write(OUT/'early_headroom_receipt.json',dict(started=start,at=now(),complete=True,clips=199,extra_workers=1,
        configurations_unchanged=True,original_queue_reuses_same_receipts=True,separate_output_namespace=True,
        existing_files_never_replaced=True,GT_used_only_for_diagnostics=True))
    for variant in ['Oracle_fixed','Oracle_augmented']:evaluate([variant],serial=True)

if __name__=='__main__':run()
