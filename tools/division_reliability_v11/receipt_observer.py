"""Preserve completed receipts from already-running older queue processes.

New queue processes archive previous executions themselves. This observer also
covers existing parents that imported the former controller before that repair.
It reads only controller metadata and never changes a running job or model.
"""
import fcntl
import hashlib
import json
import os
import time
from .common import WORK,Blocked,read,write,now


def snapshot(folder):
    saved=0
    for path in folder.glob('*.json'):
        if path.name.endswith('.resources.json'):continue
        blob=path.read_bytes();record=json.loads(blob)
        if record.get('status') not in ('complete','failed','blocked'):continue
        destination=folder/'history/observed-executions'/hashlib.sha256(blob).hexdigest()
        if (destination/path.name).exists():continue
        resource_path=path.with_name(path.stem+'.resources.json')
        if resource_path.exists():
            sample=read(resource_path)
            # A replacement job may already have published a newer sample.
            # Never attach that newer process's resource measurement here.
            if sample.get('utc','')<=record.get('finished_utc',''):
                write(destination/resource_path.name,sample,immutable=True)
        write(destination/path.name,record,immutable=True);saved+=1
    return saved


def run():
    from .report import alive
    root=WORK/'controller';saved=0
    with (root/'receipt-observer-owner.lock').open('a+') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise Blocked('A receipt observer already owns this study') from exc
        while True:
            saved+=snapshot(root/'jobs')
            active=False
            for name in ('pipeline','upstream','source-prefetch-queue'):
                path=root/(name+'.json')
                if path.exists():
                    r=read(path)
                    active |= any(alive(r.get(k)) for k in ('controller_pid','worker_pid'))
            write(root/'receipt-observer.json',dict(status='running' if active else 'complete',
                controller_pid=os.getpid() if active else None,receipts_saved_this_run=saved,updated_utc=now()))
            if not active:return
            time.sleep(2)


if __name__=='__main__':run()
