from pathlib import Path
import os, json, hashlib, shutil
from datetime import datetime, timezone
import numpy as np
from contextlib import contextmanager

@contextmanager
def cpu_batch():
    """Serialize process pools across families; at most ten pool workers overall."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'cpu_batch.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

@contextmanager
def gpu_aux():
    """Serialize auxiliary inference/calibration alongside bounded native fits."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'gpu_aux.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('V5_STUDIES', '/kaggle/working/cell-tracking'))
OUT = Path(os.environ.get('V5_OUTPUT', str(ROOT/'image-native-tracking-v5')))
V1, V2, V3, V4 = [ROOT/n for n in ('annotation-selection-v1','strong-tracker-v2','strong-tracker-v3','multidata-training-v4')]
V1=Path(os.environ.get('V5_V1',str(V1)));V2=Path(os.environ.get('V5_V2',str(V2)))
V3=Path(os.environ.get('V5_V3',str(V3)));V4=Path(os.environ.get('V5_V4',str(V4)))
DATA = Path(os.environ.get('V5_DATA','/kaggle/input/competitions/biohub-cell-tracking-during-development'))
NATIVE = V1/'public_harmonic_full/tracking_repo'
WORK = REPO/'work/image-native-tracking-v5'
BASE = {'pooled':0.934802374260586,'44b6':0.931664468721842,'6bba':0.935221784097327}
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def reserve():
    if shutil.disk_usage(ROOT).free < 8*2**30: raise RuntimeError('V5 8 GiB reserve reached')
def durable_replace(temporary,destination):
    """Flush output bytes and their renamed directory entry before acknowledging."""
    temporary=Path(temporary);destination=Path(destination)
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    temporary.replace(destination)
    directory=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
def write(p,x):
    reserve(); p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v))
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=default,allow_nan=False)+'\n');durable_replace(q,p)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(p,**kw):
    reserve();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp.npz')
    np.savez_compressed(q,**kw);durable_replace(q,p)
def inventory():return read(V1/'inventory.json')
def graph(name,variant='C0'):
    b=arrays(V3/'selected_predictions'/f'{name}.npz')
    if variant=='C0':return b
    d=arrays(OUT/'deltas'/variant/f'{name}.npz')
    n=b['nodes'];e=set(map(tuple,b['edges']))
    if 'removed_nodes' in d:n=n[~np.isin(n[:,0],d['removed_nodes'])]
    if 'added_nodes' in d:n=np.concatenate([n,d['added_nodes']])
    e.difference_update(map(tuple,d['removed_edges']));e.update(map(tuple,d['added_edges']))
    return dict(nodes=n,edges=np.asarray(sorted(e),np.int64).reshape(-1,2))
def save_delta(name,variant,nodes,edges,output_root=None):
    b=graph(name);old=set(map(tuple,b['edges']));new=set(map(tuple,edges))
    oldids=set(b['nodes'][:,0]);newids=set(nodes[:,0])
    destination=OUT if output_root is None else Path(output_root)
    save(destination/'deltas'/variant/f'{name}.npz',added_edges=np.array(sorted(new-old),np.int64).reshape(-1,2),
         removed_edges=np.array(sorted(old-new),np.int64).reshape(-1,2),
         added_nodes=nodes[np.array([n[0] not in oldids for n in nodes])],removed_nodes=np.array(sorted(oldids-newids),np.int64))
