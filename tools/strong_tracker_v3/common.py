"""Pure shared graph primitives. No legacy output globals or import-time writes."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple,set)): return [clean(x) for x in o]
    if isinstance(o,np.ndarray): return clean(o.tolist())
    if isinstance(o,np.generic): return clean(o.item())
    if isinstance(o,float) and not np.isfinite(o): return None
    if isinstance(o,Path): return str(o)
    return o
def digest(o): return hashlib.sha256(json.dumps(clean(o),sort_keys=True,allow_nan=False).encode()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text())
def write_json(p,o,immutable=False):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); o=clean(o)
    if immutable and p.exists():
        if read_json(p)!=o: raise RuntimeError(f'Locked artifact drift: {p}')
        return
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def load_graph(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def save_arrays(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(p)
def save_graph(p,nodes,edges,**arrays):
    save_arrays(p,nodes=np.asarray(nodes,np.int64).reshape(-1,5),edges=np.asarray(edges,np.int64).reshape(-1,2),**arrays)
def graph_hash(nodes,edges):
    h=hashlib.sha256()
    for x in [nodes,edges]:
        a=np.ascontiguousarray(x,dtype='<i8');h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def adjacency(nodes,edges):
    ix={int(n):i for i,n in enumerate(nodes[:,0])};pred=[[] for _ in nodes];succ=[[] for _ in nodes]
    for a,b in edges:
        i,j=ix[int(a)],ix[int(b)];succ[i].append(j);pred[j].append(i)
    return ix,pred,succ
def validate(nodes,edges,shape,reference=None,allow_legacy_bounds=False):
    n,e=np.asarray(nodes),np.asarray(edges)
    if n.ndim!=2 or n.shape[1]!=5 or e.ndim!=2 or e.shape[1]!=2: raise ValueError('Invalid array shape')
    if not np.isfinite(n).all() or not np.equal(n,np.rint(n)).all():raise ValueError('Noninteger coordinates')
    if not np.isfinite(e).all() or not np.equal(e,np.rint(e)).all():raise ValueError('Noninteger edges')
    if len(np.unique(n[:,0]))!=len(n):raise ValueError('Duplicate node IDs')
    if len(set(map(tuple,e)))!=len(e):raise ValueError('Duplicate edges')
    ix,pred,succ=adjacency(n,e)
    if any(len(x)>1 for x in pred):raise ValueError('Forbidden merge')
    if any(len(x)>2 for x in succ):raise ValueError('Too many daughters')
    if any(n[ix[int(b)],1]!=n[ix[int(a)],1]+1 for a,b in e):raise ValueError('Nonconsecutive edge')
    if not ((n[:,1]>=0)&(n[:,1]<shape[0])).all():raise ValueError('Time out of bounds')
    outside=np.any((n[:,2:]<0)|(n[:,2:]>=np.array(shape[1:])),axis=1)
    if outside.any() and not allow_legacy_bounds:raise ValueError('Coordinates out of bounds')
    return dict(nodes=len(n),edges=len(e),out_of_bounds=int(outside.sum()),forks=sum(len(s)==2 for s in succ),valid=True)
def run_pool(fn,tasks,workers=4):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(fn,t):t for t in tasks}
        for i,f in enumerate(as_completed(jobs),1):
            r=f.result();print(f'{fn.__name__} {i}/{len(jobs)} {str(r)[:180]}',flush=True);yield r
def code_hashes(ctx):
    return {str(p.relative_to(ctx.repo)):sha(p) for p in sorted((ctx.repo/'tools/strong_tracker_v3').glob('*.py'))}
