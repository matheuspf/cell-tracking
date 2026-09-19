"""Independent decoded-frame duplicate audit and retained v10 hash partition."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from .common import DATA, OLD, WORK, RESULTS, Blocked, now, sha, read, write


def audit(data=DATA):
    import numcodecs
    numcodecs.blosc.set_nthreads(1)
    folder=WORK/'decoded_inventory'
    folder.mkdir(exist_ok=True)
    started=time.monotonic()
    def one(path):
        output=folder/f'{path.stem}.json'
        frames=[]
        for t in range(100):
            chunk=path/f'0/c/{t}/0/0/0'
            packed=chunk.read_bytes()
            raw=numcodecs.blosc.decompress(packed)
            if len(raw)!=64*256*256*2:
                raise Blocked('Decoded frame size mismatch')
            frames.append(dict(t=t,compressed_sha256=hashlib.sha256(packed).hexdigest(),
                               decoded_sha256=hashlib.sha256(raw).hexdigest()))
        row=dict(dataset=path.stem,frames=frames,metadata_sha256=sha(path/'0/zarr.json'))
        write(output,row,immutable=True)
        return row
    paths=sorted((data/'train').glob('*.zarr'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one,paths):
            rows.append(row)
            if len(rows)%20==0:print(f'Decoded image audit {len(rows)}/199',flush=True)
    parent={r['dataset']:r['dataset'] for r in rows}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={};duplicates=[]
    for row in rows:
        name=row['dataset']
        for f in row['frames']:
            h=f['decoded_sha256']
            if h in seen:
                other,t=seen[h]
                if other[:4]!=name[:4]:raise Blocked('Cross-embryo decoded-frame duplicate')
                if other!=name:
                    a,b=sorted([root(name),root(other)]);parent[b]=a
                    duplicates.append([other,t,name,f['t'],h])
            else:seen[h]=(name,f['t'])
    groups=defaultdict(list)
    for n in parent:groups[root(n)].append(n)
    sources={s:dict(fit=[],calibration=[],groups=[]) for s in ('44b6','6bba')}
    for members in sorted(groups.values()):
        members=sorted(members)
        h=hashlib.sha256(json.dumps([20260918,members],sort_keys=True).encode()).hexdigest()
        part='fit' if int(h,16)*5<4*2**256 else 'calibration'
        sources[members[0][:4]][part].extend(members)
        sources[members[0][:4]]['groups'].append(dict(sha256=h,clips=members,partition=part))
    original=read(WORK/'source_partitions.json')
    for s in sources:
        for part in ('fit','calibration'):
            sources[s][part].sort()
            if sources[s][part]!=original[s][part]:
                raise Blocked('Decoded grouping does not reproduce inherited split')
    write(WORK/'duplicate_frame_pairs.json',duplicates,immutable=True)
    record=dict(status='passed',created_utc=now(),frames=19900,clips=199,
                sources=sources,duplicate_pairs=len(duplicates),cross_embryo_duplicates=0,
                decoded_manifest_hashes={p.name:sha(p) for p in sorted(folder.glob('*.json'))},
                bytecode_method_reference=sha(OLD.parents[1]/'tools/clean_validation_v10/__pycache__/splits.cpython-312.pyc')
                    if (OLD.parents[1]/'tools/clean_validation_v10/__pycache__/splits.cpython-312.pyc').exists() else None,
                method='sha256(json.dumps([20260918, sorted(group)], sort_keys=True)); integer*5 < 4*2**256',
                acquisition_independence_certified=False,wall_seconds=time.monotonic()-started)
    write(RESULTS/'split_manifest.json',record,immutable=True)
    return record
