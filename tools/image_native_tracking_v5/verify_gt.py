"""Fresh, small GEFF verification against the preserved evaluator inputs."""
import zarr
from .common import *

def run():
    rows=[]
    for r in inventory():
        name=r['dataset'];path=DATA/'train'/f'{name}.geff';g=zarr.open_group(str(path),mode='r')
        nodes=np.column_stack([np.asarray(g['nodes/ids']),*[np.asarray(g[f'nodes/props/{a}/values']) for a in 'tzyx']])
        edges=np.asarray(g['edges/ids']);cached=arrays(V1/'evaluation/gt'/f'{name}.npz')
        assert np.array_equal(nodes,cached['nodes']) and np.array_equal(edges,cached['edges']),name
        estimate=read(path/'zarr.json')['attributes']['geff']['extra']['estimated_number_of_nodes'];assert estimate==r['estimated_total']
        files={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
        rows.append(dict(dataset=name,raw_geff_content_sha256=digest(files),files=len(files),estimated_total=estimate,exact_cached_arrays=True,
            evaluator_cache_sha256=sha(V1/'evaluation/gt'/f'{name}.npz')))
    destination=OUT/'raw_gt_verification.json'
    if destination.exists():assert read(destination)['rows']==rows
    else:write(destination,dict(created=now(),clips=len(rows),raw_geff_read_only=True,rows=rows))
    print('Raw GT arrays/estimates verified',len(rows),flush=True)

if __name__=='__main__':run()
