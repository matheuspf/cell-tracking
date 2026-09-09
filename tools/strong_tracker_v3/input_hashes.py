"""Read-only content fingerprints of image chunks and neural inputs."""
from .common import *

def one(task):
    ctx,s=task;name=s['dataset'];root=Path(s['image_path']);dest=ctx.out/'input_hashes'/f'{name}.json'
    current={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}
    if dest.exists():
        old=read_json(dest)
        if old['file_stats']!=current:raise ValueError('Image content inventory changed')
        return name+' verified stat inventory'
    hashes={k:sha(root/k) for k in sorted(current)}
    native=ctx.full/'inputs'/f'pre_ilp_{name}.npz'
    write_json(dest,dict(dataset=name,files=hashes,file_stats=current,image_content_sha256=digest(hashes),
        image_bytes=sum(x[0] for x in current.values()),native_sha256=sha(native),raw_sha256=sha(ctx.v2/'raw'/f'{name}.npz'),
        incumbent_sha256=sha(ctx.incumbent(name))))
    return name+' image hashed'

def run(ctx,args):
    list(run_pool(one,[(ctx,s) for s in ctx.samples()],args.workers))
    records=[read_json(ctx.out/'input_hashes'/f'{s["dataset"]}.json') for s in ctx.samples()]
    write_json(ctx.out/'label_free_input_hash_manifest.json',dict(created=now(),samples=len(records),
        image_bytes=sum(r['image_bytes'] for r in records),
        inputs={r['dataset']:{k:v for k,v in r.items() if k not in ['files','file_stats','dataset']} for r in records},
        manifest_sha256=digest({r['dataset']:r['image_content_sha256'] for r in records})))

if __name__=='__main__':
    from .context import RunContext
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1)
    run(RunContext.default(),p.parse_args())
