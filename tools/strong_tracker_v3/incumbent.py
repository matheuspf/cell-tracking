"""Read-only sealed incumbent verification and explicit input preparation."""
import importlib.util
import platform
import subprocess
from pathlib import Path
from .common import *

def snapshot(ctx):
    p=ctx.out/'preservation_before.json'
    if p.exists():return read_json(p)
    files={}
    for label,root in [('v1',ctx.v1),('v2',ctx.v2)]:
        files[label]={str(x.relative_to(root)):[x.stat().st_size,x.stat().st_mtime_ns]
            for x in root.rglob('*') if x.is_file()}
    result=dict(created=now(),files=files,lock_sha256=sha(ctx.v2/'selected_prediction_lock.json'))
    write_json(p,result,immutable=True);return result

def preserve_check(ctx):
    old=read_json(ctx.out/'preservation_before.json')
    changes=[]
    for label,root in [('v1',ctx.v1),('v2',ctx.v2)]:
        current={str(x.relative_to(root)):[x.stat().st_size,x.stat().st_mtime_ns] for x in root.rglob('*') if x.is_file()}
        for k in current.keys()|old['files'][label].keys():
            if current.get(k)!=old['files'][label].get(k):changes.append(f'{label}/{k}')
    if changes:raise RuntimeError(f'Sealed store changed: {changes[:10]}')
    lock=read_json(ctx.v2/'selected_prediction_lock.json')
    if sha(ctx.v2/'selected_prediction_lock.json')!=old['lock_sha256']:raise RuntimeError('Incumbent lock changed')
    for k,v in lock['hashes'].items():
        if sha(ctx.incumbent(k))!=v:raise RuntimeError(f'Incumbent changed: {k}')
    result=dict(unchanged=True,files=sum(len(x) for x in old['files'].values()),incumbent_hashes_checked=len(lock['hashes']),checked=now())
    write_json(ctx.out/'preservation_check.json',result);return result

def run(ctx,args=None):
    ctx.check_outputs();ctx.out.mkdir(parents=True,exist_ok=True);ctx.work.mkdir(parents=True,exist_ok=True)
    snapshot(ctx)
    spec=importlib.util.spec_from_file_location('v3_preflight',ctx.repo/'handover/strong-tracker-v3/preflight.py')
    pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)
    receipt=pre.inspect_cache(ctx.v2)
    lock=read_json(ctx.v2/'selected_prediction_lock.json');names=sorted(lock['hashes'])
    if len(names)!=199:raise ValueError('Expected 199 clips')
    notebook=ctx.full/'harmonic_isolated.py'
    if sha(notebook)!=lock['notebook_sha256']:raise ValueError('Notebook hash mismatch')
    eval_rows={r['dataset']:r for r in ctx.eval_rows()}
    if set(eval_rows)!=set(names):raise ValueError('Evaluation coverage mismatch')
    inputs=[];details=[]
    for name in names:
        path=ctx.data/'train'/f'{name}.zarr';meta=read_json(path/'zarr.json');am=read_json(path/'0/zarr.json')
        scales=meta['attributes'].get('ome',meta['attributes'])['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][-3:]
        row=dict(dataset=name,embryo=name.split('_')[0],image_shape=am['shape'],physical_scale=scales,
            image_path=str(path),metadata_hash=digest([meta,am]))
        if row['image_shape']!=eval_rows[name]['image_shape'] or scales!=eval_rows[name]['physical_scale']:
            raise ValueError('Actual image metadata differs from scoring inventory')
        inputs.append(row);details.append(dict(**row,graph_sha256=sha(ctx.incumbent(name)),
            estimated_total=eval_rows[name]['estimated_total'],gt_sha256=sha(ctx.v1/'evaluation/gt'/f'{name}.npz')))
    write_json(ctx.out/'inputs.json',inputs,immutable=True)
    write_json(ctx.out/'expected_samples.json',names,immutable=True)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA study interpreter required')
    environment=dict(python=platform.python_version(),interpreter=__import__('sys').executable,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        paths={k:str(getattr(ctx,k)) for k in ['repo','data','v1','v2','out','work','official']},
        git_revision=subprocess.check_output(['git','-C',str(ctx.repo),'rev-parse','HEAD'],text=True).strip(),
        source=code_hashes(ctx))
    write_json(ctx.out/'environment.json',environment)
    revision=subprocess.check_output(['git','-C',str(ctx.official),'rev-parse','HEAD'],text=True).strip()
    if revision!=ctx.metric_revision:raise RuntimeError('Official metric checkout revision changed')
    metric_files={str(p.relative_to(ctx.official)):sha(p) for p in sorted((ctx.official/'src/tracking_cellmot').glob('*.py'))}
    write_json(ctx.out/'incumbent_manifest.json',dict(created=now(),preflight=receipt,samples=details,
        notebook_sha256=sha(notebook),metric_revision=revision,metric_files=metric_files,
        input_manifest_sha256=sha(ctx.out/'inputs.json'),validation='hashes verified; independent score pending'))
    print(f'Incumbent 199 hashes verified; {environment["gpu"]}; label-free manifest written',flush=True)
