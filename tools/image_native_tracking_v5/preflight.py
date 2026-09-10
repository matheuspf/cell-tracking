"""Verify existing bytes and execute C0 image-only pilots in a fresh v5 root."""
import subprocess,time,sys,os
from .common import *

def run():
    begin=time.monotonic();rows=inventory();manifest=read(V4/'inference_package/base/manifest.json')
    checked={}
    for key,p in manifest['external_checkpoint_paths'].items():
        h=sha(p);assert h==manifest[key+'_weights_sha256'];checked[p]=h
    for rel,h in manifest['tracking_source_files'].items():
        assert sha(NATIVE/rel)==h;checked[str(NATIVE/rel)]=h
    for source,p in manifest['external_teacher_paths'].items():assert sha(p)==manifest['legacy_teacher_model_hashes'][source];checked[p]=sha(p)
    assert sha(manifest['notebook_source_path'])==manifest['notebook_sha256']
    for rel,h in manifest['package_files'].items():assert sha(V4/'inference_package/base'/rel)==h
    protected={str(p.relative_to(REPO)):sha(p) for root in [REPO/'results/multidata-training-v4',REPO/'results/strong-tracker-v3',REPO/'tools/multidata_training_v4'] for p in root.rglob('*') if p.is_file()}
    for p in [V3/'selected_prediction_lock.json',V4/'prediction_lock.json',V4/'checkpoint_manifest.json',V1/'inventory.json']:
        checked[str(p)]=sha(p)
    inputs=[];density=[]
    for r in rows:
        name=r['dataset'];ip=DATA/'train'/f'{name}.zarr';g=graph(name)
        meta=read(ip/'0/zarr.json');assert meta['shape']==r['image_shape']
        inputs.append(dict(dataset=name,shape=meta['shape'],image_metadata_sha256=sha(ip/'zarr.json'),
             array_metadata_sha256=sha(ip/'0/zarr.json'),gt_cache_sha256=sha(V1/'evaluation/gt'/f'{name}.npz'),
             c0_file_sha256=sha(V3/'selected_predictions'/f'{name}.npz'),estimated_total=r['estimated_total']))
        density.append(dict(dataset=name,embryo=r['embryo'],nodes=len(g['nodes']),density=len(g['nodes'])/np.prod(r['image_shape'])))
    chosen=[]
    for em in ['44b6','6bba']:
        sub=sorted([r for r in density if r['embryo']==em],key=lambda r:(r['density'],r['dataset']))
        chosen+=[dict(**sub[len(sub)//2],source='6bba' if em=='44b6' else '44b6')]
    free=shutil.disk_usage(ROOT).free/2**30
    write(OUT/'preflight.json',dict(created=now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        critical_hashes=checked,old_sanitized_hashes=protected,free_gib=free,new_output_cap_gib=min(24,.6*free,free-8),
        pilots=chosen,training_started=False,raw_hash_reuse='v3 input_hashes.json and v4 provenance receipts; metadata and GT cache rehashed',
        seconds=time.monotonic()-begin))
    write(OUT/'input_manifest.json',inputs)
    write(OUT/'density_selection.json',dict(criterion='C0 image-derived detections per voxel-time; no annotations or score selection',rows=density,pilots=chosen))
    from strong_tracker_v3.common import graph_hash
    records=[]
    for r in chosen:
        name=r['dataset'];image_dir=OUT/'pilots/images'/name;image_dir.mkdir(parents=True,exist_ok=True)
        link=image_dir/f'{name}.zarr'
        if not link.exists():link.symlink_to(DATA/'train'/f'{name}.zarr',target_is_directory=True)
        dest=OUT/'pilots/C0'/name
        if not dest.exists():
            start=time.monotonic()
            with (OUT/'logs'/f'pilot_C0_{name}.log').open('w') as log:
                subprocess.run([str(V4/'inference_package/base/run.sh'),'--python',sys.executable,'--images',str(image_dir),
                    '--output',str(dest),'--v1',str(V1),'--v2',str(V2),'--source-model',r['source']],stdout=log,stderr=subprocess.STDOUT,check=True)
            write(dest/'v5_timing.json',dict(seconds=time.monotonic()-start))
        fresh=arrays(dest/'predictions'/f'{name}.npz');old=graph(name)
        equal=graph_hash(fresh['nodes'],fresh['edges'])==graph_hash(old['nodes'],old['edges']);assert equal,name
        records.append(dict(dataset=name,exact_C0_graph_parity=equal,**read(dest/'v5_timing.json'),
            inference_receipt_sha256=sha(dest/'inference_receipt.json')))
        print('C0 fresh pilot',records[-1],flush=True)
    write(OUT/'C0_fresh_receipt.json',dict(created=now(),clips=records,complete=True))

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
