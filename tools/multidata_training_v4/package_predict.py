"""Standalone package entry. Install guards before numerical dependencies."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--output',required=True)
    p.add_argument('--v1',required=True);p.add_argument('--v2',required=True);p.add_argument('--source-model',choices=['44b6','6bba'],required=True)
    p.add_argument('--variant');p.add_argument('--disable-new-heads',action='store_true');args=p.parse_args()
    package=Path(__file__).resolve().parents[2];output=Path(args.output).resolve();images=Path(args.images).resolve()
    if output.exists():raise ValueError('Use a fresh output directory')
    for protected in [package,images,Path(args.v1).resolve(),Path(args.v2).resolve(),*[p.resolve() for p in images.glob('*.zarr')]]:
        if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):raise ValueError('Output overlaps protected inputs')
    selected=json.loads((package/'winning_config.json').read_text())['variant'];variant='C0' if args.disable_new_heads else (args.variant or selected)
    # The preserved base has its own process-start annotation guard and source hashes.
    start=time.monotonic();output.mkdir(parents=True)
    os.environ['V4_NETWORK_AUDIT_DIR']=str(output/'network_audit')
    import sitecustomize
    assert sitecustomize.NETWORK_GUARD_INSTALLED
    sitecustomize._record()
    with (output/'base.log').open('w') as log:
        subprocess.run([str(package/'base/run.sh'),'--python',sys.executable,'--images',str(images),'--output',str(output/'base'),
            '--v1',args.v1,'--v2',args.v2,'--source-model',args.source_model],check=True,stdout=log,stderr=subprocess.STDOUT)
    sys.path.insert(0,str(package/'tools'))
    from multidata_training_v4.infer import transform,GUARD,EXTERNAL_GUARD
    from multidata_training_v4.common import read,sha,arrays,save,write
    from strong_tracker_v3.common import validate,graph_hash
    import zarr
    import torch
    import resource
    manifest=read(package/'manifest.json')
    for rel,h in manifest['package_files'].items():assert sha(package/rel)==h,'Package file changed: '+rel
    calibration=read(package/'calibration.json');results=[]
    columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id'];index=0
    with (output/'submission.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for image_path in sorted(images.glob('*.zarr')):
            name=image_path.stem;g=arrays(output/'base/predictions'/f'{name}.npz');meta=read(image_path/'zarr.json')
            spacing=meta['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][-3:]
            image=zarr.open_group(image_path,mode='r')['0'];shape=image.shape
            # Identity fallback needs no new model checkpoint or external dataset.
            n,e,receipt=transform(g['nodes'],g['edges'],image,spacing,args.source_model,variant,package/'models',calibration)
            valid=validate(n,e,shape);save(output/'predictions'/f'{name}.npz',nodes=n,edges=e)
            results.append(dict(dataset=name,graph_hash=graph_hash(n,e),**valid,**receipt))
            for node in n:
                row=dict.fromkeys(columns,-1);row.update(id=index,dataset=name,row_type='node',**dict(zip(columns[3:8],map(int,node))))
                writer.writerow(row);index+=1
            for a,b in e:
                row=dict.fromkeys(columns,-1);row.update(id=index,dataset=name,row_type='edge',source_id=int(a),target_id=int(b));writer.writerow(row);index+=1
    network=[read(p) for p in sorted((output/'network_audit').glob('*.json'))]
    assert network and all(r['guard_installed'] and not r['blocked_attempts'] and not r['blocked_external_data_reads'] for r in network)
    write(output/'inference_receipt.json',dict(variant=variant,source_model=args.source_model,graphs=results,seconds=time.monotonic()-start,
        base_receipt=read(output/'base/inference_receipt.json'),annotation_reads=0,blocked_annotation_reads=GUARD['blocked_reads'],
        external_dataset_reads=0,external_guard=EXTERNAL_GUARD,annotation_guard=GUARD,
        offline_network_guard=dict(processes_recorded=len(network),blocked_attempts=0,scope='Python socket audit hooks; loopback and Unix IPC allowed'),
        new_head_peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        parent_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        child_peak_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20,
        submission_sha256=sha(output/'submission.csv'),package_manifest_sha256=sha(package/'manifest.json')))

if __name__=='__main__':main()
