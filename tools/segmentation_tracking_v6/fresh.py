"""Two actual complete image-to-C0 executions using the immutable local package.

The inherited early audit hook is installed by sitecustomize in parent/worker
processes. Python reads/sockets are denied; this is not a Linux namespace jail.
All transient files stay in a new RAM-backed v6 directory.
"""
import csv
import os
import subprocess
import sys
import time
import numpy as np
from .common import *
from strong_tracker_v3.common import validate

def run():
    if not (OUT/'preflight.json').exists(): raise Blocked('Run preflight first')
    package=V5/'inference_package_validation'
    manifest=read_json(package/'manifest.json')
    for rel,expected in manifest['files'].items(): assert sha(package/rel)==expected,rel
    preflight=read_json(OUT/'preflight.json')
    pilots=[next(r for r in preflight['pilots'] if r['embryo']==em) for em in ['44b6','6bba']]
    records=[]
    for i,item in enumerate(pilots):
        destination=OUT/'fresh'/f'clip_{i+1}.json'
        if destination.exists():
            previous=read_json(destination)
            if previous.get('passed'): records.append(previous);continue
        reserve(SCRATCH,additional=2*2**30)
        alias=f'unknown_volume_{i+1:02d}'
        root=SCRATCH/'fresh'/f'{alias}_{time.time_ns()}'
        view=root/'images';view.mkdir(parents=True)
        (view/f'{alias}.zarr').symlink_to(DATA/'train'/f"{item['dataset']}.zarr",target_is_directory=True)
        output=root/'output';logpath=root/'run.log'
        source='6bba' if item['embryo']=='44b6' else '44b6'
        env={**os.environ,'V5_PACKAGE_ROOT':str(package),'V5_FRESH_OUTPUT':str(output),
             'V5_AUDIT_DIR':str(root/'audit'),'V5_V1':str(V1),'V5_V2':str(V2),
             'TMPDIR':str(root),'MPLCONFIGDIR':str(root/'matplotlib'),
             'PYTHONPATH':str(package/'tools')}
        # A real denial probe is separate from successful inference audit rows.
        probe = """import sitecustomize, socket, sys
from pathlib import Path
assert sitecustomize.INSTALLED_BEFORE_NUMERICAL
for p in sys.argv[1:]:
    try: Path(p).read_bytes()
    except PermissionError: pass
    else: raise AssertionError('Forbidden path was readable')
try: socket.getaddrinfo('example.com',443)
except PermissionError: pass
else: raise AssertionError('External DNS was permitted')
print('guard_denials_passed')
"""
        check=subprocess.run([sys.executable,'-c',probe,
            str(V1/'evaluation/gt'/f"{item['dataset']}.npz"),
            str(V3/'selected_predictions'/f"{item['dataset']}.npz"),
            str(V5/'observations'/f"{item['dataset']}.npz")],env={**env,'V5_AUDIT_DIR':str(root/'guard_test')},
            cwd='/tmp',text=True,capture_output=True)
        if check.returncode: raise RuntimeError(check.stderr)
        command=[str(package/'base/run.sh'),'--python',sys.executable,'--images',str(view),
                 '--output',str(output),'--v1',str(V1),'--v2',str(V2),'--source-model',source]
        start=time.monotonic()
        print('fresh full image C0',i+1,'started',flush=True)
        with logpath.open('w') as log:
            proc=subprocess.Popen(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
            import psutil
            peak_rss=0.;peak_gpu=0.
            while proc.poll() is None:
                p=psutil.Process(proc.pid)
                processes=[p]+p.children(recursive=True)
                rss=sum(q.memory_info().rss for q in processes if q.is_running())/2**30
                gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())/1024
                peak_rss=max(peak_rss,rss);peak_gpu=max(peak_gpu,gpu)
                if rss>24 or gpu>20 or time.monotonic()-start>24*3600:
                    proc.terminate();raise Blocked('v6 fresh worker resource ceiling reached')
                time.sleep(2)
        record=dict(embryo=item['embryo'],dataset=item['dataset'],unfamiliar_name=alias,source=source,
                    seconds=time.monotonic()-start,returncode=proc.returncode,
                    peak_process_tree_rss_gib=peak_rss,peak_total_gpu_gib=peak_gpu,
                    package_manifest_sha256=sha(package/'manifest.json'),scratch=str(root),
                    real_guard_denial_tests=True,learned_segmenter_executed=False,ultrack_executed=False)
        if proc.returncode:
            record.update(passed=False,error=logpath.read_text()[-5000:]);write(destination,record)
            raise RuntimeError(record['error'])
        actual=load_graph(output/'predictions'/f'{alias}.npz');expected=c0(item['dataset'])
        with (output/'submission.csv').open() as handle: csvrows=list(csv.DictReader(handle))
        nodes=np.asarray([[int(r[k]) for k in ['node_id','t','z','y','x']] for r in csvrows if r['row_type']=='node'],np.int64).reshape(-1,5)
        edges=np.asarray([[int(r[k]) for k in ['source_id','target_id']] for r in csvrows if r['row_type']=='edge'],np.int64).reshape(-1,2)
        assert {r['dataset'] for r in csvrows}=={alias}
        assert [int(r['id']) for r in csvrows]==list(range(len(csvrows)))
        exact=graph_hash(**actual)==graph_hash(**expected)
        csv_parity=graph_hash(nodes,edges)==graph_hash(**actual)
        audits=[read_json(p) for p in (root/'audit').glob('*.json')]
        audit_pass=bool(audits) and all(r['installed_before_numerical'] and not r['blocked_events'] for r in audits)
        record.update(exact_graph_parity=exact,csv_roundtrip=csv_parity,audit_processes=len(audits),
            audit_passed=audit_pass,passed=exact and csv_parity and audit_pass,
            graph_hash=graph_hash(**actual),graph=validate(nodes,edges,image_metadata(view/f'{alias}.zarr')['shape']),
            inference_receipt_sha256=sha(output/'inference_receipt.json'))
        write(destination,record);records.append(record)
        print('fresh complete',i+1,record['passed'],record['seconds'],flush=True)
        if not record['passed']: raise RuntimeError('Fresh C0 parity failed')
    write(OUT/'fresh_receipt.json',dict(passed=all(r['passed'] for r in records),clips=records,
        scope='C0 fallback only; mask-to-graph paths blocked by missing learned and Ultrack runtimes',
        fresh_label_free_image_executions=len(records)))
