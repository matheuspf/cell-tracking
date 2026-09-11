"""Run the delivered v6 CLI on two entire fresh clips, without cached inputs."""
import csv
import os
import subprocess
import time
import numpy as np
from .common import *

def run():
    earlier=read_json(OUT/'fresh_receipt.json');records=[]
    for item in earlier['clips']:
        root=SCRATCH/'cli_checks'/f"{item['embryo']}_{time.time_ns()}";root.mkdir(parents=True)
        output=root/'output';images=Path(item['scratch'])/'images';logpath=root/'run.log'
        command=[str(REPO/'scripts/run_segmentation_tracking_v6.sh'),'infer','--images',str(images),
                 '--output',str(output),'--source-model',item['source'],'--arm','P0']
        start=time.monotonic()
        env={**os.environ,'TMPDIR':str(root),'MPLCONFIGDIR':str(root/'matplotlib')}
        with logpath.open('w') as log:
            p=subprocess.Popen(command,env=env,cwd='/tmp',stdout=log,stderr=subprocess.STDOUT)
            import psutil
            max_rss=0.;max_gpu=0.
            while p.poll() is None:
                try:
                    main=psutil.Process(p.pid);rss=0
                    for child in [main]+main.children(recursive=True):
                        try:rss+=child.memory_info().rss
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:break
                gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())/1024
                max_rss=max(max_rss,rss/2**30);max_gpu=max(max_gpu,gpu)
                if max_rss>24 or max_gpu>20: p.terminate();raise Blocked('CLI child exceeded resource ceiling')
                time.sleep(2)
            p.wait()
        if p.returncode:raise RuntimeError(logpath.read_text()[-5000:])
        original=c0(item['dataset']);expected=load_graph(SCRATCH/'predictions/P0'/f"{item['dataset']}.npz")
        checks={}
        for filename,g in [('C0.csv',original),('submission.csv',expected)]:
            with (output/filename).open() as handle:rows=list(csv.DictReader(handle))
            nodes=np.asarray([[int(r[k]) for k in ['node_id','t','z','y','x']] for r in rows if r['row_type']=='node'],np.int64).reshape(-1,5)
            edges=np.asarray([[int(r[k]) for k in ['source_id','target_id']] for r in rows if r['row_type']=='edge'],np.int64).reshape(-1,2)
            checks[filename]=graph_hash(nodes,edges)==graph_hash(**g)
        audits=[read_json(p) for p in (output/'audit').glob('*.json')]
        checks['early_guard']=bool(audits) and all(r['installed_before_numerical'] and not r['blocked_events'] for r in audits)
        record=dict(embryo=item['embryo'],seconds=time.monotonic()-start,checks=checks,
            passed=all(checks.values()),peak_process_tree_rss_gib=max_rss,peak_total_gpu_gib=max_gpu,
            output=str(output),entrypoint_sha256=sha(REPO/'tools/segmentation_tracking_v6/infer.py'))
        records.append(record);print('v6 CLI full image',item['embryo'],record['passed'],record['seconds'],flush=True)
        write(OUT/'cli_check.json',dict(passed=len(records)==2 and all(r['passed'] for r in records),clips=records))
        if not record['passed']:raise RuntimeError('CLI graph/CSV/guard parity failed')
    blocked=subprocess.run([str(REPO/'scripts/run_segmentation_tracking_v6.sh'),'infer',
        '--images','/nonexistent','--output','/nonexistent','--source-model','44b6','--arm','M0'],capture_output=True,text=True)
    assert blocked.returncode==2 and 'no C0 substitution' in blocked.stderr
    write(OUT/'blocked_arm_check.json',dict(passed=True,exit_code=2,no_fallback_execution=True))

if __name__=='__main__':run()
