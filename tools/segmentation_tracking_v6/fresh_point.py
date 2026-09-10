"""Execute and check P0 on both genuinely fresh image-derived C0 outputs."""
import csv
import os
import subprocess
import sys
import time
import numpy as np
from .common import *

def run():
    receipt=read_json(OUT/'fresh_receipt.json')
    if not receipt['passed']: raise Blocked('C0 fresh checks must pass first')
    package=V5/'inference_package_validation';results=[]
    for item in receipt['clips']:
        root=Path(item['scratch']);output=root/'output';source=item['source']
        attempt=str(time.time_ns())
        env={**os.environ,'V5_PACKAGE_ROOT':str(package),'V5_FRESH_OUTPUT':str(output),
             'V5_AUDIT_DIR':str(root/('point_audit_'+attempt)),'V5_V1':str(V1),'V5_V2':str(V2),
             'TMPDIR':str(root),'MPLCONFIGDIR':str(root/'matplotlib'),
             'PYTHONPATH':str(package/'tools')+':'+str(package/'base/tools')+':'+str(REPO/'tools')}
        command=[sys.executable,str(REPO/'tools/segmentation_tracking_v6/point_child.py'),
                 '--output',str(output),'--source',source,'--package',str(package/'base'),
                 '--model',str(OUT/'models'/f'P0_{source}.json')]
        start=time.monotonic()
        logpath=root/('point_run_'+attempt+'.log')
        with logpath.open('w') as log:
            child=subprocess.run(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
        if child.returncode: raise RuntimeError(logpath.read_text()[-5000:])
        actual=load_graph(output/'predictions/P0.npz')
        expected=load_graph(SCRATCH/'predictions/P0'/f"{item['dataset']}.npz")
        fresh=load_graph(output/'point_evidence.npz');cached=load_graph(V3/'fresh_evidence'/f"{item['dataset']}.npz")
        input_checks={k:bool(np.array_equal(fresh[k],cached[k])) for k in ['pairs','edge_features']}
        with (output/'P0.csv').open() as handle: rows=list(csv.DictReader(handle))
        nodes=np.asarray([[int(r[k]) for k in ['node_id','t','z','y','x']] for r in rows if r['row_type']=='node'],np.int64).reshape(-1,5)
        edges=np.asarray([[int(r[k]) for k in ['source_id','target_id']] for r in rows if r['row_type']=='edge'],np.int64).reshape(-1,2)
        exact=graph_hash(**actual)==graph_hash(**expected);csv_match=graph_hash(nodes,edges)==graph_hash(**actual)
        audits=[read_json(p) for p in (root/('point_audit_'+attempt)).glob('*.json')]
        audit_pass=bool(audits) and all(a['installed_before_numerical'] and not a['blocked_events'] for a in audits)
        result=dict(embryo=item['embryo'],unfamiliar_name=item['unfamiliar_name'],seconds=time.monotonic()-start,
            exact_graph_parity=exact,csv_roundtrip=csv_match,exact_native_inputs=input_checks,
            guard_passed=audit_pass,guard_processes=len(audits),model_sha256=sha(OUT/'models'/f'P0_{source}.json'),
            passed=exact and csv_match and audit_pass and all(input_checks.values()))
        results.append(result); print('fresh P0',result,flush=True)
        write(OUT/'fresh_point_receipt.json',dict(passed=len(results)==2 and all(r['passed'] for r in results),clips=results))
    return results

if __name__=='__main__':run()
