"""Refresh diagnostics against corrected checkpoints, preserving old evidence."""
import shutil
import subprocess
import sys
import time
from .common import *

def run():
    while not (OUT/'sparse_edge_mask_retraining_receipt.json').exists():time.sleep(30)
    archive=OUT/'failed_fits/sparse_edge_mask/diagnostics';archive.mkdir(parents=True,exist_ok=True)
    for name in ['stress_diagnostics.csv','stress_receipt.json','source_read_audit.json','source_read_audit_44b6.json','source_read_audit_6bba.json','inference_runtime_profile.json']:
        path=OUT/name
        if path.exists() and not (archive/name).exists():shutil.copyfile(path,archive/name)
    jobs=[]
    for stage in ['read_audit','stress','profile','provenance']:
        start=time.monotonic();log=OUT/'logs'/f'repair_check_{stage}.log'
        with log.open('w') as f:r=subprocess.run([sys.executable,'-m','multidata_training_v4',stage],stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(stage=stage,returncode=r.returncode,seconds=time.monotonic()-start));assert r.returncode==0,jobs
    write(OUT/'repair_checks_receipt.json',dict(completed=True,jobs=jobs))
