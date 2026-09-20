"""Source-proven compact evaluation precision, applied before numerical imports."""
import os
import sys
import time
from .common import WORK,RESULTS,Blocked,read,write,sha,now

REPAIR=RESULTS/'compact_precision_repair.json'


def configure(stage,arm,package=None,*,source=None,seed=None,clip=None):
    compact=stage=='mine' or (arm=='C11' and stage in ('calibration-predict','calibrate','predict'))
    if stage=='infer' and package is not None:
        compact=read(package/'manifest.json')['arm']=='C11'
    if not compact:return None
    if 'torch' in sys.modules:raise Blocked('Compact evaluation precision must be selected before importing torch')
    repair=read(REPAIR)
    if repair['status']!='passed':raise Blocked('Compact evaluation precision repair is not validated')
    previous=os.environ.get('NVIDIA_TF32_OVERRIDE')
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    value=dict(stage=stage,arm='C11',source=source,seed=seed,clip=clip,pid=os.getpid(),
        environment={'NVIDIA_TF32_OVERRIDE':'0'},previous_override=previous,
        applied_before_numerical_import=True,repair_sha256=sha(REPAIR),recorded_utc=now())
    write(WORK/'precision_receipts'/f'{time.time_ns()}-{os.getpid()}.json',value)
    return value
