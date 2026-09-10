"""One fresh image-to-graph entry point, with explicit C0 disablement."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .common import OUT, V1, V2, V5, REPO, Blocked, reserve, read_json, sha

def run(args):
    arm='C0' if args.disable_v6 else args.arm
    if arm in ['B0','M0','U0','U1']:
        # Fail before baseline execution: a pure tool arm cannot silently use C0.
        raise Blocked('This study has no validated selected learned-mask recipe or complete mask tracking arm. '
                      'See preflight.json for recorded local dependency blockers; no C0 substitution.')
    images=Path(args.images).resolve();output=Path(args.output).resolve()
    if len(list(images.glob('*.zarr')))!=1: raise ValueError('v6 processes one complete clip at a time')
    if output.exists(): raise FileExistsError('Fresh output must be absent')
    reserve(output,additional=2*2**30)
    package=V5/'inference_package_validation'
    for rel,expected in read_json(package/'manifest.json')['files'].items():
        assert sha(package/rel)==expected,rel
    env={**os.environ,'V5_PACKAGE_ROOT':str(package),'V5_FRESH_OUTPUT':str(output),
         'V5_AUDIT_DIR':str(output/'audit'),'V5_V1':str(V1),'V5_V2':str(V2),
         'PYTHONPATH':str(package/'tools')}
    subprocess.run([str(package/'base/run.sh'),'--python',sys.executable,'--images',str(images),
                    '--output',str(output),'--v1',str(V1),'--v2',str(V2),
                    '--source-model',args.source_model],env=env,cwd='/tmp',check=True)
    if arm=='P0':
        subprocess.run([sys.executable,str(REPO/'tools/segmentation_tracking_v6/point_child.py'),
            '--output',str(output),'--source',args.source_model,'--package',str(package/'base'),
            '--model',str(OUT/'models'/f'P0_{args.source_model}.json')],
            env={**env,'PYTHONPATH':str(package/'tools')+':'+str(package/'base/tools')+':'+str(REPO/'tools')},cwd='/tmp',check=True)
        # The inherited C0 export remains alongside the selected output.
        (output/'submission.csv').rename(output/'C0.csv')
        (output/'P0.csv').rename(output/'submission.csv')

def main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-model',choices=['44b6','6bba'],required=True)
    p.add_argument('--arm',choices=['C0','P0','B0','M0','U0','U1'],default='C0')
    p.add_argument('--disable-v6',action='store_true')
    args=p.parse_args(argv)
    try:run(args)
    except Blocked as exc:p.exit(2,str(exc)+'\n')
