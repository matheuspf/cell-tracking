from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
WORK=REPO/'work/incumbent-comparison-20260914'
OUT=REPO/'results/incumbent-comparison-20260914'
DATA=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
PACKAGE=Path('/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1')
NATIVE=PACKAGE/'repo'
WEIGHTS=PACKAGE/'weights/unet_transformer/split_0'
GPU_LOCK=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock')


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(value,tmp);tmp.replace(path)


def modules():
    import torch
    torch.set_num_threads(2)
    for p in (NATIVE/'src',NATIVE/'scripts'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    import train_unet_transformer as training
    import predict_unet_transformer as prediction
    return training,prediction


def source_hashes():return {str(p):sha(p) for p in sorted(NATIVE.rglob('*.py'))}


def verify_sources(expected):
    for path,digest in expected.items():assert sha(path)==digest,path


def source_guard(source):
    """Keep target raw data and per-embryo caches out of source-only workers."""
    if source=='all':return
    other={'44b6':'6bba','6bba':'44b6'}[source]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=str(Path(os.fsdecode(args[0])).resolve())
            if (f'/{other}_' in path or f'/{other}/' in path) and ('biohub-cell-tracking' in path or 'incumbent-comparison' in path or 'cellpose-refine' in path):
                raise PermissionError('Embryo-excluded training rejected target access: '+path)
            if ('/weights/' in path or '/models/' in path) and '/incumbent-comparison-20260914/' not in path and path.endswith(('.pth','.pt','.pkl')):
                raise PermissionError('Scratch training rejected inherited checkpoint: '+path)
    sys.addaudithook(audit)
