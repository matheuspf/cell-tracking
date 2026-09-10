from pathlib import Path
import json
import hashlib
import os
from datetime import datetime, timezone
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('V4_OUTPUT', '/kaggle/working/cell-tracking/multidata-training-v4'))
STUDIES = OUT.parent
V3 = STUDIES/'strong-tracker-v3'
V1 = STUDIES/'annotation-selection-v1'
ARCHIVE = REPO/'work/biohub-forum-archive'
PREPARED = REPO/'work/biohub-data-guide/prepared'
SYNTH = ARCHIVE/'downloads/kaggle/biohub_synthetic'
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
BASE = {'pooled': .934802374260586, '44b6': .931664468721842, '6bba': .935221784097327}
SEED = 20260909

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    tmp.write_text(json.dumps(value, indent=2, default=default, allow_nan=False)+'\n'); tmp.replace(p)
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def save(p, **kw):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**kw);tmp.replace(p)
def inputs(): return read(V3/'inputs.json')
def status(stage, **kw): write(OUT/'status.json',dict(stage=stage,updated=now(),**kw))
