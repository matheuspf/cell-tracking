from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
WORK = REPO / 'work/annotation-selection-v1'
OUT = Path('/kaggle/working/cell-tracking/annotation-selection-v1')
OFFICIAL = WORK / 'official'
METRIC_REV = '075fc5f5a52d11077f9dc2b074644618f26939e2'
SEED = 20260908


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False) + '\n'
    if immutable and path.exists():
        if json.loads(path.read_text()) != json.loads(text):
            raise RuntimeError(f'Attempt to change locked artifact: {path}')
        return
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def stage(name, status, **details):
    path = OUT / 'status.json'
    data = read_json(path) if path.exists() else {'study_id': 'annotation-selection-v1', 'stages': {}}
    data['stages'][name] = dict(status=status, updated=now(), **clean(details))
    data['updated'] = now()
    write_json(path, data)


def graph_hash(nodes, edges):
    h = hashlib.sha256()
    for x in (nodes, edges):
        a = np.ascontiguousarray(x, dtype='<i8')
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_graph(path, nodes, edges, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, nodes=np.asarray(nodes, dtype=np.int64).reshape(-1, 5),
                        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2), **arrays)
    tmp.replace(path)


def revision():
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def load_graph(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
