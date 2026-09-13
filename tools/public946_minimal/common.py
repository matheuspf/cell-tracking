"""Small explicit-path, immutable-provenance helpers shared by the study."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HANDOVER = REPO / 'handover/public946-minimal-generalization-v1'
DEFAULT_OUT = Path('/kaggle/working/cell-tracking/public946-minimal-generalization-v1/revision-2')
DEFAULT_DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
DEFAULT_ARCHIVE = Path('/kaggle/notebooks/biohub-cell-tracking-during-development/flexonafft/biohub-harmonic-fusion')
DEFAULT_ARTIFACTS = Path('/kaggle/input/datasets/pilkwang')
DEFAULT_RUNTIME = Path('/kaggle/envs/cell-tracking-notebooks/bin/python')
METRIC_REVISION = '075fc5f5a52d11077f9dc2b074644618f26939e2'
ARMS = {r['id']: r for r in json.loads((HANDOVER / 'experiment_lock.json').read_text())['arms']}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, immutable=False):
    path = Path(path)
    value = clean(value)
    if immutable and path.exists():
        if read_json(path) != value:
            raise RuntimeError(f'Immutable receipt changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def tree_hash(path):
    """Hash file contents and relative names; never follow unrelated siblings."""
    root = Path(path)
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    return dict(sha256=digest(files), files=files,
                bytes=sum((root / p).stat().st_size for p in files))


def code_hashes():
    return {str(p.relative_to(REPO)): sha(p)
            for p in sorted((REPO / 'tools/public946_minimal').rglob('*.py'))}


def save_arrays(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def array_hash(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def check_resources(out, limits):
    import shutil
    free = shutil.disk_usage(out).free / 2**30
    if free < limits['filesystem_reserve_gib_min']:
        raise RuntimeError(f'filesystem reserve: {free:.2f} GiB remaining')
    used = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and not p.is_symlink()) / 2**30
    if used > limits['new_scratch_gib_max']:
        raise RuntimeError(f'scratch allocation: {used:.2f} GiB used')
    return dict(free_gib=free, scratch_gib=used)
