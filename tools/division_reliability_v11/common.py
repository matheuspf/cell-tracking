"""Standard-library paths, receipts, and explicit failures."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'handover/division-reliability-v11/study.json'
WORK = REPO / 'work/division-reliability-v11'
RESULTS = REPO / 'results/division-reliability-v11'
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
ARCH = Path('/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/repo')
OLD = REPO / 'work/clean-validation-division-v10'
BRANCH = 'handover/division-reliability-v11-ready'


class Blocked(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != text:
            raise Blocked(f'Immutable receipt differs: {path.name}')
        return
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                bytes=path.stat().st_size, sha256=sha(path))
