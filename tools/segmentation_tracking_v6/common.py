"""Explicit input roots; only v6 metadata may be written below the disk floor."""
from pathlib import Path
import json
import os
import shutil
from strong_tracker_v3.common import sha, digest, now, read_json, load_graph, graph_hash, clean

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/kaggle/working/cell-tracking')
OUT = ROOT / 'segmentation-tracking-v6'
SCRATCH = Path(os.environ.get('V6_SCRATCH', '/dev/shm/cell-tracking-segmentation-v6'))
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
V1, V2, V3, V4, V5 = [ROOT / n for n in (
    'annotation-selection-v1', 'strong-tracker-v2', 'strong-tracker-v3',
    'multidata-training-v4', 'image-native-tracking-v5')]
BASE = {'pooled': .934802374260586, '44b6': .931664468721842, '6bba': .935221784097327}

class Blocked(RuntimeError):
    """An unavailable asset or resource, never a substituted algorithm."""

def write(path, value):
    path = Path(path)
    value = json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n'
    if len(value.encode()) > 8 * 2**20:
        raise ValueError('Metadata receipt exceeds 8 MiB; detailed arrays belong in scratch')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)

def reserve(path, additional=0):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < 8 * 2**30 + additional:
        raise Blocked(f'8 GiB free-space floor at {path}; free={shutil.disk_usage(path).free / 2**30:.3f} GiB')

def inventory():
    rows = read_json(V1 / 'inventory.json')
    assert len(rows) == len({r['dataset'] for r in rows}) == 199
    return rows

def c0(name):
    return load_graph(V3 / 'selected_predictions' / f'{name}.npz')

def image_metadata(path):
    meta = read_json(Path(path) / 'zarr.json')
    multiscale = meta['attributes']['multiscales'][0]
    axes = ''.join(a['name'].lower() for a in multiscale['axes'])
    if axes != 'tzyx':
        raise ValueError(f'Expected TZYX; observed {axes}')
    transforms = multiscale['datasets'][0]['coordinateTransformations']
    if len(transforms) != 1 or transforms[0]['type'] != 'scale':
        raise ValueError('Unverified resampling/translation origin')
    scale = transforms[0]['scale'][1:]
    if scale != [1.625, .40625, .40625]:
        raise ValueError(f'Unexpected native spacing: {scale}')
    shape = read_json(Path(path) / '0/zarr.json')['shape']
    return dict(axes=axes, shape=shape, spacing_um=scale, origin_zyx=[0, 0, 0],
                voxel_centers='integer indices', metadata_sha256=sha(Path(path) / 'zarr.json'))
