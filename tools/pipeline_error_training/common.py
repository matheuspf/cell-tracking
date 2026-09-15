"""Study paths and immutable artifact contracts, independent of legacy globals."""
from pathlib import Path

from strong_tracker_v3.common import (
    adjacency, clean, digest, graph_hash, load_graph, now, read_json,
    save_arrays, save_graph, sha, validate, write_json,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/kaggle/working/cell-tracking')
WORK = REPO / 'work/pipeline-error-training-20260915'
RESULTS = REPO / 'results/pipeline-error-training-20260915'
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
STUDY = REPO / 'handover/pipeline-error-training-20260915/study.json'
OFFICIAL = REPO / 'work/annotation-selection-v1/official'
METRIC_REV = '075fc5f5a52d11077f9dc2b074644618f26939e2'
GPU_LOCK = ROOT / 'detector-screen-20260914.gpu.lock'
BASELINES = {
    'P0': ROOT / 'segmentation-tracking-v6-local/prior-v6/ram-outputs/predictions/P0',
    'C4_m6': ROOT / 'multidata-training-v4/predictions/C4_m6',
    'C0': ROOT / 'strong-tracker-v3/selected_predictions',
}
RECEIPTS = {
    'P0': ROOT / 'ultrack-integration-v7/evaluation/P0',
    'C4_m6': ROOT / 'multidata-training-v4/evaluation/C4_m6',
}
EVIDENCE = ROOT / 'strong-tracker-v3/fresh_evidence'
RAW = ROOT / 'annotation-selection-v1/public_harmonic_full/inputs'


class Blocked(RuntimeError):
    """A concrete missing input/resource; never an invented experimental result."""


def inputs():
    return read_json(RESULTS / 'input_manifest.json')['clips']


def check_work(path):
    path = Path(path).resolve()
    if WORK.resolve() not in path.parents:
        raise ValueError(f'Heavy output must be inside the new study work root: {path}')
    return path


def verified_graph(row, arm='P0'):
    spec = row['baselines'][arm]
    path = Path(spec['path'])
    if sha(path) != spec['sha256']:
        raise ValueError(f'Input graph bytes changed: {row["dataset"]}/{arm}')
    graph = load_graph(path)
    if graph_hash(graph['nodes'], graph['edges']) != spec['graph_hash']:
        raise ValueError('Input graph identity changed')
    return graph


def verified_evidence(row):
    spec = row['evidence']
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Incumbent evidence bytes changed')
    return load_graph(spec['path'])
