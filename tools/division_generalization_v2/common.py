"""Isolated outputs and immutable inherited inputs."""
from pathlib import Path
import csv
import json
import os

from pipeline_error_training.common import (
    DATA, ROOT, REPO, OFFICIAL, METRIC_REV, GPU_LOCK, adjacency, clean, digest,
    graph_hash, load_graph, now, read_json, save_arrays, save_graph, sha, validate,
    verified_evidence, verified_graph, write_json,
)

WORK = REPO / 'work/division-generalization-v2'
RESULTS = REPO / 'results/division-generalization-v2'
PRIOR_RESULTS = REPO / 'results/pipeline-error-training-20260915'
PRIOR_WORK = REPO / 'work/pipeline-error-training-20260915'
STUDY = REPO / 'handover/division-generalization-v2/study.json'
SEEDS = (20260916, 314159)


def inputs(source=None, partition=None):
    rows = read_json(RESULTS / 'input_manifest.json')['clips']
    if source is not None:
        rows = [r for r in rows if r['embryo'] == source]
    if partition is not None:
        split = read_json(RESULTS / 'split_manifest.json')['directions'][source]
        rows = [r for r in rows if split[r['dataset']]['partition'] == partition]
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(clean(row), sort_keys=True, allow_nan=False)+'\n')


def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)
