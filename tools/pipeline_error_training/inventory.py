"""Recover the three exact all-199 controls and hash their immutable inputs."""
import subprocess
from pathlib import Path

import numpy as np

from .common import (
    BASELINES, DATA, EVIDENCE, METRIC_REV, OFFICIAL, RAW, RECEIPTS, REPO,
    RESULTS, ROOT, STUDY, WORK, digest, graph_hash, load_graph, now,
    read_json, sha, validate, write_json,
)
from .resources import gpu_snapshot, process_snapshot


def run():
    from center_comparison.pipeline import read_geometry
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != METRIC_REV:
        raise ValueError('Pinned metric revision mismatch')
    if subprocess.check_output(['git', '-C', str(OFFICIAL), 'diff', 'HEAD', '--', 'src'], text=True):
        raise ValueError('Pinned metric has source modifications')
    old = read_json(ROOT / 'annotation-selection-v1/inventory.json')
    lock = read_json(ROOT / 'strong-tracker-v3/selected_prediction_lock.json')
    expected = sorted(lock['hashes'])
    if len(expected) != 199 or set(expected) != {r['dataset'] for r in old}:
        raise ValueError('All-199 input inventory mismatch')
    c0_graph_hashes = {r['dataset']: r['graph_hash'] for r in lock['graphs']}
    clips = []
    for source_row in old:
        name = source_row['dataset']
        path, shape, spacing = read_geometry(DATA, name)
        if shape != source_row['image_shape'] or spacing != source_row['physical_scale']:
            raise ValueError('Actual image geometry differs from the historical inventory')
        # Read the actual GEFF count estimate without loading labels into inference.
        gt_meta = read_json(DATA / 'train' / f'{name}.geff/zarr.json')
        estimate = gt_meta['attributes']['geff']['extra']['estimated_number_of_nodes']
        if estimate != source_row['estimated_total']:
            raise ValueError('Official total-node estimate changed')
        row = dict(dataset=name, embryo=source_row['embryo'], image_path=str(path),
                   image_shape=shape, physical_scale=spacing, estimated_total=estimate,
                   metadata_sha256=sha(path / 'zarr.json'),
                   array_metadata_sha256=sha(path / '0/zarr.json'), baselines={})
        nodes = None
        for arm, folder in BASELINES.items():
            p = folder / f'{name}.npz'
            graph = load_graph(p)
            check = validate(graph['nodes'], graph['edges'], shape)
            file_sha = sha(p)
            graph_sha = graph_hash(graph['nodes'], graph['edges'])
            if arm == 'C0':
                if file_sha != lock['hashes'][name] or graph_sha != c0_graph_hashes[name]:
                    raise ValueError(f'C0 lock mismatch: {name}')
            else:
                receipt = read_json(RECEIPTS[arm] / f'{name}.json')
                if (file_sha != receipt['prediction_sha256'] or graph_sha != receipt['graph_hash']
                        or receipt['metric_revision'] != METRIC_REV):
                    raise ValueError(f'{arm} receipt mismatch: {name}')
            if nodes is not None and not np.array_equal(nodes, graph['nodes']):
                raise ValueError('Control nodes are not exactly identical')
            nodes = graph['nodes']
            row['baselines'][arm] = dict(path=str(p), sha256=file_sha, graph_hash=graph_sha, **check)
        p = EVIDENCE / f'{name}.npz'
        receipt = read_json(p.with_suffix('.json'))
        if sha(p) != receipt['sha256']:
            raise ValueError(f'Native evidence receipt mismatch: {name}')
        evidence = load_graph(p)
        if not np.array_equal(nodes, evidence['incumbent_nodes']):
            raise ValueError('Native features are not at the exact incumbent observations')
        row['evidence'] = dict(path=str(p), sha256=sha(p), receipt_sha256=sha(p.with_suffix('.json')))
        p = RAW / f'pre_ilp_{name}.npz'
        row['raw'] = dict(path=str(p), sha256=sha(p))
        clips.append(row)
    payload = dict(schema_version=1, study_sha256=sha(STUDY), clips=clips,
                   metric_revision=revision, metric_sources={str(p.relative_to(OFFICIAL)): sha(p)
                       for p in sorted((OFFICIAL / 'src').rglob('*.py'))},
                   baseline_lock_sha256=sha(ROOT / 'strong-tracker-v3/selected_prediction_lock.json'))
    payload['content_sha256'] = digest(payload)
    write_json(RESULTS / 'input_manifest.json', payload, immutable=True)
    write_json(WORK / 'preflight.json', dict(created=now(), gpu=gpu_snapshot(), processes=process_snapshot(),
        git_revision=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()))
    write_json(RESULTS / 'exposure_manifest.json', dict(
        claim='source-only head; inherited upstream exposure', clean_end_to_end_transfer=False,
        public_secondary_sha256='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f',
        public_secondary_exposure='All 199 clips in checkpoint-linked training manifest; 80% of detector logits only.',
        public_primary_sha256='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771',
        public_primary_exposure='Training split unresolved.',
        teacher_exposure='Historical v2 E_hgb teacher fitted on the opposite embryo; retained in P0 upstream evidence.',
        source_labels='Only explicitly selected source embryo may be opened by each fit/calibration process.',
        external_synthetic='Excluded: historical simulator calibrated on 44b6.',
        native_clean_fits='No running native-training process observed at preflight. Existing 400-epoch fits are incomplete; left untouched.',
        sources=['docs/incumbent-provenance-20260914.md', 'results/strong-tracker-v3/final_report.md'],
    ), immutable=True)
    print(f'Verified exact hashes, graph identity and evidence for {len(clips)} clips / 597 graphs', flush=True)
