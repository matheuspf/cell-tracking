"""Read-only asset inventory and source-driven registration."""
from __future__ import annotations

import collections
import os
import subprocess
from pathlib import Path

from .common import (HANDOVER, METRIC_REVISION, REPO, check_resources, digest,
                     now, read_json, sha, tree_hash, write_json)
from .source import materialize


def choose_pilots(rows):
    import numpy as np
    pilots = []
    for embryo in sorted({r['embryo'] for r in rows}):
        group = [r for r in rows if r['embryo'] == embryo]
        for percentile in (25, 75):
            target = float(np.percentile([r['contrast_q99_q10'] for r in group], percentile))
            row = min(group, key=lambda r: (abs(r['contrast_q99_q10'] - target), r['dataset']))
            pilots.append(dict(dataset=row['dataset'], embryo=embryo, percentile=percentile,
                               target_contrast=target, contrast=row['contrast_q99_q10']))
    if len({p['dataset'] for p in pilots}) != 4:
        raise ValueError('Pilot quartiles selected duplicate clips')
    return pilots


def run(args):
    import torch
    import importlib.metadata
    args.out.mkdir(parents=True, exist_ok=True)
    limits = dict(gpu_device_hours_max=args.gpu_hours, gpu_allocated_gib_max=args.gpu_gib,
                  new_scratch_gib_max=args.scratch_gib, filesystem_reserve_gib_min=10)
    check_resources(args.out, limits)
    if not torch.cuda.is_available():
        raise RuntimeError('Registered GPU runtime unavailable')
    rows = []
    paths = sorted((args.data / 'train').glob('*.zarr'))
    if len(paths) != 199:
        raise ValueError(f'Expected 199 supplied images; found {len(paths)}')
    for p in paths:
        attrs = read_json(p / 'zarr.json')['attributes']
        meta = read_json(p / '0/zarr.json')
        axes = [a['name'].upper() for a in attrs['multiscales'][0]['axes']]
        if axes != ['T', 'Z', 'Y', 'X']:
            raise ValueError(f'Unexpected axes: {p}')
        q = attrs['image_statistics']['quantiles']
        record_path = args.out / 'input_hashes' / (p.stem + '.json')
        if record_path.exists():
            record = read_json(record_path)
            # Hash once in this preflight; worker rechecks every file before inference.
        else:
            record = tree_hash(p)
            write_json(record_path, record, immutable=True)
        rows.append(dict(dataset=p.stem, embryo=p.stem.split('_')[0], path=str(p),
                         image_shape=meta['shape'], dtype=meta['data_type'],
                         quantiles=q, scale=attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1:],
                         contrast_q99_q10=float(q['0.99']) - float(q['0.1']),
                         image_sha256=record['sha256'], image_bytes=record['bytes'],
                         overlap_group='unknown; supplied crops may overlap'))
        if len(rows) % 20 == 0:
            print(f'Input content fingerprints: {len(rows)}/199', flush=True)
    write_json(args.out / 'image_inventory.json', rows, immutable=True)
    write_json(args.out / 'pilots.json', choose_pilots(rows), immutable=True)
    artifact_records = {}
    for slug in ('biohub-tracking-support-pack-50ep-v1', 'biohub-temporal-unet3d-seed314159-v1',
                 'biohub-deepcenter-unet3d-center-prior-v1'):
        artifact_records[slug] = tree_hash(args.artifacts / slug)
    write_json(args.out / 'artifact_hashes.json', artifact_records, immutable=True)
    source = materialize(args.archive, args.out, args.artifacts)
    metric_path = REPO / 'work/annotation-selection-v1/official'
    metric_sha = subprocess.check_output(['git', '-C', str(metric_path), 'rev-parse', 'HEAD'], text=True).strip()
    if metric_sha != METRIC_REVISION:
        raise ValueError('Historical official checkout revision drift')
    current = subprocess.run(['gh', 'api', 'repos/royerlab/kaggle-cell-tracking-competition/commits/main',
                              '--jq', '.sha'], capture_output=True, text=True)
    metric = dict(locked_revision=metric_sha, current_upstream_revision=current.stdout.strip() if current.returncode == 0 else None,
                  current_verification='verified' if current.returncode == 0 else 'pending',
                  source_files={str(p.relative_to(metric_path)): sha(p) for p in (metric_path / 'src').rglob('*.py')},
                  checked_at=now())
    write_json(args.out / 'metric_identity.json', metric)
    runtime = dict(python=os.sys.version, executable=os.sys.executable,
                   packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'zarr', 'tracksdata', 'polars')},
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), limits=limits,
                   source_sha256=source['original_sha256'], created=now())
    write_json(args.out / 'preflight.json', runtime)
    if not (args.out / 'plan_freeze.json').exists():
        subprocess.run([os.sys.executable, str(HANDOVER / 'study_contract.py'), 'freeze', '--file', str(args.out / 'plan_freeze.json')], check=True)
    print('Preflight complete; metadata pilots:', [p['dataset'] for p in choose_pilots(rows)], flush=True)


def audit(args):
    root = args.out / 'public_source/tracking_repo'
    predictor = root / 'scripts/predict_unet_transformer.py'
    model = root / 'scripts/train_unet_transformer.py'
    cfg = read_json(root / 'weights/unet_transformer/split_0/config.json')
    src = predictor.read_text()
    proof = {}
    for key, needle in {
        'grid': 'coords[:, 1:] *= ds_arr', 'integer_peaks': 'return np.concatenate([t_col, coords], axis=1).astype(np.int16)',
        'normalization': 'imgs = ((imgs - q_low)', 'parent_softmax': 'probs = torch.softmax(raw, dim=0)',
        'feature_tta': "_edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA'", 'two_frame_calls': 'W = window_size',
        'detection_mix': 'secondary_det_aligned =', 'retention_guard': 'use_primary_detection = bool(',
    }.items():
        line = next(i for i, s in enumerate(src.splitlines(), 1) if needle in s)
        proof[key] = dict(file=str(predictor), line=line, text=src.splitlines()[line - 1].strip())
    applicability = {
        'E01': dict(status='applicable', seam='atomic replacement immediately after motion_relink_edges', required='complete normalized native parent columns'),
        'E02': dict(status='applicable', seam='same final scalar logits as _detect_cells_pooled; surviving unchanged originals before smoother', response='raw logits', affine=dict(stride=cfg['downsample'], offset=[0, 0, 0])),
        'E03': dict(status='pending_numerical_identity', reason='Public peak finder returns integer int16 grid coordinates; features have same spatial shape', evidence=[proof['integer_peaks'], proof['grid']]),
        'E04': dict(status='applicable', seam='per-model scalar response before public detection calibration and retention guard', shift=[0, 2, 2], alignment='shifted[q]=image[ds*q-shift]; response inverse samples q+shift/ds; retain baseline outside support'),
        'E05': dict(status='applicable', seam='mean of final normalized probabilities, fixed IDs; run full public association and primary feature TTA on X-reflected raw images', reflection='original X -> raw_width-1-X; then divide by XY stride for unchanged public truncation/gather; physical and positional coordinates use reflected values', already_present='feature-channel averaging is present; probability consensus is absent'),
        'E06': dict(status='pending_numerical_identity', reason='Actual checkpoint config window_size=2; only one fully observed window contains any consecutive pair', window_size=cfg['window_size']),
        'E07': dict(status='applicable', seam='original linefit neighborhood with anchors and no support crossing a fork', anchors='fork, immediate predecessor, immediate daughters'),
        'E08': dict(status='applicable', seam='mean -> even-count median of inverse-aligned raw detection logits independently in both branches; feature average unchanged', views=8),
    }
    if cfg['window_size'] != 2 or cfg['downsample'] != [1, 4, 4]:
        raise ValueError('Source-resolved architecture differs from audit')
    write_json(args.out / 'applicability.json', applicability)
    write_json(args.out / 'source_audit.json', dict(source_lines=proof, model_source_sha256=sha(model), config=cfg,
        correction_to_planner='Public decode context is two frames, not five. Harmonic fusion is forward/reverse primary evidence; secondary mixing remains calibrated low-margin logit fusion. Detection inter-model fusion is aligned 0.80 linear mix with retention guard.',
        prior_exact_implementation_search='No identical implementation found in committed old study code/reports; only different sub-voxel models and training mechanisms.',
        annotation_free=True))
    print('Source audit written', flush=True)
