"""Operational checks using saved training states and an earlier image-only pilot."""
from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import torch

from .common import REPO, WORK, OUT, read, write, sha, now


def training():
    results = []
    for source, target in [('44b6', '6bba'), ('6bba', '44b6')]:
        folder = WORK / 'models/native' / f'source-{source}-seed-314159'
        initial = torch.load(folder / 'initial.pt', weights_only=True, map_location='cpu')['model']
        state = torch.load(folder / 'resume.pt', weights_only=True, map_location='cpu')
        changes = {}
        for name, prefix in [('encoder', 'unet.'), ('detector', 'detect_head.'), ('association', 'transformer.')]:
            keys = [k for k in initial if k.startswith(prefix) and initial[k].is_floating_point()]
            change = sum(float((state['model'][k] - initial[k]).abs().sum()) for k in keys)
            assert keys and change > 0, name
            changes[name] = dict(keys=len(keys), l1_change=change)
        code = f"""from tools.incumbent_comparison.common import source_guard,DATA,WORK,WEIGHTS
source_guard({source!r})
paths=[DATA/{target + '_fake.geff/zarr.json'!r},WORK/'frames'/{target!r}/'fake.npy',WEIGHTS/'edge_predictor_best.pth']
for path in paths:
    try: path.open('rb')
    except PermissionError: pass
    else: raise AssertionError(path)
print('all3rejected')
"""
        proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        assert 'all3rejected' in proc.stdout
        results.append(dict(source=source, updates_at_checkpoint=state['updates'], epoch=state['epoch'],
                            offset=state['offset'], initial_checkpoint_sha256=sha(folder / 'initial.pt'),
                            changes=changes, rejected_target_metadata_cache_and_public_weights=True))
    write(OUT / 'training-launch-validation.json', dict(created_utc=now(), checks=results,
          scope='All three model components changed from scratch initialization. Negative Python audit-hook access tests; not a kernel sandbox or target evaluation.'))


def linking():
    from . import full_clips as full
    from scipy.special import softmax
    p = copy.deepcopy(read(OUT / 'full-clips-plan.json'))
    name = '44b6_81c256f0'
    keys = [f'{name}-t025', f'{name}-t026']
    p['frames'] = [next(r for r in p['frames'] if r['key'] == key) for key in keys]
    for t, row in enumerate(p['frames']):
        row['time'] = t
    p['clips'] = [next(c for c in p['clips'] if c['dataset'] == name)]
    p['clips'][0]['shape'][0] = 2
    p['scope'] = 'Inference-only preflight on original t25/t26, rebased to t0/t1. No annotation access or scoring.'
    full.OUT = WORK / 'full-clips-preflight/receipts'
    original_root = full.ROOT
    full.ROOT = WORK / 'full-clips-preflight'
    full.ROOT.mkdir(parents=True, exist_ok=True)
    link = full.ROOT / 'banks'
    if not link.exists():
        link.symlink_to(original_root / 'banks', target_is_directory=True)
    write(full.OUT / 'full-clips-plan.json', p)
    banks = [r for key in keys for r in read(original_root / 'banks-receipts' / (key + '.json'))['banks']]
    write(full.OUT / 'full-clips-banks-lock.json', dict(plan_sha256=sha(full.OUT / 'full-clips-plan.json'), banks=banks))
    full.link()
    pilot = REPO / 'work/cellpose-embedding-probe-20260914'
    pilot_lock = read(REPO / 'results/cellpose-embedding-probe-20260914/native-lock.json')
    checks = []
    for method, previous in [('incumbent', 'incumbent'), ('cellpose', 'cpdino-vitb')]:
        reference = pilot / 'native/primary' / previous / (keys[0] + '.npz')
        expected_hash = next(r['sha256'] for r in pilot_lock['predictions'] if r['weight'] == 'primary' and r['method'] == previous and r['key'] == keys[0])
        assert sha(reference) == expected_hash
        with np.load(reference, allow_pickle=False) as f:
            logits = f['image_logits']
        # Original public greedy rule, executed on the prior frozen pilot logits.
        probabilities = softmax(logits, axis=0)
        expected = full.greedy(probabilities)
        expected[:, 1] += logits.shape[0]
        with np.load(full.ROOT / 'graphs' / method / (name + '.npz'), allow_pickle=False) as f:
            actual = f['edges']
            nodes = f['nodes']
        for t, key in enumerate(keys):
            with np.load(pilot / 'banks' / previous / (key + '.npz'), allow_pickle=False) as f:
                np.testing.assert_array_equal(nodes[nodes[:, 1] == t, 2:], f['centers_zyx'])
        np.testing.assert_array_equal(actual, expected)
        checks.append(dict(method=method, edge_count=len(actual), expected_logits_sha256=expected_hash,
                           exact_edges_against_prior_native_logits=True, exact_centers_against_prior_pilot=True))
    write(OUT / 'full-clips-preflight.json', dict(created_utc=now(), checks=checks, methods_with_valid_graphs=9,
          annotation_access=False, scope=p['scope']))


if __name__ == '__main__':
    training()
    linking()
