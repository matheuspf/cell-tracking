"""Materialize v29 itself and extract declarations by semantic boundaries."""
from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path

from .common import read_json, sha, write_json


def notebook_settings(source):
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name) and target.value.value.id == 'os'
                and target.value.attr == 'environ'):
            try:
                key, value = ast.literal_eval(target.slice), ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            if isinstance(key, str) and isinstance(value, str):
                result[key] = value
    return result


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Source anchor count {text.count(old)}: {old[:120]}')
    return text.replace(old, new, 1)


def materialize(archive, out, artifacts):
    source_path = archive / 'biohub-harmonic-fusion.py'
    original = source_path.read_text()
    if sha(source_path) != '7a949f3e5ee6136175b1f3806621697e68776e87f304a37c58132166eb02d645':
        raise ValueError('Not the pinned v29 export')
    root = out / 'public_source'
    if (root / 'materialization.json').exists():
        receipt = read_json(root / 'materialization.json')
        for relative, expected in receipt['source_hashes'].items():
            if sha(root / relative) != expected:
                raise ValueError(f'Materialized source drift: {relative}')
        return receipt
    root.mkdir(parents=True, exist_ok=False)
    text = original[:original.index('test_stems = list_test_stems()')]
    text = text.replace('/kaggle/working', str(root))
    # All dataset paths remain explicit public artifacts. No package upgrade.
    for slug in ('biohub-tracking-support-pack-50ep-v1', 'biohub-temporal-unet3d-seed314159-v1',
                 'biohub-deepcenter-unet3d-center-prior-v1'):
        text = text.replace('/kaggle/input/' + slug, str(artifacts / slug))
    text += '\n'
    (root / 'materialize.py').write_text(text)
    (root / 'shared_materialization.diff').write_text(''.join(difflib.unified_diff(
        original[:original.index('test_stems = list_test_stems()')].splitlines(True),
        text.splitlines(True), fromfile='v29', tofile='explicit-path-materialization')))
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith('BIOHUB_')}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', BIOHUB_ALLOW_PIP_INSTALL='0')
    with (root / 'materialize.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'materialize.py')], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    sources = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'tracking_repo').rglob('*.py'))}
    settings = notebook_settings(original)
    settings['BIOHUB_SECONDARY_WEIGHTS'] = str(root / 'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth')
    settings['BIOHUB_DEEPCENTER_CHECKPOINT'] = str(artifacts / 'biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')
    receipt = dict(original_sha256=sha(source_path), source_hashes=sources, settings=settings)
    write_json(root / 'materialization.json', receipt, immutable=True)
    return receipt


def repair_namespace(source_path, image_dir, out, settings):
    source = Path(source_path).read_text()
    # Select config by assignment names and repairs by named functions, never stale line numbers.
    tree = ast.parse(source)
    start = source.index('def graph_from_geff(')
    end = source.index('DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()')
    first = source[:start].count('\n') + 1
    last = source[:end].count('\n') + 1
    ns = {'__name__': 'public946_repairs'}
    exec('import os, json, math, csv, sys, time\nfrom pathlib import Path\n'
         'import numpy as np\nimport torch\nimport tracksdata as td\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
         'import blosc2\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.spatial import cKDTree', ns)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names and all(n.isupper() for n in names) and node.lineno < first:
                # Only independent scalar/environment settings; workflow path/config objects are omitted.
                if any(n.startswith(('OUTPUT_', 'MOTION_', 'DIV_', 'GAP_', 'GAP2_', 'SAFE_', 'DEEPCENTER_',
                                     'USE_DEEP', 'REQUIRE_DEEP', 'SHORT_TRACK_', 'ADAPTIVE_SHORT_',
                                     'VOXEL_SCALE_', 'SUBMISSION_COLUMNS', 'CSV_COLUMNS')) for n in names):
                    nodes.append(node)
        if first <= node.lineno < last:
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), ns)
    ns.update(TEST_DIR=Path(image_dir), WORKING_DIR=Path(out), **settings)
    return ns
