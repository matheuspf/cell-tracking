from __future__ import annotations

import collections
import importlib.metadata
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import DATA, OFFICIAL, OUT, REPO, WORK, METRIC_REV, digest, now, read_json, revision, save_graph, sha, stage, write_json


def image_metadata(path):
    meta = read_json(path / 'zarr.json')
    ms = meta['attributes']['multiscales'][0]
    axes = [a['name'].lower() for a in ms['axes']]
    if axes != ['t', 'z', 'y', 'x']:
        raise ValueError(f'Unknown axes {axes}')
    transforms = ms['datasets'][0]['coordinateTransformations']
    scale = next(t['scale'] for t in transforms if t['type'] == 'scale')
    if len(scale) != 4 or not np.all(np.isfinite(scale)) or min(scale) <= 0:
        raise ValueError('Missing/invalid physical scale')
    if any(a.get('unit') != 'micrometer' for a in ms['axes'][1:]):
        raise ValueError('Unknown spatial units')
    shape = read_json(path / '0/zarr.json')['shape']
    return shape, scale, transforms, meta['attributes'].get('image_statistics', {})


def validate_graph(nodes, edges, shape, *, prediction=False, spatial_bounds=True):
    if nodes.ndim != 2 or nodes.shape[1] != 5 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('Invalid graph array shapes')
    if not np.issubdtype(nodes.dtype, np.integer) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Noninteger graph')
    ids = nodes[:, 0]
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate node IDs')
    if np.any(nodes[:, 1] < 0) or np.any(nodes[:, 1] >= shape[0]):
        raise ValueError('Time coordinates out of bounds')
    if spatial_bounds and (np.any(nodes[:, 2:] < 0) or np.any(nodes[:, 2:] >= np.asarray(shape[1:]))):
        raise ValueError('Coordinates out of bounds')
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError('Duplicate edges')
    idx = {int(n): i for i, n in enumerate(ids)}
    if not all(int(n) in idx for n in edges.ravel()):
        raise ValueError('Missing edge endpoint')
    ei = np.array([[idx[int(s)], idx[int(t)]] for s, t in edges], dtype=int).reshape(-1, 2)
    dt = nodes[ei[:, 1], 1] - nodes[ei[:, 0], 1]
    if np.any(dt <= 0) or (prediction and np.any(dt != 1)):
        raise ValueError('Invalid edge time direction')
    indeg = np.bincount(ei[:, 1], minlength=len(nodes))
    outdeg = np.bincount(ei[:, 0], minlength=len(nodes))
    if prediction and (indeg.max(initial=0) > 1 or outdeg.max(initial=0) > 2):
        raise ValueError('Prediction contains merges/invalid forks')
    return ei, dt, indeg, outdeg


def environment():
    packages = {}
    for name in ['torch', 'numpy', 'scipy', 'pandas', 'polars', 'zarr', 'geff', 'tracksdata',
                 'scikit-learn', 'pytest', 'pyarrow', 'matplotlib', 'psutil']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
    metric = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    if metric != METRIC_REV:
        raise ValueError('Official source pin changed')
    result = dict(created=now(), repo_revision=revision(), python=sys.version, executable=sys.executable,
                  platform=platform.platform(), packages=packages, gpu=gpu, metric_revision=metric,
                  metric_files={str(p.relative_to(OFFICIAL)): sha(p) for p in (OFFICIAL/'src/tracking_cellmot').glob('*metrics.py')},
                  data_manifest_hash=sha(REPO/'work/data-manifest.json'),
                  reference_manifest_hash=sha(REPO/'reference/manifest.json'),
                  environment_policy='Isolated venv inherits notebook packages; existing environments unchanged')
    write_json(OUT/'environment.json', result)
    return result


def run():
    stage('S010', 'running')
    paths = sorted((DATA/'train').glob('*.zarr'))
    if {p.stem for p in paths} != {p.stem for p in (DATA/'train').glob('*.geff')}:
        raise ValueError('Image/GEFF set mismatch')
    rows, components, frame_rows, fingerprints = [], [], [], {}
    for p in paths:
        shape, scale, transforms, stats = image_metadata(p)
        gp = p.with_suffix('.geff')
        g = zarr.open_group(gp, mode='r')
        arrays = [g['nodes/ids'][:], *[g[f'nodes/props/{a}/values'][:] for a in 'tzyx']]
        arrays.append(g['edges/ids'][:])
        if any(not np.issubdtype(x.dtype, np.integer) or x.max(initial=0) > np.iinfo(np.int64).max for x in arrays):
            raise ValueError('GEFF cannot be represented losslessly as int64')
        nodes = np.column_stack([x.astype(np.int64) for x in arrays[:-1]])
        edges = arrays[-1].astype(np.int64)
        ei, dt, indeg, outdeg = validate_graph(nodes, edges, shape)
        meta = read_json(gp/'zarr.json')['attributes']['geff']
        gs = [a['scale'] for a in meta['axes']]
        if not np.allclose(gs, scale):
            raise ValueError('GEFF/image scale disagreement')
        estimate = meta['extra']['estimated_number_of_nodes']
        valid = not isinstance(estimate, bool) and isinstance(estimate, (int, float)) and np.isfinite(estimate) and estimate > 0
        hashes = {str(f.relative_to(DATA)): sha(f) for f in sorted(gp.rglob('*')) if f.is_file()}
        hashes.update({str(f.relative_to(DATA)): sha(f) for f in [p/'zarr.json', p/'0/zarr.json']})
        fingerprint = digest(hashes)
        fingerprints[p.stem] = hashes
        row = dict(dataset=p.stem, embryo=p.stem.split('_')[0], image_shape=shape, axes='tzyx',
                   physical_scale=scale[1:], annotated_nodes=len(nodes), annotated_edges=len(edges),
                   gt_divisions=int(np.sum(outdeg == 2)), gt_merges=int(np.sum(indeg > 1)),
                   gt_multiforks=int(np.sum(outdeg > 2)), nonconsecutive_edges=int(np.sum(dt != 1)),
                   duplicate_centers=len(nodes)-len(np.unique(nodes[:, 1:], axis=0)),
                   estimated_total=estimate, estimate_type=type(estimate).__name__,
                   estimate_valid=valid, estimate_source=str(gp/'zarr.json')+'#attributes.geff.extra.estimated_number_of_nodes',
                   reference_coverage=len(nodes)/estimate if valid else None, metadata_hash=fingerprint,
                   coordinate_transforms=transforms, global_offsets_available=any(t['type']=='translation' for t in transforms))
        rows.append(row)
        ncc, labels = connected_components(coo_matrix((np.ones(len(edges)), (ei[:,0], ei[:,1])), shape=(len(nodes),len(nodes))), directed=False)
        for c in range(ncc):
            cns = nodes[labels == c]
            components.append(dict(dataset=p.stem, component=c, nodes=len(cns), t_min=int(cns[:,1].min()), t_max=int(cns[:,1].max())))
        for t in range(shape[0]):
            frame_rows.append(dict(dataset=p.stem,t=t,annotated_nodes=int(np.sum(nodes[:,1]==t))))
        save_graph(OUT/'evaluation/gt'/f'{p.stem}.npz', nodes, edges)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT/'sample_inventory.csv', index=False)
    pd.DataFrame(components).to_csv(OUT/'evaluation/annotation_components.csv', index=False)
    pd.DataFrame(frame_rows).to_csv(OUT/'evaluation/annotation_frames.csv', index=False)
    write_json(OUT/'data_fingerprints.json', fingerprints)
    write_json(OUT/'inventory.json', rows)
    write_json(OUT/'inventory_summary.json', dict(sample_count=len(rows), embryos=df.groupby('embryo')[['annotated_nodes','annotated_edges','gt_divisions','estimated_total']].sum().to_dict('index'),
               data_hash=digest(fingerprints), integrity_errors=[], global_offsets_available=bool(df.global_offsets_available.all())))
    stage('S010','census_complete_split_pending', samples=len(rows), nodes=int(df.annotated_nodes.sum()), edges=int(df.annotated_edges.sum()))
    print(df.groupby('embryo')[['annotated_nodes','annotated_edges','gt_divisions','estimated_total']].sum().to_string(),flush=True)


def provenance():
    root = DATA.parents[1]/'datasets/pilkwang'
    records=[]
    for slug in ['biohub-tracking-support-pack-50ep-v1','biohub-deepcenter-unet3d-center-prior-v1','biohub-temporal-unet3d-seed314159-v1']:
        folder=root/slug
        splits=[]
        for p in folder.rglob('*split*.json'):
            obj=read_json(p)
            if isinstance(obj,dict):
                splits.append(dict(path=str(p),hash=sha(p),train_embryos=dict(collections.Counter(n.split('_')[0] for n in obj.get('train',[]))),
                                   val_embryos=dict(collections.Counter(n.split('_')[0] for n in obj.get('val',obj.get('test',[]))))))
        weights=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(folder.rglob('*')) if p.suffix in ('.pt','.pth')]
        records.append(dict(dataset=slug,splits=splits,weights=weights,lane='diagnostic',
                            reason='Unknown training provenance' if not splits else 'Training and/or checkpoint selection used supplied embryos'))
    write_json(OUT/'public_provenance.json',records)
