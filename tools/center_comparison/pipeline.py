"""Prepare native volumes and run the same frozen FOCUS nuclei recipe as the pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = Path('/kaggle/working/cell-tracking')
DEFAULT_OUTPUT = WORK / 'center-comparison'
DEFAULT_DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
PILOT = REPO / 'work/segmentation-center-review'
FOCUS_SOURCE = Path('/home/mpf/code/kaggle/FOCUS-3D')
CHECKPOINT = WORK / 'remote-backups/20260910T224437Z-83783a2ae8c0/rootfs/root/FOCUS-3D/model_final_nuclei.pth'
CHECKPOINT_SHA = 'b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a'
RADII = (1., 2., 3., 5., 7.)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_geometry(data, dataset):
    folder = Path(data) / 'train' / f'{dataset}.zarr'
    meta = json.loads((folder / '0/zarr.json').read_text())
    root = json.loads((folder / 'zarr.json').read_text())['attributes']['multiscales'][0]
    axes = [a['name'].lower() for a in root['axes']]
    if axes != ['t', 'z', 'y', 'x'] or meta['data_type'] != 'uint16':
        raise ValueError(f'Unsupported native image layout: {dataset}')
    scale = next(t['scale'] for t in root['datasets'][0]['coordinateTransformations'] if t['type'] == 'scale')
    shape = meta['shape']
    if meta['chunk_grid']['configuration']['chunk_shape'] != [1, *shape[1:]]:
        raise ValueError('The missing-chunk check requires one complete frame per chunk')
    return folder, shape, scale[1:]


def read_gt(data, dataset, spacing):
    import zarr
    graph = zarr.open_group(str(Path(data) / 'train' / f'{dataset}.geff'), mode='r')
    axes = graph.attrs['geff']['axes']
    graph_spacing = [next(a['scale'] for a in axes if a['name'] == axis) for axis in 'zyx']
    if not np.allclose(spacing, graph_spacing):
        raise ValueError('Image and GEFF physical scales disagree')
    nodes = np.column_stack([graph['nodes/ids'][:], *[graph[f'nodes/props/{a}/values'][:] for a in 'tzyx']]).astype(np.int64)
    edges = np.asarray(graph['edges/ids'][:], dtype=np.int64).reshape(-1, 2)
    return nodes, edges


def prepare(args):
    import tifffile
    import zarr
    pilot = json.loads((args.pilot / 'manifest.json').read_text()) if not args.no_pilot else {'frames': []}
    if args.start > args.stop:
        raise ValueError('Start frame must not exceed stop frame')
    sequences = []
    # Chosen from the frozen pilot, before inspecting new FOCUS outcomes.
    for dataset in args.datasets:
        sequences.append(dict(id=f'{dataset}-movie', dataset=dataset,
                              title=f'{dataset} · continuous {args.start}–{args.stop}',
                              times=list(range(args.start, args.stop + 1)), continuous=True))
    if not args.no_pilot:
        for dataset in dict.fromkeys(r['dataset'] for r in pilot['frames']):
            sequences.append(dict(id=f'{dataset}-pilot', dataset=dataset,
                                  title=f'{dataset} · pilot snapshots 25 / 75',
                                  times=[25, 75], continuous=False))
    frames = []
    for dataset in dict.fromkeys(s['dataset'] for s in sequences):
        folder, shape, spacing = read_geometry(args.data_root, dataset)
        gt, gt_edges = read_gt(args.data_root, dataset, spacing)
        primary_path = WORK / f'annotation-selection-v1/public_harmonic_full/inputs/pre_ilp_{dataset}.npz'
        selected_path = WORK / f'strong-tracker-v3/selected_predictions/{dataset}.npz'
        watershed_path = WORK / f'image-native-tracking-v5/observations/{dataset}.npz'
        primary = np.load(primary_path)
        selected = np.load(selected_path)
        watershed = np.load(watershed_path)
        graph_dir = args.output / 'graphs'
        graph_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(graph_dir / f'{dataset}.npz', gt_nodes=gt, gt_edges=gt_edges,
                            selected_nodes=selected['nodes'], selected_edges=selected['edges'])
        array = zarr.open_group(str(folder), mode='r')['0']
        times = sorted({t for s in sequences if s['dataset'] == dataset for t in s['times']})
        for t in times:
            if not 0 <= t < shape[0]:
                raise ValueError(f'Time {t} outside {dataset}')
            # A missing chunk silently reads as zeros in Zarr; never use that as microscopy.
            chunk = folder / f'0/c/{t}/0/0/0'
            if not chunk.is_file() or chunk.stat().st_size == 0:
                raise FileNotFoundError(chunk)
            key = f'{dataset}-t{t:03}'
            dest = args.output / 'frames' / key
            dest.mkdir(parents=True, exist_ok=True)
            raw = np.asarray(array[t])
            if tuple(raw.shape) != tuple(shape[1:]):
                raise ValueError('Native frame shape mismatch')
            if (dest / 'masks.tif').exists():
                if not (dest / 'image.tif').exists() or not np.array_equal(tifffile.imread(dest / 'image.tif'), raw):
                    raise ValueError(f'Existing masks belong to a different image: {key}; use a new output directory')
            tifffile.imwrite(dest / 'image.tif', raw, metadata={'axes': 'ZYX'}, compression='zlib')
            coords = primary['coords']
            keep = coords[:, 0] == t
            detector = np.column_stack([np.flatnonzero(keep), coords[keep]])
            wn = watershed['nodes']
            valid = (wn[:, 1] == t) & watershed['valid_region'] & watershed['oldmask']
            region_nodes = np.column_stack([wn[valid, :2], watershed['optical_centroids'][valid] / spacing])
            sn = selected['nodes']
            np.savez_compressed(dest / 'points.npz', gt=gt[gt[:, 1] == t],
                                detector=detector, confidence=primary['node_probabilities'][keep],
                                selected=sn[sn[:, 1] == t], watershed=region_nodes)
            cached = args.pilot / 'frames' / key
            mask = cached / 'focus-output/image_instance_map.tif'
            cache_info = None
            if mask.exists():
                if not np.array_equal(tifffile.imread(cached / 'image.tif'), raw):
                    raise ValueError(f'Pilot image differs from canonical data: {key}')
                cache_info = json.loads((cached / 'focus.json').read_text())
                if cache_info['checkpoint_sha256'] != CHECKPOINT_SHA:
                    raise ValueError('Pilot uses a different checkpoint')
                shutil.copyfile(mask, dest / 'masks.tif')
                write_json(dest / 'focus.json', dict(cache_info, reused_from=str(mask), mask_sha256=sha256(mask),
                                                   image_sha256=sha256(dest / 'image.tif')))
            frames.append(dict(key=key, dataset=dataset, t=t, shape=shape[1:], spacing=spacing,
                               cached=cache_info is not None, image_sha256=sha256(dest / 'image.tif')))
        for s in sequences:
            if s['dataset'] == dataset:
                s.update(shape=shape[1:], spacing=spacing,
                         source_images=str(folder), source_annotations=str(folder.with_suffix('.geff')),
                         source_detector=str(primary_path), source_tracker=str(selected_path),
                         source_watershed=str(watershed_path))
        primary.close()
        selected.close()
        watershed.close()
    write_json(args.output / 'manifest.json', dict(version=1, sequences=sequences, frames=frames,
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        selection='Two frozen-pilot clips, one per embryo; consecutive times chosen before new inference. Original pilot snapshots retained.',
        checkpoint=str(args.checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
        focus_source=str(args.focus_source), radii=list(RADII)))
    print(f'Prepared {len(frames)} native frames in {len(sequences)} views', flush=True)


def focus(args):
    import gc
    import subprocess
    import sys
    import tifffile
    import torch
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('FOCUS nuclei checkpoint hash mismatch')
    sys.path.insert(0, str(args.focus_source / 'src'))
    from focus3d.segmentation.FOCUS3D.inference import infer_volume
    revision = subprocess.check_output(['git', '-C', str(args.focus_source), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    for frame in manifest['frames']:
        dest = args.output / 'frames' / frame['key']
        if (dest / 'focus.json').exists() and (dest / 'masks.tif').exists():
            receipt = json.loads((dest / 'focus.json').read_text())
            if receipt['checkpoint_sha256'] != CHECKPOINT_SHA:
                raise ValueError(f"Existing mask checkpoint mismatch: {frame['key']}")
            if receipt.get('image_sha256', frame['image_sha256']) != sha256(dest / 'image.tif'):
                raise ValueError(f"Image changed after mask inference: {frame['key']}")
            continue
        start = time.monotonic()
        spacing = frame['spacing']
        result = infer_volume(image_path=str(dest / 'image.tif'),
            config_file=str(args.focus_source / 'src/focus3d/segmentation/FOCUS3D/configs/3d_test.yaml'),
            weights_path=str(args.checkpoint), device='cuda:0', output_dir=dest / 'focus-output',
            z_ratio=spacing[0] / spacing[1], cell_radius=4.5 / spacing[1],
            lower_percentile=1., upper_percentile=99., score_thresh=.5, mask_thresh=.5,
            batch_size=1, data_loader_num_workers=0, size_filter_min_size=0,
            restore_resampled_mask=True)
        labels = np.asarray(result['instance_map'])
        if list(labels.shape) != frame['shape'] or labels.dtype.kind not in 'iu':
            raise ValueError('FOCUS failed to restore the native mask grid')
        tifffile.imwrite(dest / 'masks.tif', labels, compression='zlib', metadata={'axes': 'ZYX'})
        write_json(dest / 'focus.json', dict(seconds=time.monotonic() - start,
            checkpoint_sha256=CHECKPOINT_SHA, source_commit=revision, z_ratio=spacing[0] / spacing[1],
            cell_radius_xy_pixels=4.5 / spacing[1], score_thresh=.5, mask_thresh=.5,
            mask_sha256=sha256(dest / 'masks.tif'), image_sha256=sha256(dest / 'image.tif'), log=result['log_info']))
        print(f"VIEWER_FOCUS_DONE {frame['key']} {time.monotonic() - start:.1f}s", flush=True)
        del result, labels
        gc.collect()
        torch.cuda.empty_cache()


def main():
    from .best_predictions import DEFAULT_EVALUATION, DEFAULT_PREDICTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'focus', 'export', 'add-best', 'index-detection', 'index-tracking', 'serve'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--pilot', type=Path, default=PILOT)
    parser.add_argument('--focus-source', type=Path, default=FOCUS_SOURCE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--datasets', nargs='+', default=['6bba_e16ffc58', '44b6_8f5ab931'])
    parser.add_argument('--start', type=int, default=20)
    parser.add_argument('--stop', type=int, default=35)
    parser.add_argument('--no-pilot', action='store_true')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--audit-limit', type=int, help='Development only: restrict the detection index to this many clips')
    parser.add_argument('--best-predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--best-evaluation', type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'focus':
        focus(args)
    elif args.stage == 'export':
        from .export import export
        export(args)
    elif args.stage == 'add-best':
        from .best_predictions import add_best
        add_best(args)
    elif args.stage == 'index-detection':
        from .detection_index import build
        build(args)
    elif args.stage == 'index-tracking':
        from .tracking_index import build
        build(args)
    else:
        from http.server import ThreadingHTTPServer
        from .detection_server import make_handler
        site = args.output / 'site'
        site.mkdir(parents=True, exist_ok=True)
        for asset in ['index.html', 'viewer.js', 'style.css', 'error_view.js', 'detection_view.js', 'detection_style.css', 'tracking_view.js', 'tracking_style.css']:
            shutil.copyfile(Path(__file__).with_name(asset), site / asset)
        handler = make_handler(args.output)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        print(f'Center comparison: http://localhost:{args.port}', flush=True)
        server.serve_forever()
