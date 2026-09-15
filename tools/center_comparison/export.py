"""Export real images, native mask IDs, centers, and independently computed matches."""
from __future__ import annotations

import gzip
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from .pipeline import RADII, sha256, write_json

METHODS = {
    'detector': 'Current detector · pre-ILP',
    'selected': 'Selected v3 tracker',
    'watershed': 'V5 seeded-watershed centroid',
    'focus_centroid': 'Geometric centroid',
    'focus_weighted': 'Intensity-weighted centroid',
    'focus_peak': 'Smoothed intensity peak',
}


def extract_centers(labels, raw):
    from scipy.ndimage import gaussian_filter
    from skimage.measure import regionprops
    if labels.shape != raw.shape or labels.dtype.kind not in 'iu' or labels.min() < 0:
        raise ValueError('Masks must be nonnegative native-grid integer labels')
    smooth = gaussian_filter(raw.astype(np.float32), (.5, 1, 1))
    modes = {name: [] for name in ('centroid', 'weighted', 'peak')}
    sizes = {}
    for region in regionprops(labels, intensity_image=smooth):
        positions = np.argwhere(region.image) + np.array(region.bbox[:3])
        intensities = smooth[region.slice][region.image].astype(float)
        weights = np.maximum(intensities - np.quantile(intensities, .1), 0)
        centers = dict(centroid=region.centroid,
                       weighted=np.average(positions, axis=0, weights=weights) if weights.sum() else region.centroid,
                       peak=positions[np.argmax(intensities)])
        for mode, center in centers.items():
            modes[mode].append([region.label, *center])
        sizes[int(region.label)] = int(region.area)
    return {f'focus_{k}': np.asarray(v, float).reshape(-1, 4) for k, v in modes.items()}, sizes


def point_rows(nodes, shape):
    """[ID, submitted ZYX, exact ZYX]; use NumPy's submission rounding, also in UI."""
    nodes = np.asarray(nodes).reshape(-1, 4)
    if not np.isfinite(nodes).all():
        raise ValueError('Nonfinite center')
    if len(set(nodes[:, 0])) != len(nodes):
        raise ValueError('Duplicate point ID within frame')
    submitted = np.clip(np.rint(nodes[:, 1:]), 0, np.asarray(shape) - 1).astype(int)
    return [[int(row[0]), *map(int, rounded), *map(float, row[1:])]
            for row, rounded in zip(nodes, submitted)]


def official_matches(predictions, annotations, spacing, radius):
    """Use tracksdata itself; IDs are explicitly mapped back to exported IDs."""
    import polars as pl
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import set_options
    set_options(show_progress=False)
    if not len(predictions) or not len(annotations):
        return []
    def make(rows):
        graph = td.graph.IndexedRXGraph()
        for axis in 'zyx':
            graph.add_node_attr_key(axis, pl.Float64, -999999.)
        indices = graph.bulk_add_nodes([dict(t=0, z=float(r[1]), y=float(r[2]), x=float(r[3])) for r in rows])
        return graph, dict(zip(indices, rows))
    pred, pmap = make(predictions)
    gt, gmap = make(annotations)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred.match(gt, matching=DistanceMatching(max_distance=radius, scale=tuple(spacing), optimal=True))
    pairs = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matches = []
    for a, b in pairs.iter_rows():
        if b is None or b == -1:
            continue
        p, g = pmap[a], gmap[b]
        distance = float(np.linalg.norm((np.asarray(p[1:4]) - g[1:4]) * spacing))
        matches.append([int(p[0]), int(g[0]), distance])
    if len({p for p, _, _ in matches}) != len(matches) or len({g for _, g, _ in matches}) != len(matches):
        raise ValueError('Assignment is not one-to-one')
    return matches


def preview_links(previous, current, spacing, radius=7.):
    """GT-free, one-to-one, distance-gated preview; no division or gap model."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    if not len(previous) or not len(current):
        return []
    a = np.asarray(previous)
    b = np.asarray(current)
    distances = cdist(a[:, 4:7] * spacing, b[:, 4:7] * spacing)
    # Penalty larger than every feasible assignment's total gives maximum
    # cardinality first, then minimum distance among feasible links.
    penalty = (min(distances.shape) + 1) * (radius + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    return [[int(a[i, 0]), int(b[j, 0]), float(distances[i, j])]
            for i, j in zip(rows, cols) if distances[i, j] <= radius]


def lineage_roots(nodes, edges):
    parent = {int(n[0]): int(n[0]) for n in nodes}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        ra, rb = root(int(a)), root(int(b))
        parent[max(ra, rb)] = min(ra, rb)
    return {i: root(i) for i in parent}


def write_gzip(path, array):
    with Path(path).open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, compresslevel=6) as zipped:
            zipped.write(np.ascontiguousarray(array).tobytes())


def export(args):
    import tifffile
    manifest = json.loads((args.output / 'manifest.json').read_text())
    site = args.output / 'site'
    (site / 'frames').mkdir(parents=True, exist_ok=True)
    frames = {}
    datasets = list(dict.fromkeys(f['dataset'] for f in manifest['frames']))
    display_limits = {}
    # Fixed limits per clip, pooled over all exported frames. Never auto-normalize
    # each timepoint independently, which would introduce flicker.
    for dataset in datasets:
        hist = np.zeros(65536, np.int64)
        for frame in manifest['frames']:
            if frame['dataset'] == dataset:
                raw = tifffile.imread(args.output / 'frames' / frame['key'] / 'image.tif')
                hist += np.bincount(raw.ravel(), minlength=65536)
        cumulative = hist.cumsum()
        display_limits[dataset] = [int(np.searchsorted(cumulative, cumulative[-1] * p)) for p in (.01, .999)]
    for frame in manifest['frames']:
        key = frame['key']
        dest = args.output / 'frames' / key
        mask_path = dest / 'masks.tif'
        if not (dest / 'focus.json').exists() or not mask_path.exists():
            raise FileNotFoundError(f'FOCUS is incomplete: {key}; finish the focus stage first')
        labels = tifffile.imread(mask_path)
        raw = tifffile.imread(dest / 'image.tif')
        if sha256(dest / 'image.tif') != frame['image_sha256']:
            raise ValueError(f'Prepared image changed: {key}')
        if list(raw.shape) != frame['shape']:
            raise ValueError(f'Image dimensions changed: {key}')
        focus, sizes = extract_centers(labels, raw)
        with np.load(dest / 'points.npz') as saved:
            points = {k: point_rows(saved[k][:, [0, 2, 3, 4]], raw.shape)
                      for k in ['gt', 'detector', 'selected', 'watershed']}
            confidence = saved['confidence'].tolist()
        points.update({k: point_rows(v, raw.shape) for k, v in focus.items()})
        spacing = np.asarray(frame['spacing'])
        matches = {k: {str(int(radius)): official_matches(rows, points['gt'], spacing, radius)
                       for radius in RADII} for k, rows in points.items() if k != 'gt'}
        members = {int(label): {'gt': [], 'detector': [], 'selected': [], 'watershed': []} for label in sizes}
        for method in ['gt', 'detector', 'selected', 'watershed']:
            for row in points[method]:
                label = int(labels[tuple(row[1:4])])
                if label:
                    members[label][method].append(row[0])
        lo, hi = display_limits[frame['dataset']]
        display = np.clip((raw.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
        write_gzip(site / 'frames' / f'{key}.gray.gz', display)
        write_gzip(site / 'frames' / f'{key}.labels.gz', labels.astype('<u4'))
        frames[key] = dict(key=key, dataset=frame['dataset'], t=frame['t'], shape=frame['shape'],
                          spacing=frame['spacing'], points=points, matches=matches, sizes=sizes,
                          members=members, detector_confidence=confidence, links={}, tracks={},
                          focus=json.loads((dest / 'focus.json').read_text()),
                          display_limits=[lo, hi], image_sha256=frame['image_sha256'], mask_sha256=sha256(mask_path))
        print(f'EXPORTED {key}', flush=True)
    for dataset in datasets:
        by_time = {f['t']: f for f in frames.values() if f['dataset'] == dataset}
        with np.load(args.output / 'graphs' / f'{dataset}.npz') as graph:
            for method in ['gt', 'selected']:
                nodes, edges = graph[f'{method}_nodes'], graph[f'{method}_edges']
                roots = lineage_roots(nodes, edges)
                times = {int(n[0]): int(n[1]) for n in nodes}
                for f in by_time.values():
                    f['tracks'][method] = {r[0]: roots[r[0]] for r in f['points'][method]}
                    f['links'][method] = []
                for a, b in edges:
                    ta, tb = times[int(a)], times[int(b)]
                    if ta in by_time and tb in by_time:
                        by_time[tb]['links'][method].append([ta, int(a), int(b)])
        for mode in ['focus_centroid', 'focus_weighted', 'focus_peak']:
            for t in sorted(by_time):
                f = by_time[t]
                roots = {r[0]: t * 1000000 + r[0] for r in f['points'][mode]}
                f['links'][mode] = []
                if t - 1 in by_time:
                    previous = by_time[t - 1]
                    for a, b, _ in preview_links(previous['points'][mode], f['points'][mode], np.asarray(f['spacing'])):
                        roots[b] = previous['tracks'][mode][a]
                        f['links'][mode].append([t - 1, a, b])
                f['tracks'][mode] = roots
    for key, frame in frames.items():
        write_json(site / 'frames' / f'{key}.json', frame)
    index = dict(version=1, created=manifest['created'], sequences=manifest['sequences'], methods=METHODS,
        radii=list(RADII), frame_keys=list(frames), display_limits=display_limits,
        point_columns=['id', 'submitted_z', 'submitted_y', 'submitted_x', 'exact_z', 'exact_y', 'exact_x'],
        selection=manifest['selection'],
        matching='Installed tracksdata DistanceMatching(optimal=True), rerun at each radius; integer-rounded native-grid centers; distances in µm.',
        focus_tracking='Display-only one-to-one optimal centroid association between consecutive exported frames; 7 µm gate, exact centers, no GT, no divisions or gap closing.',
        limitations='Sparse annotations; unmatched predictions are not confirmed false positives. Two reused embryos; exploratory comparison, not independent validation. FOCUS masks have no dense ground truth.',
        center_rules={
            'focus_centroid': 'Mean native-voxel position of every voxel belonging to the instance.',
            'focus_weighted': 'Position weighted by Gaussian-smoothed image intensity (sigma ZYX 0.5,1,1), after subtracting the instance 10th-percentile intensity; geometric fallback when weights sum to zero.',
            'focus_peak': 'Brightest voxel within the instance after Gaussian smoothing (sigma ZYX 0.5,1,1).',
            'watershed': 'Cached geometric centroid of a valid incumbent-seeded V5 watershed region; missing regions omitted.'})
    write_json(site / 'index.json', index)
    for name in ['index.html', 'viewer.js', 'error_view.js', 'style.css', 'detection_view.js', 'detection_style.css', 'tracking_view.js', 'tracking_style.css']:
        shutil.copyfile(Path(__file__).with_name(name), site / name)
    receipt = dict(frames=len(frames), sequences=len(index['sequences']),
        continuous_frames=sum(len(s['times']) for s in index['sequences'] if s['continuous']),
        methods=list(METHODS), source_manifest_sha256=sha256(args.output / 'manifest.json'),
        index_sha256=sha256(site / 'index.json'),
        frame_checks=[dict(key=f['key'], annotations=len(f['points']['gt']), masks=len(f['sizes']),
                          detector_matches_7=len(f['matches']['detector']['7']),
                          focus_matches_7=len(f['matches']['focus_centroid']['7'])) for f in frames.values()])
    write_json(args.output / 'export_receipt.json', receipt)
    from .best_predictions import add_best
    add_best(args)
    print(f'Viewer ready: {site / "index.html"}', flush=True)
