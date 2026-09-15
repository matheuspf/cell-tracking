"""Bounded raw-frame reads and prediction-centered native temporal crops."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .common import read_json, sha

OFFSETS = tuple(range(-2, 5))
SHAPE = (16, 64, 64)
NORMALIZATION = dict(low_quantile=.01, high_quantile=.99, per_frame=True,
                     padding=0., native_storage='uint8', target_fitted_statistics=False)


class Images:
    def __init__(self, path, max_frames=9):
        import zarr
        self.path, self.max_frames = Path(path), max_frames
        root = read_json(self.path / 'zarr.json')['attributes']
        multi = root.get('ome', root)['multiscales'][0]
        if [a['name'].lower() for a in multi['axes']] != ['t', 'z', 'y', 'x']:
            raise ValueError('Expected native TZYX image axes')
        self.scale = np.asarray(multi['datasets'][0]['coordinateTransformations'][0]['scale'][1:])
        self.array = zarr.open_group(str(path), mode='r')['0']
        self.shape = self.array.shape
        self.frames = OrderedDict()
        self.quantiles = {}
        self.read_hashes = {}

    def raw(self, t):
        if not 0 <= t < self.shape[0]:
            return None
        if t in self.frames:
            self.frames.move_to_end(t)
            return self.frames[t]
        chunk = self.path / f'0/c/{t}/0/0/0'
        if not chunk.is_file() or not chunk.stat().st_size:
            raise FileNotFoundError(f'Missing image frame chunk: {chunk}')
        frame = np.asarray(self.array[t])
        self.read_hashes[t] = sha(chunk)
        self.quantiles[t] = tuple(np.quantile(frame, [.01, .99]))
        self.frames[t] = frame
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)
        return frame

    def get_image(self, time_point):
        from organoid_tracker.core.images import Image
        frame = self.raw(time_point.time_point_number())
        return Image(frame) if frame is not None else None


def prediction_tracklet(nodes, pred, succ, index):
    positions = {0: nodes[index, 2:].astype(float)}
    found = {0: True}
    for direction, adjacency in [(-1, pred), (1, succ)]:
        current = index
        for step in range(1, 3 if direction == -1 else 5):
            if current is not None and len(adjacency[current]) == 1:
                current = adjacency[current][0]
                positions[direction*step] = nodes[current, 2:].astype(float)
                found[direction*step] = True
            else:
                current = None
                # A fixed-center image is still observed; identity correspondence is masked.
                positions[direction*step] = positions[direction*(step-1)]
                found[direction*step] = False
    return positions, found


def sample_native(images, nodes, pred, succ, index, shape=SHAPE):
    positions, tracked = prediction_tracklet(nodes, pred, succ, index)
    t0 = int(nodes[index, 1])
    patch = np.zeros((7, 2, *shape), np.uint8)
    valid = np.zeros((7, 4), np.float32)
    offsets = np.stack(np.meshgrid(*[np.arange(n)-(n-1)/2 for n in shape], indexing='ij'))
    spatial = np.asarray(images.shape[1:])[:, None, None, None]
    for j, dt in enumerate(OFFSETS):
        frame = images.raw(t0+dt)
        if frame is None:
            continue
        lo, hi = images.quantiles[t0+dt]
        valid[j, :2] = [1., float(tracked[dt])]
        for k, scale in enumerate([1., 2.]):
            coords = positions[dt][:, None, None, None] + scale*offsets
            mask = np.all((coords >= 0) & (coords <= spatial-1), axis=0)
            values = map_coordinates(frame.astype(np.float32), coords, order=1, mode='constant', cval=0.)
            values = np.clip((values-lo)/max(float(hi-lo), 1.), 0, 1) * mask
            patch[j, k] = np.rint(values*255).astype(np.uint8)
            valid[j, 2+k] = mask.mean()
    return patch, valid


def compact_view(native):
    """Fixed three-frame 12x12 triplanes from the same normalized native context."""
    from scipy.ndimage import zoom
    out = []
    for frame in native[1:4, 0]:
        planes = [frame[frame.shape[0]//2], frame[:, frame.shape[1]//2], frame[:, :, frame.shape[2]//2]]
        out.append(np.stack([zoom(p, (12/p.shape[0], 12/p.shape[1]), order=1) for p in planes]))
    return np.asarray(out, np.uint8)


def sample_at_coordinate(frame, position):
    """Used by observation tests to require features at the actual replacement."""
    return float(map_coordinates(np.asarray(frame, np.float32), np.asarray(position)[:, None],
                                 order=1, mode='constant', cval=0.)[0])
