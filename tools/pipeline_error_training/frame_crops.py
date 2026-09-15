"""Exact inference crops from bounded, once-normalized native frames.

Integer centers and the registered even support make scale-one interpolation
an eight-voxel mean. Quantize that mean once per frame with the original NumPy
arithmetic, then gather uint8 crops. Scale two samples integer voxels directly.
This does not alter training crops, context, padding, precision or rounding.
"""
from collections import OrderedDict

import numpy as np

from .crops import OFFSETS, SHAPE, prediction_tracklet


def quantized_frames(frame, lo, hi):
    frame = np.asarray(frame)
    if frame.dtype != np.uint16:
        raise ValueError('Exact frame sampler requires the verified uint16 input')
    total = np.zeros(tuple(s-1 for s in frame.shape), np.float32)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                total += frame[dz:frame.shape[0]-1+dz,
                               dy:frame.shape[1]-1+dy,
                               dx:frame.shape[2]-1+dx]
    total *= .125
    denominator = max(float(hi-lo), 1.)
    half = np.rint(np.clip((total-lo)/denominator, 0, 1)*255).astype(np.uint8)
    direct = np.rint(np.clip((frame.astype(np.float32)-lo)/denominator, 0, 1)*255).astype(np.uint8)
    return half, direct


class FrameCache:
    def __init__(self, images):
        self.images = images
        self.frames = OrderedDict()

    def get(self, t):
        import torch
        if t in self.frames:
            self.frames.move_to_end(t)
            return self.frames[t]
        raw = self.images.raw(t)
        arrays = quantized_frames(raw, *self.images.quantiles[t])
        value = tuple(torch.as_tensor(a, device='cuda') for a in arrays)
        self.frames[t] = value
        while len(self.frames) > self.images.max_frames:
            self.frames.popitem(last=False)
        return value


def sample_gpu(images, nodes, pred, succ, indices):
    import torch
    indices = list(map(int, indices))
    if not hasattr(images, '_exact_quantized_frames'):
        images._exact_quantized_frames = FrameCache(images)
    cache = images._exact_quantized_frames
    patch = torch.zeros((len(indices), 7, 2, *SHAPE), dtype=torch.uint8, device='cuda')
    valid = torch.zeros((len(indices), 7, 4), dtype=torch.float32, device='cuda')
    by_frame = {}
    for k, i in enumerate(indices):
        positions, tracked = prediction_tracklet(nodes, pred, succ, i)
        for j, dt in enumerate(OFFSETS):
            t = int(nodes[i, 1])+dt
            if 0 <= t < images.shape[0]:
                by_frame.setdefault(t, []).append((k, j, positions[dt], tracked[dt]))
    for t, entries in sorted(by_frame.items()):
        volumes = cache.get(t)
        positions = np.asarray([e[2] for e in entries])
        if not np.equal(positions, np.rint(positions)).all():
            raise ValueError('Exact frame sampler requires integer tracklet centers')
        centers = torch.as_tensor(positions, dtype=torch.int64, device='cuda')
        ii = torch.as_tensor([e[0] for e in entries], device='cuda')
        jj = torch.as_tensor([e[1] for e in entries], device='cuda')
        valid[ii, jj, 0] = 1.
        valid[ii, jj, 1] = torch.as_tensor([e[3] for e in entries], device='cuda', dtype=torch.float32)
        for k, (scale, volume) in enumerate(zip([1, 2], volumes)):
            axes = [centers[:, a, None] + scale*torch.arange(s, device='cuda')
                    - (s//2 if scale == 1 else s-1) for a, s in enumerate(SHAPE)]
            ok = [(a >= 0) & (a < s) for a, s in zip(axes, volume.shape)]
            mask = ok[0][:, :, None, None] & ok[1][:, None, :, None] & ok[2][:, None, None, :]
            clipped = [a.clamp(0, s-1) for a, s in zip(axes, volume.shape)]
            values = volume[clipped[0][:, :, None, None], clipped[1][:, None, :, None], clipped[2][:, None, None, :]]
            patch[ii, jj, k] = values*mask
            valid[ii, jj, k+2] = mask.float().mean((1, 2, 3))
    return patch, valid
