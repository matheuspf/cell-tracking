"""Exact integer-center implementation of the registered native crop sampler.

At scale one, every sample is the mean of eight neighboring uint16 voxels.
At scale two, every sample is an integer voxel. This avoids building millions
of floating-point coordinate triples, while preserving the original quantizer.
"""
import numpy as np

from .crops import OFFSETS, SHAPE, prediction_tracklet


def values(frame, position, scale, shape=SHAPE):
    position = np.asarray(position)
    if not np.equal(position, np.rint(position)).all() or any(n % 2 for n in shape):
        raise ValueError('Exact fast sampler requires integer centers and even support')
    axes = [int(c) + scale*(np.arange(n)-(n-1)/2) for c, n in zip(position, shape)]
    valid_axes = [(a >= 0) & (a <= s-1) for a, s in zip(axes, frame.shape)]
    mask = valid_axes[0][:, None, None] & valid_axes[1][None, :, None] & valid_axes[2][None, None, :]
    lo = [np.floor(a).astype(int) for a in axes]
    if scale == 2:
        clipped = [np.clip(a, 0, s-1) for a, s in zip(lo, frame.shape)]
        out = frame[np.ix_(*clipped)].astype(np.float32)
    elif scale == 1:
        out = np.zeros(shape, np.float32)
        for z in [0, 1]:
            for y in [0, 1]:
                for x in [0, 1]:
                    clipped = [np.clip(a+d, 0, s-1) for a, d, s in zip(lo, [z, y, x], frame.shape)]
                    out += frame[np.ix_(*clipped)]
        out *= .125
    else:
        raise ValueError('Only registered scales one and two are supported')
    return out, mask


def sample_native(images, nodes, pred, succ, index, shape=SHAPE):
    positions, tracked = prediction_tracklet(nodes, pred, succ, index)
    t0 = int(nodes[index, 1])
    patch = np.zeros((7, 2, *shape), np.uint8)
    valid = np.zeros((7, 4), np.float32)
    for j, dt in enumerate(OFFSETS):
        frame = images.raw(t0+dt)
        if frame is None:
            continue
        lo, hi = images.quantiles[t0+dt]
        valid[j, :2] = [1., float(tracked[dt])]
        for k, scale in enumerate([1, 2]):
            sampled, mask = values(frame, positions[dt], scale, shape)
            normalized = np.clip((sampled-lo)/max(float(hi-lo), 1.), 0, 1)*mask
            patch[j, k] = np.rint(normalized*255).astype(np.uint8)
            valid[j, 2+k] = mask.mean()
    return patch, valid


def sample_gpu(images, nodes, pred, succ, indices):
    """Batch exact integer crops on CUDA; frame reads/quantiles remain identical."""
    import torch
    indices = list(map(int, indices))
    n = len(indices)
    patch = torch.zeros((n, 7, 2, *SHAPE), dtype=torch.uint8, device='cuda')
    valid = np.zeros((n, 7, 4), np.float32)
    tracklets = [prediction_tracklet(nodes, pred, succ, i) for i in indices]
    by_frame = {}
    for k, i in enumerate(indices):
        for j, dt in enumerate(OFFSETS):
            t = int(nodes[i, 1])+dt
            if 0 <= t < images.shape[0]:
                by_frame.setdefault(t, []).append((k, j, tracklets[k][0][dt], tracklets[k][1][dt]))
    for t, entries in sorted(by_frame.items()):
        raw = torch.as_tensor(images.raw(t).astype(np.int32), device='cuda')
        lo_q, hi_q = images.quantiles[t]
        centers = torch.as_tensor(np.asarray([e[2] for e in entries]), dtype=torch.int64, device='cuda')
        for k, scale in enumerate([1, 2]):
            axes = [centers[:, a, None]+scale*(torch.arange(s, device='cuda')-(s-1)/2)
                    for a, s in enumerate(SHAPE)]
            va = [(a >= 0) & (a <= s-1) for a, s in zip(axes, raw.shape)]
            mask = va[0][:, :, None, None] & va[1][:, None, :, None] & va[2][:, None, None, :]
            base = [a.floor().long() for a in axes]
            total = torch.zeros((len(entries), *SHAPE), dtype=torch.float32, device='cuda')
            for dz in range(2 if scale == 1 else 1):
                for dy in range(2 if scale == 1 else 1):
                    for dx in range(2 if scale == 1 else 1):
                        aa = [(a+d).clamp(0, s-1) for a, d, s in zip(base, [dz, dy, dx], raw.shape)]
                        total += raw[aa[0][:, :, None, None], aa[1][:, None, :, None], aa[2][:, None, None, :]]
            if scale == 1:
                total *= .125
            # CUDA and CPU division can fall on opposite sides of a half-integer
            # quantizer tie. Keep the registered NumPy normalization/rounding;
            # only the exact integer/half-integer image gather runs on CUDA.
            host = total.cpu().numpy()
            normalized = np.clip((host-lo_q)/max(float(hi_q-lo_q), 1.), 0, 1)*mask.cpu().numpy()
            quantized = torch.as_tensor(np.rint(normalized*255).astype(np.uint8), device='cuda')
            fractions = mask.float().mean((1, 2, 3)).cpu().numpy()
            for b, (i, j, _, tracked) in enumerate(entries):
                patch[i, j, k] = quantized[b]
                valid[i, j, :2] = [1., float(tracked)]
                valid[i, j, k+2] = fractions[b]
        del raw
    return patch, torch.as_tensor(valid, device='cuda')


def parity():
    import time
    import torch
    from .common import WORK, adjacency, inputs, verified_graph, write_json
    from .crops import Images, sample_native as reference
    from .resources import Lease, Monitor
    torch.set_num_threads(1)
    row = next(r for r in inputs() if r['embryo'] == '44b6')
    graph = verified_graph(row)
    _, pred, succ = adjacency(graph['nodes'], graph['edges'])
    indices = [int(np.flatnonzero(graph['nodes'][:, 1] == t)[0]) for t in [0, 20, 99]]
    images = Images(row['image_path'])
    begin = time.monotonic()
    expected = [reference(images, graph['nodes'], pred, succ, i) for i in indices]
    reference_seconds = time.monotonic()-begin
    begin = time.monotonic()
    actual = [sample_native(images, graph['nodes'], pred, succ, i) for i in indices]
    cpu_seconds = time.monotonic()-begin
    for a, b in zip(expected, actual):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    with Monitor(WORK / 'resources/fast_crop_parity.json'), Lease(required_gib=4.):
        begin = time.monotonic()
        p, v = sample_gpu(images, graph['nodes'], pred, succ, indices)
        torch.cuda.synchronize()
        gpu_seconds = time.monotonic()-begin
        np.testing.assert_array_equal(p.cpu().numpy(), np.stack([a[0] for a in expected]))
        np.testing.assert_array_equal(v.cpu().numpy(), np.stack([a[1] for a in expected]))
    receipt = dict(status='measured', source='44b6', cases=3, frames=[0, 20, 99],
        reference_seconds=reference_seconds, cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
        exact_pixels=True, exact_masks=True, arithmetic='uint16 integer centers; eight-voxel mean or direct integer gather',
        target_labels_read=False)
    write_json(WORK / 'fast_crop_parity.json', receipt)
    print(receipt, flush=True)
