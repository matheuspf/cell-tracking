"""One frozen module perturbation, with identical native crops across controls."""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F


def perturb(patch, nodes, indices, spacing, recipe, image_identity):
    """Float N,T,S,Z,Y,X patches; deterministic by image and physical center."""
    out = patch.clone()
    n, frames, scales, depth, height, width = out.shape
    spacing = np.asarray(spacing, float).reshape(scales, 3)
    for scale in range(scales):
        p = out[:, :, scale].reshape(n*frames, 1, depth, height, width)
        theta = torch.zeros((len(p), 3, 4), device=p.device, dtype=p.dtype)
        theta[:, :3, :3] = torch.eye(3, device=p.device)*recipe['sampling_scale']
        shift = np.asarray(recipe['center_shift_um_zyx'])/spacing[scale]
        theta[:, :, 3] = torch.as_tensor((2*shift/np.array([depth, height, width]))[::-1].copy(), device=p.device, dtype=p.dtype)
        grid = F.affine_grid(theta, p.shape, align_corners=False)
        p = F.grid_sample(p, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        out[:, :, scale] = p.reshape(n, frames, depth, height, width)
    center = out.mean(dim=(-3, -2, -1), keepdim=True)
    out = ((out-center)*recipe['contrast']+center)*recipe['brightness']
    out = out.clamp(0, 1).pow(recipe['gamma'])
    for k, index in enumerate(indices):
        key = str([image_identity, list(map(int, nodes[index, 1:])), recipe['seed']]).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
        generator = torch.Generator(device=out.device).manual_seed(seed)
        out[k] += torch.randn(out[k].shape, device=out.device, generator=generator)*recipe['read_noise_std']
    p = out.reshape(n*frames*scales, 1, depth, height, width)
    kernel = torch.as_tensor(recipe['blur_kernel'], device=p.device, dtype=p.dtype)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]; shape[axis+2] = 3
        padding = [0, 0, 0, 0, 0, 0]
        padding[2*(2-axis):2*(2-axis)+2] = [1, 1]
        p = F.conv3d(F.pad(p, padding, mode='replicate'), kernel.reshape(shape))
    return p.reshape_as(out).clamp(0, 1)


def install(recipe):
    """Patch only this study's in-memory adapters in an isolated stress worker."""
    from . import infer, organoid_adapter, scoring
    original_sampler, original_patches, original_load = infer.sample_gpu, organoid_adapter.patches, scoring.load_model
    def native(images, nodes, pred, succ, indices):
        patch, valid = original_sampler(images, nodes, pred, succ, indices)
        spacing = np.stack([images.scale, 2*images.scale])
        out = perturb(patch.float()/255, nodes, indices, spacing, recipe, str(images.path))
        out[:, 3] = 0
        valid = valid.clone(); valid[:, 3] = 0
        return (out*255).round().to(torch.uint8), valid
    def organoid(images, nodes, indices):
        original = original_patches(images, nodes, indices)
        patch = torch.as_tensor(original, device='cuda').permute(0, 4, 1, 2, 3).unsqueeze(2)
        out = perturb(patch, nodes, indices, [[2., .32, .32]], recipe, str(images.path))
        out[:, 2] = 0
        return out.squeeze(2).permute(0, 2, 3, 4, 1).cpu().numpy()
    def load(path):
        model, spec = original_load(path)
        def mask_context(module, args):
            patch, valid = args
            mask = valid.clone(); mask[:, 2 if model.family=='compact' else 3] = 0
            return patch, mask
        model.encoder.register_forward_pre_hook(mask_context)
        return model, spec
    infer.sample_gpu, organoid_adapter.patches, scoring.load_model, infer.load_model = native, organoid, load, load
