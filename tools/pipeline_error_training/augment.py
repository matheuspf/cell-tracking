"""Fixed conservative source augmentations with image/geometry symmetry."""
import torch
from torch.nn import functional as F


CONFIG = dict(xy_flips=True, xy_right_angle_rotations=True, brightness=[.9, 1.1],
              contrast=[.9, 1.1], gamma=[.9, 1.1], read_noise_std=.01,
              context_dropout_probability=.1, context_dropout_relative_frames=[-1, 1], blur_probability=.2, blur_kernel=[.25, .5, .25],
              sampling_scale=[.97, 1.03], shared_center_jitter_um_zyx=[.4, .2, .2],
              coordinate_semantics='Coherent XY symmetries preserve every distance/angle scalar feature; never exchange Z with XY or reverse time.',
              spatial_semantics='All tokens share one physical sampling shift and support-scale change; world node coordinates and their relative geometry stay fixed. This perturbs image sampling, not graph identity.')


def spatial_view(patch, family, zoom, shift_um, blur):
    """Common physical crop perturbation; Z is never treated as an XY axis."""
    if family == 'compact':
        # Each triplane spans the same native high-resolution support as its
        # parent crop. Normalize the shared physical displacement by that span.
        spans = [(63*.40625, 63*.40625), (15*1.625, 63*.40625), (15*1.625, 63*.40625)]
        shifts = [(shift_um[1], shift_um[2]), (shift_um[0], shift_um[2]), (shift_um[0], shift_um[1])]
        result = patch.clone()
        for plane, (span, displacement) in enumerate(zip(spans, shifts)):
            image = patch[:, :, plane].reshape(-1, 1, 12, 12)
            matrix = image.new_tensor([[zoom, 0, 2*displacement[1]/span[1]],
                                       [0, zoom, 2*displacement[0]/span[0]]]).expand(len(image), -1, -1)
            grid = F.affine_grid(matrix, image.shape, align_corners=True)
            sampled = F.grid_sample(image, grid, align_corners=True, padding_mode='zeros')
            if blur:
                kernel = image.new_tensor([.25, .5, .25])
                sampled = F.conv2d(F.pad(sampled, (1, 1, 0, 0)), kernel[None, None, None, :])
                sampled = F.conv2d(F.pad(sampled, (0, 0, 1, 1)), kernel[None, None, :, None])
            result[:, :, plane] = sampled.reshape_as(patch[:, :, plane])
        return result
    original_shape = patch.shape
    if family.startswith('organoid'):
        image = patch.permute(0, 4, 1, 2, 3)
        scale_groups = [(image, (2., .32, .32))]
    else:
        # Two support scales are separate channels; the physical shift must be
        # divided by their different voxel spacing before resampling.
        image = patch.reshape(-1, 2, *patch.shape[-3:])
        scale_groups = [(image[:, k:k+1], tuple(s*v for v in [1.625, .40625, .40625])) for k, s in enumerate([1., 2.])]
    parts = []
    for part, spacing in scale_groups:
        span = [(n-1)*s for n, s in zip(part.shape[-3:], spacing)]
        matrix = part.new_tensor([[zoom, 0, 0, 2*shift_um[2]/span[2]],
                                  [0, zoom, 0, 2*shift_um[1]/span[1]],
                                  [0, 0, zoom, 2*shift_um[0]/span[0]]]).expand(len(part), -1, -1)
        grid = F.affine_grid(matrix, part.shape, align_corners=True)
        part = F.grid_sample(part, grid, align_corners=True, padding_mode='zeros')
        if blur:
            channels = part.shape[1]
            kernel = part.new_tensor([.25, .5, .25])
            for axis in range(3):
                kernel_shape = [1, 1, 1]
                kernel_shape[axis] = 3
                weight = kernel.reshape(1, 1, *kernel_shape).expand(channels, 1, *kernel_shape)
                padding = [0, 0, 0]
                padding[axis] = 1
                part = F.conv3d(part, weight, padding=tuple(padding), groups=channels)
        parts.append(part)
    result = torch.cat(parts, 1)
    return result.permute(0, 2, 3, 4, 1) if family.startswith('organoid') else result.reshape(original_shape)


def view(batch, family):
    result = dict(batch)
    patch = batch['patch'].clone()
    valid = batch['valid'].clone()
    flip_x, flip_y = bool(torch.rand(()) < .5), bool(torch.rand(()) < .5)
    turns = int(torch.randint(4, ()))
    if family == 'compact':
        # Planes are YX, ZX, ZY. Apply XY transforms without rotating Z.
        if flip_x:
            patch[:, :, 0] = patch[:, :, 0].flip(-1)
            patch[:, :, 1] = patch[:, :, 1].flip(-1)
        if flip_y:
            patch[:, :, 0] = patch[:, :, 0].flip(-2)
            patch[:, :, 2] = patch[:, :, 2].flip(-1)
        for _ in range(turns):
            old_zx, old_zy = patch[:, :, 1].clone(), patch[:, :, 2].clone()
            patch[:, :, 0] = torch.rot90(patch[:, :, 0], 1, (-2, -1))
            patch[:, :, 1], patch[:, :, 2] = old_zy, old_zx.flip(-1)
    else:
        axes = (2, 3) if family.startswith('organoid') else (-2, -1)
        if flip_x:
            patch = patch.flip(axes[1])
        if flip_y:
            patch = patch.flip(axes[0])
        patch = torch.rot90(patch, turns, axes)
    zoom = float(torch.empty(()).uniform_(.97, 1.03))
    shift = (torch.rand(3)*2-1)*torch.tensor(CONFIG['shared_center_jitter_um_zyx'])
    patch = spatial_view(patch, family, zoom, shift.tolist(), float(torch.rand(())) < .2)
    brightness = float(torch.empty(()).uniform_(.9, 1.1))
    contrast = float(torch.empty(()).uniform_(.9, 1.1))
    gamma = float(torch.empty(()).uniform_(.9, 1.1))
    patch = (((patch-.5)*contrast+.5)*brightness).clamp(0, 1).pow(gamma)
    patch = (patch+torch.randn_like(patch)*.01).clamp(0, 1)
    if float(torch.rand(())) < .1:
        if family.startswith('organoid'):
            k = int(torch.randint(2, ()))*2
            patch[..., k] = 0
            valid[:, k+1] = 0
        else:
            past = float(torch.rand(())) < .5
            k = (0 if past else 2) if family == 'compact' else (1 if past else 3)
            patch[:, k] = 0
            valid[:, k] = 0
    if not family.startswith('organoid'):
        # Noise never fabricates padded future observations.
        shape = [*valid.shape[:2], *([1]*(patch.ndim-2))]
        patch = patch * valid[..., 0].reshape(shape)
    result['patch'], result['valid'] = patch, valid
    return result
