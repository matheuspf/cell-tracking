"""Frozen Cellpose features; no annotation loading or fitted preprocessing here."""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .common import CHECKPOINT, CHECKPOINT_SHA, SPACING, config, native_to_isotropic, sha


def normalize(volume):
    from cellpose import models, transforms
    x = transforms.convert_image(volume.copy(), z_axis=0, do_3D=True)
    params = {**models.normalize_default, "normalize": True, "norm3D": True}
    return transforms.normalize_img(x, **params)[..., 0]


def patches(volume, queries, shape=(7, 25, 25)):
    result = np.empty((len(queries), *shape), dtype=np.float16)
    mesh = np.stack(np.meshgrid(*[np.arange(s) - s // 2 for s in shape], indexing="ij"), axis=0)
    for begin in range(0, len(queries), 128):
        q = queries[begin:begin + 128]
        coords = q.T[:, :, None, None, None] + mesh[:, None]
        sample = map_coordinates(volume, coords, order=1, mode="nearest", prefilter=False)
        result[begin:begin + len(q)] = np.clip(sample, -1, 5)
    return result


def sample_tokens(tokens, plane_indices, uv, pad, token_shape):
    """Convolution centers are at 3.5+8*i in padded pixel coordinates."""
    h, w = token_shape
    loc = (torch.as_tensor(uv, device=tokens.device, dtype=torch.float32) +
           torch.tensor(pad, device=tokens.device) - 3.5) / 8
    lo = torch.floor(loc).long()
    fraction = loc - lo
    p = torch.as_tensor(plane_indices, device=tokens.device, dtype=torch.long)
    answer = torch.zeros((len(p), tokens.shape[-1]), device=tokens.device)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        yy = (lo[:, 0] + dy).clamp(0, h - 1)
        xx = (lo[:, 1] + dx).clamp(0, w - 1)
        weight = (fraction[:, 0] if dy else 1 - fraction[:, 0]) * (fraction[:, 1] if dx else 1 - fraction[:, 1])
        answer += tokens[p, yy * w + xx].float() * weight[:, None]
    return answer


class FrozenFeatures:
    def __init__(self, device="cuda:0"):
        from cellpose.vit import CPDINO
        if sha(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError("Cellpose checkpoint hash mismatch")
        self.device = torch.device(device)
        self.net = CPDINO(model_name="vitb", dtype=torch.bfloat16)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        errors = self.net.load_state_dict(state, strict=False)
        if errors.unexpected_keys or set(errors.missing_keys) - {"diam_labels", "diam_mean"}:
            raise ValueError(str(errors))
        self.net.eval().requires_grad_(False).to(self.device)
        self.tokens = None
        self.hook = self.net.out.register_forward_pre_hook(self._capture)
        self.settings = config()["features"]
        self.last_seconds = {}

    def _capture(self, module, inputs):
        self.tokens = inputs[0].detach()

    @torch.inference_mode()
    def batch(self, images):
        self.tokens = None
        x = torch.as_tensor(images, device=self.device, dtype=torch.bfloat16)
        logits, _unused_random_style = self.net(x)
        assert self.tokens is not None and not self.tokens.requires_grad
        return self.tokens, logits

    @torch.inference_mode()
    def extract(self, volume, queries):
        from cellpose import transforms
        started = time.perf_counter()
        queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
        if tuple(volume.shape) != (64, 256, 256):
            raise ValueError("This locked recipe requires the verified native geometry")
        if not np.isfinite(queries).all() or not ((queries >= 0) & (queries <= np.asarray(volume.shape) - 1)).all():
            raise ValueError("Invalid query coordinates")
        normalized = normalize(volume)
        local = patches(normalized, queries, tuple(self.settings["patch_shape_zyx"]))
        geometry = ((queries - np.rint(queries)) * SPACING).astype(np.float32)
        if not len(queries):
            return np.zeros((0, 3, 768), np.float16), local, geometry
        image = transforms.resize_image_3d(normalized[..., None], (256, 256, 256), no_channels=False)
        iso_queries = native_to_isotropic(queries)
        pooled = np.empty((len(queries), 3, 768), dtype=np.float16)
        batch_size = self.settings["batch_size"]
        bsize = self.settings["plane_size"]
        planes_processed = 0
        normalized_at = time.perf_counter()
        for view, permutation in enumerate(self.settings["plane_order"]):
            stack = image.transpose(*permutation, 3)
            q = iso_queries[:, permutation]
            through = np.clip(np.rint(q[:, 0]), 0, stack.shape[0] - 1).astype(int)
            planes = np.unique(through)
            y1, y2, x1, x2 = transforms.get_pad_yx(*stack.shape[1:3], min_size=(bsize, bsize))
            assert stack.shape[1] + y1 + y2 == bsize and stack.shape[2] + x1 + x2 == bsize
            for begin in range(0, len(planes), batch_size):
                selected_planes = planes[begin:begin + batch_size]
                imgs = np.pad(stack[selected_planes].transpose(0, 3, 1, 2), ((0, 0), (0, 0), (y1, y2), (x1, x2)))
                token, _ = self.batch(imgs)
                selected_queries = np.flatnonzero(np.isin(through, selected_planes))
                row = np.searchsorted(selected_planes, through[selected_queries])
                sampled = sample_tokens(token, row, q[selected_queries, 1:], (y1, x1), (bsize // 8, bsize // 8))
                pooled[selected_queries, view] = sampled.cpu().numpy().astype(np.float16)
                planes_processed += len(selected_planes)
        torch.cuda.synchronize(self.device)
        self.last_seconds = {"total": time.perf_counter() - started,
                             "normalization_resize_patch": normalized_at - started,
                             "queried_planes": planes_processed}
        assert np.isfinite(pooled).all() and np.isfinite(local).all()
        return pooled, local, geometry
