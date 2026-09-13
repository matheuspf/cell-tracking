"""Instrument the original materialized predictor; preserve unmodified arithmetic."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np

from .common import array_hash
from .modules import inverse_phase, median_views, peak_offsets, trilinear_features
from .source import replace_once


def inverse_view(value, view):
    import torch
    if view == 0:
        return value
    if view in (1, 2, 3):
        return value.flip({1: (-1,), 2: (-2,), 3: (-2, -1)}[view])
    if view in (4, 5):
        return torch.rot90(value, -{4: 1, 5: 3}[view], dims=(-2, -1))
    if view == 6:
        return value.transpose(-1, -2)
    return torch.rot90(value.transpose(-1, -2), -1, dims=(-2, -1))


def forward_view(value, view):
    import torch
    if view <= 3:
        return inverse_view(value, view)
    if view in (4, 5):
        return torch.rot90(value, {4: 1, 5: 3}[view], dims=(-2, -1))
    if view == 6:
        return value.transpose(-1, -2)
    return torch.rot90(value, 1, dims=(-2, -1)).transpose(-1, -2)


class Evidence:
    def __init__(self, modules, transform=None):
        self.modules = set(modules)
        self.transform = transform
        self.original_encode = {}
        self.views = {}
        self.offsets = []
        self.node_probabilities = []
        self.probabilities = {}
        self.summaries = dict(feature_samples=0, fractional_samples=0, integer_identity_max_abs=0.,
                              detector_frames=[], association_transitions=[])
        self.suspended = False

    def attach(self, model):
        self.original_encode[model] = model.encode
        self.views[model] = []
        original_index = model._index_features

        def index(features, coords, mask):
            result = original_index(features, coords, mask)
            valid = coords[mask]
            fraction = int((valid != valid.trunc()).any(-1).sum().item())
            self.summaries['feature_samples'] += len(valid)
            self.summaries['fractional_samples'] += fraction
            if fraction == 0:
                other = trilinear_features(features, coords, mask)
                delta = float((other - result).abs().max().item()) if result.numel() else 0.
                self.summaries['integer_identity_max_abs'] = max(self.summaries['integer_identity_max_abs'], delta)
            if 'E03' in self.modules:
                return trilinear_features(features, coords, mask)
            return result

        def encode(imgs):
            features, responses = self.original_encode[model](imgs)
            if 'E08' in self.modules and not self.suspended:
                view = len(self.views[model])
                if view >= 8:
                    raise RuntimeError('More than eight registered detection views')
                self.views[model].append([inverse_view(v, view) for v in responses])
            return features, responses
        model.encode = encode
        model._index_features = index

    def load_frame(self, original, zarr_arr, t, target_shape, downsample, *, extra_reflect=False, phase=None):
        import torch
        import torch.nn.functional as F
        if not self.transform and not extra_reflect and phase is None:
            return original(zarr_arr, t, target_shape, downsample)
        raw = np.asarray(zarr_arr[t], np.float32)
        if self.transform in ('reflect_x', 'reflect_y'):
            raw = np.flip(raw, axis=-1 if self.transform == 'reflect_x' else -2)
        elif self.transform == 'shift_x1':
            raw = np.take(raw, np.maximum(np.arange(raw.shape[-1]) - 1, 0), axis=-1)
        elif self.transform is not None:
            raise ValueError('Unregistered image transform')
        if extra_reflect:
            raw = np.flip(raw, axis=-1)
        if phase is not None:
            # Translation I_shift(q)=I(q-shift), reflection extension, no wrapping.
            indices = []
            for size, d, shift in zip(raw.shape, downsample, phase):
                q = np.arange(0, size, d) - shift
                if size == 1:
                    q[:] = 0
                else:
                    period = 2 * (size - 1)
                    q %= period
                    q = np.where(q < size, q, period - q)
                indices.append(q.astype(int))
            raw = raw[np.ix_(*indices)]
        else:
            raw = raw[::downsample[0], ::downsample[1], ::downsample[2]]
        frame = torch.from_numpy(np.array(raw, copy=True))
        if list(frame.shape) != target_shape:
            frame = F.interpolate(frame[None, None], size=target_shape, mode='trilinear', align_corners=False)[0, 0]
        return frame

    def extra_images(self, zarr_arr, frame_indices, target_shape, downsample, q_low, q_high,
                     device, extra_reflect=False, phase=None):
        import torch
        imgs = torch.stack([self.load_frame(None, zarr_arr, t, target_shape, downsample,
                                           extra_reflect=extra_reflect, phase=phase) for t in frame_indices])
        return ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.).unsqueeze(0).to(device)

    def encode_views(self, model, imgs, features=False, all_views=True):
        """Exact public D4 order; features averaged only for the primary branch."""
        enc = self.original_encode[model]
        unet, response = enc(imgs)
        feature_sum = unet.clone() if features else None
        for view in range(1, 8 if all_views else 1):
            f, r = enc(forward_view(imgs, view))
            response = [a + inverse_view(b, view) for a, b in zip(response, r)]
            if features:
                feature_sum = feature_sum + inverse_view(f, view)
            del f, r
        divisor = 8 if all_views else 1
        return feature_sum / divisor if features else unet, [r / divisor for r in response]

    def detector_response(self, model, responses, zarr_arr, frame_indices, target_shape,
                          downsample, q_low, q_high, device):
        import torch
        if 'E08' in self.modules:
            views = self.views[model]
            if len(views) != 8:
                raise RuntimeError('Missing original detection views')
            responses = [median_views([view[f] for view in views]) for f in range(len(responses))]
            self.views[model] = []
        if 'E04' in self.modules:
            shift = [0, downsample[1] // 2, downsample[2] // 2]
            if downsample[1] % 2 or downsample[2] % 2:
                raise ValueError('E04 needs even XY strides')
            imgs = self.extra_images(zarr_arr, frame_indices, target_shape, downsample,
                                     q_low, q_high, device, phase=shift)
            _, shifted = self.encode_views(model, imgs)
            merged = []
            for baseline, response in zip(responses, shifted):
                aligned, valid = inverse_phase(response, np.asarray(shift) / downsample)
                merged.append(torch.where(valid, (baseline + aligned) * .5, baseline))
            responses = merged
        return responses

    def reflected_features(self, model, secondary_model, zarr_arr, frame_indices, target_shape,
                           downsample, q_low, q_high, device):
        imgs = self.extra_images(zarr_arr, frame_indices, target_shape, downsample,
                                 q_low, q_high, device, extra_reflect=True)
        primary, _ = self.encode_views(model, imgs, features=True)
        secondary, _ = self.original_encode[secondary_model](imgs)
        return primary, secondary

    def capture_peaks(self, response, arr, downsample):
        value = response.detach().cpu().numpy()[0, 0]
        delta, supported = peak_offsets(value, arr[:, 1:])
        self.offsets.extend(delta)
        logits = value[tuple(arr[:, 1:].astype(int).T)]
        self.node_probabilities.extend(1. / (1. + np.exp(-logits)))
        self.summaries['detector_frames'].append(dict(frame=int(arr[0, 0]) if len(arr) else None,
            response_sha256=array_hash(value), detections=len(arr), supported_axes=int(supported.sum()),
            nonzero_offsets=int(np.any(delta != 0, axis=1).sum())))

    def capture_probabilities(self, time, probs, idx_src, idx_tgt):
        self.probabilities[f'p_{time}'] = probs.copy()
        self.probabilities[f's_{time}'] = idx_src.copy()
        self.probabilities[f't_{time}'] = idx_tgt.copy()
        self.summaries['association_transitions'].append(dict(time=int(time),
            shape=list(probs.shape), sha256=array_hash(probs),
            column_sum_max_error=float(np.abs(probs.sum(0) - 1).max()) if probs.size else 0.))


def predictor_source(source, modules):
    """Source-checked hooks around original detection, head, and probability blocks."""
    text = source
    text = replace_once(text, 'def _load_frame(', 'def _public_load_frame(')
    marker = '# =============================================================================\n# Inference\n# ============================================================================='
    wrapper = '''def _load_frame(zarr_arr, t, target_shape, downsample=(1, 1, 1)):
    return _p946.load_frame(_public_load_frame, zarr_arr, t, target_shape, downsample)

'''
    text = replace_once(text, marker, wrapper + marker)
    primary = '''        det_logits = _p946.detector_response(model, det_logits, zarr_arr,
            frame_indices, target_shape, downsample, q_low, q_high, device)
'''
    text = replace_once(text, '        secondary_unet_out = None', primary + '\n        secondary_unet_out = None')
    secondary = '''                secondary_det_logits = _p946.detector_response(secondary_model,
                    secondary_det_logits, zarr_arr, frame_indices, target_shape,
                    downsample, q_low, q_high, device)

'''
    text = replace_once(text, '                for f in range(W):\n                    primary_det = det_logits[f]',
                        secondary + '                for f in range(W):\n                    primary_det = det_logits[f]')
    text = replace_once(text, '                coord_lists.append(arr)',
                        '                coord_lists.append(arr)\n                _p946.capture_peaks(det_logits[f_idx], arr, downsample)')
    if 'E05' in modules:
        text = replace_once(text, '        del imgs\n', '''        _identity_primary = unet_out
        _identity_secondary = secondary_unet_out
        _reflected_primary, _reflected_secondary = _p946.reflected_features(
            model, secondary_model, zarr_arr, frame_indices, target_shape,
            downsample, q_low, q_high, device)
        del imgs
''')
        start = text.index('            # Build tensors (batch_size=1).')
        end = text.index('            candidates = sorted(', start)
        body = text[start:end]
        prefix = '''            _original_src, _original_tgt = c_src, c_tgt
            _view_probabilities = []
            for _reflect in (False, True):
                c_src = _original_src.astype(np.float32, copy=True)
                c_tgt = _original_tgt.astype(np.float32, copy=True)
                unet_out = _reflected_primary if _reflect else _identity_primary
                secondary_unet_out = _reflected_secondary if _reflect else _identity_secondary
                if _reflect:
                    c_src[:, 3] = (zarr_arr.shape[-1] - 1) / downsample[-1] - c_src[:, 3]
                    c_tgt[:, 3] = (zarr_arr.shape[-1] - 1) / downsample[-1] - c_tgt[:, 3]
'''
        suffix = '''                _view_probabilities.append(probs)
            probs = (_view_probabilities[0] + _view_probabilities[1]) * .5
            c_src, c_tgt = _original_src, _original_tgt
            unet_out, secondary_unet_out = _identity_primary, _identity_secondary

'''
        text = text[:start] + prefix + textwrap.indent(body, '    ') + suffix + text[end:]
        text = replace_once(text, '        del unet_out\n', '        del _identity_primary, _identity_secondary, _reflected_primary, _reflected_secondary\n        del unet_out\n')
    text = replace_once(text, '            candidates = sorted(',
                        '            _p946.capture_probabilities(t_src, probs, idx_src, idx_tgt)\n\n            candidates = sorted(')
    # Worker calls public predict_video/build_graph itself; workflow/evaluator are never invoked.
    compile(text, '<instrumented-public-v29>', 'exec')
    return text


def import_predictor(path, modules, evidence):
    import sys
    import types
    source = predictor_source(Path(path).read_text(), modules)
    module = types.ModuleType('public946_original_predictor')
    module.__file__ = str(path)
    module._p946 = evidence
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, source
