"""Corrected v3 division backbone at explicit incumbent observations.

The official Keras model is retained, including its last-block architecture.
No Cellpose candidate linker or cached pilot prediction is used.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import REPO, ROOT, WORK, read_json, sha, write_json

ORG_REPO = REPO.parent / 'OrganoidTracker'
ORG_MODEL = REPO / 'work/other-trackers-20260914/models/organoid/model_divisions'
ORG_SHA = '5240047e413c1e208e99f4701299735bd700145349e7b1e75c1d0137d805c034'
ORG_REV = 'db28ff26584ac6d1230ea90750a3324a2778fb31'


@lru_cache(maxsize=1)
def setup():
    import subprocess
    if sha(ORG_MODEL / 'model.keras') != ORG_SHA:
        raise ValueError('Corrected v3 Organoid checkpoint hash mismatch')
    if subprocess.check_output(['git', '-C', str(ORG_REPO), 'rev-parse', 'HEAD'], text=True).strip() != ORG_REV:
        raise ValueError('Organoid source revision mismatch')
    if subprocess.check_output(['git', '-C', str(ORG_REPO), 'diff', 'HEAD', '--'], text=True):
        raise ValueError('Organoid source is modified')
    os.environ['KERAS_BACKEND'] = 'torch'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ORG_REPO))
    sys.path.append('/kaggle/envs/detector-screen-organoid/lib/python3.12/site-packages')
    import keras
    return keras


class OrganoidEncoder(nn.Module):
    def __init__(self, random=False):
        super().__init__()
        keras = setup()
        from keras.src.backend.torch.core import device_scope
        with device_scope('cpu'):
            full = keras.saving.load_model(ORG_MODEL / 'model.keras', compile=False, safe_mode=True)
            if random:
                full = keras.models.clone_model(full)
            self.backbone = keras.Model(full.input, [full.get_layer('dense').output, full.output])
        self.projection = nn.Linear(32+2+3, 128)
        self.settings = read_json(ORG_MODEL / 'settings.json')
        self.adapt(False)

    def adapt(self, enabled):
        for layer in self.backbone.layers:
            layer.trainable = bool(enabled and (layer.name.startswith('down3>') or layer.name == 'dense'))
        # Released batch-normalization statistics always remain fixed.

    def forward(self, patch, valid):
        from keras.src.backend.torch.core import device_scope
        with device_scope(str(patch.device)):
            features, raw = self.backbone(patch, training=False)
        raw = raw.clamp(1e-10, 1-1e-7)
        logodds = torch.logit(raw)
        calibrated = self.settings['platt_intercept'] + self.settings['platt_scaling']*logodds
        mask = valid[:, 1:4, 0] if valid.shape[1] == 7 else valid[..., 0]
        return self.projection(torch.cat([features, logodds, calibrated, mask], -1))


def patches(images, nodes, indices):
    setup()
    from organoid_tracker.core import TimePoint
    from organoid_tracker.core.position import Position
    from organoid_tracker.neural_network.division_detection_cnn.division_predictor import _split_into_patches
    from other_trackers.organoid import resize_patch
    settings = read_json(ORG_MODEL / 'settings.json')
    shape = tuple(settings['patch_shape_zyx'])
    scale = tuple(images.scale / np.array([2., .32, .32]))
    result = {}
    for t in sorted({int(nodes[i, 1]) for i in indices}):
        ii = [int(i) for i in indices if int(nodes[i, 1]) == t]
        centers = [Position(x=float(nodes[i, 4]), y=float(nodes[i, 3]), z=float(nodes[i, 2]), time_point_number=t) for i in ii]
        items = _split_into_patches(images, TimePoint(t), centers, tuple(settings['time_window']),
            patch_shape_zyx_px=shape, scale_factors_zyx=scale, intensity_quantiles=(.01, .99))
        for i, item in zip(ii, items, strict=True):
            result[i] = resize_patch(item.array, shape)
    return np.stack([result[int(i)] for i in indices])


def parity(row, indices):
    from .common import verified_graph
    from .crops import Images
    from .resources import Lease
    model = OrganoidEncoder()
    graph = verified_graph(row)
    images = Images(row['image_path'])
    x = patches(images, graph['nodes'], indices)
    keras = setup()
    from keras.src.backend.torch.core import device_scope
    with device_scope('cpu'):
        full = keras.saving.load_model(ORG_MODEL / 'model.keras', compile=False, safe_mode=True)
        with torch.no_grad():
            expected = full(torch.from_numpy(x), training=False)
            _, raw = model.backbone(torch.from_numpy(x), training=False)
    cpu_error = float((expected-raw).abs().max())
    if cpu_error != 0:
        raise ValueError('Division backbone extraction changed raw released scores')
    with Lease(required_gib=3.):
        model.backbone.to('cuda')
        with device_scope('cuda:0'), torch.no_grad():
            _, actual = model.backbone(torch.from_numpy(x).cuda(), training=False)
        gpu_error = float((actual.cpu()-expected).abs().max())
        model.backbone.to('cpu')
        torch.cuda.empty_cache()
    if gpu_error > 1e-4:
        raise ValueError('Released CPU/CUDA division scores differ beyond 1e-4')
    receipt = dict(checkpoint_sha256=ORG_SHA, source_revision=ORG_REV, cpu_exact_error=cpu_error,
                   gpu_absolute_error=gpu_error, model_settings=model.settings,
                   preprocessing='Released extraction and nearest rescaling at actual P0 coordinates.',
                   missing_frames='Released nearest-frame filling retained for parity; explicit validity is a separate competition-head input.',
                   observations='P0', cases=len(indices))
    write_json(WORK / 'organoid_parity.json', receipt)
    return receipt
