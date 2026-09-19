"""Random whole upstream family; source imports are architecture-only.

The sparse-safe loss is independent of the historical trainer's dense loss.
"""
import importlib.util
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Upstream(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        root = architecture/'src/biohub_tracking/models'
        temporal = source_module(root/'temporal_unet.py', 'v11_temporal_architecture')
        association = source_module(root/'simple_node_transformer.py', 'v11_association_architecture')
        self.unet = temporal.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
        from .deterministic import TrilinearX2
        self.unet.upsamples = nn.ModuleList([TrilinearX2() for _ in self.unet.upsamples])
        self.detect_head = nn.Conv3d(32, 1, 1)
        self.transformer = association.SimpleNodeTransformer(feat_dim=64, hidden_dim=128, n_blocks=4)

    def forward(self, x):
        # 8 * 32^3 independent temporal queries exceed the fused CUDA launch
        # grid on this runtime. The math backend evaluates the same attention.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            features = self.unet(x)
        b, t, c, z, y, w = features.shape
        logits = self.detect_head(features.reshape(b*t, c, z, y, w)).reshape(b, t, z, y, w)
        return logits, features

    def query(self, features, coords):
        # Actual native origins; interpolate the zero-origin coarse lattice.
        # Explicit gathers let PyTorch use deterministic index accumulation.
        # grid_sample CUDA backward uses unordered atomics even in strict mode.
        lattice = coords/coords.new_tensor([1,4,4])
        lattice = torch.maximum(torch.zeros_like(lattice),torch.minimum(lattice,lattice.new_tensor(features.shape[-3:])-1))
        lower=lattice.floor().long(); fraction=lattice-lower
        result=features.new_zeros((len(coords),features.shape[0]),dtype=torch.float32)
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):
                    bit=lower.new_tensor([z,y,x])
                    index=torch.minimum(lower+bit,lower.new_tensor(features.shape[-3:])-1)
                    weight=torch.where(bit.bool(),fraction,1-fraction).prod(-1)
                    result=result+features[:,index[:,0],index[:,1],index[:,2]].T.float()*weight[:,None]
        return result

    def association(self, first, second, a, b, time):
        def positional(points, t):
            # Original four-axis, eight-features-per-axis sinusoidal recipe.
            full = torch.cat([points.new_full((len(points), 1), t), points], -1)
            norms = full/full.new_tensor([100,64,256,256])
            frequencies = 2.**torch.arange(4, device=points.device)*torch.pi
            phase = norms[..., None]*frequencies
            return torch.cat([phase.sin(), phase.cos()], -1).flatten(-2)
        return self.transformer(torch.cat([self.query(first, a), positional(a, time)], -1),
                                torch.cat([self.query(second, b), positional(b, time+1)], -1), a, b)


def detection_loss(logits, targets, positives, backgrounds):
    values = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none')
    pos = values[positives].mean() if positives.any() else logits.sum()*0
    bg = values[backgrounds].mean() if backgrounds.any() else logits.sum()*0
    return pos + .01*bg


def incoming_loss(logits, labels):
    terms = []
    for j in range(labels.shape[1]):
        known = labels[:, j] >= 0
        if known.any():
            terms.append(F.binary_cross_entropy_with_logits(logits[known, j].float(), labels[known, j].float()))
    return torch.stack(terms).mean() if terms else logits.sum()*0


def learning_rate(step, updates, base=1e-4, cap=500):
    warmup = min(updates-1, cap, max(1, int(.05*updates)))
    if step <= warmup:
        return base*step/warmup
    return 1e-6 + (base-1e-6)*(1+math.cos(math.pi*(step-warmup)/(updates-warmup)))/2
