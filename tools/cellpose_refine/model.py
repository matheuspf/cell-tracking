"""Small bounded residual head; the inherited encoder is never inside this module."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Refiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["model"]
        self.limit = cfg["movement_limit_um"]
        self.semantic = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, c["feature_projection_per_view"]), nn.SiLU())
        self.image = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.SiLU(),
            nn.Conv3d(8, 16, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(16, 16, 3, stride=(1, 2, 2), padding=1), nn.SiLU(), nn.Flatten())
        with torch.no_grad():
            n = self.image(torch.zeros(1, 1, *cfg["features"]["patch_shape_zyx"])).shape[-1]
        self.image_projection = nn.Sequential(nn.Linear(n, c["patch_projection"]), nn.SiLU())
        width = c["feature_projection_per_view"] * 3 + c["patch_projection"] + 3
        self.fusion = nn.Sequential(nn.Linear(width, c["fusion_width"]), nn.SiLU(), nn.Dropout(c["dropout"]))
        self.output = nn.Linear(c["fusion_width"], 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, patch, geometry):
        semantic = self.semantic(feature.float()).flatten(1)
        local = self.image_projection(self.image(patch.float()[:, None]))
        hidden = self.fusion(torch.cat((semantic, local, geometry.float()), dim=1))
        delta = self.limit * torch.tanh(self.output(hidden))
        norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        return delta / (norm / self.limit).clamp_min(1.)


def sparse_offset_loss(prediction, target, weight, beta=1.):
    if not torch.isfinite(target).all() or (weight < 0).any():
        raise ValueError("Invalid sparse target")
    per_query = F.smooth_l1_loss(prediction.float(), target.float(), beta=beta, reduction="none").mean(dim=1)
    return (per_query * weight).sum() / weight.sum().clamp_min(1.)
