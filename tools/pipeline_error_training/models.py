"""Matched image encoders and complete-decision scoring, with explicit masks."""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(*widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i != len(widths)-2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv3d(2, 16, (3, 3, 3), stride=(1, 2, 2), padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, 128),
        )
        self.gru = nn.GRUCell(128+5, 128)

    def forward(self, patch, valid):
        n, t = patch.shape[:2]
        # Encode real frames only. Padding cannot affect any valid token.
        present = valid[..., 0] > 0
        tokens = patch.new_zeros((n, t, 128))
        if present.any():
            tokens[present] = self.frame(patch[present])
        state = patch.new_zeros((n, 128))
        for k in range(t):
            relative = patch.new_full((n, 1), (k-2)/4.)
            value = torch.cat([tokens[:, k], valid[:, k], relative], -1)
            update = self.gru(value, state)
            state = torch.where(present[:, k, None], update, state)
        return state


class CompactEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from multidata_training_v4.models import FrameEncoder
        self.frame = FrameEncoder()
        self.temporal = mlp(131, 128, 128)

    def forward(self, patch, valid):
        n, t = patch.shape[:2]
        present = valid[..., 0] > 0
        f = patch.new_zeros((n, t, 64))
        if present.any():
            f[present] = self.frame(patch[present])
        w = present.to(f.dtype)
        offset = torch.arange(t, device=f.device, dtype=f.dtype)-1
        mean = (f*w[..., None]).sum(1)/w.sum(1, keepdim=True).clamp_min(1)
        moment = (f*w[..., None]*offset[None, :, None]).sum(1)/w.sum(1, keepdim=True).clamp_min(1)
        context = torch.stack([w.sum(1)/3, (w*offset).sum(1)/3, (w*offset.square()).sum(1)/3], -1)
        return self.temporal(torch.cat([mean, moment, context], -1))


class DecisionModel(nn.Module):
    def __init__(self, family='compact', encoder=None):
        super().__init__()
        self.family = family
        self.encoder = encoder or (CompactEncoder() if family == 'compact' else TemporalEncoder())
        self.identity = mlp(128*3+38, 128, 64, 1)
        self.event = mlp(128*3+40, 128, 64, 1)
        self.metric_risk = mlp(128*3+40, 64, 1)
        self.boundary = mlp(128+4, 32, 2)
        self.selector = mlp(128*3+8, 64, 3)

    def link_scores(self, z, pairs, features):
        a, b = z[pairs[:, 0]], z[pairs[:, 1]]
        return self.identity(torch.cat([a, b, (a-b).abs(), features], -1)).squeeze(-1)

    def event_scores(self, z, events, features):
        p, a, b = z[events[:, 0]], z[events[:, 1]], z[events[:, 2]]
        # Exact symmetry under daughter permutation, shared by all model families.
        combined = torch.cat([p, a+b, (a-b).abs(), features], -1)
        return self.event(combined).squeeze(-1), self.metric_risk(combined).squeeze(-1)

    def selection_scores(self, old, raw, context, evidence):
        return self.selector(torch.cat([old, raw, context, evidence], -1))


def compatible_set_loss(logits, positive, known, group):
    """One unit per biological group; all compatible alternatives form a set."""
    terms = []
    for g in torch.unique(group):
        mask = (group == g) & known
        good = mask & positive
        if not good.any() or not (mask & ~positive).any():
            continue
        terms.append(torch.logsumexp(logits[mask], 0)-torch.logsumexp(logits[good], 0))
    return torch.stack(terms).mean() if terms else logits.sum()*0


def group_binary_loss(logits, labels, group):
    terms = []
    for g in torch.unique(group):
        valid = (group == g) & (labels >= 0)
        if valid.any():
            terms.append(F.binary_cross_entropy_with_logits(logits[valid], labels[valid].to(logits.dtype)))
    return torch.stack(terms).mean() if terms else logits.sum()*0
