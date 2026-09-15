"""Annotation-free vectorized complete-decision scoring at explicit node tokens."""
import numpy as np
import torch
from torch.nn import functional as F

from .models import DecisionModel

EVENT_SCALE = np.array([1, 1, 2, 1, 2, 2, 20, 20, 20, 20, 20, 2, 1, 20, 20, 20,
                       2, 1, 1, 1, 1, 4, 1, 30, 2, 20, 20, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1], np.float32)


def load_model(package):
    from pathlib import Path
    from .common import read_json, sha
    folder = Path(package)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = folder / 'frozen_package.json'
    spec = read_json(manifest if manifest.exists() else folder / 'package.json')
    if sha(folder / 'model.pt') != spec['weights_sha256']:
        raise ValueError('Frozen model package hash mismatch')
    family = spec['recipe']['family']
    if family == 'organoid':
        from .organoid_adapter import OrganoidEncoder
        model = DecisionModel(family, OrganoidEncoder())
    else:
        model = DecisionModel(family)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, spec


def edge_features(nodes, native, pairs, scale):
    if '_pair_index' not in native:
        native['_pair_index'] = {tuple(map(int, p)): k for k, p in enumerate(native['pairs'])}
    index = native['_pair_index']
    result = np.zeros((len(pairs), 38), np.float32)
    for k, (a, b) in enumerate(pairs):
        if (a, b) in index:
            result[k] = native['edge_features'][index[a, b]]
        else:
            result[k, 1] = result[k, 24] = 1.
            result[k, 3] = np.linalg.norm((nodes[a, 2:]-nodes[b, 2:])*scale)
    return result


@torch.no_grad()
def score(model, spec, nodes, native, embeddings, valid, decisions, event_features, scale):
    """Scores every removed edge, added edge and orphan consequence exactly once."""
    device = next(model.parameters()).device
    pairs = sorted({p for d in decisions for p in d.remove | d.add})
    pair_index = {p: i for i, p in enumerate(pairs)}
    used = sorted({i for d in decisions for i in d.event[:3]} | {i for p in pairs for i in p})
    mapping = {n: i for i, n in enumerate(used)}
    z = torch.as_tensor(embeddings[used], dtype=torch.float32, device=device)
    features = edge_features(nodes, native, pairs, np.asarray(scale))
    normalized = np.clip((features-spec['mean'])/spec['scale'], -10, 10).astype(np.float32)
    pp = torch.tensor([[mapping[a], mapping[b]] for a, b in pairs], device=device, dtype=torch.long).reshape(-1, 2)
    link = model.link_scores(z, pp, torch.as_tensor(normalized, device=device)) + torch.as_tensor(features[:, 23], device=device)
    events = torch.tensor([[mapping[i] for i in d.event[:3]] for d in decisions], device=device)
    ef = torch.as_tensor(np.clip(np.asarray(event_features)/EVENT_SCALE, -10, 10), dtype=torch.float32, device=device)
    ev, risk = model.event_scores(z, events, ef)
    anchor = 1 if model.family == 'compact' else 2
    boundary = model.boundary(torch.cat([z, torch.as_tensor(valid[used, anchor], device=device)], -1))
    costs = F.softplus(boundary)
    incidence = np.zeros((len(decisions), len(pairs)), np.float32)
    births = np.zeros((len(decisions), len(used)), np.float32)
    terminations = np.zeros_like(births)
    forks = np.zeros(len(decisions), np.float32)
    for k, d in enumerate(decisions):
        for pair in d.add:
            incidence[k, pair_index[pair]] += 1
        for pair in d.remove:
            incidence[k, pair_index[pair]] -= 1
        for n in d.births:
            births[k, mapping[n]] += 1
        for n in d.terminations:
            terminations[k, mapping[n]] += 1
        forks[k] = d.kind == 'division'
    value = (torch.as_tensor(incidence, device=device) @ link
             - torch.as_tensor(births, device=device) @ costs[:, 0]
             - torch.as_tensor(terminations, device=device) @ costs[:, 1]
             + torch.as_tensor(forks, device=device)*(ev+risk))
    return value.cpu().numpy(), risk.cpu().numpy()
