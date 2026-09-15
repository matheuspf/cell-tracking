"""Score the actual incumbent fork as a reference, without adding edit proposals."""
from copy import copy
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch

from .bank import EVENT_FEATURES
from .common import load_graph, read_json, save_arrays, sha, write_json
from .fast_infer import event_values
from .resources import Lease
from .scoring import load_model


def rows(bank):
    for p, children in enumerate(bank.succ):
        if len(children) != 2 or any(len(bank.succ[d]) > 1 for d in children):
            continue
        a, b = sorted(children)
        event = (p, a, b, bank.succ[a][0] if bank.succ[a] else -1, bank.succ[b][0] if bank.succ[b] else -1)
        view = copy(bank)
        view.paths = bank.paths.copy()
        view.paths[a], view.paths[b] = [event[3]], [event[4]]
        found = [(e, f.copy()) for e, f, _ in view.iter_parent(p) if e == event]
        if len(found) != 1:
            raise ValueError('Incumbent reference fork is not uniquely representable')
        e, f = found[0]
        counts = sorted([sum(q >= 0 for q in bank.paths[a]), sum(q >= 0 for q in bank.paths[b])])
        for name, value in zip(['candidate_path_count_low', 'candidate_path_count_high'], counts):
            f[EVENT_FEATURES.index(name)] = value
        yield e, f


def prepare(bank, packages, root, wait=False, device='cuda'):
    root = Path(root)
    prepared = read_json(root/'prepared.json')
    path = root/'reference_forks.npz'
    stamp = dict(prepared_sha256=sha(root/'prepared.json'), code_sha256=sha(Path(__file__)))
    if path.exists():
        receipt = read_json(path.with_suffix('.json'))
        if receipt['inputs'] != stamp or sha(path) != receipt['sha256']:
            raise ValueError('Incumbent reference fork scores changed')
        return load_graph(path)
    cases = list(rows(bank))
    values = np.empty((len(cases), len(packages)), np.float32)
    begin = time.monotonic()
    if cases:
        with Lease(required_gib=4., wait=wait) if device=='cuda' else nullcontext():
            models, z = {}, {}
            for name in prepared['names']:
                model, spec = load_model(packages[name]); models[name] = model.to(device)
                # These embeddings were created and verified in this same staged
                # run. The reference never opens a historical feature store.
                embedding = next((root/'embeddings'/spec['weights_sha256']).glob('*.npz'))
                meta = read_json(embedding.with_suffix('.json'))
                if sha(embedding) != meta['sha256']:
                    raise ValueError('Current image embeddings changed')
                z[name] = load_graph(embedding)['embedding']
            for start in range(0, len(cases), 2048):
                values[start:start+2048] = event_values(models, z, cases[start:start+2048])
            for model in models.values():
                model.cpu()
            torch.cuda.empty_cache()
    save_arrays(path, events=np.asarray([e for e, _ in cases], np.int64).reshape(-1, 5), value=values)
    write_json(path.with_suffix('.json'), dict(inputs=stamp, sha256=sha(path), rows=len(cases),
        seconds=time.monotonic()-begin, new_edit_candidates=0, role='Actual P0 keep-state event energy only'), immutable=True)
    return load_graph(path)
