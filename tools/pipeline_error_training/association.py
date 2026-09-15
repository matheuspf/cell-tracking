"""Narrow continuation decoder with the inherited margin/cap and full fork protection."""
import numpy as np

from .actions import fork_support


def decode(nodes, edges, native, scores, *, zero_residual=False):
    if zero_residual:
        return edges.copy(), dict(changed_edges=0, zero_residual=True, edits=[])
    from strong_tracker_v3.association import decode as inherited_decode
    protected = fork_support(nodes, edges)
    ix = {int(n[0]): i for i, n in enumerate(nodes)}
    old = {(ix[int(a)], ix[int(b)]) for a, b in edges}
    values = np.asarray(scores, float).copy()
    features = native['edge_features'].copy()
    for k, (a, b) in enumerate(native['pairs']):
        if int(a) in protected or int(b) in protected:
            if (a, b) in old:
                features[k, 25:29] = 1.
            else:
                values[k] = -1e9
    out, ledger = inherited_decode(nodes, edges, dict(pairs=native['pairs'], edge_features=features),
                                    values, margin=3., max_fraction=.02)
    changed = old ^ {(ix[int(a)], ix[int(b)]) for a, b in out}
    if any(a in protected or b in protected for a, b in changed):
        raise RuntimeError('Continuation edit changed protected official fork support')
    ledger['full_fork_support_frozen'] = True
    return out, ledger
