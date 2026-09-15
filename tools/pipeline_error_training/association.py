"""Narrow continuation decoder with the inherited margin/cap and full fork protection."""
import numpy as np

from .actions import fork_support


def decode(nodes, edges, native, scores, *, zero_residual=False, max_changed_edges=None):
    if zero_residual:
        return edges.copy(), dict(changed_edges=0, zero_residual=True, edits=[])
    if max_changed_edges is not None and max_changed_edges<=0:
        return edges.copy(), dict(changed_edges=0, edits=[], explicit_edge_budget=0,
                                 no_remaining_edge_budget=True)
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
    # The half-edge offset avoids rounding an integer budget downward when the
    # inherited decoder floors fraction * edge_count. Its original 2% cap remains.
    fraction = .02 if max_changed_edges is None else min(.02,(max_changed_edges+.5)/max(1,len(edges)))
    out, ledger = inherited_decode(nodes, edges, dict(pairs=native['pairs'], edge_features=features),
                                    values, margin=3., max_fraction=fraction)
    changed = old ^ {(ix[int(a)], ix[int(b)]) for a, b in out}
    if any(a in protected or b in protected for a, b in changed):
        raise RuntimeError('Continuation edit changed protected official fork support')
    if max_changed_edges is not None:
        if len(changed)>max_changed_edges:
            raise RuntimeError('Closed association exceeded the remaining observation edge budget')
        ledger['explicit_edge_budget'] = max_changed_edges
    ledger['full_fork_support_frozen'] = True
    return out, ledger
