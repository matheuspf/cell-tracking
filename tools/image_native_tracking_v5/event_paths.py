"""Label-free division identities from predicted continuation paths.

Timing alternatives with the same parent path and unordered daughter paths are
mutually exclusive. The MILP takes the maximum complete explanation; duplicate
representations add neither score nor probability mass.
"""
from itertools import combinations
import numpy as np


def unique_edges(pairs, scores):
    best = {}
    for pair, score in zip(pairs, scores):
        pair = tuple(map(int, pair))
        if pair not in best or (np.isfinite(score) and
                (not np.isfinite(best[pair]) or score > best[pair])):
            best[pair] = float(score)
    ordered = sorted(best)
    return np.asarray(ordered, np.int64).reshape(-1, 2), np.asarray([best[p] for p in ordered])


def equivalence_classes(nodes, pairs, scores, base_edges, old_ids):
    ids = list(map(int, nodes[:, 0])); parent = {i: i for i in ids}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    incoming = {}; outgoing = {}
    for a, b in base_edges:
        a, b = int(a), int(b)
        outgoing.setdefault(a, []).append(b); incoming.setdefault(b, []).append(a)
    used_in = set(); used_out = set()
    for a, bs in outgoing.items():
        if len(bs) == 1 and len(incoming.get(bs[0], [])) == 1:
            union(a, bs[0]); used_out.add(a); used_in.add(bs[0])
    # New points get paths only from mutually best, positive model evidence.
    forward = {}; backward = {}
    for (a, b), score in zip(pairs, scores):
        a, b = int(a), int(b)
        if not np.isfinite(score) or score <= 0: continue
        if (score, -b) > forward.get(a, (-np.inf, 0)): forward[a] = (score, -b)
        if (score, -a) > backward.get(b, (-np.inf, 0)): backward[b] = (score, -a)
    for a, (_, neg_b) in sorted(forward.items()):
        b = -neg_b
        if backward[b][1] != -a or (a in old_ids and b in old_ids): continue
        if a in used_out or b in used_in or len(outgoing.get(a, [])) > 1: continue
        union(a, b); used_out.add(a); used_in.add(b)
    by_parent = {}
    for a, b in pairs: by_parent.setdefault(int(a), set()).add(int(b))
    groups = {}
    for a, bs in by_parent.items():
        for b, c in combinations(sorted(bs), 2):
            if root(b) == root(c): continue
            identity = (root(a), *sorted((root(b), root(c))))
            groups.setdefault(identity, set()).add(((a, b), (a, c)))
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}
