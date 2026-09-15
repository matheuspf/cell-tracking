"""Prediction-only streaming equivalent of the pinned v3 event union.

The full union is never cut back to six after reverse-neighbor expansion.
Only one parent's alternatives are materialized at a time.
"""
from dataclasses import asdict
from itertools import combinations, product

import numpy as np

from strong_tracker_v3.event_proposals import EVENT_FEATURES, ProposalConfig, candidate_neighbors

from .common import digest, graph_hash


class EventBank:
    def __init__(self, nodes, edges, native, scale, config=ProposalConfig()):
        self.nodes, self.edges, self.native, self.config = nodes, edges, native, config
        self.near, self.probability, (_, self.pred, self.succ), self.counts = candidate_neighbors(
            nodes, edges, native, scale, config)
        self.pos = nodes[:, 2:] * np.asarray(scale)
        self.nf = native.get('node_features', native.get('features', np.zeros((len(nodes), 13), np.float32)))
        if self.nf.shape[1] < 13:
            self.nf = np.pad(self.nf, ((0, 0), (0, 13-self.nf.shape[1])))
        self.birth = [any(not self.pred[j] for j in ns) for ns in self.near]
        self.anchors = set()
        for i, ns in enumerate(self.near):
            pp = sorted((self.probability.get((i, j), 0.) for j in ns), reverse=True)
            if (len(self.succ[i]) != 1 or self.birth[i]
                    or (len(pp) > 1 and pp[1] >= config.second_probability_trigger)
                    or int(nodes[i, 0]) % config.uniform_modulus == 0):
                self.anchors.add(i)
        self.expanded = set(self.anchors)
        for i in self.anchors:
            self.expanded.update(self.pred[i])
            self.expanded.update(self.near[i][:2])
        self.paths = [list(ns[:config.paths_per_daughter]) or [-1] for ns in self.near]
        for i in range(len(nodes)):
            for j in self.succ[i]:
                if j not in self.paths[i]:
                    self.paths[i] = ([j] + self.paths[i])[:config.paths_per_daughter]
        self.end_t = int(nodes[:, 1].max()) if len(nodes) else -1
        self.mapping = native.get('native_index', np.zeros(len(nodes)))
        self.hash = digest(dict(graph=graph_hash(nodes, edges), config=asdict(config),
                                feature_names=EVENT_FEATURES, enumeration='v3_union_lazy_v1'))

    def event_key(self, event):
        return tuple(int(self.nodes[k, 0]) if k >= 0 else -1 for k in event)

    def pairs(self, parent):
        i = int(parent)
        options = set(self.near[i]) | set(self.succ[i])
        for a, b in combinations(sorted(options), 2):
            original = set(self.succ[i]) == {a, b}
            da, db = self.pos[a]-self.pos[i], self.pos[b]-self.pos[i]
            dista, distb = np.linalg.norm(da), np.linalg.norm(db)
            sister = float(np.linalg.norm(self.pos[a]-self.pos[b]))
            if not original and (max(dista, distb) > self.config.parent_gate_um
                                 or not 1 <= sister <= self.config.sister_gate_um):
                continue
            yield a, b, original, da, db, dista, distb, sister

    def iter_parent(self, parent):
        i = int(parent)
        if i not in self.expanded:
            return
        nf, pos, probability = self.nf, self.pos, self.probability
        for a, b, original, da, db, dista, distb, sister in self.pairs(i):
            pa, pb = probability.get((i, a), 0.), probability.get((i, b), 0.)
            velocity = pos[i]-pos[self.pred[i][0]] if len(self.pred[i]) == 1 else np.zeros(3)
            bary = (da+db)/2
            owner_dist, owner_prob, owner_missing = [], [], 0
            for d in [a, b]:
                if self.pred[d] and self.pred[d][0] != i:
                    o = self.pred[d][0]
                    owner_dist.append(np.linalg.norm(pos[d]-pos[o]))
                    owner_prob.append(probability.get((o, d), 0.))
                    owner_missing += int((o, d) not in probability)
                else:
                    owner_dist.append(0.)
                    owner_prob.append(0.)
            for qa, qb in product(self.paths[a], self.paths[b]):
                if qa >= 0 and qa == qb:
                    continue
                dpath = [np.linalg.norm(pos[q]-pos[d]) if q >= 0 else 0. for d, q in [(a, qa), (b, qb)]]
                ppath = [probability.get((d, q), 0.) for d, q in [(a, qa), (b, qb)]]
                future = float(np.linalg.norm(pos[qa]-pos[qb])-sister) if qa >= 0 and qb >= 0 else 0.
                features = np.array([
                    min(pa, pb), max(pa, pb), pa+pb, abs(pa-pb),
                    int((i, a) not in probability)+int((i, b) not in probability),
                    int(a in self.succ[i])+int(b in self.succ[i]), min(dista, distb), max(dista, distb),
                    sister, np.linalg.norm(bary), np.linalg.norm(bary-.5*velocity),
                    abs(dista-distb)/max(.1, (dista+distb)/2), np.dot(da, db)/max(.01, dista*distb),
                    *sorted(dpath), future, int(qa < 0)+int(qb < 0), len(self.pred[i]) != 1,
                    nf[i, 5], min(nf[a, 5], nf[b, 5]), max(nf[a, 5], nf[b, 5]),
                    (nf[a, 6]+nf[b, 6])/(nf[i, 6]+.05),
                    min(nf[a, 6], nf[b, 6])/(max(nf[a, 6], nf[b, 6])+.05), nf[i, 8],
                    int(not self.pred[a])+int(not self.pred[b]), *sorted(owner_dist), *sorted(owner_prob),
                    owner_missing, original, min(4, int(self.nodes[i, 1]))/4,
                    min(4, self.end_t-int(self.nodes[i, 1]))/4, *sorted(ppath),
                    *sorted([sum(q >= 0 for q in self.paths[a]), sum(q >= 0 for q in self.paths[b])]),
                    not self.succ[i], self.mapping[i] < 0, self.birth[i],
                ], np.float32)
                existing = original or (bool({a, b} & set(self.succ[i])) and max(dista, distb) <= 12
                                        and sister <= 16 and a in self.near[i][:4] and b in self.near[i][:4])
                yield (i, a, b, qa, qb), features, existing

    def __iter__(self):
        for parent in sorted(self.expanded):
            yield from self.iter_parent(parent)

    def materialize(self):
        """Small test fixtures only. Production consumes the iterator."""
        rows = list(self)
        return dict(events=np.asarray([x[0] for x in rows], np.int64).reshape(-1, 5),
                    event_features=np.asarray([x[1] for x in rows], np.float32).reshape(-1, len(EVENT_FEATURES)),
                    existing_pool=np.asarray([x[2] for x in rows], bool))
