"""Fixed-node exact metric deltas with globally rematched fork ownership.

This module is source-label owning. It is never imported by model inference.
Independent full official replays are mandatory before production fitting.
"""
from collections import Counter
import numpy as np

from pipeline_error_training.labels import SourceLabels, Topology

COUNT_KEYS = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')


class Counterfactual:
    def __init__(self, nodes, edges, gt_nodes, gt_edges, scale, baseline):
        self.labels = SourceLabels(nodes, edges, gt_nodes, gt_edges, scale)
        self.nodes, self.edges, self.baseline = nodes, edges, baseline
        self.l = self.labels
        self.forks = {p for p, ch in enumerate(self.l.succ) if len(ch) == 2}
        self.base_topology = Topology(self.l.pred, self.l.succ)
        self.base_fork_info = {p: self.fork_info(self.base_topology, p) for p in self.forks}
        self.base_div = self.division_counts(self.base_topology, {})
        expected = [baseline[k] for k in COUNT_KEYS[3:]]
        if list(self.base_div['counts']) != expected:
            raise ValueError(f'Fixed-node baseline division parity failed: {self.base_div["counts"]} != {expected}')
        self.edge_cache = {}

    def edge_label(self, edge):
        if edge not in self.edge_cache:
            a, b = edge
            ga, gb = self.l.matches.get(a), self.l.matches.get(b)
            tp = (ga, gb) in self.l.gt_edges
            evaluable = bool((ga is not None and self.l.gt.successors(ga)) or
                             (gb is not None and self.l.gt.predecessors(gb)))
            self.edge_cache[edge] = (int(tp), int(evaluable and not tp))
        return self.edge_cache[edge]

    def fork_info(self, topology, parent):
        children = topology.successors(parent)
        if len(children) != 2:
            return False, set(), set()
        branches = [self.l.branch(topology, parent, d) for d in children]
        invalid = any(m for _, m in branches) or len({c for c, _ in branches if c is not None}) > 1
        evaluable = bool(parent in self.l.matches and self.l.gt.successors(self.l.matches[parent]))
        considered, compatible = set(), set()
        for event, roles in self.l.roles.items():
            if roles is None:
                continue
            ps, ds = roles
            if not ({parent, *topology.predecessors(parent)} & ps):
                continue
            considered.add(event)
            if not invalid and self.l.dm._is_strongly_connected_division(topology, parent, ps, ds):
                compatible.add(event)
        return bool(invalid or evaluable or considered), compatible, considered

    def division_counts(self, topology, overrides):
        infos = {**self.base_fork_info, **overrides} if hasattr(self, 'base_fork_info') else overrides
        candidate = {event: set() for event in self.l.roles}
        evaluable = set()
        for p, (valid, compatible, _) in infos.items():
            if valid:
                evaluable.add(p)
            for event in compatible:
                candidate[event].add(p)
        pairing = self.l.dm._bipartite_max_matching(list(candidate), candidate)
        tp = set(pairing.values())
        return dict(counts=(len(pairing), len(evaluable-tp), len(candidate)-len(pairing)),
                    pairing=pairing, fp=evaluable-tp)

    def label(self, decision):
        sparse = self.l.decision(decision)
        if not decision.remove and not decision.add:
            return dict(sparse, delta=[0]*6, supported=False, utility=0.)
        topology = Topology(self.l.pred, self.l.succ, decision.remove, decision.add)
        touched = {n for e in decision.remove | decision.add for n in e}
        affected = set(touched)
        # Changed child/grandchild ownership can affect predecessor forks too.
        for _ in range(2):
            affected.update(q for n in tuple(affected) for q in
                            [*self.l.pred[n], *topology.predecessors(n)])
        overrides = {p: self.fork_info(topology, p) for p in affected}
        div = self.division_counts(topology, overrides)
        edge = np.zeros(2, np.int64)
        for sign, edges in ((1, decision.add), (-1, decision.remove)):
            for e in edges:
                edge += sign*np.asarray(self.edge_label(e))
        delta = [int(edge[0]), int(edge[1]), -int(edge[0]),
                 *[int(a-b) for a,b in zip(div['counts'], self.base_div['counts'])]]
        supported = (any(self.edge_label(e) != (0,0) for e in decision.remove | decision.add)
                     or sparse['biological'] >= 0 or any(delta[3:]))
        return dict(sparse, delta=delta, supported=bool(supported),
                    recovered_divisions=sorted(set(div['pairing'])-set(self.base_div['pairing'])),
                    lost_divisions=sorted(set(self.base_div['pairing'])-set(div['pairing'])),
                    fp_added=sorted(div['fp']-self.base_div['fp']),
                    fp_removed=sorted(self.base_div['fp']-div['fp']))


class SourceUtility:
    """Exact aggregate metric change for a single fixed-node source edit."""
    def __init__(self, receipts):
        self.edge_den = sum(r['edge_tp']+r['edge_fp']+r['edge_fn'] for r in receipts)
        self.edge_num = sum(r['adj_edge_jaccard']*(r['edge_tp']+r['edge_fp']+r['edge_fn']) for r in receipts)
        self.div_tp = sum(r['division_tp'] for r in receipts)
        self.div_den = sum(r['division_tp']+r['division_fp']+r['division_fn'] for r in receipts)
        self.before = self.edge_num/self.edge_den + (.1*self.div_tp/self.div_den if self.div_den else 0.)

    def value(self, delta, row):
        et, ef, en, dt, df, dn = delta
        penalty = max(0., 1-.1*(row['num_pred_nodes']-row['estimated_total'])/row['estimated_total'])
        dden = self.div_den+dt+df+dn
        after = (self.edge_num+penalty*et)/(self.edge_den+et+ef+en)
        after += .1*(self.div_tp+dt)/dden if dden else 0.
        return float(after-self.before)
