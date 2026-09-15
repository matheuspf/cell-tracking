"""Source-only supervision. Biological identity and metric risk are distinct.

No quiet-chain negative rule: a single recorded child does not exclude mitosis.
"""
from collections import defaultdict

import numpy as np

from .common import adjacency


def supported_incoming(pairs, matches, gt_edges):
    """Persisted IDs in and out. Unmatched parents are not certified identities."""
    gt_edges = set(map(tuple, gt_edges))
    observed_targets = {b for _, b in gt_edges}
    result = []
    for a, b in pairs:
        ga, gb = matches.get(int(a)), matches.get(int(b))
        result.append(1 if (ga, gb) in gt_edges else
                      0 if ga is not None and gb in observed_targets else -1)
    return np.asarray(result, np.int8)


class Topology:
    """Full ownership overlay consumed by the unchanged official local-window rule."""
    def __init__(self, pred, succ, remove=(), add=()):
        self.pred, self.succ = pred, succ
        self.before, self.after = {}, {}
        for a, b in set(remove) | set(add):
            self.before.setdefault(b, set(pred[b]))
            self.after.setdefault(a, set(succ[a]))
        for a, b in remove:
            self.before[b].discard(a)
            self.after[a].discard(b)
        for a, b in add:
            self.before[b].add(a)
            self.after[a].add(b)

    def predecessors(self, node):
        return sorted(self.before[node]) if node in self.before else self.pred[node]

    def successors(self, node):
        return sorted(self.after[node]) if node in self.after else self.succ[node]


class SourceLabels:
    def __init__(self, nodes, edges, gt_nodes, gt_edges, scale):
        from annotation_selection.metric_adapter import make_graph
        from tracking_cellmot import division_metrics as dm
        self.dm, self.nodes = dm, nodes
        _, self.pred, self.succ = adjacency(nodes, edges)
        g, reverse = make_graph(nodes, edges)
        gt, gt_reverse = make_graph(gt_nodes, gt_edges)
        if not all(reverse[i] == int(n[0]) for i, n in enumerate(nodes)):
            raise ValueError('Official internal-index contract changed')
        self.gt = gt
        self.gt_reverse = gt_reverse
        self.matches = dict(dm._matched_node_attrs(dm._match_full(g, gt, tuple(scale), 7.)).iter_rows())
        self.components = dm._gt_weak_component_ids(gt)
        windows = dm.extract_divisions(gt)
        matched = dm.match_divisions(g, gt, tuple(scale), 7.)
        self.roles = {event: dm._matched_division_nodes(dm._matched_node_attrs(matched[event]), window, event)
                      for event, window in windows.items()}
        # edge_attrs also includes a library edge ID; use explicit persisted-ID mapping.
        gt_index = {int(n[0]): i for i, n in enumerate(gt_nodes)}
        self.gt_edges = {(gt_index[int(a)], gt_index[int(b)]) for a, b in gt_edges}
        self.persistent_matches = {int(nodes[p, 0]): int(gt_reverse[q]) for p, q in self.matches.items()}
        self.groups = {p: self.components[q] for p, q in self.matches.items()}
        self.event_groups = {event: self.components[event] for event in self.roles}

    def branch(self, topology, parent, child):
        return self.dm._branch_component_evidence(topology, parent, child, self.matches, self.components)

    def decision(self, decision):
        topology = Topology(self.pred, self.succ, decision.remove, decision.add)
        parent = decision.event[0]
        children = topology.successors(parent)
        fork = len(children) == 2
        valid_events = []
        considered = False
        branches = [self.branch(topology, parent, d) for d in children] if fork else []
        cross = len({c for c, _ in branches if c is not None}) > 1
        malformed = any(m for _, m in branches)
        if fork:
            for event, roles in self.roles.items():
                if roles is None:
                    continue
                ps, ds = roles
                if not ({parent, *topology.predecessors(parent)} & ps):
                    continue
                considered = True
                if not cross and not malformed and self.dm._is_strongly_connected_division(topology, parent, ps, ds):
                    valid_events.append(event)
        # A direct matched child refutes a wrong annotated incoming parent.
        # An unannotated real branch contributes neither negative nor positive loss.
        contradictions = supported = 0
        affected_sources = {a for a, _ in decision.remove | decision.add} | {parent}
        for a in affected_sources:
            for b in topology.successors(a):
                ga, gb = self.matches.get(a), self.matches.get(b)
                if (ga, gb) in self.gt_edges:
                    supported += 1
                elif ga is not None and gb is not None and self.gt.predecessors(gb):
                    contradictions += 1
        # A donor termination or redirected parent can discard a known true
        # continuation even when a different retained edge is correct. Complete
        # alternatives must pay for that loss. Official timing-compatible fork
        # positives remain a set separately from exact-edge identity targets.
        lost_supported = sum((self.matches.get(a), self.matches.get(b)) in self.gt_edges
                             for a, b in decision.remove)
        contradictions += lost_supported
        identity = 0 if contradictions else 1 if supported else -1
        biological = 1 if valid_events else 0 if cross or contradictions else -1
        # Official FP evaluability is explicitly separate from biological nondivision.
        metric_evaluable = (fork and (considered or cross or malformed or
                            (parent in self.matches and self.gt.successors(self.matches[parent]))))
        risk = 1 if valid_events else 0 if metric_evaluable else -1
        return dict(identity=identity, biological=biological, metric_fork_target=risk,
                    metric_evaluability='local_window' if considered else 'cross_lineage' if cross else
                    'matched_parent_has_recorded_child' if metric_evaluable else 'unknown',
                    compatible_events=[self.gt_reverse[e] for e in valid_events],
                    biological_group=self.groups.get(parent), supported_edges=supported,
                    contradictory_edges=contradictions, lost_supported_edges=lost_supported)

    def group_anchors(self, bank):
        """Training sampling may select labels, but cannot add deployment anchors."""
        groups = defaultdict(set)
        for p, component in self.groups.items():
            if p in bank.expanded:
                groups[component].add(p)
        for event, roles in self.roles.items():
            if roles is None:
                continue
            ps, _ = roles
            candidates = ps | {j for p in ps for j in bank.near[p]}
            groups[self.event_groups[event]].update(candidates & bank.expanded)
        return {k: sorted(v) for k, v in groups.items()}
