"""Exact edge-set queries without scanning the full graph for each raw alternative."""
from collections import defaultdict


class ObservationEdges:
    def __init__(self, incumbent_edges, raw_edges):
        self.incident = defaultdict(set)
        self.raw_outgoing = defaultdict(set)
        for a,b in incumbent_edges:
            edge = (int(a),int(b))
            self.incident[int(a)].add(edge)
            self.incident[int(b)].add(edge)
        for a,b in raw_edges:
            self.raw_outgoing[int(a)].add(int(b))

    def removed(self, replacement):
        return {edge for node in replacement for edge in self.incident.get(node,())}

    def restored(self, raw_ids, prefix):
        selected = set(raw_ids)
        return {(prefix+a,prefix+b) for a in selected
                for b in self.raw_outgoing.get(a,()) if b in selected}
