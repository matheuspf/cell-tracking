"""Complete competing local graphs, atomic ownership and bounded exact solving."""
from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np

from strong_tracker_v3.decode import Action, BoundedActionComponents, legal_edges, solve_actions

from .common import adjacency, digest


def fork_support(nodes, edges):
    """Protect all evidence in the official predecessor/child/grandchild window."""
    _, pred, succ = adjacency(nodes, edges)
    protected = set()
    for p, children in enumerate(succ):
        if len(children) == 2:
            protected.update([p, *pred[p], *children])
            protected.update(q for d in children for q in succ[d])
    return protected


@dataclass
class Decision:
    kind: str
    event: tuple
    remove: frozenset
    add: frozenset
    owners: tuple
    births: tuple
    terminations: tuple
    resources: frozenset

    @property
    def key(self):
        return digest([sorted(self.remove), sorted(self.add)])


class Alternatives:
    def __init__(self, bank, replace=False, possible=None):
        self.bank, self.replace = bank, replace
        self.nodes, self.pred, self.succ = bank.nodes, bank.pred, bank.succ
        self.old = {(i, j) for i, children in enumerate(self.succ) for j in children}
        self.protected = set() if replace else fork_support(bank.nodes, bank.edges)
        self.rejections = Counter()
        self.possible = possible

    def complete(self, event, desired, kind):
        desired = dict(desired)
        occupied = {j for ch in desired.values() for j in ch}
        if sum(map(len, desired.values())) != len(occupied):
            self.rejections['shared_daughters'] += 1
            return
        donors = sorted({self.pred[j][0] for j in occupied if self.pred[j] and self.pred[j][0] not in desired})
        donor_options = []
        for owner in donors:
            if len(self.succ[owner]) == 2 and not self.replace:
                self.rejections['existing_owner_fork_window'] += 1
                return
            remaining = tuple(j for j in self.succ[owner] if j not in occupied)
            available = [j for j in self.bank.near[owner] if j not in occupied and
                         (not self.pred[j] or self.pred[j][0] == owner or
                          (self.pred[j][0] in desired and j not in desired[self.pred[j][0]]))]
            # All legal donor continuations in the frozen neighbor union, plus termination.
            options = [remaining, *[(j,) for j in available], ()]
            donor_options.append(list(dict.fromkeys(options)))
        if self.possible is not None and not self.possible(event, desired, donors, donor_options, kind):
            self.rejections['nonpositive_complete_score_upper_bound'] += 1
            return
        for choices in product(*donor_options):
            full = {**desired, **dict(zip(donors, choices))}
            new = {(i, j) for i, children in full.items() for j in children}
            if len({j for _, j in new}) != len(new):
                continue
            old = {(i, j) for i in full for j in self.succ[i]}
            remove, add = old-new, new-old
            touched = {n for edge in remove | add for n in edge}
            if touched & self.protected:
                self.rejections['existing_fork_full_window'] += 1
                continue
            if not remove and not add:
                continue
            # A node cannot be a source/target in another neighboring-anchor edit.
            event_nodes = {n for n in event if n >= 0}
            event_nodes.update(self.pred[event[0]])
            resources = frozenset(('node', n) for n in touched | event_nodes)
            births = tuple(sorted(j for _, j in remove if j not in {b for _, b in new}))
            terminations = tuple(sorted(i for i, ch in full.items() if not ch and self.succ[i]))
            owners = tuple((o, tuple(self.succ[o]), tuple(full[o])) for o in donors)
            yield Decision(kind, tuple(event), frozenset(remove), frozenset(add), owners, births, terminations, resources)

    def event(self, event):
        p, a, b, qa, qb = map(int, event)
        yield Decision('keep', tuple(event), frozenset(), frozenset(), (), (), (), frozenset())
        seen = set()
        families = [('continuation_a', {p: (a,)}), ('continuation_b', {p: (b,)})]
        fork = {p: (a, b)}
        for d, q in [(a, qa), (b, qb)]:
            if q >= 0:
                fork[d] = (q,)
        families.append(('division', fork))
        # Explicit two-parent continuation alternatives use the same frozen bank.
        for d, other in [(a, b), (b, a)]:
            if self.pred[other] and self.pred[other][0] != p:
                families.append(('two_parents', {p: (d,), self.pred[other][0]: (other,)}))
        for kind, desired in families:
            for decision in self.complete(event, desired, kind):
                if decision.key not in seen:
                    seen.add(decision.key)
                    yield decision


def apply_decisions(nodes, edges, decisions, values, *, max_fraction=.02, time_limit=2.):
    """Zero residual preserves byte-order as well as the P0 edge set."""
    collector = BoundedActionComponents()
    for k, (d, value) in enumerate(zip(decisions, values, strict=True)):
        if not np.isfinite(value):
            raise ValueError('Nonfinite decision objective')
        if d.kind == 'keep' or value <= 1e-9:
            continue
        action = Action(k, float(value), set(d.remove), set(d.add), set(d.resources),
                        [dict(event=d.event), *d.owners], d.kind)
        collector.add(action)
    actions, streaming = collector.finish()
    chosen, solver = solve_actions(actions, max_changed_edges=int(np.floor(max_fraction*len(edges))), time_limit=time_limit)
    ix = {int(n[0]): i for i, n in enumerate(nodes)}
    old = {(ix[int(a)], ix[int(b)]) for a, b in edges}
    result = set(old)
    ledger = []
    convert = lambda es: [[int(nodes[a, 0]), int(nodes[b, 0])] for a, b in sorted(es)]
    for k in chosen:
        action = actions[k]
        if not action.remove <= result:
            raise RuntimeError('Atomic component removed an edge owned by another edit')
        result.difference_update(action.remove)
        result.update(action.add)
        ledger.append(dict(kind=action.kind, value=action.value, event=action.owners[0]['event'],
                           removed=convert(action.remove), added=convert(action.add),
                           resources=sorted(action.resources), owners=action.owners[1:], solver_status='optimal'))
    out = edges.copy() if result == old else np.asarray(convert(result), np.int64).reshape(-1, 2)
    if not legal_edges(nodes, set(map(tuple, out))):
        raise RuntimeError('Complete decision graph violates ownership or consecutive-time rules')
    changed = old ^ result
    explained = set()
    for k in chosen:
        explained |= actions[k].remove | actions[k].add
    if changed != explained:
        raise RuntimeError('Changes outside logged atomic components')
    return out, dict(**streaming, **solver, changed_edges=len(changed),
                     exact_unchanged_outside_components=True, edits=ledger)
