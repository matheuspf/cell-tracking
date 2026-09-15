"""Fixed decoder ablation: pay for every displaced or changed incumbent fork."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .actions import Alternatives
from .common import graph_hash, load_graph, read_json, save_graph, sha, write_json
from .fast_infer import AdditiveScores


def fork_signatures(bank, decision):
    """Before/after fork windows affected by the entire atomic ownership edit."""
    changed = {a for a, _ in decision.remove | decision.add}
    successors = {i: set(bank.succ[i]) for i in changed}
    for a, b in decision.remove:
        successors[a].remove(b)
    for a, b in decision.add:
        successors[a].add(b)
    after = lambda i: sorted(successors.get(i, bank.succ[i]))
    owners = set(changed)
    # An edited daughter continuation changes its parent's event window too.
    owners.update(p for i in changed for p in bank.pred[i])
    owners.update(a for a, b in decision.add if b in changed)
    def signature(i, succ):
        children = sorted(succ(i))
        if len(children) != 2:
            return None
        a, b = children
        if len(succ(a)) > 1 or len(succ(b)) > 1:
            return ('unrepresented_branching_grandchild', i)
        return (i, a, b, succ(a)[0] if len(succ(a)) else -1, succ(b)[0] if len(succ(b)) else -1)
    before = {s for i in owners if (s := signature(i, lambda j: bank.succ[j])) is not None}
    new = {s for i in owners if (s := signature(i, after)) is not None}
    return before-new, new-before


class ReplacementAlternatives(Alternatives):
    def __init__(self, bank, scalar=None):
        self.old_forks = {i for i, children in enumerate(bank.succ) if len(children) == 2}
        self.scalar = scalar
        super().__init__(bank, replace=True, possible=self.possible_replacement if scalar is not None else None)

    def possible_replacement(self, event, desired, donors, options, kind):
        sources = set(desired) | set(donors)
        affected_parents = sources | {p for i in sources for p in self.pred[i]}
        if affected_parents & self.old_forks:
            # Removing or changing an old fork can add energy. Never apply the
            # additive bound in this region; enumerate every complete assignment.
            return True
        # No old fork energy changes here. Only the event parent can acquire two
        # children, so the proven additive objective and bound apply exactly.
        return self.scalar.possible(event, desired, donors, options, kind)

    def complete(self, event, desired, kind):
        choices = [dict(desired)]
        if kind == 'division':
            p = event[0]
            for parent in self.pred[p]:
                if len(self.succ[parent]) == 2:
                    # An explicit later-anchor alternative converts the former
                    # parent's fork to a continuation and pays for the orphan.
                    choices.append({**desired, parent: (p,)})
        seen = set()
        for complete in choices:
            for decision in super().complete(event, complete, kind):
                if decision.key in seen:
                    continue
                seen.add(decision.key)
                before, after = fork_signatures(self.bank, decision)
                context = set()
                for signature in before | after:
                    if len(signature) != 5:
                        continue
                    parent = signature[0]
                    context.update(i for i in signature if i >= 0)
                    context.update(self.pred[parent])
                yield replace(decision, resources=decision.resources | frozenset(('node', i) for i in context))


def complete_value(scalar, decision, event_values):
    before, after = fork_signatures(scalar.bank, decision)
    if any(s not in event_values for s in before | after):
        return None
    value = scalar.sum_links(decision.add)-scalar.sum_links(decision.remove)
    if decision.births:
        value -= scalar.costs[list(decision.births), 0].sum(0)
    if decision.terminations:
        value -= scalar.costs[list(decision.terminations), 1].sum(0)
    for s in after:
        value += event_values[s]
    for s in before:
        value -= event_values[s]
    return np.array([value[k]/scalar.specs[n]['calibration']['temperature']+scalar.specs[n]['calibration']['intercept']
                     for k, n in enumerate(scalar.names)])


def solve(row, graph, bank, root, destinations, monitor=None, *, bound=True):
    from strong_tracker_v3.decode import Action, BoundedActionComponents, legal_edges, solve_actions
    root = Path(root)
    prepared = read_json(root/'prepared.json')
    if any(sha(root/k) != v for k, v in prepared['files'].items()):
        raise ValueError('Replacement ablation changed its frozen neural inputs')
    names = prepared['names']
    chosen_names = [n for n in names if n in destinations]
    dtype = np.dtype([('event', '<i8', (5,)), ('value', '<f4', (len(names),))])
    records = np.memmap(root/'events.bin', mode='r', dtype=dtype, shape=(prepared['records'],)) if prepared['records'] else []
    values = {tuple(map(int, r['event'])): r['value'].astype(np.float64) for r in records}
    candidates = list(values)
    reference = load_graph(root/'reference_forks.npz')
    for event, value in zip(reference['events'], reference['value']):
        key = tuple(map(int, event))
        if key in values:
            np.testing.assert_allclose(values[key], value, rtol=1e-5, atol=1e-6)
        values[key] = value.astype(np.float64)
    arrays = load_graph(root/'scalars.npz')
    scalar = AdditiveScores.__new__(AdditiveScores)
    scalar.names, scalar.bank = names, bank
    scalar.specs = {n: dict(calibration=prepared['calibration'][n]) for n in names}
    scalar.index = {tuple(p): i for i, p in enumerate(arrays['pairs'])}
    scalar.links, scalar.costs = arrays['links'], arrays['costs']
    scalar.threshold = arrays['threshold']
    alternatives = ReplacementAlternatives(bank, scalar if bound else None)
    collectors = {n: BoundedActionComponents() for n in chosen_names}
    parent, seen, count = None, set(), Counter()
    begin = time.monotonic()
    for event in candidates:
        if parent != event[0]:
            parent, seen = event[0], set()
        count['events'] += 1
        scalar.set_event(event, values[event])
        for decision in alternatives.event(event):
            if decision.kind == 'keep' or decision.key in seen:
                continue
            seen.add(decision.key)
            value = complete_value(scalar, decision, values)
            if value is None:
                count['unrepresented_complete_fork_window_abstentions'] += 1
                continue
            count['complete_assignments_scored'] += 1
            for n in chosen_names:
                k = names.index(n)
                if value[k] > 1e-9:
                    collectors[n].add(Action(count['complete_assignments_scored'], float(value[k]), set(decision.remove),
                        set(decision.add), set(decision.resources), [dict(event=decision.event), *decision.owners], decision.kind))
        if count['events'] % 102400 == 0:
            if monitor:
                monitor.check()
            print(f'Full fork replacement: {count["events"]}/{len(candidates)} events', flush=True)
    results = {}
    old = {(i, j) for i, children in enumerate(bank.succ) for j in children}
    convert = lambda es: [[int(graph['nodes'][a, 0]), int(graph['nodes'][b, 0])] for a, b in sorted(es)]
    for name, collector in collectors.items():
        actions, streaming = collector.finish()
        chosen, solving = solve_actions(actions, max_changed_edges=int(.02*len(graph['edges'])))
        final, edits = set(old), []
        for k in chosen:
            action = actions[k]
            if not action.remove <= final:
                raise RuntimeError('Replacement ablation atomic ownership drift')
            final.difference_update(action.remove); final.update(action.add)
            edits.append(dict(kind=action.kind, value=action.value, event=action.owners[0]['event'],
                removed=convert(action.remove), added=convert(action.add), owners=action.owners[1:], solver_status='optimal'))
        edges = graph['edges'].copy() if final == old else np.asarray(convert(final), np.int64).reshape(-1, 2)
        if not legal_edges(graph['nodes'], set(map(tuple, edges))):
            raise RuntimeError('Replacement ablation created an illegal graph')
        if old ^ final != {p for k in chosen for p in actions[k].remove | actions[k].add}:
            raise RuntimeError('Replacement changed edges outside complete components')
        destination = Path(destinations[name])
        save_graph(destination, graph['nodes'], edges)
        receipt = dict(status='measured', arm=prepared['arms'][name]+'_replacement', source=prepared['sources'][name],
            same_weights_and_calibration_as_protected_arm=True, replacement_forks_scored_as_complete_before_after=True,
            source_model_sha256=prepared['inputs']['model_sha256'][name], prediction_sha256=sha(destination),
            graph_hash=graph_hash(graph['nodes'], edges), bank_sha256=bank.hash, unchanged_nodes=True,
            seconds=time.monotonic()-begin, measured_gpu_lease_seconds=0., enumeration=count,
            bound_outside_existing_fork_regions=bound,
            ledger=dict(**streaming, **solving, changed_edges=len(old ^ final), edits=edits,
                        exact_unchanged_outside_components=True), annotation_reads=0)
        write_json(destination.with_suffix('.json'), receipt)
        results[name] = receipt
    return results
