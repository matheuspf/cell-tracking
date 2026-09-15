"""Shared-bank inference with exact additive ownership bounds.

Neural identity scores and birth/termination costs are node/pair quantities. They
are evaluated once, instead of once per combinatorial donor assignment. An
optimistic bound drops only families whose every complete assignment is
non-improving for every supplied frozen model; it does not change the bank.
"""
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .actions import Alternatives, apply_decisions
from .bank import EventBank
from .common import WORK, digest, graph_hash, save_graph, sha, write_json
from .infer import embeddings
from .scoring import EVENT_SCALE, edge_features, load_model


class AdditiveScores:
    def __init__(self, models, specs, bank, z, valid):
        self.models, self.specs, self.bank = models, specs, bank
        self.names = list(models)
        pairs = sorted(set(map(tuple, bank.native['pairs'])) |
                       {(i, j) for i in range(len(bank.nodes)) for j in set(bank.near[i]) | set(bank.succ[i])})
        self.pairs = pairs
        self.index = {p: k for k, p in enumerate(pairs)}
        x = edge_features(bank.nodes, bank.native, pairs, np.ones(3))
        # Missing-pair physical distances must use the same input metadata as the
        # registered bank, not the unit placeholder used to fetch known rows.
        missing = x[:, 24] > 0
        for k in np.flatnonzero(missing):
            a, b = pairs[k]
            x[k, 3] = np.linalg.norm(bank.pos[a]-bank.pos[b])
        self.links = np.empty((len(pairs), len(models)), np.float64)
        self.costs = np.empty((len(bank.nodes), 2, len(models)), np.float64)
        self.z = z
        self.valid = valid
        self.current = None
        with torch.inference_mode():
            for m, name in enumerate(self.names):
                model, spec = models[name], specs[name]
                device = next(model.parameters()).device
                zz = torch.as_tensor(z[name], device=device)
                for start in range(0, len(pairs), 4096):
                    pp = torch.as_tensor(pairs[start:start+4096], device=device)
                    xx = x[start:start+4096]
                    normalized = np.clip((xx-spec['mean'])/spec['scale'], -10, 10).astype(np.float32)
                    out = model.link_scores(zz, pp, torch.as_tensor(normalized, device=device))
                    self.links[start:start+len(pp), m] = out.cpu().numpy()+xx[:, 23]
                anchor = 1 if model.family == 'compact' else 2
                vv = torch.as_tensor(valid[name][:, anchor], device=device)
                self.costs[:, :, m] = F.softplus(model.boundary(torch.cat([zz, vv], -1))).cpu().numpy()
        self.threshold = np.array([-specs[n]['calibration']['intercept']*specs[n]['calibration']['temperature'] for n in self.names])

    def sum_links(self, edges):
        ids = [self.index[tuple(p)] for p in edges]
        return self.links[ids].sum(0) if ids else np.zeros(len(self.names))

    def set_event(self, event, values):
        self.current, self.event_value = tuple(event), np.asarray(values)

    def possible(self, event, desired, donors, options, kind):
        if tuple(event) != self.current:
            raise RuntimeError('Bound used an event score from another time anchor')
        # Ignore all nonnegative birth/termination costs and donor collisions.
        # The resulting objective is an upper bound, never a heuristic gate.
        new = [(i, j) for i, children in desired.items() for j in children]
        old = [(i, j) for i in desired for j in self.bank.succ[i]]
        value = self.sum_links(new)-self.sum_links(old)
        for owner, choices in zip(donors, options):
            best = np.max([self.sum_links([(owner, j) for j in choice]) for choice in choices], axis=0)
            value += best-self.sum_links([(owner, j) for j in self.bank.succ[owner]])
        if kind == 'division':
            value += self.event_value
        return bool(np.any(value > self.threshold-1e-4))

    def decision(self, decision):
        value = self.sum_links(decision.add)-self.sum_links(decision.remove)
        if decision.births:
            value -= self.costs[list(decision.births), 0].sum(0)
        if decision.terminations:
            value -= self.costs[list(decision.terminations), 1].sum(0)
        if decision.kind == 'division':
            value += self.event_value
        return np.array([value[k]/self.specs[n]['calibration']['temperature']+self.specs[n]['calibration']['intercept']
                         for k, n in enumerate(self.names)])


def event_batches(bank, size=2048):
    pending = []
    for event, f, _ in bank:
        pending.append((event, f))
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


@torch.no_grad()
def event_values(models, z, rows):
    events = np.array([e[:3] for e, _ in rows], np.int64)
    features = np.clip(np.array([f for _, f in rows])/EVENT_SCALE, -10, 10).astype(np.float32)
    result = []
    for name, model in models.items():
        device = next(model.parameters()).device
        zz = torch.as_tensor(z[name], device=device)
        a, b = model.event_scores(zz, torch.as_tensor(events, device=device), torch.as_tensor(features, device=device))
        result.append((a+b).cpu().numpy())
    return np.stack(result, axis=-1)


def predict_many(row, graph, native, packages, destinations, cache_root, *, bound=True, device='cuda'):
    from strong_tracker_v3.decode import Action, BoundedActionComponents, legal_edges, solve_actions
    started = time.monotonic()
    models, specs, z, valid = {}, {}, {}, {}
    bank = EventBank(graph['nodes'], graph['edges'], native, row['physical_scale'])
    for name, path in packages.items():
        model, spec = load_model(path)
        if spec['recipe']['arm'] == 'A10':
            raise ValueError('The continuation-only decoder is a separate independent lane')
        models[name], specs[name] = model.to(device), spec
        z[name], valid[name] = embeddings(model, row, graph, bank,
            Path(cache_root)/spec['weights_sha256']/f'{row["dataset"]}.npz', spec['weights_sha256'])
    scoring = AdditiveScores(models, specs, bank, z, valid)
    alternatives = Alternatives(bank, possible=scoring.possible if bound else None)
    collectors = {name: BoundedActionComponents() for name in models}
    parent, seen, counts = None, set(), Counter()
    for batch in event_batches(bank):
        ev = event_values(models, z, batch)
        for (event, _), values in zip(batch, ev):
            if parent != event[0]:
                parent, seen = event[0], set()
            scoring.set_event(event, values)
            counts['events'] += 1
            for d in alternatives.event(event):
                if d.kind == 'keep' or d.key in seen:
                    continue
                seen.add(d.key)
                value = scoring.decision(d)
                counts['complete_assignments_scored'] += 1
                for k, name in enumerate(models):
                    if value[k] > 1e-9:
                        collectors[name].add(Action(counts['complete_assignments_scored'], float(value[k]),
                            set(d.remove), set(d.add), set(d.resources), [dict(event=d.event), *d.owners], d.kind))
        if counts['events'] % 102400 == 0:
            print(f'Shared frozen bank: {counts["events"]} events, {counts["complete_assignments_scored"]} potentially improving assignments', flush=True)
    receipts = {}
    old = {(i, j) for i, children in enumerate(bank.succ) for j in children}
    for name, collector in collectors.items():
        actions, streaming = collector.finish()
        chosen, solving = solve_actions(actions, max_changed_edges=int(.02*len(graph['edges'])))
        final = set(old)
        edits = []
        convert = lambda es: [[int(graph['nodes'][a, 0]), int(graph['nodes'][b, 0])] for a, b in sorted(es)]
        for k in chosen:
            a = actions[k]
            if not a.remove <= final:
                raise RuntimeError('Atomic ownership drift')
            final.difference_update(a.remove)
            final.update(a.add)
            edits.append(dict(kind=a.kind, value=a.value, event=a.owners[0]['event'], removed=convert(a.remove),
                              added=convert(a.add), owners=a.owners[1:], solver_status='optimal'))
        edges = graph['edges'].copy() if final == old else np.asarray(convert(final), np.int64).reshape(-1, 2)
        if not legal_edges(graph['nodes'], set(map(tuple, edges))):
            raise RuntimeError('Shared-bank graph contract failed')
        changed = old ^ final
        if changed != {p for k in chosen for p in actions[k].remove | actions[k].add}:
            raise RuntimeError('Graph changed outside its atomic edit ledger')
        destination = Path(destinations[name])
        save_graph(destination, graph['nodes'], edges)
        ledger = dict(**streaming, **solving, changed_edges=len(changed), edits=edits,
                      bank_structural_rejections=dict(alternatives.rejections), exact_unchanged_outside_components=True)
        receipt = dict(arm=specs[name]['recipe']['arm'], source=specs[name]['recipe']['source'],
            model_sha256=specs[name]['weights_sha256'], graph_hash=graph_hash(graph['nodes'], edges),
            prediction_sha256=sha(destination), unchanged_nodes=True, bank_sha256=bank.hash, ledger=ledger,
            seconds=time.monotonic()-started, shared_model_count=len(models), enumeration=dict(counts),
            optimization='Additive upper bound; all legal improving complete assignments retained', annotation_reads=0)
        write_json(destination.with_suffix('.json'), receipt)
        receipts[name] = receipt
        models[name].cpu()
    torch.cuda.empty_cache()
    return receipts
