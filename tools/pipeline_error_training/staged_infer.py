"""Serialize exact neural quantities, then release the GPU for CPU graph solving.

This preserves the full shared event bank and the additive upper bound verified
against the original per-parent implementation. Independent clips can solve on
the CPU while the next clip uses the single cooperative GPU lease.
"""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch

from .actions import Alternatives
from .bank import EventBank
from .common import digest, graph_hash, load_graph, read_json, save_arrays, save_graph, sha, write_json
from .fast_infer import AdditiveScores, event_batches
from .infer import embeddings
from .resources import Lease
from .scoring import EVENT_SCALE, load_model


def prepare(row, graph, native, packages, root, monitor=None, wait=False, device='cuda'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bank = EventBank(graph['nodes'], graph['edges'], native, row['physical_scale'])
    names = list(packages)
    models, specs = {}, {}
    for name, path in packages.items():
        model, spec = load_model(path)
        if not spec['recipe']['arm'].startswith('D'):
            raise ValueError('Only division models share the complete-event bank')
        models[name], specs[name] = model, spec
    stamp = dict(bank_sha256=bank.hash, model_sha256={n: specs[n]['weights_sha256'] for n in names},
                 package_sha256={n: sha(Path(packages[n])/'frozen_package.json') for n in names},
                 implementation_sha256=sha(Path(__file__)), device=device)
    final = root/'prepared.json'
    if final.exists():
        receipt = read_json(final)
        if receipt['inputs'] != stamp or any(sha(root/k) != v for k, v in receipt['files'].items()):
            raise ValueError('Frozen staged neural quantities changed')
        # Every used image chunk is checked, even when all neural outputs exist.
        for name in names:
            embeddings(models[name], row, graph, bank, root/'embeddings'/specs[name]['weights_sha256']/f'{row["dataset"]}.npz',
                       specs[name]['weights_sha256'])
        return bank, receipt
    dtype = np.dtype([('event', '<i8', (5,)), ('value', '<f4', (len(names),))])
    z, valid = {}, {}
    begin = time.monotonic()
    with Lease(required_gib=8., wait=wait) if device == 'cuda' else nullcontext():
        leased = time.monotonic()
        for name, model in models.items():
            model.to(device)
            if device == 'cpu' and not (root/'embeddings'/specs[name]['weights_sha256']/f'{row["dataset"]}.npz').exists():
                raise ValueError('CPU parity requires verified existing image-derived embeddings')
            z[name], valid[name] = embeddings(model, row, graph, bank,
                root/'embeddings'/specs[name]['weights_sha256']/f'{row["dataset"]}.npz', specs[name]['weights_sha256'])
            if monitor:
                monitor.check()
        scalar = AdditiveScores(models, specs, bank, z, valid)
        zz = {name: torch.as_tensor(z[name], device=device) for name in names}
        count = 0
        with (root/'events.partial').open('wb') as stream, torch.inference_mode():
            for rows in event_batches(bank):
                events = np.asarray([e for e, _ in rows], np.int64)
                indices = torch.as_tensor(events[:, :3], device=device)
                features = torch.as_tensor(np.clip(np.asarray([f for _, f in rows])/EVENT_SCALE, -10, 10).astype(np.float32), device=device)
                record = np.empty(len(rows), dtype=dtype)
                record['event'] = events
                for k, name in enumerate(names):
                    value, risk = models[name].event_scores(zz[name], indices, features)
                    record['value'][:, k] = (value+risk).cpu().numpy()
                stream.write(record.tobytes())
                count += len(rows)
                if monitor:
                    monitor.check()
        save_arrays(root/'scalars.npz', pairs=np.asarray(scalar.pairs, np.int64), links=scalar.links, costs=scalar.costs,
                    threshold=scalar.threshold)
        for model in models.values():
            model.cpu()
        del zz, scalar
        torch.cuda.empty_cache()
        gpu_seconds = time.monotonic()-leased if device == 'cuda' else 0.
    (root/'events.partial').replace(root/'events.bin')
    receipt = dict(status='measured', inputs=stamp, names=names, records=count,
        dtype='event:5*i8; value:M*f4', files={k: sha(root/k) for k in ['events.bin', 'scalars.npz']},
        calibration={n: specs[n]['calibration'] for n in names}, arms={n: specs[n]['recipe']['arm'] for n in names},
        sources={n: specs[n]['recipe']['source'] for n in names}, elapsed_seconds=time.monotonic()-begin,
        measured_gpu_lease_seconds=gpu_seconds, complete_bank=True, no_annotation_reads=True)
    write_json(final, receipt, immutable=True)
    return bank, receipt


def solve(row, graph, bank, root, destinations, monitor=None):
    from strong_tracker_v3.decode import Action, BoundedActionComponents, legal_edges, solve_actions
    root = Path(root)
    prepared = read_json(root/'prepared.json')
    if any(sha(root/k) != v for k, v in prepared['files'].items()):
        raise ValueError('Staged event data changed before solving')
    names = prepared['names']
    a = load_graph(root/'scalars.npz')
    scalar = AdditiveScores.__new__(AdditiveScores)
    scalar.names = names
    scalar.specs = {n: dict(calibration=prepared['calibration'][n]) for n in names}
    scalar.bank, scalar.pairs = bank, a['pairs']
    scalar.index = {tuple(p): k for k, p in enumerate(a['pairs'])}
    scalar.links, scalar.costs, scalar.threshold = a['links'], a['costs'], a['threshold']
    alternatives = Alternatives(bank, possible=scalar.possible)
    collectors = {name: BoundedActionComponents() for name in names}
    dtype = np.dtype([('event', '<i8', (5,)), ('value', '<f4', (len(names),))])
    events = np.memmap(root/'events.bin', mode='r', dtype=dtype, shape=(prepared['records'],)) if prepared['records'] else np.empty(0, dtype=dtype)
    parent, seen, count = None, set(), Counter()
    begin = time.monotonic()
    for record in events:
        event = tuple(map(int, record['event']))
        if parent != event[0]:
            parent, seen = event[0], set()
        scalar.set_event(event, record['value'])
        count['events'] += 1
        for decision in alternatives.event(event):
            if decision.kind == 'keep' or decision.key in seen:
                continue
            seen.add(decision.key)
            value = scalar.decision(decision)
            count['complete_assignments_scored'] += 1
            for k, name in enumerate(names):
                if value[k] > 1e-9:
                    collectors[name].add(Action(count['complete_assignments_scored'], float(value[k]), set(decision.remove),
                        set(decision.add), set(decision.resources), [dict(event=decision.event), *decision.owners], decision.kind))
        if count['events'] % 102400 == 0:
            if monitor:
                monitor.check()
            print(f'CPU complete bank: {count["events"]}/{len(events)} events, {count["complete_assignments_scored"]} assignments', flush=True)
    old = {(i, j) for i, children in enumerate(bank.succ) for j in children}
    results = {}
    for name, collector in collectors.items():
        actions, streaming = collector.finish()
        chosen, solving = solve_actions(actions, max_changed_edges=int(.02*len(graph['edges'])))
        final, edits = set(old), []
        convert = lambda es: [[int(graph['nodes'][a, 0]), int(graph['nodes'][b, 0])] for a, b in sorted(es)]
        for k in chosen:
            action = actions[k]
            if not action.remove <= final:
                raise RuntimeError('Staged atomic owner assignment drift')
            final.difference_update(action.remove); final.update(action.add)
            edits.append(dict(kind=action.kind, value=action.value, event=action.owners[0]['event'],
                removed=convert(action.remove), added=convert(action.add), owners=action.owners[1:], solver_status='optimal'))
        edges = graph['edges'].copy() if old == final else np.asarray(convert(final), np.int64).reshape(-1, 2)
        if not legal_edges(graph['nodes'], set(map(tuple, edges))):
            raise RuntimeError('Staged complete graph violates ownership')
        if old ^ final != {p for k in chosen for p in actions[k].remove | actions[k].add}:
            raise RuntimeError('Graph changes escaped the complete-action ledger')
        destination = Path(destinations[name])
        save_graph(destination, graph['nodes'], edges)
        receipt = dict(status='measured', arm=prepared['arms'][name], source=prepared['sources'][name],
            model_sha256=prepared['inputs']['model_sha256'][name], graph_hash=graph_hash(graph['nodes'], edges),
            prediction_sha256=sha(destination), unchanged_nodes=True, bank_sha256=bank.hash,
            ledger=dict(**streaming, **solving, changed_edges=len(old ^ final), edits=edits,
                        bank_structural_rejections=dict(alternatives.rejections), exact_unchanged_outside_components=True),
            seconds=time.monotonic()-begin+prepared['elapsed_seconds'], cpu_solve_seconds=time.monotonic()-begin,
            measured_gpu_lease_seconds=prepared['measured_gpu_lease_seconds'], shared_model_count=len(names),
            enumeration=dict(count), prepared_sha256=sha(root/'prepared.json'), annotation_reads=0)
        write_json(destination.with_suffix('.json'), receipt)
        results[name] = receipt
    return results


def predict_many(row, graph, native, packages, destinations, root, monitor=None, wait=False):
    bank, _ = prepare(row, graph, native, packages, root, monitor, wait)
    return solve(row, graph, bank, root, destinations, monitor)
