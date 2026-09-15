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
from .shared_embeddings import embeddings_many


def prepare(row, graph, native, packages, root, monitor=None, wait=False, device='cuda', embedding_packages=None):
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
    # Continuation uses the same P0 tracklets and temporal crop contract. Its
    # separately trained encoder can consume the identical batches; its head
    # and graph solver remain in the independent continuation process.
    auxiliary_models, auxiliary_specs = {}, {}
    for name, path in (embedding_packages or {}).items():
        if name in models:
            raise ValueError('Auxiliary embedding names must not shadow division models')
        model, spec = load_model(path)
        if spec['recipe']['arm'] != 'A10':
            raise ValueError('Only continuation shares the full P0 embedding query')
        auxiliary_models[name], auxiliary_specs[name] = model, spec
    all_models, all_specs = dict(models, **auxiliary_models), dict(specs, **auxiliary_specs)
    stamp = dict(bank_sha256=bank.hash, model_sha256={n: specs[n]['weights_sha256'] for n in names},
                 package_sha256={n: sha(Path(packages[n])/'frozen_package.json') for n in names},
                 shared_embedding_code_sha256=sha(Path(__file__).with_name('shared_embeddings.py')),
                 implementation_sha256=sha(Path(__file__)), device=device)
    if auxiliary_models:
        stamp['auxiliary_embeddings'] = {n:dict(model_sha256=s['weights_sha256'],
            package_sha256=sha(Path(embedding_packages[n])/'frozen_package.json')) for n,s in auxiliary_specs.items()}
    final = root/'prepared.json'
    if final.exists():
        receipt = read_json(final)
        if receipt['inputs'] != stamp or any(sha(root/k) != v for k, v in receipt['files'].items()):
            raise ValueError('Frozen staged neural quantities changed')
        # Every used image chunk is checked, even when all neural outputs exist.
        for name in all_models:
            embeddings(all_models[name], row, graph, bank, root/'embeddings'/all_specs[name]['weights_sha256']/f'{row["dataset"]}.npz',
                       all_specs[name]['weights_sha256'])
        return bank, receipt
    dtype = np.dtype([('event', '<i8', (5,)), ('value', '<f4', (len(names),))])
    z, valid = {}, {}
    begin = time.monotonic()
    # Geometry enumeration is CPU work independent of every neural model. Keep
    # its exact 2048-row order and normalization, and acquire the GPU afterwards.
    feature_dtype = np.dtype([('event', '<i8', (5,)), ('features', '<f4', (40,))])
    feature_path = root/'event_features.partial'
    feature_count = 0
    with feature_path.open('wb') as stream:
        for rows in event_batches(bank):
            record = np.empty(len(rows), dtype=feature_dtype)
            record['event'] = np.asarray([e for e, _ in rows], np.int64)
            record['features'] = np.clip(np.asarray([f for _, f in rows])/EVENT_SCALE, -10, 10).astype(np.float32)
            stream.write(record.tobytes())
            feature_count += len(rows)
            if monitor:
                monitor.check()
    feature_sha256 = sha(feature_path)
    cpu_geometry_seconds = time.monotonic()-begin
    feature_records = np.memmap(feature_path, mode='r', dtype=feature_dtype, shape=(feature_count,)) if feature_count else []
    with Lease(required_gib=8., wait=wait) if device == 'cuda' else nullcontext():
        leased = time.monotonic()
        for name, model in all_models.items():
            model.to(device)
            if device == 'cpu' and not (root/'embeddings'/all_specs[name]['weights_sha256']/f'{row["dataset"]}.npz').exists():
                raise ValueError('CPU parity requires verified existing image-derived embeddings')
        shared = embeddings_many(all_models, row, graph, bank,
            {name:root/'embeddings'/all_specs[name]['weights_sha256']/f'{row["dataset"]}.npz' for name in all_models},
            {name:all_specs[name]['weights_sha256'] for name in all_models})
        for name in names:
            z[name], valid[name] = shared[name]
        if monitor:
            monitor.check()
        scalar = AdditiveScores(models, specs, bank, z, valid)
        zz = {name: torch.as_tensor(z[name], device=device) for name in names}
        count = 0
        with (root/'events.partial').open('wb') as stream, torch.inference_mode():
            for start in range(0, feature_count, 2048):
                rows = feature_records[start:start+2048]
                events = rows['event'].copy()
                indices = torch.as_tensor(events[:, :3], device=device)
                features = torch.as_tensor(rows['features'].copy(), device=device)
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
        for model in all_models.values():
            model.cpu()
        del zz, scalar
        torch.cuda.empty_cache()
        gpu_seconds = time.monotonic()-leased if device == 'cuda' else 0.
    (root/'events.partial').replace(root/'events.bin')
    del feature_records
    # This deterministic temporary geometry is reproducible from the immutable
    # bank. Keep its digest; persistent neural quantities retain all event IDs.
    feature_path.unlink()
    receipt = dict(status='measured', inputs=stamp, names=names, records=count,
        dtype='event:5*i8; value:M*f4', files={k: sha(root/k) for k in ['events.bin', 'scalars.npz']},
        calibration={n: specs[n]['calibration'] for n in names}, arms={n: specs[n]['recipe']['arm'] for n in names},
        sources={n: specs[n]['recipe']['source'] for n in names}, elapsed_seconds=time.monotonic()-begin,
        measured_gpu_lease_seconds=gpu_seconds, cpu_geometry_seconds=cpu_geometry_seconds,
        geometry_materialization_sha256=feature_sha256,
        CPU_candidate_geometry_outside_GPU_lease=True, complete_bank=True, no_annotation_reads=True)
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


def predict_many(row, graph, native, packages, destinations, root, monitor=None, wait=False, embedding_packages=None):
    bank, _ = prepare(row, graph, native, packages, root, monitor, wait, embedding_packages=embedding_packages)
    return solve(row, graph, bank, root, destinations, monitor)
