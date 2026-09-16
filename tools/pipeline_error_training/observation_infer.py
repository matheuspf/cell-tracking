"""One-for-one raw selection and a separate, count-scored restoration control."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .actions import fork_support
from .common import adjacency, digest, graph_hash, save_graph, sha, write_json, read_json
from .infer import embeddings
from .observations import CONFIG, ObservationBank, raw_graph
from .scoring import load_model


def construct_actions(bank, gain, restore, *, indexed_edges=True):
    """Complete observation/incident-edge replacements with stable raw provenance."""
    from strong_tracker_v3.decode import Action, BoundedActionComponents
    protected = fork_support(bank.nodes, bank.graph['edges'])
    collector = BoundedActionComponents()
    rejected = Counter()
    prefix = int(bank.nodes[:, 0].max())+1
    pair_index = {tuple(p): i for i, p in enumerate(bank.pairs)}
    seen = set()
    original_edges = set(map(tuple, bank.graph['edges']))
    from .observation_edges import ObservationEdges
    lookup = ObservationEdges(bank.graph['edges'],bank.raw['edges']) if indexed_edges else None
    for k, (old, raw) in enumerate(bank.pairs):
        state = 2 if restore else 1
        tube = bank.tube(int(old), int(raw))
        # Every time-anchor alternative uses the same closed correspondence bank.
        if any((i, j) not in pair_index and bank.exact_alias.get(j) != i for i, j in tube):
            rejected['tracklet_correspondence_outside_closed_bank'] += 1
            continue
        support = [pair_index[i, j] for i, j in tube if (i, j) in pair_index]
        value = float(np.mean(gain[support, state-1]))
        if value <= 1e-9:
            continue
        if any(i in protected for i, _ in tube):
            rejected['existing_fork_full_window'] += 1
            continue
        if restore:
            raw_ids = sorted({j for _, j in tube if j not in bank.exact_alias})
            if not raw_ids:
                continue
            removed_nodes = []
            new_nodes = [[prefix+j, *bank.raw_nodes[j, 1:]] for j in raw_ids]
            remove = set()
            add = (lookup.restored(raw_ids,prefix) if lookup is not None else
                   {(prefix+int(a), prefix+int(b)) for a, b in bank.raw['edges'] if int(a) in raw_ids and int(b) in raw_ids})
            # Fragment endpoints are explicit births/terminations. Attaching them
            # to an occupied incumbent endpoint requires the later closed native
            # association step to score the complete ownership consequences.
        else:
            if any(j in bank.exact_alias and bank.exact_alias[j] != i for i, j in tube):
                rejected['raw_proposal_owned_by_distinct_existing_observation'] += 1
                continue
            changed = [(i, j) for i, j in tube if not np.array_equal(bank.nodes[i, 1:], bank.raw_nodes[j, 1:])]
            if not changed:
                continue
            removed_nodes = [int(bank.nodes[i, 0]) for i, _ in changed]
            new_nodes = [[prefix+j, *bank.raw_nodes[j, 1:]] for _, j in changed]
            replacement = {int(bank.nodes[i, 0]): prefix+j for i, j in changed}
            remove = (lookup.removed(replacement) if lookup is not None else
                      {(int(a), int(b)) for a, b in original_edges if a in replacement or b in replacement})
            add = {(replacement.get(a, a), replacement.get(b, b)) for a, b in remove}
            raw_ids = [j for _, j in changed]
        key = digest([removed_nodes, new_nodes, sorted(remove), sorted(add)])
        if key in seen:
            continue
        seen.add(key)
        context = {i for i, _ in tube}
        context.update(j for i, _ in tube for j in [*bank.pred[i], *bank.succ[i]])
        resources = {('P0', int(bank.nodes[i, 0])) for i in context} | {('raw', int(j)) for j in raw_ids}
        descriptor = dict(removed_nodes=removed_nodes, new_nodes=new_nodes, raw_ids=raw_ids,
                          changed_node_cost=len(new_nodes), correspondence=tube,
                          kind='retain_both' if restore else 'substitute')
        collector.add(Action(k, value, remove, add, resources, [descriptor], descriptor['kind']))
    actions, streaming = collector.finish()
    return actions, dict(streaming, structural_rejections=dict(rejected))


def apply_observations(bank, actions, streaming, restore):
    from strong_tracker_v3.decode import legal_edges, solve_actions
    edge_cap = int(.02*len(bank.graph['edges']))
    chosen, solving = solve_actions(actions, max_changed_edges=edge_cap)
    cap = int(CONFIG['max_changed_node_fraction']*len(bank.nodes))
    remaining = cap
    accepted = []
    for k in sorted(chosen, key=lambda k: (-actions[k].value, actions[k].candidate)):
        cost = actions[k].owners[0]['changed_node_cost']
        if cost <= remaining:
            remaining -= cost
            accepted.append(k)
    nodes = {int(n[0]): n.copy() for n in bank.nodes}
    edges = set(map(tuple, bank.graph['edges']))
    ledger = []
    for k in accepted:
        action, detail = actions[k], actions[k].owners[0]
        if not action.remove <= edges:
            raise RuntimeError('Observation atomic ownership drift')
        for node in detail['removed_nodes']:
            del nodes[node]
        for node in detail['new_nodes']:
            if int(node[0]) in nodes:
                raise RuntimeError('A raw proposal was selected more than once')
            nodes[int(node[0])] = np.asarray(node, np.int64)
        edges.difference_update(action.remove)
        edges.update(action.add)
        ledger.append(dict(detail, value=action.value, removed_edges=sorted(action.remove),
                           added_edges=sorted(action.add), solver_status='optimal'))
    nn = bank.nodes.copy() if not accepted else np.asarray([nodes[k] for k in sorted(nodes)], np.int64)
    ee = bank.graph['edges'].copy() if not accepted else np.asarray(sorted(edges), np.int64).reshape(-1, 2)
    if not legal_edges(nn, set(map(tuple, ee))):
        raise RuntimeError('Observation selection created an invalid graph')
    if not restore and len(nn) != len(bank.nodes):
        raise RuntimeError('One-for-one observation selection changed the node count')
    if restore and len(nn)-len(bank.nodes) > cap:
        raise RuntimeError('Restoration exceeded its frozen node budget')
    changed_edges = len(set(map(tuple, bank.graph['edges'])) ^ set(map(tuple, ee)))
    if changed_edges>edge_cap:
        raise RuntimeError('Observation selection exceeded the frozen changed-edge budget')
    return dict(nodes=nn, edges=ee), dict(streaming, **solving, edits=ledger, changed_node_cap=cap,
        selected_node_cost=cap-remaining, accepted_observation_actions=len(accepted),
        node_cap_abstentions=len(chosen)-len(accepted), node_count_before=len(bank.nodes), node_count_after=len(nn),
        exact_one_for_one=not restore, stable_raw_ids=True,
        changed_edge_cap=edge_cap, observation_changed_edges=changed_edges)


def close_associations(graph, old_graph, native, p0_spec, changed_ids, *, remaining_edge_budget):
    from .association import decode
    nodes = graph['nodes']
    ix, pred, succ = adjacency(nodes, graph['edges'])
    seed = {ix[i] for i in changed_ids}
    mutable = set(seed)
    for a, b in native['pairs']:
        if a in seed or b in seed:
            mutable.update([int(a), int(b)])
            mutable.update(pred[int(b)])
            mutable.update(succ[int(a)])
    x, model = native['edge_features'], p0_spec['model']
    scores = x[:, 23]+model['beta'][0]+((x[:, p0_spec['columns']]-model['mean'])/model['scale'])@np.asarray(model['beta'][1:])
    constrained = dict(native, edge_features=x.copy())
    old = {(ix[int(a)], ix[int(b)]) for a, b in graph['edges']}
    for k, (a, b) in enumerate(native['pairs']):
        if a not in mutable or b not in mutable:
            if (a, b) in old:
                constrained['edge_features'][k, 25:29] = 1.
            else:
                scores[k] = -1e9
    edges, receipt = decode(nodes, graph['edges'], constrained, scores,
                            max_changed_edges=remaining_edge_budget)
    changed = old ^ {(ix[int(a)], ix[int(b)]) for a, b in edges}
    if any(a not in mutable or b not in mutable for a, b in changed):
        raise RuntimeError('Native observation association escaped its closed affected region')
    receipt['mutable_nodes'] = sorted(int(nodes[i, 0]) for i in mutable)
    return edges, receipt


def predict(row, graph, native, package, destinations, cache_root, p0_model, *, raw_override=None):
    from .native_refresh import refresh
    begin = time.monotonic()
    model, spec = load_model(package)
    raw = raw_graph(row) if raw_override is None else raw_override
    bank = ObservationBank(row, graph, raw)
    model.cuda()
    old_indices = set(map(int, bank.pairs[:, 0]))
    old_indices.update(j for i in list(old_indices) for j in [*bank.pred[i], *bank.succ[i]])
    raw_indices = set(map(int, bank.pairs[:, 1]))
    z, _ = embeddings(model, row, graph, bank, Path(cache_root)/'P0.npz', spec['weights_sha256'], indices=old_indices)
    rz, _ = embeddings(model, row, raw, SimpleNamespace(pred=bank.raw_pred, succ=bank.raw_succ),
        Path(cache_root)/'raw.npz', spec['weights_sha256'], indices=raw_indices)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(bank.pairs), 2048):
            pairs = bank.pairs[start:start+2048]
            context = np.stack([z[sorted({int(i), *bank.pred[i], *bank.succ[i]})].mean(0) for i, _ in pairs])
            evidence = np.stack([bank.evidence(int(i), int(j)) for i, j in pairs])
            logits = model.selection_scores(torch.as_tensor(z[pairs[:, 0]], device='cuda'),
                torch.as_tensor(rz[pairs[:, 1]], device='cuda'), torch.as_tensor(context, device='cuda'),
                torch.as_tensor(evidence, device='cuda')).cpu().numpy()
            calibrated = logits/spec['calibration']['temperature']+spec['calibration']['state_intercepts']
            probabilities.append(calibrated[:, 1:]-calibrated[:, :1])
    gain = np.concatenate(probabilities) if probabilities else np.empty((0, 2))
    model.cpu()
    torch.cuda.empty_cache()
    p0 = read_json(p0_model)
    receipts = {}
    for arm, destination in destinations.items():
        restore = arm == 'O10_restore'
        actions, streaming = construct_actions(bank, gain, restore)
        selected, ledger = apply_observations(bank, actions, streaming, restore)
        fresh = None
        if ledger['accepted_observation_actions']:
            from .cache_reuse import native_query
            reused = native_query(Path(cache_root)/'native_query_reference',Path(destination).parent/'fresh_native',
                                  selected['nodes'],row['metadata_sha256'])
            ledger['identical_native_query_reused'] = reused is not None
            fresh = refresh(row, selected, graph, native, Path(destination).parent/'fresh_native')
            changed_ids = {int(n[0]) for e in ledger['edits'] for n in e['new_nodes']}
            remaining = ledger['changed_edge_cap']-ledger['observation_changed_edges']
            selected['edges'], association = close_associations(selected, graph, fresh, p0, changed_ids,
                                                                remaining_edge_budget=remaining)
            ledger['fresh_association'] = association
        final_changed_edges = len(set(map(tuple, graph['edges'])) ^ set(map(tuple, selected['edges'])))
        if final_changed_edges>ledger['changed_edge_cap']:
            raise RuntimeError('Observation plus association exceeded the original P0 edge budget')
        ledger['final_changed_edges'] = final_changed_edges
        save_graph(destination, selected['nodes'], selected['edges'])
        receipt = dict(arm=arm, source=spec['recipe']['source'], model_sha256=spec['weights_sha256'],
            selector_identical_between_swap_and_restore=True, bank_sha256=bank.hash,
            prediction_sha256=sha(destination), graph_hash=graph_hash(selected['nodes'], selected['edges']),
            seconds=time.monotonic()-begin, ledger=ledger, coordinates_changed=bool(ledger['accepted_observation_actions']),
            native_features_refreshed=fresh is not None, p0_model_sha256=sha(p0_model),
            annotation_reads=0, unknown_raw_proposals_not_training_negatives=True)
        write_json(Path(destination).with_suffix('.json'), receipt)
        receipts[arm] = receipt
    return receipts
