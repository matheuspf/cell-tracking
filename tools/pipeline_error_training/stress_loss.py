"""Source-owning CPU scorer for paired supported losses from stressed embeddings."""
import json
from pathlib import Path
import sys


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    from .guard import install
    guard = install(source=job['source'])
    from .resources import cpu_budget
    cpu_budget(4)
    import numpy as np
    import torch
    from scipy.special import logsumexp
    torch.set_num_threads(2)
    from .calibration import group_loss, targets
    from .common import WORK, load_graph, read_json, sha, write_json
    from .feasibility import from_record, records
    from .scoring import edge_features, load_model, score
    package, row = Path(job['package']), job['row']
    model, spec = load_model(package)
    if spec['recipe']['source'] != job['source']:
        raise PermissionError('Supported loss model direction mismatch')
    output = Path(job['destination'])
    cache = Path(read_json(output.with_suffix('.stress.json'))['cache_root'])
    arm, seed, source = spec['recipe']['arm'], spec['recipe']['seed'], job['source']
    if arm == 'O10_swap':
        from .observations import ObservationBank, raw_graph
        graph, raw = load_graph(job['graph_path']), raw_graph(row)
        bank = ObservationBank(row, graph, raw)
        z, rz = load_graph(cache/'P0.npz')['embedding'], load_graph(cache/'raw.npz')['embedding']
        data = load_graph(WORK/'observation_source'/source/f'{row["dataset"]}.npz')
        known = (data['labels'] >= 0) & (data['group'] >= 0)
        pairs, y, groups = data['pairs'][known], data['labels'][known], data['group'][known]
        logits = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 2048):
                pp = pairs[start:start+2048]
                context = np.stack([z[sorted({int(i), *bank.pred[i], *bank.succ[i]})].mean(0) for i, _ in pp])
                evidence = np.stack([bank.evidence(int(i), int(j)) for i,j in pp])
                logits.extend(model.selection_scores(torch.from_numpy(z[pp[:,0]]), torch.from_numpy(rz[pp[:,1]]),
                    torch.from_numpy(context), torch.from_numpy(evidence)).numpy())
        x = np.asarray(logits).reshape(-1, 3)
        normal = load_graph(WORK/'observation_calibration'/source/str(seed)/f'{row["dataset"]}.npz')['logits']
        ordinary = logsumexp(normal, axis=1)-normal[np.arange(len(y)), y]
        stressed = logsumexp(x, axis=1)-x[np.arange(len(y)), y]
        pair = lambda a: float(np.mean([a[groups==g].mean() for g in np.unique(groups)])) if len(y) else None
        result = dict(ordinary_grouped_selection_loss=pair(ordinary), stressed_grouped_selection_loss=pair(stressed), supported_rows=len(y))
    else:
        graph, native = load_graph(job['graph_path']), load_graph(job['evidence_path'])
        embedding = cache/spec['weights_sha256']/f'{row["dataset"]}.npz'
        if (cache/'prepared.json').exists():
            embedding = cache/'embeddings'/spec['weights_sha256']/f'{row["dataset"]}.npz'
        embedded = load_graph(embedding)
        z, valid = embedded['embedding'], embedded['valid']
        normal = load_graph(WORK/'calibration'/arm/source/str(seed)/f'{row["dataset"]}.npz')
        rr = list(records(source, row['dataset']))
        decisions = [from_record(r) for r in rr]
        values = []
        for start in range(0, len(rr), 1024):
            part = rr[start:start+1024]
            vv, _ = score(model, spec, graph['nodes'], native, z, valid, decisions[start:start+1024],
                          [r['event_features'] for r in part], row['physical_scale'])
            values.extend(vv)
        good, known = targets(rr)
        groups = np.asarray([r['group'] for r in rr])
        result = dict(ordinary_grouped_decision_loss=group_loss(normal['value'], good, known, groups),
                      stressed_grouped_decision_loss=group_loss(np.asarray(values), good, known, groups))
        data = load_graph(WORK/'source'/source/'pairs'/f'{row["dataset"]}.npz')
        pp, y, pg = native['pairs'][data['index']], data['labels'], data['group']
        xx = native['edge_features'][data['index']]
        normalized = np.clip((xx-spec['mean'])/spec['scale'], -10, 10).astype(np.float32)
        with torch.inference_mode():
            link = model.link_scores(torch.from_numpy(z), torch.from_numpy(pp), torch.from_numpy(normalized)).numpy()+xx[:,23]
        loss = np.logaddexp(0, link)-y*link
        before = np.logaddexp(0, normal['pair_score'])-y*normal['pair_score']
        mean = lambda a: float(np.mean([a[pg==g].mean() for g in np.unique(pg)])) if len(pg) else None
        result.update(ordinary_grouped_pair_nll=mean(before), stressed_grouped_pair_nll=mean(loss), supported_pairs=len(pg))
    if guard['blocked_reads'] or guard['blocked_network']:
        raise RuntimeError('Source-supported stress loss accessed forbidden data')
    write_json(output.with_suffix('.loss.json'), dict(status='measured', arm=arm, source=source, seed=seed,
        dataset=row['dataset'], model_sha256=spec['weights_sha256'], source_only=True, guard=guard,
        raw_model_losses_before_calibration=True, **result), immutable=True)


if __name__ == '__main__':
    main()
