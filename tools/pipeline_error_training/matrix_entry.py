"""Startup-guarded multi-model inference for one explicit complete image clip."""
import argparse
import atexit
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    root = Path(job['root'])
    allowed = [job['graph_path'], job['evidence_path'], *job['packages'].values(),
               *[d['path'] for d in job['dependencies']]]
    from .guard import install
    guard = install(fresh_root=root, allowed_models=allowed, images=[job['row']['image_path']])
    if not guard['installed_before_numerical']:
        raise RuntimeError('Matrix prediction guard must precede numerical imports')
    def record():
        root.mkdir(parents=True, exist_ok=True)
        (root/'guard.json').write_text(json.dumps(dict(guard, source_model=job['source'],
            cached_operational_inputs=True, fresh_end_to_end=False))+'\n')
    atexit.register(record)
    from .resources import Monitor, cpu_budget
    # All four workers inherit the same bounded 16-CPU mask. Restricting every
    # worker to its first four CPUs would concentrate them on the same cores.
    cpu_budget()
    import torch
    torch.set_num_threads(2)
    from .common import load_graph, read_json, sha
    for kind in ['graph', 'evidence']:
        if sha(job[kind+'_path']) != job[kind+'_sha256']:
            raise ValueError('Matrix input hash mismatch')
    for dep in job['dependencies']:
        if sha(dep['path']) != dep['sha256']:
            raise ValueError('Architecture dependency hash mismatch')
    for name, path in job['packages'].items():
        spec = read_json(Path(path)/'frozen_package.json')
        if spec['recipe']['source'] != job['source'] or sha(Path(path)/'frozen_package.json') != job['package_sha256'][name]:
            raise PermissionError('Explicit frozen source model mismatch')
    from .staged_infer import predict_many
    from .serialization import export_csv
    with Monitor(root/'resources.json') as monitor:
        predict_many(job['row'], load_graph(job['graph_path']), load_graph(job['evidence_path']),
            job['packages'], job['destinations'], root/'neural', monitor=monitor, wait=True)
        if job.get('replacement_destinations'):
            from .bank import EventBank
            from .fork_reference import prepare as reference_prepare
            from .replacement import solve as replacement_solve
            graph, native = load_graph(job['graph_path']), load_graph(job['evidence_path'])
            bank = EventBank(graph['nodes'], graph['edges'], native, job['row']['physical_scale'])
            reference_prepare(bank, job['packages'], root/'neural', wait=True)
            replacement_solve(job['row'], graph, bank, root/'neural', job['replacement_destinations'], monitor)
        for destination in [*job['destinations'].values(), *job.get('replacement_destinations', {}).values()]:
            graph = load_graph(destination)
            export_csv(Path(destination).with_suffix('.csv'), job['row']['dataset'], graph['nodes'], graph['edges'])
            monitor.check()


if __name__ == '__main__':
    main()
