"""Explicit operational prediction job; annotations denied before dependency loads."""
import argparse
import atexit
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    destination = Path(job['destination'])
    package = Path(job['package'])
    from .guard import install
    guard = install(fresh_root=destination.parent, allowed_models=[package, job['graph_path'], job['evidence_path'],
        *[d['path'] for d in job.get('dependencies', [])]],
                    images=[job['row']['image_path']])
    if not guard['installed_before_numerical']:
        raise RuntimeError('Operational inference did not install its label guard before imports')
    def record():
        path = destination.with_suffix('.guard.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(guard, cached_operational_inputs=True,
            fresh_end_to_end=False, explicit_source_model=job['source']))+'\n')
    atexit.register(record)
    from .resources import Lease, Monitor, cpu_budget
    cpu_budget()
    from .common import load_graph, read_json, sha
    for dependency in job.get('dependencies', []):
        if sha(dependency['path']) != dependency['sha256']:
            raise ValueError('Architecture dependency hash drift')
    for name in ['graph', 'evidence']:
        if sha(job[name+'_path']) != job[name+'_sha256']:
            raise ValueError('Explicit operational input hash drift')
    spec = read_json(package / 'frozen_package.json')
    if spec['recipe']['source'] != job['source'] or sha(package / 'frozen_package.json') != job['package_sha256']:
        raise PermissionError('Frozen explicit source model mismatch')
    graph, native = load_graph(job['graph_path']), load_graph(job['evidence_path'])
    cache_root = destination.parent/'current_image_cache'
    if job.get('stress_recipe'):
        from .stress import install as install_stress
        from .common import digest, write_json
        recipe = job['stress_recipe']
        if sha(Path(__file__).with_name('stress.py')) != job['stress_code_sha256']:
            raise ValueError('Frozen perturbation implementation changed')
        cache_root = destination.parent/('stress_cache_'+digest([recipe, job['stress_code_sha256']])[:16])
        write_json(cache_root/'namespace.json', dict(recipe=recipe, code_sha256=job['stress_code_sha256']), immutable=True)
        install_stress(recipe)
        write_json(destination.with_suffix('.stress.json'), dict(recipe=recipe, cache_root=str(cache_root),
            code_sha256=job['stress_code_sha256'], scope='Frozen module perturbation; upstream P0 points and evidence are fixed'), immutable=True)
    from .infer import predict
    from .serialization import export_csv
    with Monitor(destination.with_suffix('.resources.json')) as monitor:
        if job.get('operator') in ['complete_replacement', 'bounded_additive']:
            from .staged_infer import prepare
            bank, _ = prepare(job['row'], graph, native, {'model': package}, cache_root, monitor,
                              wait=job.get('wait_for_lease', False))
            if job['operator'] == 'complete_replacement':
                from .fork_reference import prepare as reference_prepare
                from .replacement import solve
                reference_prepare(bank, {'model': package}, cache_root, wait=job.get('wait_for_lease', False))
            else:
                from .staged_infer import solve
            solve(job['row'], graph, bank, cache_root, {'model': destination}, monitor)
        else:
            with Lease(required_gib=8., wait=job.get('wait_for_lease', False)):
                if job.get('operator') == 'observation':
                    from .observation_infer import predict as observation_predict
                    observation_predict(job['row'], graph, native, package, {job['arm']: destination}, cache_root, job['p0_model'])
                else:
                    predict(job['row'], graph, native, package, destination, cache_root=cache_root)
    result = load_graph(destination)
    export_csv(destination.with_suffix('.csv'), job['row']['dataset'], result['nodes'], result['edges'])


if __name__ == '__main__':
    main()
