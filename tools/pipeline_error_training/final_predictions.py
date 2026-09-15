"""Frozen all-199 operational prediction matrix, with one GPU and four CPU jobs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import RESULTS, WORK, inputs, read_json, sha, write_json
from .resources import Monitor, cpu_budget


def division_job(row, freeze):
    source = '6bba' if row['embryo'] == '44b6' else '44b6'
    selected = [p for p in freeze['packages'] if p['source'] == source and p['arm'].startswith('D')]
    root = WORK/'final_inference'/row['dataset']
    safe = {k: row[k] for k in ['dataset', 'image_path', 'image_shape', 'physical_scale', 'metadata_sha256']}
    packages = {p['experiment']: str(Path(p['manifest_path']).parent) for p in selected}
    deps = {}
    for p in selected:
        spec = read_json(p['manifest_path'])
        if 'architecture_dependency' in spec:
            dep = spec['architecture_dependency']; deps[dep['path']] = dep
    return dict(row=safe, source=source, root=str(root), packages=packages,
        package_sha256={p['experiment']: p['manifest_sha256'] for p in selected}, dependencies=list(deps.values()),
        graph_path=row['baselines']['P0']['path'], graph_sha256=row['baselines']['P0']['sha256'],
        evidence_path=row['evidence']['path'], evidence_sha256=row['evidence']['sha256'],
        destinations={n: str(root/n/f'{row["dataset"]}.npz') for n in packages},
        replacement_destinations={n: str(root/(n+'_replacement')/f'{row["dataset"]}.npz')
                                  for n in packages if n in freeze['replacement_arms']})


def one(row):
    freeze = read_json(RESULTS/'target_freeze.json')
    job = division_job(row, freeze)
    root = Path(job['root']); root.mkdir(parents=True, exist_ok=True)
    path = root/'job.json'
    write_json(path, job, immutable=True)
    start = time.monotonic()
    complete = (root/'complete.json').exists()
    if not complete:
        with (root/'predict.log').open('a') as log:
            child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.matrix_entry', str(path)],
                                   stdout=log, stderr=subprocess.STDOUT)
        if child.returncode:
            write_json(root/'failure.json', dict(status='failed', returncode=child.returncode,
                log_sha256=sha(root/'predict.log'), dataset=row['dataset']))
            return dict(status='failed', dataset=row['dataset'], returncode=child.returncode)
    guard = read_json(root/'guard.json')
    if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
        raise RuntimeError('Operational prediction guard failed')
    outputs = {}
    all_destinations = dict(job['destinations'], **{n+'_replacement': p for n, p in job.get('replacement_destinations', {}).items()})
    for arm, output in all_destinations.items():
        source = Path(output)
        receipt = read_json(source.with_suffix('.json'))
        if sha(source) != receipt['prediction_sha256']:
            raise ValueError('Serialized complete prediction hash mismatch')
        destination = WORK/'predictions'/arm/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ['.npz', '.json', '.csv']:
            if destination.with_suffix(suffix).exists():
                if sha(destination.with_suffix(suffix)) != sha(source.with_suffix(suffix)):
                    raise ValueError('Existing final prediction artifact drift')
            else:
                shutil.copyfile(source.with_suffix(suffix), destination.with_suffix(suffix))
        write_json(destination.with_suffix('.guard.json'), guard, immutable=True)
        outputs[arm] = dict(prediction_sha256=sha(destination), source=job['source'],
                           model_sha256=receipt.get('model_sha256', receipt.get('source_model_sha256')))
    result = dict(status='measured', dataset=row['dataset'], outputs=outputs, wall_seconds=time.monotonic()-start)
    if not complete:
        write_json(root/'complete.json', result, immutable=True)
    return result


def run():
    if not (RESULTS/'target_freeze.json').exists():
        raise PermissionError('Complete directional model and replication freeze is required')
    rows = []
    with Monitor(WORK/'resources/final-division-matrix.json') as monitor:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(one, inputs()):
                rows.append(result)
                write_json(WORK/'final_inference/progress.json', dict(completed=len(rows), total=len(inputs()), jobs=rows))
                monitor.check()
                print(f'Final frozen division predictions: {len(rows)}/{len(inputs())}; {result["status"]}', flush=True)
    result = dict(status='measured' if all(r['status']=='measured' for r in rows) else 'failed', jobs=rows)
    write_json(WORK/'final_inference/status.json', result)
    return result


if __name__ == '__main__':
    cpu_budget()
    raise SystemExit(1 if run()['status']=='failed' else 0)
