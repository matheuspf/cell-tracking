"""Independent continuation and closed-bank observation predictions, all 199 clips."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import RESULTS, WORK, inputs, read_json, sha, write_json
from .source_screen import job as prediction_job
from .resources import Monitor, cpu_budget


def run():
    freeze = read_json(RESULTS/'target_freeze.json')
    experiments = sorted({p['experiment'] for p in freeze['packages'] if p['arm'] in ['A10', 'O10_swap']})
    # Restoration is the same frozen head and a separate graph/count policy.
    experiments += [e.replace('O10_swap', 'O10_restore') for e in experiments if e.startswith('O10_swap') and not e.endswith('_replication')]
    done = []
    with Monitor(WORK/'resources/final-point-lanes.json') as monitor:
        for experiment in experiments:
            for row in inputs():
                source = '6bba' if row['embryo']=='44b6' else '44b6'
                original = experiment.replace('O10_restore', 'O10_swap')
                model = next(p for p in freeze['packages'] if p['experiment']==original and p['source']==source)
                package = Path(model['manifest_path']).parent
                arm = 'O10_restore' if experiment.startswith('O10_restore') else model['arm']
                root = WORK/'final_point_inference'/experiment/row['dataset']
                destination = root/f'{row["dataset"]}.npz'
                path = root/'job.json'
                job = prediction_job(row, source, package, destination, arm=arm)
                job['wait_for_lease'] = True
                write_json(path, job, immutable=True)
                start = time.monotonic()
                completed = root/'complete.json'
                if not completed.exists():
                    with (root/'predict.log').open('a') as log:
                        child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.prediction_entry', str(path)],
                                               stdout=log, stderr=subprocess.STDOUT)
                    if child.returncode:
                        failure = dict(status='failed', arm=experiment, dataset=row['dataset'], returncode=child.returncode,
                                       log_sha256=sha(root/'predict.log'))
                        write_json(root/'failure.json', failure)
                        done.append(failure)
                        print(f'Independent {experiment} prediction failed for {row["dataset"]}; continuing', flush=True)
                        continue
                guard = read_json(destination.with_suffix('.guard.json'))
                if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
                    raise RuntimeError('Point prediction label/cache guard failed')
                receipt = read_json(destination.with_suffix('.json'))
                if sha(destination) != receipt['prediction_sha256']:
                    raise ValueError('Point prediction bytes changed')
                out = WORK/'predictions'/experiment/destination.name
                out.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ['.npz', '.json', '.csv', '.guard.json', '.resources.json']:
                    if out.with_suffix(suffix).exists():
                        if sha(out.with_suffix(suffix)) != sha(destination.with_suffix(suffix)):
                            raise ValueError('Preserved final point output changed')
                    else:
                        shutil.copyfile(destination.with_suffix(suffix), out.with_suffix(suffix))
                result = dict(status='measured', arm=experiment, dataset=row['dataset'], source=source,
                    model_sha256=model['weights_sha256'], prediction_sha256=sha(out), seconds=time.monotonic()-start)
                if not completed.exists():
                    write_json(completed, result, immutable=True)
                done.append(result)
                write_json(WORK/'final_point_inference/progress.json', dict(jobs=done))
                monitor.check()
                print(f'Independent {experiment}: {row["dataset"]} complete', flush=True)
    result = dict(status='measured' if all(r['status']=='measured' for r in done) else 'failed', jobs=done)
    write_json(WORK/'final_point_inference/status.json', result)
    return result


if __name__ == '__main__':
    cpu_budget()
    raise SystemExit(1 if run()['status']=='failed' else 0)
