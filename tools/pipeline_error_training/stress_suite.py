"""One paired full-graph stress run on the six predeclared source representatives."""
from pathlib import Path
import subprocess
import sys

from .common import DATA, RESULTS, WORK, inputs, load_graph, read_json, sha, write_json
from .evaluate import summary, write_csv
from .source_screen import job as prediction_job


def run():
    freeze = read_json(RESULTS/'target_freeze.json')
    lock = read_json(RESULTS/'diagnostic_lock.json')
    from annotation_selection.metric_adapter import evaluate_graph
    from center_comparison.pipeline import read_gt
    scored, ordinary, jobs = [], [], []
    # All primary fitted representations/independent lanes and their controls;
    # replication concerns the unperturbed acceptance comparison.
    for package in [p for p in freeze['packages'] if p['seed'] == 20260915]:
        source, arm = package['source'], package['arm']
        selected = lock['representative_clips'][source]
        for row in [r for r in inputs() if r['dataset'] in selected]:
            root = WORK/'stress'/arm/source/row['dataset']
            output = root/f'{row["dataset"]}.npz'
            job = prediction_job(row, source, Path(package['manifest_path']).parent, output)
            job.update(stress_recipe=lock['stress'], stress_code_sha256=sha(Path(__file__).with_name('stress.py')),
                       wait_for_lease=True)
            path = root/'job.json'
            write_json(path, job, immutable=True)
            success = root/'complete.json'
            if not success.exists():
                with (root/'predict.log').open('a') as log:
                    child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.prediction_entry', str(path)],
                                           stdout=log, stderr=subprocess.STDOUT)
                if child.returncode:
                    jobs.append(dict(status='failed', arm=arm, source=source, dataset=row['dataset'],
                                     log_sha256=sha(root/'predict.log')))
                    continue
                with (root/'loss.log').open('a') as log:
                    loss = subprocess.run([sys.executable, '-m', 'pipeline_error_training.stress_loss', str(path)],
                                          stdout=log, stderr=subprocess.STDOUT)
                if loss.returncode:
                    jobs.append(dict(status='failed', arm=arm, source=source, dataset=row['dataset'],
                                     stage='supported_source_losses', log_sha256=sha(root/'loss.log')))
                    continue
            guard = read_json(output.with_suffix('.guard.json'))
            if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
                raise RuntimeError('Stress prediction guard failed')
            graph = load_graph(output)
            gn, ge = read_gt(DATA, row['dataset'], row['physical_scale'])
            score, _, _ = evaluate_graph(row['dataset'], graph['nodes'], graph['edges'], gn, ge, row['physical_scale'], row['estimated_total'])
            score.update(arm=arm, embryo=source, source=source, status='measured')
            screen = read_json(WORK/'source_screen'/arm/source/'20260915/summary.json')
            control = next(r for r in screen['per_clip'] if r['dataset']==row['dataset'])
            ordinary.append(dict(control, arm=arm, embryo=source, source=source, status='measured'))
            scored.append(score)
            record = dict(status='measured', arm=arm, source=source, dataset=row['dataset'],
                model_sha256=package['weights_sha256'], prediction_sha256=sha(output),
                supported_losses=read_json(output.with_suffix('.loss.json')), guard=guard)
            write_json(success, record, immutable=True)
            jobs.append(record)
            print(f'Fixed paired stress: {arm}/{source}/{row["dataset"]}', flush=True)
    table = []
    for arm in sorted({r['arm'] for r in scored}):
        for source in ['44b6', '6bba']:
            use = [r for r in scored if r['arm']==arm and r['embryo']==source]
            if not use:
                continue
            ids = {r['dataset'] for r in use}
            normal = [r for r in ordinary if r['arm']==arm and r['dataset'] in ids]
            measured, baseline = summary(use, arm, source), summary(normal, arm, source)
            table.append(dict(measured, source=source, ordinary_score=baseline['score'],
                paired_score_change=measured['score']-baseline['score'],
                complete_registered_representatives=len(use)==len(lock['representative_clips'][source]),
                scope='Source-calibration representatives; module perturbation; not out-of-embryo or independent validation'))
    if table:
        write_csv(RESULTS/'stress_scores.csv', table)
    result = dict(status='measured' if all(j['status']=='measured' for j in jobs) else 'failed',
        diagnostic_lock_sha256=sha(RESULTS/'diagnostic_lock.json'), jobs=jobs,
        model_or_threshold_selection_from_stress=False, no_repeated_stress_optimization=True)
    write_json(RESULTS/'stress_validation.json', result)
    return result


if __name__ == '__main__':
    from .resources import cpu_budget
    cpu_budget()
    raise SystemExit(1 if run()['status']=='failed' else 0)
