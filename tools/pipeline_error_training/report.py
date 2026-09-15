"""Export concise, rebuildable evidence from measured study receipts only."""
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess

from .common import RESULTS, STUDY, WORK, now, read_json, sha, write_json
from .evaluate import write_csv
from .freeze import PRIMARY


def optional(path):
    return read_json(path) if Path(path).exists() else None


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def fit_rows():
    result = []
    for arm in [*PRIMARY, 'D10_random']:
        for seed in [20260915, 314159]:
            for source, target in [('44b6','6bba'), ('6bba','44b6')]:
                root = WORK/'training'/arm/source/str(seed)
                package = optional(root/'package.json')
                frozen = optional(root/'frozen_package.json')
                progress = optional(root/'progress.json')
                history = [json.loads(line) for line in (root/'history.jsonl').read_text().splitlines() if line] if (root/'history.jsonl').exists() else []
                if package:
                    if sha(root/'model.pt') != package['weights_sha256']:
                        raise ValueError('Trained package changed while exporting results')
                    status, reason = 'measured', None
                elif progress:
                    status, reason = 'not run', 'Incomplete directional training; see execution phase and continuation state'
                else:
                    status, reason = 'not run', 'Conditional arm, replication not nominated, or directional job not reached'
                recipe = optional(root/'recipe.json')
                result.append(dict(arm=arm, source=source, target=target, seed=seed, status=status, reason=reason,
                    execution_phase=progress.get('status') if progress else 'not started',
                    completed_updates=package['completed_updates'] if package else (progress or {}).get('step'),
                    locked_updates=recipe.get('updates') if recipe else None,
                    unique_optimizer_steps=len({r['step'] for r in history}),
                    measured_update_lease_seconds=sum(r['seconds'] for r in history) if history else None,
                    full_fit_wall_seconds=package.get('seconds') if package else None,
                    first_recorded_loss=history[0]['total'] if history else None,
                    final_recorded_loss=history[-1]['total'] if history else None,
                    model_sha256=package['weights_sha256'] if package else None,
                    recipe_sha256=sha(root/'recipe.json') if recipe else None,
                    frozen_package_sha256=sha(root/'frozen_package.json') if frozen else None,
                    calibrated=bool(frozen), calibration_sha256=sha(root/'calibration.json') if frozen else None,
                    source_groups_visited=package.get('source_groups_visited', len(package.get('group_visits', {}))) if package else None,
                    claim='Source-only head; inherited upstream exposure'))
    return result


def source_rows():
    result = []
    for path in sorted((WORK/'source_screen').glob('*/*/*/summary.json')):
        receipt = read_json(path)
        row = dict(receipt['measured'], source=receipt['source'], seed=receipt['seed'],
            control_score=receipt['baseline']['score'], source_delta=receipt['delta'],
            source_checkpoint_sha256=receipt.get('model_sha256'),
            full_source_clip_replay=True, independent_source_validation=False,
            source_only=True, target_comparison=False)
        calibration = optional(WORK/'training'/receipt['arm'].removesuffix('_replacement').replace('O10_restore','O10_swap')/
                               receipt['source']/str(receipt['seed'])/'calibration.json')
        if calibration:
            diagnostic = calibration.get('source_grouped_decision', {})
            row['raw_source_grouped_decision_loss'] = diagnostic.get('before', {}).get('loss')
            row['calibrated_source_grouped_decision_loss'] = diagnostic.get('after', {}).get('loss')
            row['calibration_temperature'] = calibration.get('calibration', {}).get('temperature')
        result.append(row)
    return result


def matched_training_controls():
    """Check observed common prefixes; this does not claim later model equality."""
    training = read_json(RESULTS/'execution_repair_lock.json')['training']
    pairs = [('D10_frozen','D10_adapted',max(1,training['updates']//8)),
             ('D20_temporal','A10',training['pretrain_updates']),
             ('D20_temporal','O10_swap',training['pretrain_updates'])]
    records = []
    for left, right, count in pairs:
        for source in ['44b6','6bba']:
            paths = [WORK/'training'/arm/source/'20260915/history.jsonl' for arm in [left,right]]
            histories = [[json.loads(line) for line in p.read_text().splitlines() if line]
                         if p.exists() else [] for p in paths]
            if any(len(h)<count for h in histories):
                records.append(dict(left=left,right=right,source=source,status='not run',
                    reason='Both declared common training prefixes have not completed'))
                continue
            fields = ['total','identity','decision','metric_risk','contrastive','consistency']
            delta = {k:max(abs(a[k]-b[k]) for a,b in zip(histories[0][:count],histories[1][:count])) for k in fields}
            records.append(dict(left=left,right=right,source=source,status='measured',
                common_updates=count,max_abs_recorded_loss_difference=delta,
                exact_recorded_losses=all(v==0 for v in delta.values()),
                history_sha256=[sha(p) for p in paths],
                scope='Observed losses while training objectives and trainable blocks are identical; no later weight-equality or advantage claim'))
    write_json(RESULTS/'matched_training_controls.json',dict(records=records,new_target_metrics_read=False))
    return records


def matrix(fits):
    frozen = optional(RESULTS/'target_freeze.json') or {}
    nomination = optional(RESULTS/'nomination.json') or {}
    replication = optional(RESULTS/'replication.json') or {}
    experiments = [r['id'] for r in read_json(STUDY)['experiments']]
    experiments += [a for a in frozen.get('experiments', []) if a not in experiments]
    result = []
    for arm in experiments:
        score = optional(WORK/'full_evaluation'/arm/'summary.json')
        if score:
            pooled = next(g for g in score['groups'] if g['embryo']=='pooled')
            result.append(dict(experiment=arm, status='measured', score=pooled['score'], clips=pooled['clips'], reason=None))
            continue
        dependencies = {'S00':'baseline_validation.json', 'S10':'source_feasibility.json',
                        'R10':'replication.json', 'C10':'composition.json', 'V10':'validation.json'}
        if arm in dependencies:
            receipt = optional(RESULTS/dependencies[arm])
            status = receipt.get('status', 'not run') if receipt else 'not run'
            if status not in ['measured','failed','blocked','not run']:
                status = 'not run'
            reason = None if status=='measured' else (receipt or {}).get('reason', 'Required execution or validation is incomplete')
        else:
            base = arm.removesuffix('_replication').removesuffix('_replacement')
            base = 'O10_swap' if base=='O10_restore' else base
            seed = 314159 if arm.endswith('_replication') else 20260915
            picked = [r for r in fits if r['arm']==base and r['seed']==seed]
            trained = sum(r['status']=='measured' for r in picked)
            status, reason = 'not run', f'{trained}/2 directional fits complete; all-199 target score unavailable'
            if arm=='D10_random' and replication and not replication.get('conditional_random_control'):
                reason = 'Organoid family did not qualify in both source directions; conditional control was not authorized'
            if arm=='C10' and nomination and not (nomination.get('division_nominee') and nomination.get('identity_nominee')):
                reason = 'Both independent component families did not source-qualify'
            failures=[]
            for q in ['queue','observation_queue']:
                jobs=(optional(WORK/q/'progress.json') or {}).get('jobs',[])
                failures.extend(j for j in jobs if base in j.get('job','') and j.get('status') in ['failed','blocked'])
            # Old failed attempts remain in their receipts. A successful repair
            # and completed package supersede them for directional availability.
            if failures and not all(r['calibrated'] for r in picked):
                status='blocked' if any(j['status']=='blocked' for j in failures) else 'failed'
                reason='Directional execution failed; independent lanes continued; exact logs are in the ignored work root'
            prediction_jobs = (optional(WORK/('final_inference' if base.startswith('D') else
                'final_point_inference')/'status.json') or {}).get('jobs', [])
            prediction_failures = [j for j in prediction_jobs if j['status'] in ['failed', 'blocked']
                and (base.startswith('D') or j.get('arm')==arm)]
            scorer_jobs = (optional(WORK/'finish_queue/progress.json') or {}).get('jobs', [])
            scorer_failures = [j for j in scorer_jobs if j['job']=='score-'+arm and j['status']=='failed']
            if prediction_failures or scorer_failures:
                status = 'blocked' if prediction_failures and all(j['status']=='blocked' for j in prediction_failures) else 'failed'
                reason = 'Complete prediction or official scoring failed; partial clips do not receive a pooled score'
        result.append(dict(experiment=arm,status=status,score=None,clips=None,reason=reason))
    return result


def recommendation():
    nomination = optional(RESULTS/'nomination.json') or {}
    candidates = [a for a in [nomination.get('division_nominee'),nomination.get('identity_nominee')] if a]
    if len(candidates)==2:
        candidates.append('C10')
    baseline=read_json(RESULTS/'baseline_validation.json')
    controls={r['embryo']:r['score'] for r in [next(r for r in baseline['pooled'] if r['arm']=='P0'),
        *[r for r in baseline['embryos'] if r['arm']=='P0']]}
    c4=next(r for r in baseline['pooled'] if r['arm']=='C4_m6')['score']
    validation=optional(RESULTS/'validation.json') or {}
    assessed=[]
    for arm in candidates:
        seed_records=[]
        for suffix in ['', '_replication']:
            receipt=optional(WORK/'full_evaluation'/(arm+suffix)/'summary.json')
            if receipt:
                delta={g['embryo']:g['score']-controls[g['embryo']] for g in receipt['groups']}
                pooled=next(g['score'] for g in receipt['groups'] if g['embryo']=='pooled')
                seed_records.append(dict(seed=20260915 if not suffix else 314159,score=pooled,delta=delta,
                    passes=delta['pooled']>0 and delta['44b6']>=0 and delta['6bba']>=0))
        metric_pass=len(seed_records)==2 and all(r['passes'] for r in seed_records) and seed_records[0]['score']>c4
        fresh=optional(RESULTS/f'fresh_candidate_{arm}.json') or {}
        correctness=validation.get('hard_correctness_pass',False) and fresh.get('status')=='measured'
        assessed.append(dict(candidate=arm,seed_results=seed_records,metric_gate_pass=metric_pass,
            candidate_fresh_validation=fresh.get('status','not run'),hard_correctness_pass=correctness,
            recommended=metric_pass and correctness))
    passing=[r for r in assessed if r['recommended']]
    selected=max(passing,key=lambda r:r['seed_results'][0]['score'])['candidate'] if passing else None
    return dict(candidate=selected,status='Candidate recommended' if selected else 'P0 retained',
        assessed=assessed,production_default_changed=False,clean_end_to_end_transfer='unestablished',
        leaderboard_improvement_claim=False,matched_control_second_seed_advantage_claim=False)


def resources():
    from .budget import report as budget_report
    records=[]
    for path in WORK.rglob('*.json'):
        if 'resource' not in path.name and path.parent.name!='resources':
            continue
        receipt=optional(path)
        if not isinstance(receipt,dict) or 'peak_total_gpu_gib' not in receipt:
            continue
        records.append(dict(path=str(path.relative_to(WORK)),sha256=sha(path),**{k:receipt[k] for k in [
            'wall_seconds','peak_total_gpu_gib','peak_tree_rss_gib','peak_all_study_process_rss_gib','peak_os_threads','resource_error'] if k in receipt}))
    result=dict(status='measured',gpu_budget=budget_report(),measurements=len(records),
        peak_total_gpu_gib=max((r['peak_total_gpu_gib'] for r in records),default=None),
        peak_process_tree_rss_gib=max((r['peak_tree_rss_gib'] for r in records),default=None),
        peak_all_study_process_rss_gib=max((r.get('peak_all_study_process_rss_gib',r['peak_tree_rss_gib']) for r in records),default=None),
        resource_errors=[r for r in records if r.get('resource_error')],
        cpu_contract='Process affinity restricted to 16 logical CPUs; BLAS threads 1, data loaders 0, scorer processes 4. Initial Zarr pools had excess idle OS threads; corrected nested configuration is retained with the original measurements.',
        total_device_measurement_includes_other_programs=True,records=records,
        historical_or_other_studies_restarted=False,Kaggle_runtime_measured=False)
    write_json(RESULTS/'resource.json',result)
    return result


def pad_metric_tables(experiments):
    """A missing full graph result is a null row with its execution reason."""
    wanted={r['experiment']:r for r in experiments if r['experiment'].startswith(('D','A','O','C10'))}
    for filename,embryos in [('scores.csv',['pooled']),('per_embryo_scores.csv',['44b6','6bba'])]:
        measured=[r for r in csv_rows(RESULTS/filename) if r.get('status')=='measured']
        fields=list(dict.fromkeys(k for r in measured for k in r))
        present={(r['arm'],r['embryo']) for r in measured}
        for arm,experiment in wanted.items():
            for embryo in embryos:
                if (arm,embryo) in present:continue
                measured.append(dict({k:None for k in fields},arm=arm,embryo=embryo,
                    status=experiment['status'] if experiment['status']!='measured' else 'not run',
                    reason=experiment['reason'] or 'Complete all-199 metric unavailable',
                    seed=314159 if arm.endswith('_replication') else 20260915))
        write_csv(RESULTS/filename,measured)


def continuation(status):
    pending=[r for r in status['experiments'] if r['status']!='measured']
    lines=['# Continuation — pipeline error training', '',
        'Read REPORT.md and STATUS.json first. This file is generated from the current execution receipts.', '',
        '## Current state', '',
        f"State: {status['status']}. Production default: P0. No new target fitting or threshold selection is authorized.", '',
        'Inspect actual processes before launching any resumable queue; active.json can refer to a completed child.',
        'Do not kill or resume a different study. Historical native 400-epoch fits remain untouched.', '',
        '```sh', "ps -eo pid,ppid,etime,rss,args | rg 'pipeline_error_training|train_native'",
        'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits', '```', '',
        '## Runtime and exact commands', '',
        'Run from the repository root. Do not start a duplicate queue while its process is alive.', '',
        '```sh', 'export PYTHONNOUSERSITE=1', 'export PYTHONPATH=tools:.',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1',
        'STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python',
        '$STUDY_PY -m pipeline_error_training queue',
        '$STUDY_PY -m pipeline_error_training.observation_queue',
        '$STUDY_PY -m pipeline_error_training.finish_queue',
        '$STUDY_PY -m pipeline_error_training.report', '```', '',
        'The post-primary queue requires both primary progress files to say complete. It performs source-only',
        'qualification and replication before the target freeze. Every failed independent prediction lane is retained.',
        'If a frozen stage fails, preserve its files under the new invalid/ root, record a named implementation repair,',
        'and rerun the same stage with unchanged scientific weights, calibration, bank and decision policy.', '',
        '## Pending or unsuccessful registered experiments', '']
    lines.extend(f"- {r['experiment']}: {r['status']} — {r['reason']}" for r in pending)
    if not pending:lines.append('All registered experiments have complete measured receipts.')
    lines += ['', '## Artifact locations and restrictions', '',
        '- Resumable weights, optimizer/RNG state, full graphs, source labels, image embeddings and logs: work/pipeline-error-training-20260915/.',
        '- Queue state: queue/progress.json, observation_queue/progress.json, finish_queue/progress.json under that root.',
        '- GPU accounting: gpu_budget/ledger.json; limits are 20 GiB total device memory, 50 GiB study RSS, and the frozen 4/24/12/8 lease-hour reservations.',
        '- Concrete validation failures: fresh_image_validation.json and native_refresh_validation.json, with log hashes; original failed attempts stay under invalid/.',
        '- Missing clean upstream fits block a clean end-to-end transfer claim; they do not block the independent operational lanes.',
        '- Do not change the production default, merge, upload to Kaggle or publish weights.', '']
    (RESULTS/'CONTINUATION.md').write_text('\n'.join(lines))


def run(complete=False):
    from .clean_readiness import run as check_clean_upstream
    clean = check_clean_upstream()
    fits=fit_rows();source=source_rows()
    write_csv(RESULTS/'training_fits.csv',fits)
    if source:write_csv(RESULTS/'source_model_scores.csv',source)
    lock=read_json(RESULTS/'execution_repair_lock.json')
    training=dict(status='measured' if any(r['status']=='measured' for r in fits) else 'not run',
        matched_primary_updates=lock['training']['updates'],effective_batch_groups=32,
        source_checkpoint_selection='Fixed final checkpoint after the common source-only throughput lock',
        fits=fits,source_scores=source,matched_common_prefixes=matched_training_controls(),
        source_independence_certified=False,clean_end_to_end_out_of_fold=False,
        target_labels_used_for_new_fitting_or_calibration=False,
        resumption='Exact model, optimizer, RNG and source-group visit state; recipe hash checked before resuming',
        repairs_preserved=[str(p.relative_to(WORK)) for p in sorted((WORK/'invalid').iterdir())])
    write_json(RESULTS/'training_summary.json',training)
    table=matrix(fits);write_csv(RESULTS/'experiment_matrix.csv',table)
    pad_metric_tables(table)
    resource=resources();recommended=recommendation()
    write_json(RESULTS/'recommendation.json',recommended)
    queues={q:(optional(WORK/q/'progress.json') or {}).get('status','not started') for q in ['queue','observation_queue']}
    finished=optional(WORK/'finish_queue/progress.json') or {}
    if complete and (any(v!='complete' for v in queues.values()) or finished.get('status')!='complete'):
        raise RuntimeError('Cannot mark this study complete while its execution queues are unfinished')
    status=dict(created=now(),status='executing' if any(v=='running' for v in queues.values()) else 'awaiting_final_validation',
        experiments=table,queues=queues,recommendation=recommended['candidate'],production_default='P0',
        clean_transfer=dict(status='blocked',reason=clean['reason'],
            readiness_sha256=sha(RESULTS/'clean_upstream_readiness.json')),
        uploads=False,weights_published=False,merged=False)
    if complete:status['status']='complete'
    write_json(RESULTS/'STATUS.json',status)
    continuation(status)
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','zarr','tracksdata','scikit-image','scikit-learn','pandas','polars','pytest','blosc2','pydantic','joblib']}
    extra={d.metadata['Name']:d.version for d in importlib.metadata.distributions(
        path=['/kaggle/envs/detector-screen-organoid/lib/python3.12/site-packages'])
        if d.metadata['Name'].lower() in ['keras','h5py','namex','optree']}
    write_json(RESULTS/'environment.json',dict(python=subprocess.check_output([__import__('sys').executable,'--version'],text=True).strip(),
        packages=versions,organoid_overlay_packages=extra,python_no_user_site=True,implementation_runtime='cell-tracking-annotation-selection-v1 isolated environment',
        inherited_notebook_dependencies='cell-tracking-notebooks',organoid_additional_environment='detector-screen-organoid',
        official_scorer_revision='075fc5f5a52d11077f9dc2b074644618f26939e2',runtime_installations_by_this_study=False))
    measured=sum(r['status']=='measured' for r in fits)
    report=f'''{recommended['status']}

# Pipeline error training — September 15, 2026

Execution status: **{status['status']}**. {measured} directional fits have completed the locked {lock['training']['updates']} updates.
The adopted production default remains P0. The recommendation is recorded separately in recommendation.json.

## Exact controls and evidence

All 199 complete clips reproduce P0 **0.934864986413134**, C4_m6 **0.935178370257** and C0 **0.934802374260586**.
Each control has 4,108,943 nodes. The zero-head D00 reproduces P0 exactly.
Both embryos and all recovered/lost TP and removed/introduced FP counts are reported; error families overlap.

- [Experiment status](experiment_matrix.csv)
- [Pooled full metrics](scores.csv) and [each embryo](per_embryo_scores.csv)
- [Every clip](per_clip_scores.csv) and [changed errors](error_transitions.csv)
- [Directional fits](training_fits.csv) and [source-only full-clip screens](source_model_scores.csv)
- [Source feasibility witnesses](source_feasibility_scores.csv)

## Interpretation

These modules use source-only direct fitting and calibration on the exposed P0 proposal pipeline.
Inherited primary/secondary checkpoints and E teachers prevent a clean end-to-end out-of-fold claim.
Exact overlapping frames were unioned before deterministic whole-clip source partitioning, but missing acquisition offsets
prevent independent inner-validation certification. Source calibration and family nomination are exploratory.
Stress diagnostics describe perturbations of these same clips; they do not supply independent biological validation.

Organoid last-block adaptation competes with its frozen backbone control. The native temporal head competes with
the compact representation and the same temporal architecture without identity pretraining. Continuation-only and
closed-bank observation selection run independently. Complete decisions include continuation, birth and other-parent alternatives.

## Execution and limits

The common source-only throughput lock reduced the proposed 16,000-update ceiling to {lock['training']['updates']} updates,
with equal budgets across matched fits. Models use the fixed final checkpoint, source-only regularized calibration,
and no target threshold selection. Failed implementation attempts remain under the new ignored invalid/ directory.
Cumulative charged GPU lease time is {resource['gpu_budget']['total_hours']:.3f} hours of 48; detailed memory/runtime evidence is in resource.json.
No measured Kaggle 12-hour runtime or leaderboard improvement is claimed. No weights were published or defaults changed.

See [CONTINUATION.md](CONTINUATION.md) for exact commands and remaining work.
'''
    (RESULTS/'REPORT.md').write_text(report)
    return status


if __name__=='__main__':
    run('--complete' in __import__('sys').argv)
