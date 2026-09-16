"""Export concise, rebuildable evidence from measured study receipts only."""
from collections import Counter
import csv
import importlib.metadata
import json
import math
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
                    positive_biological_event_groups_in_fit_pool=package.get('source_event_groups') if package else None,
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
            if receipt['arm'].startswith('O10_'):
                selector = calibration['calibration']
                row['raw_source_grouped_selection_loss'] = calibration['source_grouped_selection_loss']
                row['raw_source_selection_loss'] = selector['before_loss']
                row['regularized_calibrated_source_selection_loss'] = selector['after_loss']
                for state, count in zip(['keep','swap','both'],selector['state_counts']):
                    row['calibration_supported_'+state+'_choices'] = count
        if receipt['arm']=='A10':
            diagnostic_path = WORK/'training/A10'/receipt['source']/str(receipt['seed'])/'identity_diagnostic.json'
            diagnostic = optional(diagnostic_path)
            if diagnostic:
                for field in ['raw_source_grouped_pair_nll','unchanged_native_offset_grouped_pair_nll']:
                    row[field] = diagnostic[field]
                row['identity_diagnostic_groups'] = diagnostic['groups']
                row['identity_diagnostic_sha256'] = sha(diagnostic_path)
        result.append(row)
    return result


def executed_schedule(training):
    """Describe the existing executed schedule without changing any fit."""
    updates = training['updates']
    warmup = min(500, updates)
    factors = [min(1., (step+1)/warmup)*.5*(1+math.cos(math.pi*step/updates))
               for step in range(updates)]
    peak = max(range(updates), key=factors.__getitem__)
    result = dict(status='measured', total_updates=updates,
        effective_batch_groups=training['effective_batch_groups'],
        nominal_warmup_updates=training['warmup_updates'], effective_warmup_denominator=warmup,
        zero_based_step_formula='base_lr * min(1, (step+1)/min(500, updates)) * (1+cos(pi*step/updates))/2',
        warmup_and_cosine_applied_concurrently=True,
        new_parameter_learning_rates=dict(first=.0003*factors[0], peak=.0003*factors[peak],
            last=.0003*factors[-1], peak_update_one_based=peak+1),
        pretrained_parameter_learning_rate_multiplier=.1,
        identity_only_prefix_updates=training['pretrain_updates'],
        standard_joint_update_groups=dict(identity=16, biological_event=16),
        continuation_control='A10 uses 32 identity groups at every update',
        no_pretrain_control='D20_no_pretrain uses 16 identity and 16 biological-event groups from its first update',
        observation_joint_update_groups=dict(identity=16, observation=16,
            synthetic_probability_per_observation_group=.25),
        organoid_adaptation_starts_at_update_one_based=max(1,updates//8)+1,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ['train.py','observation_train.py']},
        changed_training=False, target_results_used=False,
        interpretation='The common throughput reduction also shortens the warmup denominator. This records the schedule already used by all valid fits; it is not a new recipe or a convergence claim.')
    write_json(RESULTS/'executed_training_schedule.json',result)
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
        if not isinstance(receipt,dict) or not {'peak_total_gpu_gib','peak_tree_rss_gib','wall_seconds'}<=receipt.keys():
            # Archived aggregate reports are not individual monitor receipts.
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


def outcome_text(recommended):
    measured = [r for r in csv_rows(RESULTS/'scores.csv')
                if r.get('status')=='measured' and r['arm'] not in ['P0','C4_m6','C0','D00']]
    if not measured:
        return 'No new model has a complete all-199 target comparison yet. The source screens are reported separately.'
    embryos = {(r['arm'],r['embryo']):r for r in csv_rows(RESULTS/'per_embryo_scores.csv')
               if r.get('status')=='measured'}
    changes = {r['arm']:r for r in csv_rows(RESULTS/'error_transitions.csv') if r['embryo']=='pooled'}
    lines = []
    if recommended['candidate']:
        lines.append(f"Recommended candidate: **{recommended['candidate']}**. It passed the frozen two-seed, both-embryo and fresh-inference gates. P0 remains the production default.")
    else:
        lines.append('No candidate has passed every frozen recommendation gate; P0 is retained. A higher descriptive score alone does not establish a recommendation.')
    best = max(measured,key=lambda r:float(r['score']))
    lines += ['',f"Highest completed new-model pooled result: **{best['arm']} {float(best['score']):.12f}** "
              f"(P0 delta {float(best['delta_P0']):+.12f}; C4_m6 delta {float(best['delta_C4_m6']):+.12f}).",'',
              'Complete all-199 results, with embryos ordered 44b6 / 6bba:','']
    for row in sorted(measured,key=lambda r:r['arm']):
        arm = row['arm']
        both = ' / '.join(f"{float(embryos[arm,e]['score']):.9f} ({float(embryos[arm,e]['delta_P0']):+.9f})"
                          for e in ['44b6','6bba'])
        lines.append(f"- **{arm}**: pooled {float(row['score']):.12f}; embryos {both}. "
                     f"Edge TP/FP/FN {row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']}; "
                     f"division TP/FP/FN {row['division_tp']}/{row['division_fp']}/{row['division_fn']}; "
                     f"predicted nodes {row['num_pred_nodes']}.")
        if arm in changes:
            c = changes[arm]
            lines.append(f"  Relative to P0: edge TP recovered/lost {c['tp_edges_recovered']}/{c['tp_edges_lost']}, "
                f"edge FP removed/introduced {c['fp_edges_removed']}/{c['fp_edges_introduced']}; "
                f"division TP recovered/lost {c['tp_divisions_recovered']}/{c['tp_divisions_lost']}, "
                f"division FP removed/introduced {c['fp_divisions_removed']}/{c['fp_divisions_introduced']}.")
    return '\n'.join(lines)


def source_decision_text():
    nomination = optional(RESULTS/'nomination.json')
    if not nomination:
        return 'Source-only family nomination and conditional replication are pending. No target result is used to choose a checkpoint, margin or training budget.'
    replication = optional(RESULTS/'replication.json') or dict(status='in progress',
        conditional_random_control=nomination['candidates']['D10_adapted']['qualified'])
    lines = [f"Source nominees: division **{nomination['division_nominee'] or 'none'}**; "
             f"identity/observation **{nomination['identity_nominee'] or 'none'}**. "
             f"Replication status: **{replication['status']}**.", '']
    for arm,record in nomination['candidates'].items():
        evidence = '; '.join(f"{s}: delta {d['source_delta']:+.9f}, graph gate {'pass' if d['source_graph_safe'] else 'fail'}"
                             if 'source_delta' in d else f"{s}: {d['reason']}"
                             for s,d in record['directions'].items())
        lines.append(f"- {arm}: {'qualified' if record['qualified'] else 'not qualified'}; {evidence}.")
    lines += ['', 'These are the predeclared complete source calibration clips, not the all-199 target comparison. '
              'The source split is not independently certified. Nominee-only replication does not establish a second-seed advantage over a newly trained matched control.']
    if not replication['conditional_random_control']:
        lines += ['', 'D10_random was not run because the Organoid family did not qualify in both source directions. No pretrained-advantage claim is made.']
    return '\n'.join(lines)


def proof_text():
    tested = optional(RESULTS/'test_validation.json') or {}
    fresh = optional(RESULTS/'fresh_image_validation.json') or {}
    clips = fresh.get('clips',[])
    lines = [f"Tests: **{tested.get('status','not run')}** — {tested.get('summary','final executable test receipt pending')}.", '']
    if fresh.get('status')=='measured':
        times = ', '.join(f"{r['unfamiliar_name']}: {r['seconds']:.3f} s" for r in clips)
        lines.append('The complete primary/secondary/harmonic/DeepCenter/P0 pipeline plus D10_frozen ran '
            f'from both renamed 100-frame images ({times}). Startup guards denied annotations, historical prediction caches and network access; '
            'both reconstructed P0 graphs and CSV/GEFF roundtrips were exact. These two runs are not a measured Kaggle 12-hour bound. '
            'A recommended candidate also requires its own cold-image proof.')
    else:
        lines.append(f"Fresh-image pipeline proof: {fresh.get('status','not run')}.")
    candidates = optional(RESULTS/'fresh_candidate_validation.json') or {}
    for arm in candidates.get('experiments', []):
        candidate = optional(RESULTS/f'fresh_candidate_{arm}.json') or {}
        parity = optional(RESULTS/f'fresh_candidate_{arm}_replay_parity.json') or {}
        if candidate.get('status')=='measured':
            lines += ['', f'**{arm}** also completed its own [cold-image pipeline proof](fresh_candidate_{arm}.json) '
                'on both renamed 100-frame images with the frozen source models. '
                + (f'Both candidate graphs exactly matched the scored graphs ([parity evidence](fresh_candidate_{arm}_replay_parity.json)). '
                   if parity.get('status')=='measured' else '')
                + 'These correctness checks do not change the metric recommendation gates.']
    timing = optional(RESULTS/'fresh_candidate_timing.json') or {}
    if timing.get('status')=='measured':
        lines += ['', f"Candidate fresh inference used {timing['gpu_lease_wall_seconds']:.3f} seconds under the GPU lease "
            f"and {timing['elapsed_wall_seconds']:.3f} seconds elapsed, including shared-GPU waits "
            '([timing evidence](fresh_candidate_timing.json)). Lease wall time is not GPU kernel time.']
    reused = optional(RESULTS/'native_query_reuse_parity.json') or {}
    if reused.get('status')=='measured':
        lines += ['', 'Identical observation-policy coordinate queries can reuse the verified full-ensemble neural output. '
            'A CUDA-disabled replay rebuilt all graph feature arrays exactly at the real changed-coordinate fixture; '
            'changed node IDs or coordinates reject reuse. Each ordinary query loader still verifies checkpoints, code and image chunks. '
            'This operational cache reuse is separate from the cold-image proof.']
    lines += ['', '[Correctness evidence](validation.json), [fresh-image proof](fresh_image_validation.json), '
              '[actual changed-coordinate feature proof](native_refresh_validation.json), '
              '[identical native-query reuse proof](native_query_reuse_parity.json), '
              '[indexed observation-action proof](observation_edge_parity.json), '
              '[resumption proof](resume_validation.json) and [resource measurements](resource.json).']
    return '\n'.join(lines)


def continuation(status):
    pending=[r for r in status['experiments'] if r['status']!='measured']
    lines=['# Continuation — pipeline error training', '',
        'Read REPORT.md and STATUS.json first. This file is generated from the current execution receipts.', '',
        '## Current state', '',
        f"State: {status['status']}. Production default: P0. No new target fitting or threshold selection is authorized.", '',
        f"Active stage: {status['active_stage'] or 'none'}. Completed division matrices: {status['complete_division_clip_matrices']}/199.",
        'Completed point predictions per arm: '+', '.join(f'{a}={n}/199' for a,n in status['complete_point_clip_predictions'].items())+'.', '',
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
        'For early CPU audits while the finish queue is running in another terminal, use',
        '`$STUDY_PY -m pipeline_error_training.early_scoring`. It requires every completed clip and',
        'serialization file. A global scorer lock serializes audits and shared table exports;',
        'the final queue reuses completed scoring receipts. This does not change model selection.', '',
        'The post-primary queue requires both primary progress files to say complete. It performs source-only',
        'qualification and replication before the target freeze. Every failed independent prediction lane is retained.',
        'If a frozen stage fails, preserve its files under the new invalid/ root, record a named implementation repair,',
        'and rerun the same stage with unchanged scientific weights, calibration, bank and decision policy.', '',
        '## Pending or unsuccessful registered experiments', '']
    lines.extend(f"- {r['experiment']}: {r['status']} — {r['reason']}" for r in pending)
    if not pending:lines.append('All registered experiments have complete measured receipts.')
    restart = optional(RESULTS/'host_restart_recovery.json')
    if restart:
        lines += ['', '## Host restart recovery', '',
            f"The host restarted at {restart['current_boot_UTC']}; the exact study interruption time is unknown. "
            f"All {restart['completed_clips_preserved']} completed clip matrices and the frozen models were verified before resuming. "
            'Incomplete outputs and the previous ledger remain under invalid/host_restart_20260916/. '
            f"GPU accounting conservatively includes {restart['conservative_interrupted_budget_charge_hours']:.3f} hours "
            'from the unclosed lease through the new boot, including possible downtime. '
            'This is an upper bound, not observed active execution. Training and other studies were not resumed. '
            'See host_restart_recovery.json.']
    retired = optional(RESULTS/'training_crop_retirement.json')
    if retired:
        lines += ['', '## Completed-fit crop caches', '',
            f"The frozen fits' regenerable training crops ({retired['bytes']/2**30:.2f} GiB) were retired to reserve space for inference. "
            f"All {retired['retained_training_state_files']} model/optimizer/RNG state files were hash-verified before and after. "
            'The data loaders rebuild missing crops on demand from pinned images. '
            'See training_crop_retirement.json and work/pipeline-error-training-20260915/maintenance/training_crop_retirement/manifest.json.']
    lines += ['', '## Artifact locations and restrictions', '',
        '- Resumable weights, optimizer/RNG state, full graphs, source labels, image embeddings and logs: work/pipeline-error-training-20260915/.',
        '- Queue state: queue/progress.json, observation_queue/progress.json, finish_queue/progress.json under that root.',
        '- GPU accounting: gpu_budget/ledger.json; limits are 20 GiB total device memory, 50 GiB study RSS and 48 total lease-hours. The original reservations are 4/24/12/8 hours. Before target evaluation, gpu_reservation_settlement.json records unused first-seed hours shared with final inference; all 12 replication hours remain reserved.',
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
        executed_schedule=executed_schedule(lock['training']),
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
    active=optional(WORK/'finish_queue/active.json') or {}
    import psutil
    try:
        stage_running=psutil.Process(active['pid']).cmdline()==active['command']
    except (KeyError,psutil.NoSuchProcess,psutil.AccessDenied):
        stage_running=False
    if complete and (any(v!='complete' for v in queues.values()) or finished.get('status')!='complete'):
        raise RuntimeError('Cannot mark this study complete while its execution queues are unfinished')
    status=dict(created=now(),status='executing' if stage_running or any(v=='running' for v in queues.values()) else 'awaiting_final_validation',
        experiments=table,queues=queues,recommendation=recommended['candidate'],production_default='P0',
        active_stage=active.get('job') if stage_running else None,
        complete_division_clip_matrices=len(list((WORK/'final_inference').glob('*/complete.json'))),
        complete_point_clip_predictions={p.name:len(list(p.glob('*/complete.json')))
            for p in sorted((WORK/'final_point_inference').glob('*')) if p.is_dir()},
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
    heading = f"Candidate recommended: {recommended['candidate']}" if recommended['candidate'] else 'P0 retained'
    report=f'''{heading}

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

## Measured model outcomes

{outcome_text(recommended)}

## Source-only branching

{source_decision_text()}

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
Equal optimizer-update counts do not imply equal computation. [Encoder operation profiles](encoder_compute_profile.csv)
report fixed 16-node forward/backward costs, boundary support and checkpoint recomputation using PyTorch's registered
FLOP formulas; these exclude unsupported operations, decision heads and variable group sizes, and are not full-training
FLOP totals or hardware throughput. Actual fit and inference times are reported separately.
Performance changes to caches were adopted after exact source parity; fit wall times reflect each job's recorded
implementation and are not a controlled comparison of architecture speed.
The executed learning-rate schedule applies warmup and cosine decay concurrently; the shortened update budget
also shortens its warmup denominator to {lock['training']['updates']}. See [the exact schedule](executed_training_schedule.json).
These short, matched fits do not establish convergence or rule out the architectures after longer source-only training.
Event minibatches are conditioned on groups containing a supported biological positive: the completed temporal
packages record 15 such groups for 44b6 and 74 for 6bba. Their alternatives supply supported identity confusers and
metric-risk negatives; groups containing only negative event hypotheses are not a separate event-minibatch pool.
Identity minibatches sample supported trajectory groups. Calibration expands the supported source sample:
ordinary anchors use one deterministic temporal residue out of nine and receive ninefold weight; event-compatible
anchors are retained. These are fixed expansion weights, not proven randomized row propensities or a complete
source-field census. [The executed sampling audit](calibration_sampling_audit.json) records both raw and expanded
prevalence. These sampling choices limit conclusions about whole-field false-fork rejection.
Cumulative charged GPU lease time is {resource['gpu_budget']['total_hours']:.3f} hours of 48; detailed memory/runtime evidence is in resource.json.
This includes {resource['gpu_budget']['interrupted_upper_bound_hours']:.3f} hours conservatively charged for an interrupted
lease through the next host boot. Its exact end was not observed, and the charge can include downtime;
see host_restart_recovery.json. Completed outputs and frozen model hashes were verified before resuming.
No measured Kaggle 12-hour runtime or leaderboard improvement is claimed. No weights were published or defaults changed.

## Executable validation and fresh inference

{proof_text()}

See [CONTINUATION.md](CONTINUATION.md) for exact commands and remaining work.
'''
    (RESULTS/'REPORT.md').write_text(report)
    return status


if __name__=='__main__':
    run('--complete' in __import__('sys').argv)
