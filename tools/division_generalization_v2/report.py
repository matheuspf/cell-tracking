"""Concise honest receipts, curves and resumable commands; no invented results."""
import csv
import json
import os
from pathlib import Path
import psutil

from .common import WORK,RESULTS,REPO,read_json,write_json,write_csv,sha,now


def completion_gates(complete,arm,target,fresh,gates,fits,freeze,image_gate,matrix_gate):
    """Completion is measured separately from whether a score happens to win."""
    by_key={(r['arm'],r['seed']):r for r in target['pooled']}
    expected={(a,s) for a in freeze['qualified_exports']
              for s in ((20260916,) if a=='G30' else (20260916,314159))}
    matched={(a,s) for a in ('J_uniform','J_mined') for s in (20260916,314159)}
    directional=[r for r in fits if r['arm']!='prefix']
    model_hash=sha(Path(__file__).with_name('model.py'))
    return dict(
        queue_complete=bool(complete),
        all_qualified_exports_complete=set(by_key)==expected and all(r['clips']==199 for r in by_key.values()),
        matched_image_controls_complete=matched<=set(by_key),
        ten_adequate_directional_fits=len(directional)==10 and all(
            r['status']=='complete' and r['joint_optimizer_updates']>=4096 for r in directional),
        all_full_source_screens=bool(directional) and all(r['full_source_screen'] for r in directional),
        fresh_nominated_module=fresh.get('status')=='measured' and fresh.get('cold_pipeline_artifacts') is True and len(fresh.get('clips',[]))==2 and all(
            r['arm']==arm and r['exact_P0'] and r['exact_scored_candidate'] for r in fresh.get('clips',[])),
        active_zero_and_tests=bool(gates.get('literal_zero_path_test') and gates.get('unit_tests_passed')),
        current_image_zero=image_gate.get('model_code_sha256')==model_hash and len(image_gate.get('clips',[]))==2 and all(
            r['full_clip'] and r['exact_active_zero'] for r in image_gate.get('clips',[])),
        current_matrix_parity=matrix_gate.get('model_code_sha256')==model_hash and bool(matrix_gate.get('image_models_exact')))


def run(args=None):
    RESULTS.mkdir(parents=True,exist_ok=True)
    fits=[];curves=[];source_scores=[];calibrations=[];runtime=[]
    crash_replays=[]
    timing_fields=('loader_seconds','transfer_seconds','augmentation_seconds','loader_transfer_seconds',
                   'encoder_head_seconds','backward_seconds','optimizer_seconds','compute_seconds')
    for path in sorted(RESULTS.glob('crash_recovery_*.json')):
        recovery=read_json(path)
        source,arm,seed=(recovery['resume_source'],recovery['resume_arm'],recovery['resume_seed'])
        history=Path(recovery['archive'])/f'training-{arm}-{source}-{seed}'/'history.jsonl'
        discarded=[json.loads(line) for line in history.read_text().splitlines()
                   if json.loads(line)['step']>recovery['resume_step']]
        crash_replays.append(dict(source=source,arm=arm,seed=seed,updates=len(discarded),
            archived_history_sha256=sha(history),recovery_receipt_sha256=sha(path),
            totals={k:sum(r.get(k,0.) for r in discarded) for k in timing_fields},
            scope='Executed before poweroff, then replayed from the saved checkpoint; additional to retained history timings',
            interrupted_invocation_wall_and_checkpoint_times_complete=False))
    prefix_pairs=[]
    for source in ('44b6','6bba'):
        for seed in (20260916,314159):
            roots=[WORK/'training'/arm/source/str(seed) for arm in ('J_uniform','J_mined')]
            if not all((p/'common_prefix.json').exists() and (p/'mining-2048.json').exists() for p in roots):continue
            same_prefix=read_json(roots[0]/'common_prefix.json')==read_json(roots[1]/'common_prefix.json')
            same_mining=sha(roots[0]/'mining-2048.json')==sha(roots[1]/'mining-2048.json')
            if not same_prefix or not same_mining:raise ValueError('Matched branches did not share the same starting state')
            prefix_pairs.append(dict(source=source,seed=seed,same_prefix_receipt=True,initial_mining_byte_exact=True,
                checkpoint_sha256=read_json(roots[0]/'common_prefix.json')['sha256'],
                initial_mining_sha256=sha(roots[0]/'mining-2048.json')))
    write_json(RESULTS/'matched_prefix_audit.json',dict(status='measured' if len(prefix_pairs)==4 else 'partial',
        expected_source_seed_pairs=4,pairs=prefix_pairs,shared_prefix_not_independent=True))
    for path in sorted((WORK/'training').glob('*/*/*/training_receipt.json')):
        row=read_json(path)
        invocation_paths=list((path.parent/'invocations').glob('*.json'))
        invocations={sha(p):(p,read_json(p)) for p in invocation_paths}
        invocations.setdefault(sha(path),(path,row))
        invocation_totals={k:sum(r.get(k,0.) for _,r in invocations.values()) for k in
            ('compute_seconds','lease_seconds','wait_seconds','checkpoint_seconds','wall_seconds')}
        prefix=path.parent/'common_prefix.json'
        prefix_receipt=read_json(prefix) if prefix.exists() else None
        if prefix_receipt and sha(prefix_receipt['path'])!=prefix_receipt['sha256']:
            raise ValueError('Shared prefix artifact drift')
        screens=list((WORK/'screens'/row['arm']/row['source']/str(row['seed'])).glob('*/summary.json'))
        exposure=dict(row['exposure'])
        for field in ('group_visits','anchor_visits'):
            visits=exposure.pop(field,{})
            exposure[field+'_summary']=dict(unique=len(visits),draws=sum(visits.values()),
                minimum=min(visits.values(),default=0),maximum=max(visits.values(),default=0))
        fits.append(dict(row,exposure=exposure,receipt_path=str(path),receipt_sha256=sha(path),
            common_prefix=prefix_receipt,invocation_totals=invocation_totals,
            training_log_hashes={name:sha(path.parent/name) for name in
                ('history.jsonl','diagnostics.jsonl','mining-2048.json','mining-3072.json')
                if (path.parent/name).exists()},
            invocation_receipts=[dict(path=str(p),sha256=key) for key,(p,_) in invocations.items()],
            full_source_screen=bool(screens) and all(read_json(p)['full_source_screen'] for p in screens),
            source_screen_receipts=[dict(path=str(p),sha256=sha(p)) for p in sorted(screens)]))
        folder=path.parent
        history=folder/'history.jsonl'
        if history.exists():
            updates=[json.loads(line) for line in history.read_text().splitlines()]
            runtime.append(dict(arm=row['arm'],source=row['source'],seed=row['seed'],
                recorded_updates=len(updates),first_step=min(r['step'] for r in updates),
                last_step=max(r['step'] for r in updates),
                totals={k:sum(r.get(k,0.) for r in updates) for k in timing_fields},
                maximum_gradient_norm=max(r['gradient_norm'] for r in updates),
                checkpoint_seconds=invocation_totals['checkpoint_seconds'],wall_seconds=invocation_totals['wall_seconds'],
                measured_invocations=len(invocations),
                crash_interrupted_invocation=any((r['source'],r['arm'],r['seed'])==
                    (row['source'],row['arm'],row['seed']) for r in crash_replays),
                includes_common_prefix=row['arm'] not in ('J_uniform','J_mined'),
                cuda_kernel_profiler_seconds=None,
                timing_scope='Synchronized operation wall times; encoder/head includes CPU dispatch, not pure kernel time'))
        if (folder/'diagnostics.jsonl').exists():
            for line in (folder/'diagnostics.jsonl').read_text().splitlines():
                d=json.loads(line)
                for split in ('fit','calibration'):
                    curves.append(dict(arm=row['arm'],source=row['source'],seed=row['seed'],step=d['step'],partition=split,**d[split]))
    for path in sorted((WORK/'screens').glob('*/*/*/*/summary.json')):
        r=read_json(path)
        for app,score in r['applications'].items():
            source_scores.append(dict(arm=r['arm'],source=r['source'],seed=r['seed'],step=r['step'],application=app,
                score=score['score'],delta=score['delta'],lost_supported_edges=score['lost_supported_edges'],clips=score['clips'],
                division_tp=score['division_tp'],division_fp=score['division_fp'],division_fn=score['division_fn']))
    for path in sorted((WORK/'screens').glob('*/*/*/*/calibration.json')):
        r=read_json(path)
        calibrations.append(dict(arm=path.parts[-5],seed=int(path.parts[-3]),step=int(path.parts[-2]),
            **{k:v for k,v in r.items() if k!='rows'},
            calibration_sha256=sha(path),full_source_graph_screen_complete=(path.parent/'summary.json').exists()))
    write_json(RESULTS/'training_receipts.json',dict(fits=fits,actual_receipts_required=True))
    write_csv(RESULTS/'training_curves.csv',curves)
    if (RESULTS/'diagnostic_panel_composition.json').exists():
        from .plot_curves import render
        render(curves)
    write_csv(RESULTS/'source_scores.csv',source_scores)
    inference=[]
    for kind,paths in [('source_matrix',(WORK/'frozen_matrix').glob('*/*/*/complete.json')),
                       ('target_matrix',(WORK/'target_matrix').glob('*/complete.json'))]:
        for path in sorted(paths):
            r=read_json(path)
            inference.append(dict(kind=kind,dataset=path.parent.name,receipt=str(path),sha256=sha(path),
                seconds=r['seconds'],timings=r['timings'],counts=r['counts'],packages=r['packages'],
                worker_times_overlap=r['worker_times_overlap'],shared_work_counted_once=True))
    for output in sorted((WORK/'screens').glob('*/*/*/*/predictions/*')):
        if output.is_symlink():continue
        path=output/'protected'/(output.name+'.json')
        if not path.exists():continue
        r=read_json(path)
        inference.append(dict(kind='single_source',dataset=output.name,receipt=str(path),sha256=sha(path),
            seconds=r['seconds'],timings=r['timings'],counts=r['counts']))
    write_json(RESULTS/'runtime_breakdown.json',dict(training=runtime,inference=inference,crash_replays=crash_replays,
        independent_workers_overlap=True,wall_times_must_not_be_summed_into_total_elapsed=True,
        reason=None if runtime else 'Production optimizer histories are not complete yet'))
    write_json(RESULTS/'calibration_audit.json',dict(status='measured' if calibrations else 'pending',fits=calibrations,
        full_source_graph_screens_are_separate=True,
        reason=None if calibrations else 'Adequate training and full source screens have not completed'))
    ledger=WORK/'resources/leases.jsonl';events=[]
    if ledger.exists():events=[json.loads(line) for line in ledger.read_text().splitlines()]
    resource=dict(exclusive_lease_seconds=sum(e.get('lease_seconds',0.) for e in events if e['event']=='end'),
        wait_seconds=sum(e.get('wait_seconds',0.) for e in events if e['event']=='end'),
        open_leases=[e for e in events if e['event']=='begin' and e['token'] not in {x['token'] for x in events if x['event']=='end'}],
        scope='Full exclusive lease intervals, including loader and CPU work; not kernel time',
        study_disk_gib=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file())/2**30)
    resource['crash_estimated_lease_seconds']=sum(e.get('lease_seconds',0.) for e in events
        if e['event']=='end' and e.get('reconciled_utc'))
    resource['measured_closed_lease_seconds']=(resource['exclusive_lease_seconds']-
        resource['crash_estimated_lease_seconds'])
    recovery=RESULTS/'crash_recovery_20260917.json'
    if recovery.exists():
        resource['crash_recovery_receipt_sha256']=sha(recovery)
        resource['crash_downtime_charged']=False
    resource['lease_seconds_by_stage']={stage:sum(e.get('lease_seconds',0.) for e in events
        if e['event']=='end' and e['purpose'].split('/')[0]==stage)
        for stage in sorted({e['purpose'].split('/')[0] for e in events})}
    monitors=[]
    for p in WORK.rglob('*.json'):
        if p.name=='resources.json' or p.name.endswith('.resources.json'):
            try:monitors.append(read_json(p))
            except (OSError,ValueError):pass
    resource.update(peak_total_gpu_gib=max((r.get('peak_total_gpu_gib',0.) for r in monitors),default=None),
                    peak_process_tree_rss_gib=max((r.get('peak_process_tree_rss_gib',0.) for r in monitors),default=None))
    write_json(RESULTS/'resource.json',resource)
    active=[]
    for p in psutil.process_iter(['cmdline','create_time']):
        try:
            if p.pid == os.getpid():continue
            cmd=p.info['cmdline'] or []
            if any(x.startswith('division_generalization_v2') for x in cmd):
                active.append(dict(pid=p.pid,command=cmd,started=p.info['create_time']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    complete=(WORK/'queue/complete.json').exists()
    target=read_json(RESULTS/'target_evaluation.json') if (RESULTS/'target_evaluation.json').exists() else None
    baseline=read_json(RESULTS/'baseline_validation.json') if (RESULTS/'baseline_validation.json').exists() else None
    if (RESULTS/'validation.json').exists():
        validation=read_json(RESULTS/'validation.json')
        validation['target_comparisons']='complete' if target and target['status']=='complete' else 'pending'
        if target:validation['target_evaluation_sha256']=sha(RESULTS/'target_evaluation.json')
        if (RESULTS/'fresh_image_validation.json').exists():
            validation['fresh_image_proof']=read_json(RESULTS/'fresh_image_validation.json')['status']
            validation['fresh_image_validation_sha256']=sha(RESULTS/'fresh_image_validation.json')
        write_json(RESULTS/'validation.json',validation)
    status=dict(study='division-generalization-v2',updated=now(),status='complete' if complete else 'incomplete_resumable',
        production_default='P0',target_met=False,new_target_scores=target,
        completed_directional_fits=len([r for r in fits if r['status']=='complete' and r['arm']!='prefix']),
        required_directional_fits=10,active_processes=active,
        prepared_clips={s:len(list((WORK/'source'/s).glob('*/receipt.json'))) for s in ('44b6','6bba')},
        reason=None if complete else 'Full training, full source screens, frozen target comparison and fresh proof remain required')
    write_json(RESULTS/'STATUS.json',status)
    replication=dict(status='measured' if target else 'pending',seeds=[20260916,314159],
                     paired_shared_prefix=True,independent_experiments=False,recommended=False,
                     reason='Recommendation requires all complete graphs, both seeds and fresh gates')
    recommendation=None
    if target and (RESULTS/'target_freeze.json').exists():
        freeze=read_json(RESULTS/'target_freeze.json');arm=freeze['nominee']
        pooled=[r for r in target['pooled'] if r['arm']==arm]
        embryos=[r for r in target['embryos'] if r['arm']==arm]
        bp={r['embryo']:r['score'] for r in baseline['embryos'] if r['arm']=='P0'}
        fresh=read_json(RESULTS/'fresh_image_validation.json') if (RESULTS/'fresh_image_validation.json').exists() else {}
        gates=read_json(RESULTS/'validation.json')
        checks=completion_gates(complete,arm,target,fresh,gates,fits,freeze,
            read_json(RESULTS/'corrected_image_validation.json'),read_json(RESULTS/'matrix_parity.json'))
        checks['matched_prefix_and_initial_mining']=len(prefix_pairs)==4
        ready=all(checks.values()) and len(pooled)==2 and len(embryos)==4
        if ready and all(r['score']>.934864986413134 for r in pooled) and all(r['score']>=bp[r['embryo']] for r in embryos) \
                and next(r['score'] for r in pooled if r['seed']==20260916)>.935178370257:
            recommendation=arm
        replication.update(recommended=bool(recommendation),nominee=arm,
            target_met_by_seed={str(r['seed']):r['score']>=.95 for r in pooled},
            robust_target_met=bool(recommendation and all(r['score']>=.95 for r in pooled)))
        by_key={(r['arm'],r['seed']):r['score'] for r in target['pooled']}
        mining={str(seed):by_key['J_mined',seed]-by_key['J_uniform',seed]
            for seed in (20260916,314159) if ('J_mined',seed) in by_key and ('J_uniform',seed) in by_key}
        by_embryo={(r['arm'],r['seed'],r['embryo']):r['score'] for r in target['embryos']}
        mining_embryos={f'{seed}/{embryo}':by_embryo['J_mined',seed,embryo]-by_embryo['J_uniform',seed,embryo]
            for seed in (20260916,314159) for embryo in ('44b6','6bba')
            if ('J_mined',seed,embryo) in by_embryo and ('J_uniform',seed,embryo) in by_embryo}
        replication.update(correctness_and_completion_gates=bool(ready),completion_checks=checks,
            mining_delta_by_seed=mining,mining_delta_by_seed_and_embryo=mining_embryos,
            replicated_mining_advantage=len(mining)==2 and all(v>0 for v in mining.values()))
        status['target_met']=replication['robust_target_met'];write_json(RESULTS/'STATUS.json',status)
    write_json(RESULTS/'replication.json',replication)
    if not (RESULTS/'stage_attribution.json').exists():
        write_json(RESULTS/'stage_attribution.json',dict(status='pending',value=None,
            reason='Requires complete frozen source and target inference ledgers'))
    if not (RESULTS/'error_transitions.csv').exists():write_csv(RESULTS/'error_transitions.csv',[])
    target_outcome='achieved' if status['target_met'] else ('not achieved' if complete else 'not established')
    report=[f'Candidate recommended: {recommendation}' if recommendation else 'P0 retained','',
        f'Execution: **{status["status"]}**. Requested replicated score ≥0.95: **{target_outcome}**.','',
        '| Arm / seed | Scope | Score | Δ P0 | Δ C4_m6 | Edge TP / FP / FN | Division TP / FP / FN | Edge raw / adjusted | Selected / matched nodes | Node hash | Status |',
        '|---|---|---:|---:|---:|---|---|---|---|---|---|']
    measured=((baseline['pooled']+baseline['embryos']) if baseline else [])+((target['pooled']+target['embryos']) if target else [])
    refs={(r['arm'],r['embryo']):r['score'] for r in (baseline['pooled']+baseline['embryos'] if baseline else [])}
    node_ids=read_json(RESULTS/'node_identity.json')['corpora'] if (RESULTS/'node_identity.json').exists() else {}
    for r in measured:
        scope=r['embryo']
        edges=' / '.join(str(r['edge_'+k]) for k in ('tp','fp','fn'))
        divisions=' / '.join(str(r['division_'+k]) for k in ('tp','fp','fn'))
        report.append(f'| {r["arm"]}{" / "+str(r["seed"]) if "seed" in r else ""} | {scope} | {r["score"]:.12f} | '
            f'{r["score"]-refs["P0",scope]:+.12f} | {r["score"]-refs["C4_m6",scope]:+.12f} | {edges} | {divisions} | '
            f'{r["edge_jaccard"]:.9f} / {r["adj_edge_jaccard"]:.9f} | {r["num_pred_nodes"]} / {r["matched_nodes"]} | '
            f'{r.get("node_identity_sha256",node_ids.get(scope,"pending"))[:12]} | {r["status"]} |')
    for arm in ('G30','J_uniform','J_mined'):
        if not any(r['arm']==arm for r in measured):
            reason='source qualification pending' if not target else 'source failed; target export not qualified'
            report.append(f'| {arm} | pooled + both embryos | null | null | null | null | null | null | null | null | {reason} |')
    if (RESULTS/'target_freeze.json').exists():
        freeze=read_json(RESULTS/'target_freeze.json')
        nominee=freeze['nominee'] or 'none qualified in both source directions'
        exports=', '.join(freeze['qualified_exports']) or 'none'
        report+=['',f'Source-frozen nominee: **{nominee}**. Qualified complete exports: {exports}. '
            'The [frozen source decisions](target_freeze.json) record each direction and seed\'s selected '
            'checkpoint, application, calibration and source score before target predictions.']
        for source in ('44b6','6bba'):
            for seed in (20260916,314159):
                path=RESULTS/f'extension-{source}-{seed}.json'
                if path.exists():
                    decision='extended both arms to 8,192 updates' if read_json(path)['extend'] else 'stopped both arms at 4,096 updates'
                    report.append(f'[{source} / {seed} duration decision]({path.name}): {decision}.')
    calibration_counts={name:sum(r['status']==name for r in calibrations) for name in
        ('held_source_fitted','grouped_small_head_oof_fitted','calibration_unestablished')}
    report+=['',f'{status["completed_directional_fits"]}/10 directional fits have completed their required updates. '
        'Each main arm requires 4,096 joint updates; uniform and mined arms share their first 2,048 updates per direction/seed. '
        'The shared prefix is counted once in compute and does not make independent experiments.','',
        'The fixed diagnostic panels contain 32 hashed groups each. The 6bba held panel has no positive-utility anchors, '
        'so its loss measures supported rejection and cannot establish division recovery. The 44b6 held panel has two '
        'positive-utility anchors. Full source graph screens govern qualification. The panels and extension rule remain '
        'as originally locked. [Panel composition](diagnostic_panel_composition.json).','',
        f'Completed checkpoint calibrations: {calibration_counts["held_source_fitted"]} direct held-source fits, '
        f'{calibration_counts["grouped_small_head_oof_fitted"]} grouped small-head fallbacks, and '
        f'{calibration_counts["calibration_unestablished"]} unestablished. The fallback fits three heads for 4,096 updates '
        'each on frozen features, holding complete overlap groups out. Those updates do not count toward the joint-training '
        'floor. Only the heads are out of fit: the frozen source encoder retains its original label exposure. '
        '[Calibration audit](calibration_audit.json).','',
        'The new module keeps native evidence inside learned relative complete-action scores. It uses fixed-seed probability '
        'sampling, separate positive exposure, masked unknown alternatives, negative-only groups, complete lost-link utility, '
        'and one two-scale raw scene shared across candidate pairs. The 490,804-parameter encoder/head uses ordered attention.','',
        'Both inherited baselines retain all 4,108,943 observations. All graph hashes and baseline metric receipts were verified, '
        'with fresh ordinary/crowded replays in both embryos and independent official aggregation. '
        'P0 and C4_m6 remain unchanged.','',
        'This is exploratory source-only direct fitting on exposed P0 observations. Reused whole-clip source splits union exact '
        'frame overlaps, but missing global acquisition offsets prevent independent inner-validation certification. '
        'Neither a local score nor a new raw-scene encoder establishes clean OOF or hidden leaderboard performance.','',
        '[Training receipts](training_receipts.json), [fixed source curves](training_curves.csv), '
        '[source curve plot](training_event_curves.png), '
        '[complete source screens](source_scores.csv), [per-embryo scores](per_embryo_scores.csv), '
        '[sampling audit](sampling_audit.json), [validation](validation.json), [resources](resource.json).','',
        f'Accounted exclusive GPU leases: {resource["exclusive_lease_seconds"]/3600:.3f} h; waits: {resource["wait_seconds"]:.1f} s. '
        f'This includes {resource["crash_estimated_lease_seconds"]:.3f} s estimated for an interrupted lease; '
        'host downtime is excluded. '
        'Full lease intervals include preprocessing. [Operation wall timings](runtime_breakdown.json) separate loading, transfers, '
        'augmentation, encoder/head, backward, optimizer and checkpoint work; pure CUDA kernel time is not measured.','',
        'An early image implementation sampled CNN feature maps at shifted coordinates. Those image fits were archived as '
        'implementation-invalid and restarted from scratch; their updates do not count toward training adequacy, and their '
        'GPU leases remain in the cost ledger. The cached-feature controls are unaffected, verified by exact output parity. '
        '[Corrected image proof](corrected_image_validation.json), [profile provenance](profile_provenance.json), '
        '[control compatibility](model_code_compatibility.json).','',
        'No production promotion, merge, Kaggle submission, weight publication or leaderboard claim has been made.']
    (RESULTS/'REPORT.md').write_text('\n'.join(report)+'\n')
    if complete:
        continuation=f'''Read [REPORT.md](REPORT.md) and [STATUS.json](STATUS.json). Status: complete.

The execution queue finished. The report records whether the measured nominee passed all recommendation and replication gates; completing the study does not itself establish a score of ≥0.95. P0 remains the production default.

Review [target_freeze.json](target_freeze.json) for the source-only checkpoint, application and family choices; [replication.json](replication.json) for completion and recommendation checks; and [fresh_image_validation.json](fresh_image_validation.json) for the renamed-image replays. Training, calibration, source screens, target scores, error transitions and resource receipts are linked from REPORT.md.

Heavy artifacts remain under `work/division-generalization-v2`, a symlink to `/kaggle/working/cell-tracking/division-generalization-v2`. Input paths and hashes are recorded in input_manifest.json. Preserve those artifacts, the recovery archives and the original baselines. No completed training or inference needs to be rerun to inspect these results.

To regenerate the concise report from existing receipts, run from `{REPO}`:

```sh
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m division_generalization_v2 report
```

For a deliberate independent reconstruction, use the committed study configuration and verified inputs in a separate isolated work root. Preserve the 4,096 joint-update floor, paired prefixes, source-only decisions and startup access guards. The original execution, invalid attempts and crash costs remain part of this study's provenance.

No production promotion, merge, Kaggle submission, weight publication or leaderboard claim was performed.
'''
    else:
        continuation=f'''Read REPORT.md and STATUS.json. Status: {status['status']}.

Run from `{REPO}` on the current branch. Inspect actual processes first; never resume an older study or launch duplicate workers.

```sh
ps -eo pid,ppid,etime,rss,args | rg division_generalization_v2
nvidia-smi
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
$STUDY_PY -m division_generalization_v2 queue
$STUDY_PY -m division_generalization_v2 report
```

The queue checkpoints every 128 joint updates and at milestones. Each matched arm loads its seed/source's identical 2,048-update model, optimizer, RNG and data-order state. Do not shrink 4,096 updates or treat interrupted fits as failed architectures.

Heavy artifacts are under `work/division-generalization-v2`, a symlink to `/kaggle/working/cell-tracking/division-generalization-v2`. Recreate that empty isolated root on another machine; inherited inputs resolve through input_manifest.json. Existing raw inputs and work/pipeline-error-training-20260915 are read-only references. No shared environment changes are needed.

Any stage failure is preserved with its exact command and log in queue/failure.json and logs/. Fix implementation defects, archive invalid attempts with reasons, and retain scientific settings. An unclosed GPU lease requires conservative accounting reconciliation before restart.

Outstanding gates are explicit in STATUS.json, validation.json, training_receipts.json and source_scores.csv. Target export requires source qualification; failed cheap control does not block main training. Both directional packages and both seeds must be frozen before target scoring. Full target scoring and fresh image proof, not partial clips, govern recommendation and the ≥0.95 claim.
'''
    (RESULTS/'CONTINUATION.md').write_text(continuation)
    return status


if __name__=='__main__':run()
