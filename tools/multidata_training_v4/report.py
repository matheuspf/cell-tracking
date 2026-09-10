"""Measured, sanitized results. Never substitute a loss curve for transfer scores."""
import shutil
import html
import pandas as pd
from .common import *

def select(scores):
    candidates=[]
    for v in scores.variant.unique():
        if v.endswith('_seed2'):continue  # Never pick a lucky replication seed.
        rows=scores[scores.variant==v]
        if len(rows)!=3 or (rows.samples<1).any():continue
        pooled=rows[rows.embryo=='pooled'].iloc[0]
        repeated=scores[scores.variant==v+'_seed2']
        replicated=len(repeated)==3 and (repeated.delta_v3>=-1e-10).all() and repeated[repeated.embryo=='pooled'].iloc[0].delta_v3>1e-12
        if pooled.delta_v3>1e-12 and (rows.delta_v3>=-1e-10).all() and replicated:candidates.append((pooled.score,v))
    return max(candidates)[1] if candidates else 'C0'

def measured_observations(scores):
    """Interpret the already frozen comparisons without changing selection rules."""
    indexed=scores.set_index(['variant','embryo'])
    def value(variant,embryo='pooled',column='score'):
        return float(indexed.loc[(variant,embryo),column])
    lines=['', '## What the measurements establish', '']
    for variant in ['C4','C5']:
        lines.append(f'{variant} scored {value(variant):.12f}, a pooled delta of {value(variant,column="delta_v3"):+.12f}. '
            f'Its embryo deltas were {value(variant,"44b6","delta_v3"):+.12f} on 44b6 and '
            f'{value(variant,"6bba","delta_v3"):+.12f} on 6bba. The 6bba regression fails the prespecified adoption rule.')
    lines.append(f'The descriptive margin-6 C4 control scored {value("C4_m6"):.12f}, '
        f'but its 6bba delta was {value("C4_m6","6bba","delta_v3"):+.12f}. '
        'A higher pooled score alone does not qualify it. The full real-only C1 and fish-geometry C3 primary policies abstained and preserved the incumbent exactly; '
        'the shorter real-only, synthetic-only C2 and appearance-randomized C6 policies regressed in pooled score.')
    lines.append(f'Removing C4 optical evidence reduced the pooled score by {value("C4")-value("C4_Gonly"):.12f}. '
        f'When continuations were rebuilt, external C4 exceeded matched real-only C1 by '
        f'{value("C4_relinked")-value("C1_relinked"):.12f}, but remained '
        f'{value("C4_relinked",column="delta_v3"):+.12f} below v3. '
        'These controls support a role for image evidence and learned associations inside this pipeline; they do not establish an improvement over the incumbent.')
    lines.append('All detector variants regressed relative to v3. detector_attribution.csv compares each coordinate refinement with the identical association policy on unmoved centers, separating refinement from the larger cost of rebuilding continuations.')
    for variant in ['C1_seed2','C4_seed2','C7','C7_seed2']:
        if (variant,'pooled') not in indexed.index:continue
        lines.append(f'{variant}: pooled {value(variant):.12f}, delta v3 {value(variant,column="delta_v3"):+.12f}; '
            f'44b6 {value(variant,"44b6","delta_v3"):+.12f}, 6bba {value(variant,"6bba","delta_v3"):+.12f}; '
            f'pooled difference from its same-seed C1 control {value(variant,column="delta_vs_C1"):+.12f}.')
    return lines

def run():
    scores=pd.read_csv(OUT/'ablation_scores.csv');rows=pd.read_csv(OUT/'score_rows.csv')
    assert (scores[scores.embryo=='pooled'].samples==199).all()
    receipts=[read(p) for p in sorted((OUT/'models').glob('*.json'))]
    write(OUT/'training_receipts.json',receipts)
    learning=[]
    for r in receipts:
        for h in r['history']:
            row=dict(model=r['config']['name'],**{k:v for k,v in h.items() if k!='validation'})
            v=h.get('validation',{});row.update({'validation_'+k:x for k,x in v.items()})
            if v.get('positive'):row['validation_pair_recall']=v['positive_pair_correct']/v['positive']
            learning.append(row)
    pd.DataFrame(learning).to_csv(OUT/'learning_curves.csv',index=False)
    ledger=pd.read_csv(OUT/'dataset_use.csv').fillna('')
    coverage=pd.read_csv(OUT/'source_coverage.csv').fillna(0)
    if 'C7' in set(scores.variant) and (OUT/'rendered_dataset_index.json').exists():
        rendered=pd.DataFrame([dict(source=r['source'],partition=r['partition'],samples=1,
            bags=r['bags'],positive_groups=r['positive_groups'],independent_acquisition=False)
            for r in read(OUT/'rendered_dataset_index.json')['records']])
        rendered.to_csv(OUT/'rendered_source_coverage.csv',index=False)
        coverage=pd.concat([coverage,rendered],ignore_index=True)
    control={(r.variant,r.embryo):r.score for r in scores[scores.variant.isin(['C1','C1_seed2'])].itertuples()}
    scores['matched_control']=scores.variant.map(lambda v:'C1_seed2' if v.endswith('_seed2') else 'C1')
    scores['delta_vs_C1']=scores.apply(lambda r:r.score-control[(r.matched_control,r.embryo)],axis=1);scores.to_csv(OUT/'ablation_scores.csv',index=False)
    selected=select(scores);winner=scores[scores.variant==selected];pool=winner[winner.embryo=='pooled'].iloc[0]
    external_arm=selected in ['C2','C3','C4','C5','C6','C7']
    repeated=scores[scores.variant==selected+'_seed2']
    primary_external_advantage=bool(external_arm and (winner.delta_vs_C1>0).all())
    external_advantage=bool(primary_external_advantage and len(repeated)==3 and (repeated.delta_vs_C1>0).all())
    config=dict(variant=selected,score=float(pool.score),delta_v3=float(pool.delta_v3),
        embryo_scores={r.embryo:dict(score=r.score,delta_v3=r.delta_v3,delta_vs_C1=r.delta_vs_C1) for r in winner.itertuples()},
        baseline='v3 A_residual_m3.0',baseline_score=BASE['pooled'],external_advantage_over_matched_control=external_advantage,
        primary_external_advantage=primary_external_advantage,matched_control_is_same_seed=True,
        preselected_external='C4',selection_is_operational_exploratory=True,
        no_target_independence_claim=True,new_heads_disabled=selected=='C0',replication_required_for_adoption=True,
        model_manifest_sha256=sha(OUT/'checkpoint_manifest.json'),prediction_lock_sha256=sha(OUT/'prediction_lock.json'))
    config['model_manifests_sha256']={p.name:sha(p) for p in sorted(OUT.glob('checkpoint_manifest*.json'))}
    write(OUT/'winning_config.json',config)
    regret=[]
    for variant in scores.variant.unique():
        gained=lost=0
        for name in rows[rows.variant==variant].dataset:
            baseline=set(map(tuple,arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['recovered']))
            current=set(map(tuple,arrays(OUT/'evaluation_matches'/variant/f'{name}.npz')['recovered']))
            gained+=len(current-baseline);lost+=len(baseline-current)
        regret.append(dict(variant=variant,recovered_gt_edges=gained,lost_gt_edges=lost))
    pd.DataFrame(regret).to_csv(OUT/'edge_regret.csv',index=False)
    train=[]
    for r in receipts:
        c=r['config'];train.append(dict(model=c['name'],component=c['component'],source=c.get('source'),updates=r['actual_updates'],
            parameters=r['trainable_parameters'],seconds=r['elapsed_seconds'],peak_gpu_gib=r['peak_gpu_gib'],
            unique_groups=sum(v['consumed_unique_groups'] for v in r['coverage'].values()),
            unique_positive_groups=sum(v['consumed_unique_positive_groups'] for v in r['coverage'].values()),
            external_examples=sum(v['consumed_unique_examples'] for k,v in r['coverage'].items() if k not in ['44b6','6bba']),
            direct_sources=','.join(r['direct_sources']),primary_real_supervised_rows=r.get('sparse_source_supervised_rows')))
    pd.DataFrame(train).to_csv(OUT/'training_summary.csv',index=False)
    detection=[]
    for (variant,embryo),rr in rows.groupby(['variant','embryo']):
        detection.append(dict(variant=variant,embryo=embryo,matched_centers=int(rr.matched_nodes.sum()),
            observed_gt_centers=int(rr.gt_node_count.sum()),observed_center_recall=float(rr.matched_nodes.sum()/rr.gt_node_count.sum()),
            mean_matched_distance_um=float(rr.match_distance_sum_um.sum()/rr.matched_nodes.sum()),
            rms_matched_distance_um=float(np.sqrt(rr.match_distance_squared_sum_um2.sum()/rr.matched_nodes.sum())),
            count_ratio=float(rr.num_pred_nodes.sum()/rr.estimated_total.sum()),exact_duplicates=int(rr.exact_duplicate_centers.sum()),
            gt_division_parents=int(rr.gt_division_parents.sum()),division_daughters_matched=int(rr.division_daughters_matched.sum())))
    pd.DataFrame(detection).to_csv(OUT/'detection_diagnostics.csv',index=False)
    attribution=[]
    for variant,control in [('D_real','C1_relinked'),('D_synthetic','C1_relinked'),('D_synthetic_C4','C4_relinked')]:
        for embryo in ['44b6','6bba','pooled']:
            r=scores[(scores.variant==variant)&(scores.embryo==embryo)].iloc[0]
            c=scores[(scores.variant==control)&(scores.embryo==embryo)].iloc[0]
            attribution.append(dict(variant=variant,embryo=embryo,score=r.score,delta_v3=r.delta_v3,
                association_control=control,association_control_score=c.score,delta_refinement_with_same_association=r.score-c.score))
    pd.DataFrame(attribution).to_csv(OUT/'detector_attribution.csv',index=False)
    refinement=[]
    for p in sorted((OUT/'prediction_receipts').glob('*.json')):
        receipt=read(p)
        for detector,stats in receipt['detector'].items():
            refinement.append(dict(dataset=receipt['dataset'],embryo=receipt['dataset'][:4],detector=detector,**stats))
    pd.DataFrame(refinement).to_csv(OUT/'refinement_diagnostics.csv',index=False)
    fresh=[]
    for filename in ['inference_receipt.json','additional_pilot_receipt.json']:
        if (OUT/filename).exists():fresh+=read(OUT/filename).get('fresh_image_pilots',[])
    if fresh:
        ff=pd.DataFrame(fresh);fs=ff.groupby('variant').agg(pilots=('seconds','size'),mean_seconds=('seconds','mean'),
            median_seconds=('seconds','median'),max_seconds=('seconds','max'),max_parent_rss_gib=('parent_peak_rss_gib','max'),
            max_child_rss_gib=('child_peak_rss_gib','max'),new_head_peak_gpu_gib=('new_head_peak_gpu_gib','max')).reset_index()
        fs['linear_199_clip_hours_estimate']=fs.mean_seconds*199/3600;fs.to_csv(OUT/'fresh_runtime_summary.csv',index=False)
    hours=sum(r['elapsed_seconds'] for r in receipts)/3600;updates=sum(r['actual_updates'] for r in receipts)
    candidate=pd.read_csv(OUT/'candidate_budget.csv')
    second=[read(p) for p in sorted((OUT/'secondary_prediction_receipts').glob('*/*.json'))]
    second += [read(p) for p in sorted((OUT/'rendering_prediction_receipts').glob('*/*.json'))]
    if second:
        candidate=pd.concat([candidate,pd.DataFrame([dict(dataset=r['dataset'],variant=r['variant'],**r['candidate_budget']) for r in second])],ignore_index=True)
        candidate=candidate.drop_duplicates(['dataset','variant'],keep='last');candidate.to_csv(OUT/'candidate_budget.csv',index=False)
    cs=candidate.groupby('variant').sum(numeric_only=True).reset_index()
    for stage in ['after_gate','after_margin','selected']:cs[stage+'_per_million_parents']=cs[stage]/cs.parent_observations.clip(lower=1)*1e6
    cs.to_csv(OUT/'candidate_budget_summary.csv',index=False)
    decision='Retain the v3 incumbent.' if selected=='C0' else f'Retain the measured candidate {selected}; selection remains exploratory.'
    lines=[f'# Multi-dataset training v4: measured transfer', '',decision,
        f'Selected score **{pool.score:.15f}**, delta **{pool.delta_v3:+.15f}** against v3 A_residual_m3.0 = **{BASE["pooled"]:.15f}**, on all 199 clips.',
        '', 'This study is operational exploratory. Both embryos have been repeatedly reused. The released simulator was calibrated on 44b6, including when 44b6 is the transfer target. The incumbent public checkpoints and historical E teachers retain inherited label exposure. No local delta is a leaderboard gain.',
        '', '## Measured whole-graph comparisons', '',
        '| Variant | Pooled score | Delta v3 | Delta C1 (same seed) | 44b6 delta | 6bba delta | Division TP/FP/FN |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in scores[scores.embryo=='pooled'].sort_values('score',ascending=False).itertuples():
        er=scores[scores.variant==r.variant].set_index('embryo')
        lines.append(f'| {r.variant} | {r.score:.12f} | {r.delta_v3:+.12f} | {r.delta_vs_C1:+.12f} | {er.loc["44b6","delta_v3"]:+.12f} | {er.loc["6bba","delta_v3"]:+.12f} | {int(r.division_tp)}/{int(r.division_fp)}/{int(r.division_fn)} |')
    lines += measured_observations(scores)
    lines += ['', '## Actual training and source coverage', '',
        f'Training completed **{updates:,} optimizer updates** across {len(receipts)} saved production fits, taking {hours:.3f} summed hours in timed optimizer/validation loops on the RTX 4090. Initial cache loading and separate sanity fits are outside that sum; it is not a measure of continuously saturated GPU compute. Peak training allocation was {max(r["peak_gpu_gib"] for r in receipts):.3f} GiB.',
        'D uses native static-image center queries and subvoxel offsets. I uses a shared triplanar frame encoder, explicit masked temporal pooling, and separate parent and daughter evidence. G uses only relative per-axis-normalized point geometry with label-blind nearest-frame context; clean links and clone IDs never enter the geometry input. Primary C1–C6 never use Zoo as image input; optional C7 uses only explicitly rendered Zoo images.',
        'The fixed component budgets are D 12,000, G 20,000, and I 20,000 pretraining updates, then 8,000 adaptation updates per applicable component and source. C1 repeats source Biohub for the full matched budget. C1short receives only G/I adaptation from random initialization. Reused pretraining checkpoints are accounted for once in storage and process totals; see individual histories for each arm.',
        'Adaptation batches contain 75% identically sampled supported source observations and 25% external replay. In real-only controls, the remaining 25% performs source-only consistency without extra ground-truth loss, matching primary real supervision and optimizer steps. Exact sampled rows and visited groups are recorded; alternatives are not counted as independent events.',
        'Non-fork event targets require two recorded single-child transitions on each side of the anchor. This is conditional annotation support: sparse graphs cannot certify absence of every biological daughter. Quiet-chain event labels remain a weak-supervision assumption even after unsupported edge negatives are masked. Event-bank sampling excludes censored/unreachable bags; their usable partial edges are not separately exploited.',
        'An initial real-only detector adaptation diverged under raw-logit consistency and was stopped after its 5,000-update checkpoint. It is preserved under failed_fits and excluded from production comparisons. All real-only adaptations restarted with bounded probability consistency before comparative predictions. Real-only consistency also performs a detached teacher forward; optimizer steps and supervised rows match, but exact FLOPs differ.',
        'A subsequent pre-outcome audit found unsupported edge negatives: a parent with one recorded child incorrectly contradicted unmatched possible second daughters. Those labels were changed to unknown for Biohub and weak Zoo supervision. All affected G/I pretraining and adaptations, including the early second-seed real-only controls, were preserved and restarted from their original initializations for the full budgets. Event targets, input features, partitions and sampling were unchanged. Dense synthetic pretraining and D fits were unaffected. The correction and failed compute accounting are separate receipts.',
        'D-to-I transfer copies only the compatible convolutional encoder. Native D patches and isotropically sampled I patches have different physical support; this is an explicit scale-transfer limitation. The compact D experiment refines existing center proposals; it is not a replacement full-volume U-Net or a new-peak recall experiment.',
        'Real-only detector supervision includes 1,893 center queries and 8,781 image-validated dark-background queries for 44b6, and 9,783 center queries and 17,446 background queries for 6bba; every clip supplies background. Unmatched detections are not labeled negative. Dark background remains heuristic supervision, so very dim unannotated cells cannot be ruled out.',
        '', '| Source | Partition | Examples/blocks | Event bags | Reachable positive groups |', '|---|---|---:|---:|---:|']
    for r in coverage.itertuples():lines.append(f'| {r.source} | {r.partition} | {int(r.samples)} | {int(r.bags)} | {int(r.positive_groups)} |')
    lines += ['', 'Synthetic train/holdout identities are preserved. Generator validation and test are engineering diagnostics. Zoo train/validation/test use contiguous acquisition time blocks with six-frame purges and are not independent embryos; their coordinate IQR normalization uses the whole unlabeled acquisition, including held-out blocks. Only the eligible ascidian acquisition supplements zebrafish, capped at 25% of geometry pretraining/replay batches; other species and all RIKEN acquisitions remain excluded as individually recorded in dataset_use.csv.',
        '', '## Decoder, calibration and attribution', '',
        'The installed raw solver was tested with the original 1.2 division cost and a parent-specific learned event term: zero gain selected no fork, positive gain selected a true fork, and the nondivision control retained a continuation. Deployment uses the tested local binary event objective, explicit no-op and conflicting-parent/daughter/owner constraints. Existing fork evidence is protected through two generations. A blanket lower division penalty is not used.',
        'Each parent has at most six daughter candidates and 15 unordered pair alternatives. The geometry gate is fixed at 0.5. A pair must beat the log-sum-exp of competing pairs and no-fork by margin 4.0 after temperature scaling. C4_m2 and C4_m6 are the handover-prescribed descriptive margin-2 and margin-6 controls, both wired in before prediction freezing. The original-objective and zero-new-head decoder controls preserve v3.',
        'The 199-clip original-objective negative control is explicitly a structural no-op on the frozen v3 graph. The installed raw solver was executed in the positive/negative fixtures; it was not rerun globally for every identity-control clip.',
        'Temperature is fit on unbalanced observed generator validation bags, using the same calibration access for every arm. Thus “real-only” describes neural weight training; it does not exclude shared external calibration. This is not a biological posterior. Independent source inner groups could not be certified; the fallback primary margin was fixed before new target outcomes. No prior multiplier or target-label threshold search was used.',
        'Primary C1–C6 modify fixed incumbent nodes only. C1_relinked/C4_relinked rebuild continuations from current-point geometry. D_real and D_synthetic refine center coordinates and rebuild all v4 proposals/associations under C1; D_synthetic_C4 is the limited combination. Count and duplicate diagnostics are reported separately from tracking score. No old residual feature array is reused after moving centers.',
        'Final adoption additionally requires a complete qualifying second-seed replication. A positive exploratory variant without that replication is retained in the measured tables but does not replace v3. The second seed is never selected instead of the primary seed.',
        f'External-data advantage for the selected result: **{external_advantage}**. delta_vs_C1 uses the control from the same seed; a verified external advantage requires positive differences in both seeds and both embryos. Gains caused by architecture, compute or decoding alone are not credited to external data.',
        '', '## Validation, scope and artifacts', '',
        'All 41,468 archived/prepared files passed fresh hashes. All 3,713 synthetic raw/prepared grids were checked. Actual source overfits passed for G/I/D, including encoder gradients, unordered daughter and padding invariance, future censoring, and actual solver fixtures. Thirteen additional contract tests check empty observations, detector displacement units and rounding, sparse unknowns and possible unannotated second daughters, missing future frames, unit scaling, duplicate identity, and conflicting/protected forks, including displaced old children.',
        'The six generator stress conditions alter point observations while retaining the original optical frames. Short history tests missing tracking history, and clipped daughters test missing candidate detections. Stress anchors are known simulated cells; robustness to false-positive parent detections is not established by these diagnostics.',
        'Predictions for both directions and every primary comparison were serialized before comparative official scoring. Every complete graph was validated and scored using the pinned official metric, revision 075fc5f5a52d11077f9dc2b074644618f26939e2; independent aggregation checks passed.',
        'W470 trajectory-backed rendering is conditional on a predeclared external-utility/real-transfer-gap/time test after the required comparisons. The rendering_trial_receipt.json states whether it ran. C7, when measured, replaces half of optical pretraining and replay with Gaussian triplanes rendered from eligible Zoo-fish points using source-only appearance and density statistics. It retains C4 geometry histories and full matched budgets. These are explicitly simulated patches from weak trajectories, not experimental Zoo microscopy; existing acquisition and simulator exposure caveats persist.',
        'Fresh-image/package, second-seed, preservation, and resource receipts state their actual completion independently. The local 4090 runtime is not a verified Kaggle 12-hour guarantee. No submission, notebook publication, gated agreement, new hardware, or paid API was initiated.',
        'The container rejected the Linux user/mount/network namespace probe with Operation not permitted. Offline fresh inference therefore uses process-start Python audit hooks denying annotation and external-training paths and nonlocal sockets. Those datasets remain mounted but are inaccessible through the audited reads; this is dependency-use evidence, not an OS isolation or physical-unmount claim. Local IPC remains allowed.',
        '', 'Local models, detailed predictions, matched identities and image crops remain under /kaggle/working/cell-tracking/multidata-training-v4/. Sanitized score rows, source ledger, learning/transfer curves, calibration caveats, model hashes and this report are exported to results/multidata-training-v4/. Git is not a backup of microscopy or weights.',
        '', 'Reproduction: see docs/multidata-training-v4.md. Inference: see the local inference_package/README.md and inference_receipt.json.']
    if (OUT/'head_parity_receipt.json').exists():
        parity=read(OUT/'head_parity_receipt.json')
        lines += ['', f'The individual-graph C4 deployment path was replayed against the primary shared-patch path on {parity["samples"]} clips. Exact whole-graph parity passed: **{parity["passed"]}**. No annotations or new score tuning entered this check.']
    if (OUT/'candidate_gt_coverage.csv').exists():
        c=pd.read_csv(OUT/'candidate_gt_coverage.csv');c=c[c.variant=='C4'].sum(numeric_only=True)
        lines += ['', '## Observed candidate attrition', '',
            f'For primary C4, {int(c["parent_matched"])}/{int(c["observed_gt_forks"])} observed GT fork parents matched incumbent detections; '
            f'{int(c["both_daughters_matched"])} also had both daughters matched and {int(c["within_six_cap"])} fit the six-candidate, exact-next-frame proposal set. '
            f'The fixed gate retained {int(c["after_gate"])} of those exact-ID forks, and {int(c["after_margin"])} exact daughter pairs passed the margin. '
            f'{int(c["after_decoder_new_edits"])} were newly accepted exact-ID edits after incumbent protection and conflict resolution. '
            'This strict immediate-ID diagnostic differs from the official temporal-tolerance division matching, so it is not an alternative official TP count or a biological recall estimate. '
            'The full stage counts and control comparisons are in candidate_gt_coverage.csv; they were computed after freezing and were not used to change the gate or threshold.']
    if fresh:
        lines += ['', '## Fresh-image delivery measurements', '',
            f'{len(fresh)} fresh image-to-graph executions on {len({r["dataset"] for r in fresh})} fixed clips completed with exact scored-graph parity and integer CSV roundtrips. The external C4 heads were actually loaded; C0 exercised the disabled-head fallback. Additional clips were selected from frozen incumbent density, without labels or scores.']
        for r in fs.itertuples():
            lines.append(f'{r.variant}: {r.pilots} clips, mean {r.mean_seconds:.1f} seconds, maximum {r.max_seconds:.1f} seconds; '
                f'a simple 199-clip extrapolation is {r.linear_199_clip_hours_estimate:.2f} hours on this 4090. '
                f'Peak reported parent/child RSS was {r.max_parent_rss_gib:.2f}/{r.max_child_rss_gib:.2f} GiB; '
                f'the new-head GPU allocation peaked at {r.new_head_peak_gpu_gib:.3f} GiB. '
                'The RSS figures are per-process maxima, and the new-head allocation excludes upstream detector allocations. '
                'The small density-selected runtime sample does not certify all-clip or Kaggle runtime.')
    if (OUT/'failed_fit_accounting.json').exists():
        failed=read(OUT/'failed_fit_accounting.json')
        lines += ['', f'Preserved failed/superseded checkpoints additionally account for {failed["checkpointed_updates"]:,} saved optimizer updates and {failed["timed_seconds"]/3600:.3f} summed timed hours. Unsaved work may add to that total; sanity updates are separate. These weights never enter the reported comparisons.']
    if (OUT/'rendering_geometry_drift.json').exists():
        drift=read(OUT/'rendering_geometry_drift.json')
        lines += ['', f'C7 geometry-history parity was checked before its target outcomes: all {len(drift)} source fits have configurations identical to same-seed C4 except their names; bitwise weight equality holds for {sum(r["bitwise_equal"] for r in drift)} fits. Maximum relative parameter L2 difference was {max(r["relative_parameter_l2"] for r in drift):.3g}. The intended treatment changes the optical data.']
    if (OUT/'unknown_name_pilot_receipt.json').exists():
        alias=read(OUT/'unknown_name_pilot_receipt.json')
        lines += ['', f'An additional fresh C4 run used the same fixed image under the unfamiliar name {alias["alias"]}, with source model {alias["source_model"]} supplied explicitly. Complete graph parity and CSV renaming passed: **{alias["passed"]}**, in {alias["seconds"]:.1f} seconds. This is a deployment-routing test, not a new biological sample.']
    if (OUT/'rendered_holdout_diagnostics.csv').exists():
        held=pd.read_csv(OUT/'rendered_holdout_diagnostics.csv')
        lines += ['', 'A post-outcome check evaluated the frozen C4/C7 optical towers on the rendered Zoo holdouts at temperature 1, without updates or calibration changes.']
        for source in ['44b6','6bba']:
            h=held[(held.source==source)&(held.partition=='test')].set_index('arm')
            lines.append(f'Source {source}: rendered-test correct-pair recall was '
                f'{h.loc["C4","positive_pair_correct"]/h.loc["C4","positive"]:.3f} for C4 and '
                f'{h.loc["C7","positive_pair_correct"]/h.loc["C7","positive"]:.3f} for C7 '
                f'on {int(h.loc["C7","positive"])} observed positive bags. Wrong-pair fractions among all observed bags were '
                f'{h.loc["C4","false_pair"]/h.loc["C4","rows"]:.3f} for C4 and {h.loc["C7","false_pair"]/h.loc["C7","rows"]:.3f} for C7. '
                'These are weak, conditionally sampled labels from time blocks of the same acquisition; they do not establish biological generalization.')
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'study_summary.json',dict(selected=config,actual_updates=updates,training_wall_hours=hours,completed_graph_variants=len(scores.variant.unique()),
        score_rows=len(rows),source_records=len(ledger),external_advantage=external_advantage,all_scores_official=True))
    dashboard(scores,train,ledger,cs,config,receipts)
    print('measured decision',selected,pool.score,flush=True)

def dashboard(scores,train,ledger,budget,config,receipts):
    def records(df):return json.loads(df.to_json(orient='records'))
    payload=dict(scores=records(scores),training=train,sources=records(ledger),budget=records(budget),config=config,
        curves={r['config']['name']:[dict(step=h['step'],loss=h['loss'],validation=h.get('validation')) for h in r['history']] for r in receipts},
        stress=records(pd.read_csv(OUT/'stress_diagnostics.csv')) if (OUT/'stress_diagnostics.csv').exists() else [])
    payload['calibration']={}
    payload['rendered_holdout']=records(pd.read_csv(OUT/'rendered_holdout_diagnostics.csv')) if (OUT/'rendered_holdout_diagnostics.csv').exists() else []
    payload['validation_domains']={r['config']['name']:('synthetic' if 'synthetic' in r['direct_sources'] else 'zebrafish' if 'zebrafish' in r['direct_sources'] else None) for r in receipts}
    for name in ['calibration.json','calibration_secondary.json','calibration_rendering.json']:
        if (OUT/name).exists():payload['calibration'].update(read(OUT/name)['models'])
    encoded=json.dumps(payload,default=lambda x:None if pd.isna(x) else x).replace('</','<\\/')
    template=(Path(__file__).parent/'dashboard.html').read_text()
    (OUT/'dashboard.html').write_text(template.replace('__DATA__',encoded))
