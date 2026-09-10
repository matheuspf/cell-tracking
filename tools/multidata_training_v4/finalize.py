"""Measured completion, preserved inputs, and explicit sanitized result export."""
import csv
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from .common import *

EXPORT_FILES=[
    'dataset_use.csv','source_manifest.json','source_web_receipt.json','protocol.json','execution_contract.json',
    'implementation_corrections.json','adaptation_correction.json','training_concurrency.json','equivalent_history_drift.json',
    'sparse_edge_mask_correction.json','sparse_edge_mask_tests.json','sparse_edge_mask_retraining_receipt.json','repair_checks_receipt.json','failed_fit_accounting.json',
    'rendering_followup_lock.json','rendering_plan.json','rendering_trial_receipt.json','rendering_progress.json','rendering_sanity.json','rendering_numeric_fixture.json','rendering_geometry_drift.json',
    'rendered_zoo_44b6_receipt.json','rendered_zoo_6bba_receipt.json','rendered_dataset_index.json','rendered_source_coverage.csv','rendered_holdout_diagnostics.csv','rendered_holdout_receipt.json',
    'calibration_rendering.json','calibration_rendering_C7.json','calibration_rendering_C7_seed2.json',
    'checkpoint_manifest_rendering.json','checkpoint_manifest_rendering_C7.json','checkpoint_manifest_rendering_C7_seed2.json',
    'rendering_inference_config_C7.json','rendering_inference_config_C7_seed2.json','rendering_prediction_receipt_C7.json','rendering_prediction_receipt_C7_seed2.json',
    'index_metadata_correction.json','partition_audit.json',
    'authorized_run_window.json','followup_policy.json','preprediction_protocol_completion.json','numerical_promotion_tolerance.json','seed2_bootstrap_lock.json','seed2_bootstrap_receipt.json',
    'baseline_verification.json','source_coverage.csv',
    'adapter_audit.json','native_metadata_audit.json','biohub_metadata_audit.json','sanity_tests.json','source_read_audit.json','decoder_window_audit.json',
    'inference_contract_tests.json','offline_guard_test.json','rendered_guard_probe.json','isolation_probe.json','serialization_probe.json','test_receipt.json',
    'head_parity_lock.json','head_parity_receipt.json',
    'training_receipts.json','training_summary.csv','arm_training_histories.csv','learning_curves.csv','calibration.json',
    'calibration_secondary.json','transfer_diagnostics.csv','stress_lock.json','stress_fixture_summary.csv','stress_fixture_receipt.json',
    'stress_diagnostics.csv','stress_receipt.json','queued_stress.json','ablation_scores.csv','score_rows.csv',
    'candidate_budget.csv','candidate_budget_summary.csv','candidate_gt_coverage.csv','candidate_coverage_receipt.json',
    'detection_diagnostics.csv','refinement_diagnostics.csv','detector_supervision_coverage.json','detector_attribution.csv','edge_regret.csv','final_report.md','dashboard.html','dashboard_validation.json',
    'status.json','study_summary.json','winning_config.json','checkpoint_manifest.json','checkpoint_manifest_secondary.json',
    'inference_config.json','prediction_generation_receipt.json','secondary_plan.json','secondary_progress.json',
    'secondary_verification.json','secondary_prediction_receipt.json','secondary_inference_config.json','secondary_training_receipt.json','scheduled_tests.json','scheduled_tests_receipt.json',
    'inference_receipt.json','inference_package_manifest.json','inference_archive.json','archive_check.json','fresh_runtime_summary.csv','additional_pilot_lock.json',
    'additional_pilot_receipt.json','additional_pilot_schedule.json','unknown_name_pilot_receipt.json','preservation_check.json','runtime_versions.json','runtime_summary.json',
    'score_completeness.json','frozen_inputs_check.json','inference_runtime_profile.json','artifact_manifest.json','NEXT_AGENT.md',
]

def frozen_inputs():
    import zipfile
    models={}
    for p in sorted(OUT.glob('checkpoint_manifest*.json')):
        for name,record in read(p).items():
            if name in models:assert models[name]['sha256']==record['sha256'],name
            models[name]=record
    for name,record in models.items():assert sha(OUT/'models'/f'{name}.pt')==record['sha256'],name
    configs=[('inference_config.json','calibration.json'),('secondary_inference_config.json','calibration_secondary.json')]
    configs += [(p.name,'calibration_rendering_'+p.stem.removeprefix('rendering_inference_config_')+'.json')
        for p in sorted(OUT.glob('rendering_inference_config_*.json'))]
    checked_code=set()
    for config,calibration in configs:
        lock=read(OUT/config);assert sha(OUT/calibration)==lock['calibration_sha256'],calibration
        for name,digest in lock['inference_code_sha256'].items():
            assert sha(REPO/'tools/multidata_training_v4'/name)==digest,(config,name)
            checked_code.add(name)
        if 'driver_code_sha256' in lock:
            assert sha(REPO/'tools/multidata_training_v4/rendering_trial.py')==lock['driver_code_sha256']
        if 'checkpoint_manifest_sha256' in lock:
            variant=lock['variant'];assert sha(OUT/f'checkpoint_manifest_rendering_{variant}.json')==lock['checkpoint_manifest_sha256']
    graphs=read(OUT/'prediction_lock.json')['graphs'];count=0
    for variant,names in graphs.items():
        root=V3/'selected_predictions' if variant=='C0' else OUT/'predictions'/variant
        for name,digest in names.items():assert sha(root/f'{name}.npz')==digest;count+=1
    write(OUT/'frozen_inputs_check.json',dict(passed=True,created=now(),checkpoint_files=len(models),
        graph_files=count,calibration_locks=len(configs),core_code_files=sorted(checked_code),
        all_frozen_predictions_weights_calibration_and_inference_code_unchanged=True))
    archive=read(OUT/'inference_archive.json');path=Path(archive['path'])
    assert sha(path)==archive['sha256'] and path.stat().st_size==archive['bytes']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,digest in read(OUT/'inference_package/manifest.json')['package_files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        entries=len(z.infolist())
    write(OUT/'archive_check.json',dict(passed=True,created=now(),zip_entries=entries,
        zip_sha256=archive['sha256'],crc_and_packaged_file_hashes_verified=True))

def histories():
    receipts={p.stem:read(p) for p in (OUT/'models').glob('*.json')};rows=[]
    def ancestors(name):
        if not name:return []
        return ancestors(receipts[name]['config']['init'])+[name]
    for name,r in receipts.items():
        if not name.startswith('G_C'):continue
        comp,rest=name.split('_',1);arm,source=rest.rsplit('_',1)
        image='I_'+rest
        if image not in receipts:continue  # Unfinished optional fits never form a reported arm.
        lineage=list(dict.fromkeys(ancestors(name)+ancestors(image)))
        rr=[receipts[x] for x in lineage]
        rows.append(dict(arm=arm,source=source,target='6bba' if source=='44b6' else '44b6',seed=r['config']['seed'],
            lineage=';'.join(lineage),effective_optimizer_updates=sum(x['actual_updates'] for x in rr),
            D_updates=sum(x['actual_updates'] for x in rr if x['config']['component']=='D'),
            G_updates=sum(x['actual_updates'] for x in rr if x['config']['component']=='G'),
            I_updates=sum(x['actual_updates'] for x in rr if x['config']['component']=='I'),
            G_final_source_supervised_rows=r['sparse_source_supervised_rows'],
            I_final_source_supervised_rows=receipts[image]['sparse_source_supervised_rows'],
            final_G_parameters=r['trainable_parameters'],final_I_parameters=receipts[image]['trainable_parameters'],
            direct_sources=','.join(sorted({d for x in rr for d in x['direct_sources']}))))
    import pandas as pd
    df=pd.DataFrame(rows);df.to_csv(OUT/'arm_training_histories.csv',index=False)
    full=df[~df.arm.str.startswith('C1short')]
    assert (full.effective_optimizer_updates==68000).all()
    assert (full.G_final_source_supervised_rows==384000).all()
    assert (full.I_final_source_supervised_rows==96000).all()
    assert full.final_G_parameters.nunique()==1 and full.final_I_parameters.nunique()==1

def preservation():
    from .sources import snapshot
    before=read(OUT/'preservation_before.json');after=snapshot();changes=[]
    for root,files in before.items():
        for path,stat in files.items():
            if after[root].get(path)!=stat:changes.append(dict(root=root,file=path,change='modified_or_missing'))
        for path in set(after[root])-set(files):changes.append(dict(root=root,file=path,change='added'))
    hashes=read(OUT/'source_manifest.json')['incumbent_hashes']
    assert all(sha(V3/'selected_predictions'/f'{n}.npz')==h for n,h in hashes.items())
    write(OUT/'preservation_check.json',dict(passed=not changes,created=now(),changes=changes,
        prior_files_checked={k:len(v) for k,v in before.items()},selected_v3_graphs_rehashed=len(hashes),
        raw_inputs_opened_read_only=True,prior_roots=list(before)))
    assert not changes,changes[:10]

def runtime():
    import torch
    failed=[]
    for path in sorted((OUT/'failed_fits').rglob('*.pt')):
        if path.name.endswith('.tmp.pt'):continue
        state=torch.load(path,map_location='cpu',weights_only=False)
        failed.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),model=state['config']['name'],
            checkpointed_updates=state['step'],timed_seconds=state['elapsed'],excluded_from_comparative_models=True))
        del state
    write(OUT/'failed_fit_accounting.json',dict(created=now(),fits=failed,
        checkpointed_updates=sum(r['checkpointed_updates'] for r in failed),
        timed_seconds=sum(r['timed_seconds'] for r in failed),
        additional_unsaved_steps_may_have_run=True,sanity_updates_separate=True))
    versions={}
    for package in ['numpy','scipy','pandas','torch','torchvision','zarr','tracksdata','polars','scikit-learn','playwright']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    write(OUT/'runtime_versions.json',dict(created=now(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),packages=versions,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()))
    samples=[json.loads(s) for s in (OUT/'resource_samples.jsonl').read_text().splitlines()]
    receipts=[read(p) for p in (OUT/'models').glob('*.json')]
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
    write(OUT/'runtime_summary.json',dict(created=now(),resource_observations=len(samples),sampling_interval_seconds=30,
        monitoring_began_after_initial_pretraining=True,peak_sampled_gpu_gib=max(r['gpu_memory_mib'] for r in samples)/1024,
        peak_sampled_process_rss_sum_gib=max(r['total_observed_process_rss_gib'] for r in samples),
        peak_training_allocated_gpu_gib=max(r['peak_gpu_gib'] for r in receipts),
        peak_training_process_rss_gib=max(r['peak_process_rss_gib'] for r in receipts),
        training_wall_seconds=sum(r['elapsed_seconds'] for r in receipts),optimizer_updates=sum(r['actual_updates'] for r in receipts),
        local_output_bytes=size,free_disk_gib=shutil.disk_usage(OUT).free/2**30,
        total_budget_gpu_hours=36,user_wall_window=read(OUT/'authorized_run_window.json'),
        measured_4090_only=True,kaggle_12_hour_guarantee=False))

def tests():
    jobs=[]
    for directory,pattern in [('tests','test_multidata_v4.py'),('handover/multidata-training-v4','test_contracts.py')]:
        cmd=[sys.executable,'-m','unittest','discover','-s',directory,'-p',pattern]
        result=subprocess.run(cmd,cwd=REPO,text=True,capture_output=True)
        jobs.append(dict(command=cmd,returncode=result.returncode,output=result.stdout+result.stderr))
        assert result.returncode==0,jobs[-1]
    write(OUT/'test_receipt.json',dict(created=now(),passed=True,jobs=jobs))

def run():
    import pandas as pd
    from .report import run as report
    required=['inference_receipt.json','secondary_verification.json','stress_receipt.json','repair_checks_receipt.json','head_parity_receipt.json','unknown_name_pilot_receipt.json']
    if read(OUT/'rendering_trial_receipt.json').get('primary_measured'):required.append('rendered_holdout_receipt.json')
    for name in required:assert read(OUT/name)['completed'],name
    assert read(OUT/'head_parity_receipt.json')['passed']
    assert read(OUT/'unknown_name_pilot_receipt.json')['passed']
    frozen_inputs();histories();preservation();runtime();tests();report()
    assert read(OUT/'winning_config.json')==read(OUT/'inference_package/winning_config.json'),'Reported selection differs from the verified package'
    browser_python=os.environ.get('V4_BROWSER_PYTHON','/root/.conda/envs/cell-tracking/bin/python')
    subprocess.run([browser_python,'-m','multidata_training_v4','dashboard_check'],check=True,cwd=REPO)
    scores=pd.read_csv(OUT/'ablation_scores.csv');rows=pd.read_csv(OUT/'score_rows.csv');config=read(OUT/'winning_config.json')
    names=sorted(r['dataset'] for r in inputs())
    for variant,rr in rows.groupby('variant'):assert sorted(rr.dataset)==names
    assert (scores[scores.embryo=='pooled'].samples==199).all() and rows.variant.nunique()<=24
    write(OUT/'score_completeness.json',dict(passed=True,variants=rows.variant.nunique(),samples_per_variant=199,
        official_score_rows=len(rows),both_source_directions=True,current_baseline=BASE,
        round_locks={p.name:sha(p) for p in sorted((OUT/'round_locks').glob('*.json'))}))
    stages={code:dict(state='complete') for code in ['W400','W410','W420','W430','W440','W450','W460','W480','W490']}
    rendering=read(OUT/'rendering_trial_receipt.json')
    stages['W470']=dict(state='complete' if rendering['completed'] else 'not_run_optional' if rendering.get('optional_skipped') else 'partial_optional',receipt=rendering)
    status('W400-W490',state='completed',study='multidata-training-v4',stages=stages,selected=config['variant'],score=config['score'],
        delta_v3=config['delta_v3'],external_data_advantage=config['external_advantage_over_matched_control'],
        measured=True,operational_exploratory=True,submission_made=False,publication_made=False,
        local_inference_package_verified=True)
    summary=read(OUT/'study_summary.json');runtime_info=read(OUT/'runtime_summary.json')
    measured=scores[scores.embryo=='pooled'].set_index('variant')
    replicated_arm=read(OUT/'secondary_plan.json')['arm']
    rendering_result=(f"C7's pooled delta was {measured.loc['C7','delta_v3']:+.12f}." if 'C7' in measured.index else 'C7 has no complete target comparison.')
    next_agent=f'''# Multi-dataset training v4: completed measured execution

Selected **{config['variant']}**, pooled **{config['score']:.15f}**, delta **{config['delta_v3']:+.15f}** against v3 A_residual_m3.0 = 0.934802374260586.
Verified external-data advantage over matched real-only controls: **{config['external_advantage_over_matched_control']}**.

Read final_report.md, ablation_scores.csv, arm_training_histories.csv, dashboard.html and the inference/preservation receipts. There are {summary['completed_graph_variants']} complete variants and {summary['score_rows']} freshly pinned official score rows. Actual training completed {summary['actual_updates']:,} optimizer updates in {summary['training_wall_hours']:.3f} summed fit wall hours on GPU 0, RTX 4090.

Primary C4's pooled delta was {measured.loc['C4','delta_v3']:+.12f}; the repeated {replicated_arm} arm's second-seed delta was {measured.loc[replicated_arm+'_seed2','delta_v3']:+.12f}. Same-seed full C1 controls had deltas {measured.loc['C1','delta_v3']:+.12f} and {measured.loc['C1_seed2','delta_v3']:+.12f}. {rendering_result} Per-embryo failures and replication determine adoption; a positive pooled row alone is insufficient. The report separates exact-ID candidate attrition from the official temporal-tolerance division counts and coordinate refinement from the cost of rebuilding associations.

W400–W460 and W480–W490 were executed. W470 status is **{stages['W470']['state']}**; its conditional treatment and any measurements are in rendering_trial_receipt.json. Native static images, six-frame paired synthetic movies, Zoo-fish geometry and one eligible ascidian acquisition entered actual parameter updates. Other species and all seven RIKEN collections remain excluded with reasons in dataset_use.csv. D is a compact center/refinement experiment; it does not generate a replacement dense detector population.

Both Biohub source directions have separate sparse fits. Full C1 controls match D/G/I update budgets, initialization architecture and final supervised-source sampling. C1short is intentionally shorter. All calibration temperatures use common external validation access; real-only describes neural weight training. Source-44b6 simulator calibration and public/historical teacher exposure persist. The repeated local embryos and Zoo time blocks do not establish independent biological generalization.

Do not rerun or mutate sealed v1–v3 or alter frozen v4 checkpoint/prediction locks. Scientific changes need another output namespace. Detailed inputs, crops, weights, graphs and matching identities remain local at `{OUT}`; Git contains sanitized measurements and code. The local inference package is `{OUT}/inference_package`, with transfer archive selected_inference_package.zip. Use its README and run.sh with explicit image, output, v1, v2, source-model and Python paths. `--disable-new-heads` is verified v3 identity. New weights load in the separate C4 fresh-image pilots even when C0 is selected.

Preserve the existing v1/v2 neural/source artifacts, v3 package and original DeepCenter checkpoint, plus the new models, source caches and archive, before destroying this container. Dependencies are explicit in the base package README and runtime_versions.json. Local fresh-image tests do not certify Kaggle runtime; no Kaggle submission, forum/notebook publication, remote microscopy upload, paid API or new hardware was used.

Reproduction commands are maintained in docs/multidata-training-v4.md. The user's seven-hour window ends at 2026-09-10 06:09:09 UTC. Required work and the actual follow-up tests have their own completion receipts. At final measurement, outputs occupied {runtime_info['local_output_bytes']/2**30:.3f} GiB; free disk was {runtime_info['free_disk_gib']:.3f} GiB.
'''
    (OUT/'NEXT_AGENT.md').write_text(next_agent)
    files={name:dict(sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in EXPORT_FILES if name!='artifact_manifest.json' and (OUT/name).exists()}
    local={name:dict(sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in ['runtime_dataset_index.jsonl','prediction_lock.json','selected_inference_package.zip']}
    write(OUT/'artifact_manifest.json',dict(created=now(),sanitized=files,local_only=local,
        code={str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'tools/multidata_training_v4').glob('*')) if p.is_file()},
        external_microscopy_or_weights_exported=False))
    target=REPO/'results/multidata-training-v4';target.mkdir(parents=True,exist_ok=True)
    for name in EXPORT_FILES:
        if (OUT/name).exists():shutil.copyfile(OUT/name,target/name)
    # Preserve authoring history while marking this handover's actual execution.
    handover=read(REPO/'handover/multidata-training-v4/STATUS.json')
    handover.update(status='executed_measured',next_stage=None,completed_at=now(),new_score=config['score'],
        delta_v3=config['delta_v3'],external_data_gain=config['external_advantage_over_matched_control'],
        results='results/multidata-training-v4/final_report.md',selected_variant=config['variant'],
        execution_validation=dict(competition_data_read=True,external_raw_data_read=True,training_run=True,
            official_graph_evaluation_run=True,complete_variants=summary['completed_graph_variants'],
            actual_optimizer_updates=summary['actual_updates'],local_inference_package_verified=True))
    write(REPO/'handover/multidata-training-v4/STATUS.json',handover)
    readme=REPO/'handover/multidata-training-v4/README.md'
    old_status='**Status: execution handover, not measured v4 results.**'
    measured_status=('**Status: executed and measured.** See the '
        '[measured report](../../results/multidata-training-v4/final_report.md), '
        '[interactive dashboard](../../results/multidata-training-v4/dashboard.html), and '
        '[NEXT_AGENT](../../results/multidata-training-v4/NEXT_AGENT.md). '
        'The remaining text preserves the original execution handover. '
        'MANIFEST.json remains the original authoring snapshot; measured artifacts have their own manifest.')
    if old_status in readme.read_text():readme.write_text(readme.read_text().replace(old_status,measured_status,1))
    print('Sanitized measured findings exported',target,flush=True)
