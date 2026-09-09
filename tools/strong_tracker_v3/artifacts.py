"""Final local dependency manifest and explicit sanitized publication allowlist."""
from pathlib import Path
import shutil
import zipfile
from .common import *

SAFE_JSON=['census_endpoint_summary.json','heatmap_rescue_summary.json','rescue_quantization_diagnostic.json',
    'teacher_reconciliation_summary.json','raw_decoder_trace.json','event_candidate_coverage.json',
    'source_association_oracle_score_summary.json','source_event_oracle_summary.json',
    'conditional_ablation_receipt.json','fresh_receipt.json','fresh_startup_benchmark.json',
    'fresh_pilot_precision_audit.json','fresh_input_resume_validation.json',
    'fresh_teacher_pilots_receipt.json',
    'dashboard_validation.json','association_source_bias_summary.json','association_summary.json',
    'rescue_fixed_summary.json','event_decode_summary.json','event_grid_lock.json',
    'association_regret_summary.json','source_oracle_division_regret_summary.json',
    'fresh_full_precision_audit.json','fresh_selected_score_summary.json',
    'source_sampling_metadata_audit.json','event_correctness_audit.json','event_table_probe.json','runtime_versions.json',
    'fresh_heatmap_optimization_parity.json','cpu_affinity_receipt.json',
    'event_dense_decode_profile.json','event_table_manifest.json','event_decode_resources.json',
    'event_lane_findings.json','fresh_teacher_native_parity_summary.json',
    'event_io_loader_parity.json','event_io_loader_lock.json','event_io_handoff_runtime.json',
    'event_score_disagreement.json','event_table_schema_audit.json',
    'candidate_manifest.json','round_lock.json','ledger_delivery_summary.json',
    'event_measured_summary.json','source_event_oracle_manifest.json','source_event_oracle_measured.json',
    'combination_worker_handoff.json','combination_tail_helper_receipt.json',
    'resource_monitor_revision.json','package_smoke_receipt.json',
    'fresh_numeric_validation_receipt.json','fresh_delivery_validation_receipt.json',
    'export_scoring_manifest.json','ledger_delivery_validation.json']

def run(ctx):
    public=ctx.repo/'results/strong-tracker-v3';public.mkdir(parents=True,exist_ok=True)
    for name in SAFE_JSON:
        path=ctx.out/name
        if path.exists():shutil.copyfile(path,public/name)
    p=ctx.out/'event_model_lock.json'
    if p.exists():
        lock=read_json(p);models={}
        for k,r in lock['models'].items():
            models[k]={key:value for key,value in r.items() if key!='source_inputs'}
        write_json(public/'event_training_summary.json',dict(models=models,source_pools=lock['source_pools'],
            total_elapsed_seconds=lock['total_elapsed_seconds']))
    p=ctx.out/'association_model_lock.json'
    if p.exists():
        d=read_json(p)
        write_json(public/'association_training_summary.json',{k:v for k,v in d.items() if k not in ['models']})
    p=ctx.out/'label_free_input_hash_manifest.json'
    if p.exists():shutil.copyfile(p,public/p.name)
    p=ctx.out/'inference_package/manifest.json'
    if p.exists():
        shutil.copyfile(p,public/'inference_package_manifest.json')
        package=p.parent;manifest=read_json(p)
        archive=ctx.out/'selected_inference_package.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
            for relative in sorted([*manifest['package_files'],'manifest.json','README.md']):
                output.write(package/relative,arcname='inference_package/'+relative)
        archive_record=dict(variant=manifest['variant'],path=str(archive),bytes=archive.stat().st_size,
            sha256=sha(archive),package_manifest_sha256=sha(p),
            scope='Selected code and repair weights; upstream source/checkpoints remain explicit local dependencies',
            annotations_included=False,cached_predictions_included=False,external_backup_created=False)
        write_json(ctx.out/'inference_archive.json',archive_record)
        write_json(public/'inference_archive.json',archive_record)
    # Include the full exact selected lock (hashes/counts only, no coordinates).
    code={str(p.relative_to(ctx.repo)):sha(p) for root in [ctx.repo/'tools/strong_tracker_v3',ctx.repo/'tests/strong_tracker_v3']
        for p in sorted(root.glob('*.py'))}
    for p in [ctx.repo/'scripts/run_strong_tracker_v3.sh',
              ctx.repo/'tools/strong_tracker_v3/dashboard_template.html',ctx.repo/'docs/strong-tracker-v3.md']:
        code[str(p.relative_to(ctx.repo))]=sha(p)
    dependencies=dict(
        data=dict(root=str(ctx.data),content_manifest_sha256=sha(ctx.out/'label_free_input_hash_manifest.json')),
        v1=dict(root=str(ctx.v1),required=['inventory.json (evaluation only)','evaluation/gt (evaluation only)',
            'baseline/public (teacher validation)','public_harmonic_full: source, upstream checkpoints, pre-ILP raw inputs']),
        v2=dict(root=str(ctx.v2),required=['selected_predictions','selected_prediction_lock.json','raw','heatmaps',
            'models and model_lock.json','predictions (teacher validation)','replay/no_motion']),
        v3=dict(root=str(ctx.out),required=['All model/candidate/feature/input fingerprints','events','event_models',
            'association_models','features','candidate_graphs','selected_predictions','selected_prediction_lock.json',
            'evaluation (reproduction only, excluded from inference)','fresh_full','fresh_evidence','fresh_candidates',
            'edit_ledger.parquet','event_features.parquet','inference_package','logs']),
        metric=dict(root=str(ctx.official),revision=ctx.metric_revision),
        environment='Existing CUDA study interpreter; environment.json records exact software and paths')
    write_json(ctx.out/'artifact_manifest.json',dict(created=now(),code=code,dependencies=dependencies,
        preservation_receipt_sha256=sha(ctx.out/'preservation_check.json'),
        public_files={p.name:sha(p) for p in sorted(public.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'},
        local_storage_note='Instance disk only; no external backup or persistence was created or promised'))
    shutil.copyfile(ctx.out/'artifact_manifest.json',public/'artifact_manifest.json')
    return dependencies
