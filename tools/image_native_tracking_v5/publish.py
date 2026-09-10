"""Copy an explicit sanitized result allowlist into Git; never stage or push automatically."""
import shutil
from .common import *

ALLOW=['status.json','ablation_scores.csv','score_rows.csv','family_outcomes.csv','coverage.csv',
    'training_summary.csv','learning_curves.csv','exposure.json','score_budget.json','selection.json',
    'execution_protocol.json','input_manifest.json','native_architecture.json','C0_fresh_receipt.json',
    'native_original_probability_parity.json','fresh_observation_parity.json','resume_determinism.json',
    'dashboard.html','dashboard_validation.json','progress_report.md','final_report.md',
    'regret_rows.csv','score_decomposition.csv','model_manifest.json','inference_dependency_manifest.json',
    'fresh_validation.json','validation_receipt.json','family_analysis.json','P2_decision.json',
    'inference_package_receipt.json','fresh_validation_plan.json','fresh_node_only_rows.csv','fresh_node_only_scores.csv',
    'architecture_details.json','actual_native_pixel_test.json','regret_summary.json','division_regret.csv',
    'score_attrition.csv','score_attrition_rows.csv','matched_controls.csv','decoder_runtime.csv','coverage_rows.csv',
    'dataset_use.csv','runtime_versions.json','source_manifest.json','optical_review_receipt.json','preservation_check.json',
    'resource_summary.json','execution_repairs.json','source_event_coverage.json','early_package_validation.json',
    'event_index_equivalence.json','event_index_deployment.json','CPU_worker_profile.json','serialization_receipt.json',
    'fresh_payload_compatibility.json','fresh_validation_schedule.json','fresh_primary_comparison.json','evaluation_schedule.json',
    'early_paths_validation.json','early_headroom_receipt.json','training_concurrency_schedule.json','GPU_concurrency_benchmark.json',
    'source_lane_schedule.json','candidate_cap_coverage.csv','candidate_cap_coverage_rows.csv','division_evidence.csv','division_evidence_rows.csv']


def run():
    destination=REPO/'results/image-native-tracking-v5';destination.mkdir(parents=True,exist_ok=True)
    copied=[]
    for name in ALLOW:
        source=OUT/name
        if source.exists():shutil.copyfile(source,destination/name);copied.append(name)
    status=read(OUT/'status.json');write(REPO/'handover/image-native-tracking-v5/STATUS.json',status)
    recovery=read(OUT/'recovery_20260910.json')
    write(destination/'recovery.json',{k:recovery[k] for k in ['at','cache_hash_checks','critical_old_hashes_preserved',
        'native_resume_step','native_checkpoint_sha256','last_logged_step','repeated_updates_required','reason','restart_policy']})
    verification={}
    for name in ['implementation_tests','handover_contract_tests']:
        path=max((OUT/'logs').glob(f'{name}*.log'),key=lambda p:p.stat().st_mtime)
        text=path.read_text()
        assert text.rstrip().endswith('OK'),name
        verification[name]=dict(sha256=sha(path),summary='\n'.join(text.splitlines()[-4:]))
    write(destination/'test_receipt.json',dict(at=now(),tests=verification))
    (destination/'README.md').write_text(f'''# Image-native tracking v5 — {status['status']}

This snapshot contains measured results and execution progress. Read
[the continuation](../../handover/image-native-tracking-v5/CONTINUATION.md) before
resuming work. The [offline dashboard](dashboard.html) identifies completed and
pending work explicitly. C0 is reproduced at 0.934802374260586; the target is 0.95.

This snapshot has {status['complete_configurations']} of {status['registered_configurations']}
registered complete configurations and {status['complete_native_fits']} of eight
production native fits. Its status is **{status['status']}**. Do not infer complete
learned-model results from partial source losses.

Only aggregate/per-sample official counts, safe hashes, model architecture,
source training curves and code/configuration receipts are tracked. Microscopy,
GEFF identities, detailed matches/oracles, model tensors, raw submissions and the
inference archive remain outside Git. Reused embryos and upstream checkpoint
exposure make these operational exploratory comparisons.
''')
    files={str(p.relative_to(destination)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in destination.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write(destination/'artifact_manifest.json',dict(at=now(),status=status['status'],files=files,
        allowlist_only=True,weights=False,raw_images=False,GT_node_or_edge_IDs=False,submissions=False))
    print('Sanitized v5 result files copied:',len(files),flush=True)

if __name__=='__main__':run()
