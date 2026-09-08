"""Finalize validated local artifacts and write sanitized, aggregate-only results."""
from __future__ import annotations

import json
import csv
import shutil
import time
from pathlib import Path

import pandas as pd

from .common import OUT,REPO,WORK,clean,now,read_json,revision,sha,stage,write_json


def run(args):
    required=['final_report.md','dashboard.html','coverage.csv','sample_inventory.csv','retention.csv',
              'score_rows.csv','summary.json','fold_manifest.json','environment.json','candidate_features.parquet',
              'matched_budget_summary.csv','endpoint_correlations.csv','final_integrity_receipt.json','dashboard_validation.json']
    required+=['validation_receipt.json','public_score_manifest.json','audit_browser_validation.json']
    for name in required:
        if not (OUT/name).exists():raise ValueError(f'Missing required artifact {name}')
    if not read_json(OUT/'dashboard_validation.json')['passed']:raise ValueError('Dashboard validation failed')
    if read_json(OUT/'dashboard_validation.json')['html_sha256']!=sha(OUT/'dashboard.html'):
        raise ValueError('Dashboard changed after browser validation')
    validation=read_json(OUT/'validation_receipt.json')
    if not validation['passed'] or not read_json(OUT/'audit_browser_validation.json')['passed']:
        raise ValueError('Required implementation/browser validation failed')
    for path,h in validation['code_hashes'].items():
        if sha(REPO/path)!=h:raise ValueError('Code changed after final validation')
    result=read_json(OUT/'summary.json')
    score=read_json(OUT/'score_manifest.json')
    expected=read_json(OUT/'fold_manifest.json')['expected_samples']
    if not score['complete'] or sorted(score['samples'])!=sorted(expected):raise ValueError('Incomplete main score sample set')
    if not result.get('public') or result['public']['sample_count']!=len(expected):
        raise ValueError('Full public diagnostic lane not complete; document any blocker before finalizing')
    public_manifest=read_json(OUT/'public_score_manifest.json')
    if not public_manifest['complete'] or sorted(public_manifest['samples'])!=sorted(expected):
        raise ValueError('Incomplete public score sample set')
    if 'Verified 24,886 files, 87,609,892,618 bytes (CRC32).' not in (WORK/'data-crc-verification.log').read_text():
        raise ValueError('Full input CRC verification receipt missing')
    if (WORK/'evaluation-supplement.log').exists() and not (OUT/'supplemental_scoring_execution.json').exists():
        raise ValueError('Supplemental evaluator is still running')
    # Close the resource CSV before hashing the artifact tree.
    write_json(OUT/'resource_monitor.stop',dict(requested=now(),reason='Final artifact sealing'))
    deadline=time.monotonic()+35
    while not (OUT/'resource_monitor_stopped.json').exists():
        if time.monotonic()>deadline:raise ValueError('Resource monitor did not acknowledge a closed CSV')
        time.sleep(.25)
    if not read_json(OUT/'resource_monitor_stopped.json').get('csv_closed'):
        raise ValueError('Resource monitor did not confirm closure')
    resource=pd.read_csv(OUT/'resource_samples.csv')
    if resource.summed_rss_gib.max()>32 or resource.device_used_mib.max()>20*1024:
        raise ValueError('Measured soft resource cap exceeded; report the affected run before finalizing')
    # Copy receipts into the self-contained local artifact tree.
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for pattern in ['*.log','validation-*.xml']:
        for p in WORK.glob(pattern):shutil.copy2(p,logs/p.name)
    stage('S000','complete',data_files_crc_verified=24886,helper_tests=validation['helper_tests'],upstream_tests=validation['upstream_tests'],
          study_environment_tests=validation['study_tests'],inspection_environment_tests=validation['inspection_tests'],inspection_optional_skips=validation['inspection_skips'])
    stage('S020','complete',clean_samples=len(expected),public_samples=len(expected),public_graph_parity=True,
          public_provenance='contaminated_diagnostic_only',
          public_spatial_exception_nodes=read_json(OUT/'public_coordinate_audit.json')['out_of_bounds_nodes'])
    stage('S090','complete',samples=len(expected),score_rows=score['rows'],variants=score['variants'],
          public_samples=len(expected),public_score_rows=public_manifest['rows'],public_variants=public_manifest['variants'])
    stage('S040','automated_audit_complete_manual_census_unavailable',blinded_rois=48,blinded_candidate_crops=48,
          offline_viewers_validated=True,exact_true_cell_prevalence=None)
    public=read_json(OUT/'public_summary.json')
    public_pooled=next(r for r in public['primary'] if r['embryo']=='pooled')
    summary=dict(schema_version=1,study_id='annotation-selection-v1',status=result['status'],created=result['created'],
                 clean_primary=[{k:r[k] for k in ['embryo','n','baseline_score','score','delta','realized_keep','annotation_recall','phi']} for r in result['primary']],
                 public_diagnostic_primary=[{k:r[k] for k in ['embryo','n','baseline_score','score','delta','realized_keep']} for r in public['primary']],
                 coverage=result['coverage'],gt_unavailable_reproduction=True,true_cell_prevalence=None,
                 independent_groups_per_embryo=1,source_selection='preregistered_fixed_setting_fallback',
                 primary_outcome=result['primary_outcome'],
                 metric_revision=result['metric_revision'],data_hash=result['data_hash'],split_hash=result['split_hash'],
                 local_report=str(OUT/'final_report.md'),local_dashboard=str(OUT/'dashboard.html'),
                 no_hidden_test_claim=True,no_submission_or_publication=True)
    dest=REPO/'results/annotation-selection-v1';dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'summary.json',summary)
    md=f'''# Annotation selection v1 — measured local result

Status: **{result['status']}**. Both embryo directions were evaluated on all 199
training clips. The primary coherent keep-90% rule used fixed source-trained models;
unresolved crop overlap prevented independent source inner tuning and informative
within-embryo bootstrap intervals. All 199 prediction files reproduced byte-for-byte
with annotation reads blocked before outer evaluation.

| Outer embryo | Baseline score | Filtered score | Delta | Realized retention |
| --- | ---: | ---: | ---: | ---: |
'''
    for r in result['primary']:
        md+=f'| {r["embryo"]} | {r["baseline_score"]:.6f} | {r["score"]:.6f} | {r["delta"]:+.6f} | {r["realized_keep"]:.4%} |\n'
    md+=f'''
The public Harmonic Fusion lane is separately marked contaminated and diagnostic.
Its transferred primary rule changes pooled score from {public_pooled['baseline_score']:.6f}
to {public_pooled['score']:.6f} ({public_pooled['delta']:+.6f}); this rule harms that tracker.
Exact sparse annotations, supplied-count coverage, candidate-conditional membership,
all fixed budgets, random/confidence controls, exact-cost coherent controls, fresh
official graph/division counts, count/graph attribution, and both image-probe seeds
are in the local artifacts. Exact all-cell prevalence remains unidentified; the
48-ROI blinded census pack has no manual labels. No hidden-test gain is claimed.

- [Final report]({OUT/'final_report.md'})
- [Offline dashboard]({OUT/'dashboard.html'})
- [Artifact manifest]({OUT/'artifact_manifest.json'})
- [Reproduction instructions](../../docs/annotation-selection-v1.md)
- [Sanitized aggregate JSON](summary.json)

Raw inputs, notebook originals and existing environments were preserved. Predictions,
raw patches, weights and detailed GT matching tables remain outside Git. No submission,
push, notebook publication, forum post, paid API call or hardware rental was performed.
'''
    (dest/'README.md').write_text(md)
    initial=read_json(REPO/'handover/annotation-selection-v1/STATUS.json')
    initial.update(status=result['status'],local_execution_completed=True,local_execution_date=now(),
                   competition_data_accessed_in_local_execution=True,gpu_experiments_run_in_local_execution=True,
                   official_graph_integration_tests_run=True,real_annotation_prevalence=None,
                   annotation_estimated_coverage=result['coverage'],heldout_classifier_results=str(OUT/'classifier_metrics.csv'),
                   official_score_delta={r['embryo']:r['delta'] for r in result['primary']},next_stage=None,
                   note='Protocol executed locally; fixed-setting fallback because independent source overlap groups could not be certified. True-cell prevalence remains unknown. See local final report and sanitized results.',
                   local_report=str(OUT/'final_report.md'),local_dashboard=str(OUT/'dashboard.html'))
    write_json(REPO/'handover/annotation-selection-v1/STATUS.json',initial)
    handover=REPO/'handover/annotation-selection-v1/README.md'
    marker='<!-- local-execution-result -->'
    if marker not in handover.read_text():
        handover.write_text(f'{marker}\nLocal execution is complete. See [measured results](../../results/annotation-selection-v1/README.md) and [reproduction instructions](../../docs/annotation-selection-v1.md). The original handover below is preserved.\n\n'+handover.read_text())
    # Include runnable source and execution helpers in local storage, without raw inputs.
    sources=[*sorted((REPO/'tools/annotation_selection').glob('*.py')),
             *sorted((REPO/'tests/annotation_selection').glob('*.py')),
             *sorted((REPO/'handover/annotation-selection-v1').glob('*.py')),
             *sorted((REPO/'handover/annotation-selection-v1').glob('*.md')),
             REPO/'handover/annotation-selection-v1/STATUS.json',
             REPO/'scripts/run_annotation_selection.sh',REPO/'docs/annotation-selection-v1.md',
             REPO/'handover/annotation-selection-v1/experiments.json']
    code_hashes={str(p.relative_to(REPO)):sha(p) for p in sources}
    for p in sources:
        target=OUT/'reproduction'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in WORK.glob('*.py'):
        target=OUT/'reproduction/execution_helpers'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    tables=[]
    for p in sorted(OUT.glob('*.csv')):
        with p.open(newline='') as f:
            fields=next(csv.reader(f))
        tables.append(dict(path=p.name,columns=fields,sha256=sha(p)))
    write_json(OUT/'table_manifest.json',dict(schema_version=1,run_id='annotation-selection-v1',code_revision=revision(),
               code_hashes=code_hashes,producer_commands='commands.jsonl',tables=tables,
               parquet_contract_metadata='contract_table_receipt.json'))
    stage('S110','complete',decision=result['status'],report=str(OUT/'final_report.md'),dashboard=str(OUT/'dashboard.html'))
    status=read_json(OUT/'status.json');status.update(status=result['status'],outer_results_revealed=True);write_json(OUT/'status.json',status)
    try:
        files=[]
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='artifact_manifest.json' and not p.is_symlink():
                files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
        size=sum(x['bytes'] for x in files)
        patch_bytes=sum(p.stat().st_size for d in ['patches','public_patches'] for p in (OUT/d).glob('*.npy'))
        if size>150*2**30 or patch_bytes>50*2**30:raise ValueError('Artifact/patch soft cap exceeded')
        write_json(OUT/'artifact_manifest.json',dict(schema_version=1,created=now(),study_id='annotation-selection-v1',
                   repository_base=read_json(OUT/'environment.json')['repo_revision'],finalization_parent_revision=revision(),
                   code_hashes=code_hashes,artifact_root=str(OUT),files=files,total_bytes=size,patch_bytes=patch_bytes,
                   commands='commands.jsonl',raw_data_mutated=False,existing_notebooks_mutated=False,existing_environments_rebuilt=False))
    except Exception:
        stage('S110','finalization_failed',report=str(OUT/'final_report.md'),dashboard=str(OUT/'dashboard.html'))
        raise
    print(json.dumps(clean(summary),indent=2),flush=True)
