"""Verify complete v5 evidence and write final artifacts; no Git mutation here."""
import subprocess,sys,time,zipfile
import pandas as pd
from .common import *


def preserve():
    initial=read(OUT/'preflight.json')
    for path,h in initial['critical_hashes'].items():assert sha(path)==h,path
    for path,h in initial['old_sanitized_hashes'].items():assert sha(REPO/path)==h,path
    rows=read(OUT/'input_manifest.json')
    for row in rows:
        name=row['dataset'];image=DATA/'train'/f'{name}.zarr'
        assert sha(image/'zarr.json')==row['image_metadata_sha256']
        assert sha(image/'0/zarr.json')==row['array_metadata_sha256']
        assert sha(V1/'evaluation/gt'/f'{name}.npz')==row['gt_cache_sha256']
        assert sha(V3/'selected_predictions'/f'{name}.npz')==row['c0_file_sha256']
    from .verify_gt import run as verify_gt
    verify_gt()
    r=dict(at=now(),passed=True,critical_legacy_files=len(initial['critical_hashes']),
        old_sanitized_files=len(initial['old_sanitized_hashes']),C0_complete_graphs=199,GT_caches=199,
        raw_GT_arrays_and_estimates_reverified=True,image_metadata_pairs=199,
        raw_image_hash_scope='initial inherited full-image hashes reused; final fresh tests also rehash their six full image clips',
        raw_images_modified=False,prior_models_modified=False,previous_study_outputs_modified=False,
        prior_v4_execution_restarted=False)
    write(OUT/'preservation_check.json',r);return r


def resources():
    records=[]
    for line in (OUT/'resources.jsonl').read_text().splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:continue
    output_size=sum(p.stat().st_size for root in [OUT,WORK] for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    fits=[read(p) for p in (OUT/'models').glob('*/*.json') if 'steps' in read(p)]
    training_seconds=sum(r['seconds'] for r in fits)
    # Preserve logged restart overhead separately; source metadata already includes
    # the saved precrash history. The all-precrash allowance below double counts it
    # intentionally as conservative cap accounting.
    before_restart=[]
    for line in (OUT/'logs/44b6_N1_20260910.log').read_text().splitlines():
        if 'Resumed complete optimizer/RNG state' in line:break
        parts=line.split()
        if len(parts)==4 and parts[0]=='44b6_N1_20260910':
            try:before_restart.append(float(parts[-1]))
            except ValueError:pass
    allowance=max(before_restart,default=0)
    r=dict(at=now(),max_observed_GPU_gib=max(x['gpu_mib'] for x in records)/1024,
        max_observed_summed_RSS_gib=max(x['summed_process_rss_gib'] for x in records),
        minimum_observed_free_gib=min(x['free_gib'] for x in records),current_free_gib=shutil.disk_usage(ROOT).free/2**30,
        new_output_and_work_gib=output_size/2**30,registered_output_cap_gib=read(OUT/'preflight.json')['new_output_cap_gib'],
        completed_fit_seconds_including_tiny_tests=training_seconds,conservative_training_hours=(training_seconds+allowance)/3600,
        precrash_all_logged_training_seconds_allowance=allowance,repeated_optimizer_updates=941,
        RSS_scope='later observations include recursive inference subprocesses; earliest samples included named v5 roots only',
        samples=len(records),resource_sampling_seconds=30,
        caveat='sampled process/GPU peaks; per-fit CUDA/RSS high-water marks are also in model receipts')
    assert r['max_observed_GPU_gib']<=20 and r['max_observed_summed_RSS_gib']<=28
    assert r['minimum_observed_free_gib']>=8 and r['current_free_gib']>=8
    assert r['new_output_and_work_gib']<=r['registered_output_cap_gib'] and r['conservative_training_hours']<=48
    write(OUT/'resource_summary.json',r);return r


def validate_complete():
    from strong_tracker_v3.common import validate,graph_hash
    from annotation_selection.metric_adapter import aggregate
    expected=read(OUT/'execution_protocol.json')['variants'];rows=inventory();all_counts=[]
    for variant in expected:
        paths=list((OUT/'evaluation'/variant).glob('*.json'));assert len(paths)==199,variant
        counts=[]
        for row in rows:
            name=row['dataset'];g=graph(name,variant);validate(g['nodes'],g['edges'],row['image_shape'])
            r=read(OUT/'evaluation'/variant/f'{name}.json')
            assert r['graph_hash']==graph_hash(g['nodes'],g['edges'])
            assert r['estimated_total']==row['estimated_total'] and r['gt_sha256']==sha(V1/'evaluation/gt'/f'{name}.npz')
            counts.append(r)
        result=aggregate(counts,[r['dataset'] for r in rows]);all_counts.append(dict(variant=variant,samples=len(counts),score=result['score']))
        if variant=='C0':assert abs(result['score']-BASE['pooled'])<1e-12
    fresh=read(OUT/'fresh_validation.json');assert fresh['completed'] and fresh['export_passed'] and fresh['clips']==6 and fresh['clip_variant_runs']==72
    assert fresh['payload_compatibility']['passed'] and fresh['final_guard_selftest']['passed']
    assert len(fresh['final_default'])==1 and fresh['final_default'][0]['passed']
    assert len(fresh['C0_disablement'])==1 and fresh['C0_disablement'][0]['exact_graph_parity']
    assert read(OUT/'actual_native_pixel_test.json')['passed']
    assert read(OUT/'final_analysis_complete.json')['complete']
    assert read(OUT/'optical_review_receipt.json')['complete']
    assert len(read(OUT/'P2_decision.json')['source_pilots'])==6
    package=OUT/'inference_package';manifest=read(package/'manifest.json')
    assert sha(package/'manifest.json')==fresh['package_manifest_sha256']
    for name,h in manifest['files'].items():assert sha(package/name)==h,name
    with zipfile.ZipFile(OUT/'inference_package.zip') as z:assert z.testzip() is None
    assert sha(OUT/'inference_package.zip')==read(OUT/'inference_package_receipt.json')['sha256']
    return all_counts


def continuation(selection):
    previous=REPO/'handover/image-native-tracking-v5/CONTINUATION.md'
    backup=OUT/'continuation_execution_checkpoint.md'
    if not backup.exists():backup.write_text(previous.read_text())
    r=read(OUT/'resource_summary.json');fresh=read(OUT/'fresh_validation.json');text=f'''# V5 completed execution continuation

X500–X580 were implemented and executed on the existing RTX 4090. All 18 registered
configurations have fresh official results on all 199 clips: 16 operational
configurations including C0, plus two truth-assisted legal feasibility oracles.
All eight production native fits and four HOCT probes completed. This is the v5
completion handover; v4 remains completed and must not be restarted.

The selected export is **{selection['selected']}**, score **{selection['score']:.15f}**,
delta C0 **{selection['delta_C0']:+.15f}**. The >=0.95 local target
{'passed the registered gates' if selection['target_met'] else 'was not achieved'}.
C0 remains available at **0.934802374260586**. The report distinguishes the selected
primary seed from exploratory best points and requires both-seed pooled improvement
and no embryo regression beyond 1e-8. Repeated embryos and inherited checkpoints
prevent an independent biological-generalization claim.

Read `results/image-native-tracking-v5/final_report.md` and its offline
`dashboard.html` for measured outcomes, source losses, complete per-sample counts,
matching/edge/division regret, coverage and seed gates. The source/weight/input and
dependency manifests have hashes. All raw microscopy, GT identities, detailed
oracle/matching records, model tensors and submissions remain outside Git.

The ignored root is `/kaggle/working/cell-tracking/image-native-tracking-v5`.
Its `inference_package/` and `inference_package.zip` contain the tested package;
`inference_package_receipt.json` pins the archive. Six full density-spanning clips
from both embryos exercised all 12 primary/control pipelines from images, under
unfamiliar names and early Python read/network auditing. These executions used a
validation bundle during replica training. Its complete runtime, primary weights,
calibrations and external dependency pins are byte-identical to the final package;
the final configuration and extra replica files are recorded separately. Two actual
final-package invocations tested the selected default and `--disable-new-heads` on
opposite embryos. CSV roundtrip passed. {len(fresh['verified_variants'])} of 12 pipelines
passed the unchanged graph/count parity criterion on all six clips. Failed parity
pipelines: {', '.join(fresh['failed_parity_variants']) or 'none'}. Their executed tests
and integer-count differences remain in the report; they are excluded from promotion.
The selected export and explicit C0 fallback passed their fresh tests.
The hooks are not OS namespace isolation; Kaggle runtime is untested.

Use the preserved runtime `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`.
The package requires the pinned v1 native source/primary/secondary/DeepCenter models
and v2 E teacher files/configuration listed in its manifest. Example:

```bash
/kaggle/working/cell-tracking/image-native-tracking-v5/inference_package/run.sh \\
  --images IMAGE_DIRECTORY --output NEW_OUTPUT_DIRECTORY \\
  --v1 /kaggle/working/cell-tracking/annotation-selection-v1 \\
  --v2 /kaggle/working/cell-tracking/strong-tracker-v2 \\
  --source-model 44b6
```

Choose the training-source model explicitly. Local transfer tests use the opposite
source from each input embryo. `--variant` selects a registered operational pipeline;
`--disable-new-heads` executes C0. Do not read cached selected graphs as inference.

The native N1 recipe used 8,000 updates per source/seed; N2 warm-started N1 and used
12,000 full-image representation updates. Each has batch four and the frozen source
calibration/decoder recipe. HOCT used the official JIT and genuine region features;
its edge probe used 2,000 updates and disabled hard ILP consistency. P1 exports new
full-field image peaks, rebuilds their features, and selects competing observation
and temporal hypotheses. DeepCenter is a frozen confirmation arm. P2 dense training
was conditional and not scheduled; its source-only decision is explicit. No FOCUS
checkpoint, new synthetic replay, Zoo rendering, or small fork-head threshold sweep
was substituted.

The server reboot interrupted the first production N1 fit at logged update 2,941.
Recovery verified saved hashes and resumed optimizer/RNG state at update 2,000;
48 repeated logged losses matched to five decimals. Failure logs and the original
resume copy remain preserved. Later resume snapshots were saved every 250 updates.
After a throughput benchmark, scheduling allowed one native optimizer per source,
with per-fit locks coordinating source replica helpers and the original queue.
Calibration and inference shared one auxiliary GPU lane. The complete optimizer
function body, data, precision, batches and update counts remained unchanged;
the scheduling and benchmark receipts document the transition.
The cause of the reboot is unknown. Early package trials exposed missing declared
model-config/hash-lock dependencies and a guard rejecting newly generated GEFF
outputs; precise pinned exceptions were tested. An extra pixel fixture's incorrect
channel rank was repaired. These did not change the frozen model/decoder recipe.

Observed resource peaks were {r['max_observed_GPU_gib']:.3f} GiB GPU and
{r['max_observed_summed_RSS_gib']:.3f} GiB summed RSS. The new output/work footprint is
{r['new_output_and_work_gib']:.3f} GiB, leaving {r['current_free_gib']:.3f} GiB free.
Training cap accounting, including a conservative precrash allowance, is
{r['conservative_training_hours']:.3f} hours. Resource receipts distinguish sampled
peaks from per-fit high-water marks.

`optical_review/index.html` is a local post-freeze review pack, not new labels or
invented human judgments. Use the measured family/attrition evidence to choose a
new study with the user. Do not silently tune on these opposite-source results or
describe a subtarget/hindsight point as success. Preserve the incumbent and all
v1–v5 artifacts. No Kaggle submission, notebook/forum publication, merge, additional
hardware or paid API call was performed.

The user's unrelated dirty files were preserved. Stage only an explicit v5 allowlist
for any follow-up; never reset or clean the repository. The instance disk is local
to this server, so copy ignored weights/outputs out before replacing it.
'''
    previous.write_text(text);(OUT/'CONTINUATION.md').write_text(text)


def run(wait=False):
    while not (OUT/'final_analysis_complete.json').exists():
        if not wait:raise RuntimeError('Final analysis is not complete')
        time.sleep(30)
    if wait:os.execv(sys.executable,[sys.executable,'-m','image_native_tracking_v5.finalize'])
    write(OUT/'report_halt.json',dict(at=now(),reason='freeze final dashboard bytes for browser validation'))
    import psutil
    while any('image_native_tracking_v5.live' in (p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):time.sleep(2)
    with cpu_batch():counts=validate_complete();preserve();resource=resources()
    tests={}
    for root,pattern,label in [('tests','test_image_native_v5.py','implementation_tests'),
            ('handover/image-native-tracking-v5','test_contracts.py','handover_contract_tests')]:
        path=OUT/'logs'/f'{label}_final.log'
        with path.open('w') as log:subprocess.run([sys.executable,'-m','unittest','discover','-s',root,'-p',pattern,'-v'],cwd=REPO,check=True,stdout=log,stderr=subprocess.STDOUT)
        tests[label]=dict(sha256=sha(path),summary='\n'.join(path.read_text().splitlines()[-4:]))
    from .report import decision,csv_rows,build
    selection=decision(csv_rows('ablation_scores.csv'))
    assert selection['selected']==read(OUT/'inference_package/winning_config.json')['variant']
    write(OUT/'validation_receipt.json',dict(at=now(),passed=True,complete_graph_variants=counts,tests=tests,
        exact_C0_score=True,all_199_GT_estimates_preserved=True,final_package_bytes_verified=True,
        fresh_validation_sha256=sha(OUT/'fresh_validation.json'),preservation_sha256=sha(OUT/'preservation_check.json')))
    write(OUT/'execution_complete.json',dict(at=now(),complete=False,numerical_experiments_complete=True,optional_P2_run=False,
        selected=selection['selected'],selected_export_fresh_inference_passed=True,
        all_primary_fresh_parity_passed=read(OUT/'fresh_validation.json')['passed'],
        failed_fresh_parity_variants=read(OUT/'fresh_validation.json')['failed_parity_variants'],
        final_validation_sha256=sha(OUT/'validation_receipt.json')))
    browser_python=Path('/root/.conda/envs/cell-tracking/bin/python')
    def browser_check():
        with (OUT/'logs/dashboard_final_check.log').open('a') as log:
            subprocess.run([str(browser_python),'-m','image_native_tracking_v5.dashboard_check'],cwd=REPO,check=True,stdout=log,stderr=subprocess.STDOUT)
    build(final=True);browser_check()
    complete=read(OUT/'execution_complete.json');complete.update(at=now(),complete=True,browser_validation_passed=True)
    write(OUT/'execution_complete.json',complete)
    build(final=True)
    try:browser_check()
    except Exception:
        complete.update(complete=False,browser_validation_passed=False);write(OUT/'execution_complete.json',complete)
        raise
    final_validation=read(OUT/'validation_receipt.json');final_validation['dashboard_validation_sha256']=sha(OUT/'dashboard_validation.json')
    write(OUT/'validation_receipt.json',final_validation)
    complete['final_validation_sha256']=sha(OUT/'validation_receipt.json');write(OUT/'execution_complete.json',complete)
    selection=read(OUT/'selection.json');continuation(selection)
    print('V5 complete artifacts and browser checks ready for sanitized Git review',selection['selected'],selection['score'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();run(a.wait)
