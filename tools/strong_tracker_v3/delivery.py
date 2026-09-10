"""Publish execution status and the next-agent handover from measured receipts."""
from pathlib import Path
import shutil
from .common import now, read_json, sha, write_json


def run(ctx):
    summary=read_json(ctx.out/'summary.json')
    preservation=read_json(ctx.out/'preservation_check.json')
    validation=read_json(ctx.out/'validation_summary.json')
    dashboard=read_json(ctx.out/'dashboard_validation.json')
    if not preservation['unchanged'] or preservation['incumbent_hashes_checked']!=199:
        raise ValueError('The sealed v2 incumbent must remain unchanged')
    if summary['status']!='measured_complete' or not dashboard['passed'] or not validation['passed']:
        raise ValueError('Delivery requires completed measured results and validation')
    for relative,expected in validation['code_sha256'].items():
        if sha(ctx.repo/relative)!=expected:
            raise ValueError('Delivered code changed after final validation')
    if summary['variants']!=32 or summary['score_rows']!=32*199:
        raise ValueError('The bounded executed grid is incomplete')
    selected=summary['selected_variant'];score=summary['selected']['score'];delta=summary['delta_v2']
    if dashboard['dashboard_sha256']!=sha(ctx.out/'dashboard.html'):
        raise ValueError('The measured dashboard changed after browser validation')
    package=ctx.out/'inference_package'
    manifest=read_json(package/'manifest.json')
    config=read_json(ctx.out/'winning_config.json')['selection_configuration']
    if manifest['variant']!=selected or read_json(package/'policy.json')!=config.get('policy',config):
        raise ValueError('The package policy differs from the measured selection')
    for relative,expected in manifest['package_files'].items():
        if sha(package/relative)!=expected:
            raise ValueError('The selected package changed after its manifest was frozen')
    smoke=read_json(ctx.out/'package_smoke_receipt.json')
    if (smoke['package_manifest_sha256']!=sha(package/'manifest.json')
            or smoke['policy_sha256']!=sha(package/'policy.json') or not smoke['passed']):
        raise ValueError('The actual copied-package smoke belongs to another package')
    stages={
        'V300':dict(status='complete',clips=199,incumbent_score=summary['baseline_v2']['score']),
        'V310':dict(status='complete',division_observations=151,conditional_variants=5,
                    teacher_canonical_diagnostics=4,source_oracles='evaluation-only, fresh official scoring'),
        'V320':dict(status='complete',association_variants=5,incumbent_native_features=True),
        'V330':dict(status='complete',proposal_pools=['base','fallback'],coverage=summary['events']['coverage']),
        'V340':dict(status='complete',event_variants=10,image_optimizer_steps=40000,
                    source_seeds=['44b6/20260909','44b6/314159','6bba/20260909','6bba/314159']),
        'V350':dict(status='complete',rescue_variants=4,
                    invalid_control='R_image_local; flat-background regression fixed in separate R_image_persistent'),
        'V360':dict(status='complete',variants=['AD_primary','ADR_primary'],
                    graph_dependent_features_regenerated=True,whole_graph_regret=True,
                    unified_edit_ledger=True),
        'V370':dict(status='complete',fresh_selected=summary['fresh_inference'],
                    fresh_official_evaluation=summary['fresh_official_evaluation'],
                    dashboard_validated=True,selected_inference_package=True),
    }
    status=dict(study_id='strong-tracker-v3',status='measured_complete',completed=now(),
        branch='handover/strong-tracker-v3',next_stage=None,
        v2_incumbent_preserved=True,incumbent_score_from_v2=0.9342063149703403,
        selected_variant=selected,v3_score=score,v3_gain_against_v2=delta,decision=summary['decision'],
        stages=stages,validation=validation,validation_scope=summary['validation'],
        report='../../results/strong-tracker-v3/final_report.md',
        next_agent='NEXT_AGENT.md',raw_artifacts=str(ctx.out),
        preservation_receipt_sha256=sha(ctx.out/'preservation_check.json'),
        note='Execution is complete. Historical authoring instructions are retained as specifications; do not restart v1/v2 or V300 based on old planned status.')
    target=ctx.repo/'handover/strong-tracker-v3'
    write_json(target/'STATUS.json',status)
    embryo_lines='\n'.join(f"- {e}: {r['score']:.15f}, delta versus v2 {r['delta_v2']:+.15f}."
                          for e,r in summary['embryos'].items())
    next_agent=f'''# Strong tracker v3 — completed execution handover

V300–V370 is complete. Selected **{selected}**, pooled score **{score:.16f}**,
**{delta:+.16f} versus the preserved v2 incumbent 0.9342063149703403**.
Decision: `{summary['decision']}`. Every promotion comparison uses v2. V1 is a
historical secondary reference only.

{embryo_lines}

Read `final_report.md`, `summary.json`, `operating_points.csv`, the offline
`dashboard.html`, and `artifact_manifest.json` in `results/strong-tracker-v3/`.
There are 32 complete graph configurations  × 199 clips  = 6,368 official score rows.
The source oracle evaluations and the independent selected fresh-image evaluation
are separate diagnostics/reproduction checks, not additional selected policies.

## What was executed

The exact v2 incumbent was fingerprinted and independently rescored. The new census
covers all 151 division observations, 5,860 edge FN and 4,930 edge FP. Conditional
safe-division/smoothing, no-gap and no-pruning controls all kept motion relinking
off. Four teachers were reconciled at original and canonical coordinates; guessed
correspondence for inserted IDs was excluded.

Five opposite-embryo association settings used rebuilt incumbent-center features.
Both expanded division proposal pools ran, followed by four real 10,000-step CUDA
event-image fits, event logistic fits, source positive-group diagnostics and ten
frozen event configurations. Four rescue configurations include the preserved
invalid plateau control and its separately measured foreground-persistence fix.
The two combinations regenerated graph-dependent features after association edits.

The full selected fresh-image reproduction and an additional official evaluation
are recorded in the fresh receipts. The package exports validated integer graphs
and `submission.csv` with annotations denied from process startup. Nothing was
submitted to Kaggle or uploaded as a notebook.

## Findings that constrain the next experiment

- All gain claims remain operational exploratory: two repeatedly reused embryos,
  public checkpoint contamination, correlated teachers and unknown crop overlap.
  There is no independent inner validation or justified confidence interval.
- The strongest association setting recovers 112 GT edges without losing a GT edge,
  while adding 35 FP. Margin selection is itself exploratory. The source-6bba
  residual optimizer hit its fixed 250-iteration cap; finite losses do not establish
  optimizer convergence. Frozen historical E-hgb teachers carry prior target-label
  exposure beyond the new direct source-only fits.
- Freezing immediate fork edges does not freeze official timing-window evidence.
  The source association oracle loses one division by removing a globally FP edge
  that supports an early daughter path. Do not optimize edge counts as a complete
  surrogate for the official combined objective.
- Raw neural forks are structurally dominated by the ILP objective: division cost
  1.2 exceeds any normalized second-edge reward at most 1, while daughter birth is
  free. This was proved and tested against the actual solver; the original decoder
  was preserved. A future change to this cost needs a new bounded experiment.
- Base event coverage is 20/26 and 86/125. Wider source-selected coverage and its
  cost are recorded in `event_candidate_coverage.json`. Millions of alternatives
  do not create additional independent annotated events. Check the measured
  source-oracle feasibility and candidate/owner abstention counts before growing
  models or candidate pools further.
- All ten learned event policies lose in both embryos. The best event arm adds
  40 division TP but 1,646 division FP. The expanded annotation-assisted source
  oracle adds 81 division TP and only two FP, reaching 0.9672051875911127
  (+0.0329988726207724 versus v2). This is candidate feasibility on observed
  annotations, not a deployable result or a bound. Reliable event discrimination
  remains unresolved; more candidate alternatives alone did not solve it.
- A flat image patch cannot establish temporal object persistence. Keep the new
  foreground regression tests. The 0.75 um refinement cap applies before integer
  voxel rounding; measured realized shifts reach 0.908403 um. Do not describe it as
  a strict final integer-coordinate bound.

## Resume and transfer dependencies

Do not rerun the old studies. Do not overwrite sealed v1/v2 artifacts or change
an existing v3 prediction/model/config lock. Exact fingerprint-compatible resumes
reuse completed work; scientific changes require another namespace and an explicit
record. The wrapper accepts an explicit CUDA interpreter and dynamically resolves
the checkout. See `docs/strong-tracker-v3.md` for entry points.

Preserve these local dependencies before destroying the instance:

- Raw competition images, the full content manifest and original model inputs.
- `{ctx.v1}` including pinned tracking source, primary/secondary weights and raw
  pre-ILP evidence. Its evaluation inventory/GT are needed only to reproduce scores.
- `{ctx.v2}` including the selected incumbent lock/graphs, frozen E teacher models,
  source model lock, heatmaps and legacy prediction evidence.
- `{ctx.out}` including model/feature/proposal/graph locks, both event pools,
  source training labels, model weights, detailed ledgers, score/matching evidence,
  fresh checkpoints, selected graphs and `inference_package/`.
- `{ctx.official}` at the pinned revision and the existing CUDA study environment.
- The locally supplied DeepCenter checkpoint at its manifest path.

The portable package bundles code and selected repair weights but lists upstream
model/source dependencies. `selected_inference_package.zip` is the local transfer
archive; its hash and dependency scope are in `inference_archive.json`.
Unknown deployment images require an explicit source
fit argument; filenames do not choose an embryo-specific policy. Full local raw
artifacts have not been externally backed up. Git contains sanitized measurements,
hashes, configs and code; it is not a backup of images, predictions or weights.

## Next work is a new experiment

Use the measured candidate coverage, event errors, abstentions and source-oracle
regret to select one bounded mechanism. Preserve this selected policy and v2 as
separate controls. Any attempt to change native fork costs, protect longer daughter
paths, or improve image-triggered missing points must measure fresh whole-graph
matching in both directions and state the existing data reuse. Do not restart a
completed stage or treat the original +0.02 aspiration as an achieved result.
'''
    (target/'NEXT_AGENT.md').write_text(next_agent)
    (ctx.out/'NEXT_AGENT.md').write_text(next_agent)
    public=ctx.repo/'results/strong-tracker-v3'
    shutil.copyfile(target/'NEXT_AGENT.md',public/'NEXT_AGENT.md')
    shutil.copyfile(ctx.out/'validation_summary.json',public/'validation_summary.json')
    (target/'README.md').write_text(f'''# Strong tracker v3 — measured complete

V300–V370 was implemented and executed on all 199 expected clips. Selected
**{selected}**, score **{score:.16f}**, **{delta:+.16f} versus v2**.
The v2 incumbent **0.9342063149703403** remains preserved.

Read the [measured report](../../results/strong-tracker-v3/final_report.md),
[offline dashboard](../../results/strong-tracker-v3/dashboard.html),
[STATUS.json](STATUS.json), [NEXT_AGENT.md](NEXT_AGENT.md), and
[reproduction instructions](../../docs/strong-tracker-v3.md).

The 32 complete inference configurations and 6,368 official score rows are
operational exploratory evidence on reused embryos and public checkpoints.
The final selected pipeline also ran from fresh images with annotations unavailable.
No Kaggle submission, notebook publication or raw-artifact backup was performed.

[CODEX_PROMPT.md](CODEX_PROMPT.md), [EXPERIMENTS.md](EXPERIMENTS.md),
[IMPLEMENTATION.md](IMPLEMENTATION.md), [REVIEW.md](REVIEW.md) and
[experiments.json](experiments.json) retain the original specification. Their
authoring language does not mean execution is unfinished. Do not rerun v1/v2.
The manifest describes this completed handover and excludes itself.
''')
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.iterdir())
           if p.is_file() and p.name!='MANIFEST.json'}
    write_json(target/'MANIFEST.json',dict(schema_version=2,study_id='strong-tracker-v3',
        scope='Completed execution handover payload hashes; manifest excludes itself',files=files))
    return status
