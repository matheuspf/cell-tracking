# Strong tracker v3 — completed execution handover

V300–V370 is complete. Selected **A_residual_m3.0**, pooled score **0.9348023742605860**,
**+0.0005960592902456 versus the preserved v2 incumbent 0.9342063149703403**.
Decision: `small_local_gain`. Every promotion comparison uses v2. V1 is a
historical secondary reference only.

- 44b6: 0.931664468721842, delta versus v2 +0.000052081944511.
- 6bba: 0.935221784097327, delta versus v2 +0.000696534227794.

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
- `/kaggle/working/cell-tracking/annotation-selection-v1` including pinned tracking source, primary/secondary weights and raw
  pre-ILP evidence. Its evaluation inventory/GT are needed only to reproduce scores.
- `/kaggle/working/cell-tracking/strong-tracker-v2` including the selected incumbent lock/graphs, frozen E teacher models,
  source model lock, heatmaps and legacy prediction evidence.
- `/kaggle/working/cell-tracking/strong-tracker-v3` including model/feature/proposal/graph locks, both event pools,
  source training labels, model weights, detailed ledgers, score/matching evidence,
  fresh checkpoints, selected graphs and `inference_package/`.
- `/root/code/kaggle/cell-tracking/work/annotation-selection-v1/official` at the pinned revision and the existing CUDA study environment.
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
