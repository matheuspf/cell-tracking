# Execution plan: one public-baseline ablation, no tuning loop

## Non-negotiable scope

Scientific controls: **B0**, original Harmonic Fusion v29; **B1**, the same pipeline
with effective `OUTPUT_MOTION_RELINK=False`. Exactly one candidate. No retraining,
residual classifiers, new checkpoints, segmentation, Ultrack, new TTA, ensembling
changes, threshold sweeps, per-embryo policy selection, per-clip overrides or
leaderboard-guided revisions. Old C0/P0 results are context only.

The objective is a credible minimal candidate for **0.95+ LB**, not 0.95 on reused
local clips. Preserve the baseline if the hypothesis fails. This plan nominates
an already known local ablation; it is not a newly preregistered discovery.
Freeze it before new execution to prevent further adaptive selection.

New code: `tools/public946_minimal/` and `tests/test_public946_minimal.py`.
Runner to implement: `scripts/run_public946_minimal.sh` with `preflight`,
`audit`, `replay`, `fresh`, `robustness`, `package`, `report`, and `run` commands.
All commands must accept explicit image/artifact/output/runtime paths and be
resumable via content hashes. They are planned interfaces, not implemented here.

Outputs: `/kaggle/working/cell-tracking/public946-minimal-generalization-v1/`,
with sanitized summaries under `results/public946-minimal-generalization-v1/`.
Raw data, microscopy, graphs, weights, detailed labels and generated notebooks/
submissions stay ignored and outside Git. Commit code, concise receipts and docs.
Never edit old result files, selection locks, cached predictions or model inputs.

## P000 — preserve, identify and freeze

1. Confirm this branch descends from the pinned main commit. Inspect local git
   status and active processes. Preserve unrelated user changes and running
   studies; do not reset, clean, kill jobs, change branches or apply old handovers.
2. Discover actual mounts and runtimes. Known inference interpreter:
   `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`; alternate
   notebook runtime `/kaggle/envs/cell-tracking-notebooks/bin/python`. Existing
   wrappers use `scripts/root_remote_env.sh` to avoid a CUDA library override.
   Use the working stack, `PYTHONNOUSERSITE=1`, bounded CPU threads, one RTX 4090.
   Do not upgrade packages, install anything, download models or accept licenses.
3. Read the archived v29 notebook, Python export, metadata, local-verification
   records and the actual notebook-patched `tracking_repo` source. Pin script
   version 347965685, checkpoint hashes, source hashes, full effective settings,
   input-dataset versions and dependency versions. A file name alone is not identity.
4. Write `source_review.md` with source-line references for preprocessing,
   coordinate units, temporal windows, eight-view TTA and inverse maps, both
   checkpoints, exact harmonic fusion, candidates, ILP, motion relinking and all
   later repairs. Trace every assignment/use of the motion flag, including
   environment variables, notebook cells and worker processes. Confirm B0's
   **effective runtime value is True**; external environment settings may be
   overwritten by the notebook. Do not assume setting an env var is enough.
5. Read current official references through the existing throttled extractor.
   The previously pinned evaluator is revision
   `075fc5f5a52d11077f9dc2b074644618f26939e2`. If rules/metric have changed, record
   the change and score both arms on the same valid revision. Do not compare
   historical scores across revisions or replace the public source with 'latest'.
   If network access is unavailable, retain the historical pin and mark current
   rule verification pending; do not falsely certify Kaggle readiness.
6. Inventory storage, models and cache paths without modifying them. V6 recorded
   only 6.838 GiB free and used volatile `/dev/shm`; do not assume those artifacts
   survive. Stream per clip, share read-only inputs, and budget scratch before
   running. Never delete earlier studies to make room. Missing old caches are
   rebuildable from public source; they are not a reason to require old residuals.
7. Copy `experiment_lock.json` to output, append local source/runtime/input hashes
   in a separate immutable `execution_lock.json`, and commit its sanitized digest
   before new comparative scores. Store planned and actual state separately.

Produce `preflight.json`, `source_review.md`, `input_manifest.json`,
`execution_lock.json`, `preservation_manifest.json`. Missing assets must be
reported specifically. Continue all independent audits/tests rather than aborting
all work at the first missing optional cache or clean confirmation set.

## P010 — construct a genuine one-change pair

Use the archived public source, not a v3/v5/v6 'selected' inference export.
`strong_tracker_v3.replay.repair_namespace` hard-codes motion off; unmodified use
cannot reproduce B0. Older replay helpers also select code by line ranges. Reuse
interfaces after review, not their global defaults or mutable output roots.

Create isolated B0 and B1 runtime adapters with exactly the same necessary path,
offline-install and annotation-validator handling. Keep the archived originals
unchanged. Exclude the public notebook's GT-reading validation section from both
inference paths; its cut point was `TRAIN_DIR = COMP_DIR / "train"` in the v1
adapter. Verify the current source structure rather than applying blind replaces.

The supplied `baseline_contract.py` checks a unique module-level flag assignment
and changes only its right-hand-side expression in a Python export. It rejects
multiple writes and refuses to overwrite files. Example, after P000:

```bash
python handover/public946-minimal-generalization-v1/baseline_contract.py \
  --source /kaggle/working/biohub-harmonic-fusion.py \
  --candidate /kaggle/working/cell-tracking/public946-minimal-generalization-v1/source/candidate.py \
  --receipt /kaggle/working/cell-tracking/public946-minimal-generalization-v1/source/patch_receipt.json
```

A helper rejection is an engineering audit task: inspect the exact notebook and
write an equally strict cell-aware patch if needed. Do not weaken it into an
unbounded replace, silently patch a different version, or start tuning. Compile
and inspect the .ipynb cells and generated prediction script as well. The helper
does not resolve IPython magics, environment overrides or runtime side effects.

Assert at runtime B0=True and B1=False immediately before graph filtering; log
whether the relinker executes. Hash the raw neural detections, pre-ILP candidates
and raw solved graph. They must be identical across the two repair arms when
fed the same evidence. All remaining configuration, code and model hashes match.
No additional scientific settings are allowed to differ.

Keep any identical deployment glue in a separate diff from the scientific patch.
Do not claim the complete notebook differs by one line if packaging also adds
shared path/offline/validation code. The meaningful claim is one scientific change.

## P020 — reproduce and characterize the known local effect

Reuse validated v1/v2 raw evidence where available, read-only. Existing locations:

- `/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/`
- `/kaggle/working/cell-tracking/strong-tracker-v2/raw/`
- `/kaggle/working/cell-tracking/strong-tracker-v2/heatmaps/`

Make a new image-only inventory, never pass the old GT-bearing inventory to
inference. Run B0/B1 repair on all 199 available supplied clips. Evaluate in a
separate process with fresh official matching, division handling and exact
run-level weighting. Require the full expected sample set; report failed/missing
clips, never omit them or average clip scores as the competition score.

Under the historical metric expect B0 ~0.911774 and B1 ~0.934206 and the counts in
BASELINE_AND_EVIDENCE.md. Differences beyond score rounding must be investigated
as reproducibility issues, not 'improvements'. Compare per-embryo results and
TP/FP/FN counts. Record true candidate gains/losses via rematching, not merely
counting edges added/deleted. Report the number of changed edges/nodes/clips,
existing correct links lost, recovered links, division impact, speed and memory.

Measure contributions for the raw graph, immediate relink output, and final
pipeline to distinguish direct link changes from downstream cascade effects.
Use intensity, crowding, time boundaries and physical displacement for descriptive
error tables only. Freeze their definitions before scoring; never derive a
per-subgroup deployment policy. These are reused, correlated diagnostics.

### Serialization and bounds are not a hidden second experiment

The old replay reconciled two half-integer floating ties against sealed baseline
predictions and separately clipped six out-of-bounds points. Do not import that
compatibility operation into deployed inference or use saved coordinates to force
fresh parity. Preserve natural public serialization for the raw A/A comparison.
Document actual tie behavior and use canonical graph comparison plus metric counts.

Validate the official output contract. If strict bounds require a correction,
register it as identical deterministic deployment sanitation for both B0 and B1,
never a candidate-specific fix. Score raw and sanitized versions of both arms;
list changed nodes and all metric deltas. The scientific comparison is then
B1-sanitized versus B0-sanitized, alongside the exact original B0 reference.
Require zero score/count effect of sanitation on this cohort to retain the
one-change evidence claim. Otherwise stop promotion and report the confound.
Never silently round away a score difference or clamp the metric itself.

## P030 — fresh inference and bounded robustness checks

Cached replay alone is insufficient for a public-LB candidate. The new driver
must operate from images and immutable public artifacts only, in a fresh output
namespace, with annotation/evaluation/old prediction paths denied before importing
inference code. Reuse existing audit-hook patterns after reviewing allowlists;
child workers must inherit guards. Evaluation runs separately with labels allowed.

First choose four pilots without labels/scores: per embryo, the clips nearest
25th and 75th percentile of image-metadata q99-q10 contrast, ties by stem. Freeze
identities and metadata hashes. Process each entire 100-frame clip, not a crop or
truncated window, through fresh public neural inference and both repair arms.
Cached evidence is allowed only as a comparison target in evaluation, not as input
in a run labelled fresh. Run both baseline and candidate .ipynb/export paths on
at least one full pilot to verify packaging has not changed the graph semantics.

Then complete one all-199 fresh neural pass, streaming clips, with B0/B1 repairs
branching from the same freshly generated raw graph. This avoids paying twice for
identical model inference. Persist enough identity evidence for restart/parity,
not duplicate dense volumes. If compute/storage prevents the full fresh pass,
record the exact completed scope and leave full-workload verification pending;
never equate four fresh pilots with a full hidden-run validation.

On the same four frozen pilots run these fixed non-selection checks:

- Rename stems to unrelated identifiers, remove training annotations from the
  input view, and reverse filesystem enumeration order. Canonical output must
  be unchanged after reversing the naming map. No prefix-based embryo/model
  selector or list of training filenames may enter deployment.
- Apply one X-axis reflection to the image. Invert output X with `X-1-x` for
  comparison and transform evaluation annotations only in the evaluator. Preserve
  voxel scale and time direction. Quantify B0/B1 consistency and metric deltas;
  floating ties or model non-equivariance require explanation, not an assertion
  that exact invariance is guaranteed. No Z/X swaps or time reversal of divisions.
- Repeat the same untransformed inference in a fresh process. Compare raw graph,
  final canonical graph and CSV; quantify any actual nondeterminism and its A/A
  score range. Do not hide variation with sealed-output lookups.

These tests check reproducibility and a narrow nuisance transformation. They are
NOT new embryos, clean holdouts or statistical proof of generalization. Do not
optimize to them or add arbitrary perturbations until a desirable result appears.
Report a reflection sign reversal as a limitation requiring review before the LB
trial, not an invitation to search reflection-specific knobs.

If genuinely unexamined, embryo-disjoint, in-domain labelled data are ALREADY
available, document overlap and upstream-model provenance and evaluate the locked
pair once. Otherwise mark clean confirmation unavailable. Do not launch another
external-data study, rename a seen split 'holdout', bootstrap overlapping clips as
independent embryos, or make confidence-interval claims from just two embryos.
Lack of clean confirmation alone does not forbid packaging this transparent LB trial.

## P040 — decision and standalone Kaggle package

The registered readiness rule is a conjunction, not a competition between many
variants: exact source/config identity apart from the flag; trustworthy B0
reproduction; positive B1 full-cohort delta and positive delta on each supplied
embryo; nondecreasing pooled division Jaccard; valid graph/CSV contract;
annotation-free fresh execution; no unexplained portability/A/A failure.
The 1e-6 tolerance in the lock is only for rounded historical score reproduction,
not a tunable effect-size threshold. Do not require local score >=0.95.

If these fail, retain the results and B0, mark B1 not recommended, and do not
invent B2 in this study. Engineering failures can be repaired with the scientific
lock unchanged. Never add an adaptive 'safe relinker', margin gate or confidence
threshold merely to salvage the candidate after seeing scores.

Produce the following local package, even when its manifest must say unvalidated:

- `baseline_public946.ipynb` and `candidate_no_motion.ipynb`, plus reviewed Python
  exports; source attribution/licenses and exact public input dependencies.
- `scientific_change.diff` and separately `shared_runtime_changes.diff`.
- `manifest.json` with base/source/notebook/model/config hashes, metric revision,
  effective flags, test scope, package checksums and explicit readiness status.
- `LOCAL_RUN.md`, `KAGGLE_RUN.md`, valid kernel metadata without invented account
  identifiers, and minimal offline dependency/runtime instructions using the
  original inputs. No imports from old work directories or local fitted models.

The inference entry must enumerate arbitrary test stems, handle shapes from actual
metadata, reset per-dataset state, produce the documented node/edge CSV schema,
check every input dataset was processed, and reject malformed graphs. Verify
consecutive-frame links, endpoint existence, time ordering, degree/merge limits,
duplicate rows/edges, finite integer coordinates and applicable bounds. Keep
track IDs local per dataset with an appropriate global CSV row index.

Do not ship training-score caches as a 'submission'. Visible test examples overlap
training and are packaging smoke tests only. The submitted notebook must infer
again on Kaggle's substituted hidden data. No notebook may read GT GEFFs or rely
on the optional public validator's label-reading cells.

Measure real stage times, peak GPU/RSS/scratch and full local wall time. Confirm
current Kaggle runtime limits and available accelerator. A local RTX 4090 timing
is not a Kaggle runtime guarantee; distinguish actual Kaggle smoke evidence from
hardware/workload extrapolation. Preserve baseline complexity; skipping one CPU
repair stage must not quietly remove TTA or any other model work for speed.

## P050 — one manual leaderboard trial, no automatic submission

The user requested a plan/branch, not a Kaggle write. Do not upload, submit, post
to a forum or change a Kaggle dataset. Prepare these instructions for the user:

1. Lock candidate notebook checksum before seeing new LB feedback. Reuse a
   comparable, currently scored exact-baseline run when its source/metric identity
   is known; otherwise run the prepared baseline control once for comparison.
2. Submit the single locked B1 candidate once after notebook packaging checks.
   Record kernel version, submitted notebook hash, metric/date, runtime and score.
   Do not compare a stale pre-rescore baseline with a current candidate.
3. Classify the result: >=0.95 and above matched B0 is target met on public LB;
   above B0 but <0.95 is smaller LB improvement; equal is no demonstrated uplift;
   below is failed transfer. Do not convert any of these into a private-LB claim.
   A result known to be within measured execution variability is inconclusive.

Any follow-on scientific idea requires a new explicit study, not a sequence of
threshold changes chosen by LB feedback. The plan does not consume all daily
submission slots or automatically resubmit the same candidate looking for a high score.

## P060 — report and handoff

Write `results/public946-minimal-generalization-v1/START_HERE.md`, `final_report.md`,
`status.json`, aggregate `scores.csv`, `source_audit.json`, `fresh_receipt.json`,
`robustness_summary.json` and `package_manifest.json`. Mark every stage planned,
complete, failed or blocked with exact evidence; use null for unmeasured scores.
Include old vs new metric provenance, observed semantic edit footprint, both
embryos, raw/validity-control distinction, test scope, resource measurements,
artifact locations/checksums, and one exact command for the next local agent.

Run new source-contract tests plus relevant existing graph/metric/inference tests.
Do not call 10 synthetic source tests successful microscopy validation. Save
failure logs and identify the smallest next engineering action for blocked runs.
Commit only an explicit allowlist of this study's code and sanitized records to
this branch. Confirm old artifact hashes and unrelated work are unchanged.

Terminal states: `ready_for_manual_lb_test`, `not_recommended_local_failure`,
`blocked_with_partial_artifacts`, or, only after a real user-supplied Kaggle result,
`public_lb_target_met` / `public_lb_improved_below_target` / `public_lb_no_uplift` /
`public_lb_transfer_failed` / `public_lb_inconclusive`. No LB result is expected
from local Codex execution alone.
