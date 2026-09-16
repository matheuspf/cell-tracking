# Implementation and validation contracts

The v5 training/inference pipeline is not implemented in this handover. Local Codex
implements it under `tools/image_native_tracking_v5/` with tests in a separate v5
namespace. The included arithmetic/supervision helpers are reference checks, not
an official graph scorer or a trained tracker. Keep the previous studies intact.

## Reuse, do not rerun

Read `docs/multidata-training-v4.md`, the measured v4 continuation, and its inference
package contract. Existing v4 `--disable-new-heads` is a verified C0 path. V3's
native feature/ID reconciliation and v2's replay source are reusable as read-only
references. Their hard-coded output globals must not receive new artifacts.

Primary filesystem inputs are explicit in `config.json` and `preflight.py`. The
canonical v4 archive alone is insufficient: primary/secondary native weights and
the patched predictor come from the v1 root; E_hgb teachers from v2; DeepCenter
has its own checkpoint path. Read exact package manifests, not checkpoint titles.
Do not invent filenames for dependencies that have not been discovered locally.

## Suggested modules and deliverables

| Module | Required behavior |
|---|---|
| paths / manifest | Explicit old/new roots, source hashes, no old-study writes, resumable per-stage fingerprints. |
| observations | Stream native heatmaps, new peaks and watershed hypotheses; physical transforms; stable IDs; node/region mapping and exclusivity. |
| native_adapter | Exact installed architecture/preprocessing parity; dense candidate neighborhoods; frozen-head and trainable-backbone paths. |
| hoct_adapter | Official feature-complete graph, checkpoint checksum, preserved C0 centers in fixed-node tests, source edge-embedding probe. |
| supervision | Source-only matches, supported incoming-parent targets, censored missing parents, unknown second daughters, optional pseudo-label provenance. |
| temporal_decode | Consecutive-edge graph objective with valid birth/continuation/fork/no-op choices, ownership, observation exclusions, complete path protection. |
| train | H/N module-specific parameter updates; data coverage, sampling correction, gradient and checksum receipts; source-only checkpoints/calibration. |
| infer | Image/model-only execution, no matching/oracle imports, scores for new nodes, deterministic overlap reconciliation, safe fallback. |
| evaluate | Fresh official graph/division matching, exact all-sample aggregation, GT-edge identity regret, complete score table. |
| package / report | Fresh image package with base dependency hashes and C0 fallback; final report, offline dashboard, CONTINUATION and execution ledger. |

Add `scripts/run_image_native_tracking_v5.sh` only when implemented. It must resolve
its repository root from its own path and run from a foreign cwd, use an explicitly
chosen Python interpreter, and never bootstrap/download the entire competition.
No command in the initial handover claims that this future runner already exists.

## Data structures

Keep raw detailed tables local. Every artifact includes schema version, family,
source embryo, code/config/model hashes, candidate-universe hash and producer command.

- `nodes`: stable dataset-scoped ID, t, floating native ZYX, physical ZYX, origin,
  optional region ID, confidence/missing flags, observation-exclusion group.
- `edges`: source/target IDs, actual time delta, native and H/N scores, window/core
  owner, lineage-context masks. Output edges must have delta1.
- `regions`: candidate ID -> optical mask/property provenance and feature validity.
  Region IDs and per-frame labels are not persistent biological identities.
- `training_labels` in evaluation-only storage: GT match, label {-1,0,1}, support
  type, predecessor-represented flag, optional weak-label weight. Not input features.
- `run_manifest`: complete ordered clip list, source direction, training and
  checkpoint exposure, input/model hashes, prediction-freeze time, score-reveal time.
- `score_rows.csv`: variant, seed, dataset, embryo, edge TP/FP/FN, division TP/FP/FN,
  num_pred_nodes, estimated_total, exact score components, graph and scorer hashes.
- `coverage.csv`: fixed/new node coverage; supported transition coverage; exact-ID
  versus official-local division evidence; score/gate/conflict/timeout attrition.

A model's source exposure follows its teachers and preprocessing, not just the
last optimizer. A cached source feature derived from an opposite-embryo-trained
teacher retains that exposure. Publish a readable exposure table for each family.

## Sparse-label and native-network specifics

`contracts.supported_edge_labels` demonstrates conservative incoming-parent
supervision on a fixed, one-to-one matched candidate bank. It labels a non-GT
incoming edge negative only when that target's positive parent edge is represented
in the same bank. It does not certify an absent second daughter from a mother
with one recorded child. Match geometry and time validation remain the local
implementation's job. Unknown remains -1, and the dense external-label case is a
separately declared path.

For normalized incoming-parent learning, each daughter column has its own parent
choice; a mother may be selected by two columns. Logit dimensions must match the
installed predictor. Empty/censored columns contribute no false birth/negative
loss. If the correct predecessor falls outside a crop/neighbor list, log the
coverage miss and censor that target rather than training all candidates negative.

Low-weight stability/pseudo-label losses must not reuse GT-generated teacher graphs
at inference or certify every missing incumbent edge as false. Verify new positive
fork gradients survive the consistency term. Change encoder parameters only in N2;
N1 must prove their equality, and all models must retain finite probabilities.

## Required integration tests on the execution machine

1. Full C0 reproduction and source package dependency identity, with all199 samples.
2. Actual HOCT import/model/region feature shapes and numeric transforms; point-stub
   detection; fixed-node ID/count/coordinate parity before and after feature creation.
   A mask collision, empty region or duplicate label must not silently erase a cell.
3. Sparse incoming-parent masks: two daughters positive; possible unlabeled second
   daughter unknown; no predecessor in view censored; dense-only negatives never leak
   to Biohub; empty-frame losses finite with correct zero contribution.
4. Native checkpoint/preprocessing and original logits parity; finite overfit on a
   tiny real-source set; N1 frozen encoder checksum; N2 gradients/changed encoder.
5. Actual solver tests for true fork, no-fork+birth, inherited fork-dominance failure,
   competing owner, duplicate observation alternatives, boundary censoring, no-op,
   and temporal support beyond immediate daughter edges. No blanket fork promotion.
6. Equivalent event alternatives duplicated without increasing/decreasing confidence;
   unique alternatives still supported. Daughter order and window batching do not
   change results. Padded frames and omitted competitors cannot supply fake evidence.
7. Full-frame newly detected center absent from incumbent enters a scored trajectory;
   split-vs-single hypotheses are exclusive. Flat background produces no persistence.
   Optical centroid and official exported center are distinguished in fixed-node tests.
8. Multiple windows reconcile exactly without duplicate links, backward edges, merges
   or dropped samples. Chunked and monolithic fixtures match; timeouts report fallback.
9. Annotation-unavailable fresh-image executions actually load/run each selected head,
   detector and solver, instead of reading cached final graphs. Pixel perturbation on
   a synthetic fixture affects the optical path; frozen C0 identity remains verified.
10. Exact official aggregation/parity; expected clip-set equality and fixed GT/count
    manifests; no skipped read errors; unit/axis anisotropy, duplicate detections,
    timing-shifted division evidence and zero-event cases handled by official code.
11. Resource/resume contracts: cache bounded, free-space reserve honored, no stale
    checkpoint/config reuse, atomic outputs, foreign-cwd CLI, immutable old artifacts.
12. Same-recipe second seed and matched controls; report inherited exposure and reused
    embryos. Seeds do not make independent biological validation. No falsely precise CI.

## Final report contract

Provide `final_report.md`, `dashboard.html`, `score_rows.csv`, `family_outcomes.csv`,
`coverage.csv`, `training_summary.csv`, `learning_curves.csv`, `exposure.json`,
`inference_dependency_manifest.json`, `validation_receipt.json`, `status.json`,
`artifact_manifest.json`, selected package and `CONTINUATION.md`.

The report must answer: which representations and new observation populations
were actually tested; how much changed relative to C0; whether >=0.95 and both
embryo/replication requirements were reached; whether fixed-node or new-node changes
caused gains; where each failure occurred; and what a next iteration should not
repeat. Mark every pilot/subset/unrun/invalid family explicitly. No completed score
can be inferred from a source loss or partial clip score.
