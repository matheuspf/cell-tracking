# Validation, promotion and hand-back

## Clean does not mean untouched research history

Use the inherited v10 recursive source-exclusion contract. Source44 models may read only source44 fit/calibration data in their permitted stages; source6 models only source6. Training excludes calibration labels/images from encoder fitting and excludes all target images/labels, teachers and fitted statistics. Fixed image-local normalization at inference is allowed. Each checkpoint, normalizer, calibration file, feature cache and model parent has explicit provenance. Unknown ancestry fails reuse.

Retain the exact duplicate-connected clip groups and fixed source split from v10. Do not select a different split because it has better source or target scores. New local v10 target inspections must appear in reconciliation and the final exposure statement. Both public embryos have prior research use; report `source_isolated_reused_embryos`, not pristine independent generalization. Calibration groups are not acquisition-independent unless actually established. Two seeds are optimization replication, not new animals.

Raw target inference and label-based evaluation run in separate processes. Freeze every retained model, calibrator, bank/config and complete per-frame prediction hash before comparative target evaluation. All target directions/seed cells remain in the matrix. A failed/missing cell has null results, never P0 values, zero, or the intersection of available clips. A source-selected disabled policy is separately labeled and proves no new learned improvement.

## Metric and correctness

Pin the official scorer and dependencies at `075fc5f5a52d11077f9dc2b074644618f26939e2`; inspect upstream at execution. A changed current scorer receives separate columns and freshly matched controls, not mixed-version deltas. Verify actual T/Z/Y/X Zarr v3 axes and spacing. Expected native shape is 100x64x256x256 and spacing [1.625,0.40625,0.40625] um; metadata is authoritative.

Use 7-um one-to-one per-frame matching and the official independent local division-window rules. Map persisted IDs explicitly. Sparse unmatched predictions/edges/forks are not automatically FP. Coarse node totals are evaluator inputs only. Round once, clamp native integer coordinates, and score the actual exported CSV roundtrip.

Per clip w=edge_TP+edge_FP+edge_FN. Adjusted edge contribution is weighted by w; division Jaccard is computed from pooled division counts. Score each seed separately. Verify official empty-denominator behavior rather than assuming a value; the handover's independent aggregation helper requires explicit empty-case conventions. Do not average clip scores or invent public/private correlation.

Tests must cover negative-only and unknown groups; no invented biological nondivision; compatible timing sets; conditional denominator masking; unknown zero supervised gradients; daughter/permutation and duplicate-edit invariance; sequential LR and exact resume; missing score handling; full parent/action census; complete ownership and no-fork competition; solver/cap abstention; provenance cycles and exposed parents; coordinate/ID/CSV roundtrips; and complete population/seed accounting.

Actual execution proofs must include a real source optimizer/gradient check, tiny supervised overfit without hiding detector collapse, positive and negative legal source actions, same-recipe resume, clean feature/crop parity and complete source graphs. After freeze, cold-run at least one renamed complete target clip in each of the four source/seed cells; choose clips by fixed image-only density strata, including a crowded example in each target. Refuse GT/network/old predictions, and check exact same-environment graph/CSV equality. Report tolerance differences explicitly when nondeterminism prevents bit identity; never claim an exact proof then.

## How to interpret outcomes

Report the original user milestone separately: official pooled C11 >=0.95 in each seed with complete clean provenance, no target fitting and correct CSV scoring. Report the stronger robustness milestone only when each target embryo also reaches 0.95 in both seeds. Neither is an LB/private-score guarantee.

A promising compact improvement requires C11-C01 pooled >=0.002 in each seed and every direction delta >=-0.001; C11 must not regress against C00 in any direction by more than 0.001. Require full correctness/cold proof and nonzero beneficial edits, with exact edge/division/count decomposition. These are practical prespecified thresholds, not significance tests. Scores below the gate remain measured. A strong C01 may be the preferred next candidate; do not nominate C11 merely because it is neural. P0 remains unchanged pending review.

When results fail, distinguish missing endpoint/anchor/legal action, weak conditional ranking, unsupported occurrence calibration, excessive accepted FP, protected/conflicting actions, resource abstention and no-op safety. Source witnesses are label-guided diagnostics, not achievable score bounds. Error groups overlap and cannot be summed as independent repairs.

No new model/hyperparameter is chosen after target scoring. REPORT_BACK.md proposes the next decision; it does not silently execute a target-selected follow-up. Final all-data fitting and actual Kaggle submission require their own subsequent decision and runtime verification.

## Required committed return package

Write under `results/division-reliability-v11/`:

- `REPORT_BACK.md` and `STATUS.json`: completed/partial/blocked stages, reconciliation, what was reused, exact recommendation, exposed-data caveats, failures and tested resume commands.
- `scores.csv`, `per_embryo_scores.csv`, `per_clip_scores.csv`: arm/source/target/seed, full edge/division counts, adjusted contribution, node totals, score, baseline deltas, number of clips/frames and disabled-policy flag.
- `reconciliation.json`, `execution_lock.json`, `exposure_manifest.json`, `prediction_manifest.json`: code/scorer/data identities, recursive model parents, target-access history and exact coverage.
- `training_summary.json`, `source_diagnostics.json`, `calibration.json`, `error_funnel.csv`, `resource.json`, `validation.json`: actual schedules and group counts, positive/negative/unknown labels, witnesses, candidate multiplicity and selected-tail reliability, score safety, recovered/lost events, hardware and cold proofs.

Use blank/null missing metrics with a reason. Logs, raw images, weights, caches, per-action detailed graphs and optimizer state remain ignored locally. Commit portable model/package hashes and explicit retrieval paths, not their bytes, personal sessions, credentials or unnecessary personal information. Keep reports compact and machine-readable; another dashboard is not required.

STATUS must include `trained`, `scored`, `clean_provenance_passed`, `target_freeze_passed`, `all_four_cells_complete`, `cold_inference_passed`, `upstream_reused_cells`, `blocked_stages`, and `next_command`. A timestamp or progress JSON alone does not prove a process is running.

The final hand-back must answer: Does C01 already solve much of the fork problem? Does C11 improve it on both excluded embryos and seeds? How much of remaining error is clean detection versus actions/ranking/calibration? Was 0.95 actually reached, under which definition? What exact next experiment is justified? No local delta is added to the historical public 0.946 score.
