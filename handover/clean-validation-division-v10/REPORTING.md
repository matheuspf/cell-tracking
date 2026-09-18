# Required results and hand-back

Create `results/clean-validation-division-v10/` only for compact, sanitized measurements. Add a restrictive ignore policy before generating large artifacts. Keep raw images, graph banks, labels, weights, optimizer states, embeddings, full logs and submission files in the ignored work root. Preserve existing study outputs.

## Required portable files

- `REPORT_BACK.md`: self-contained decision report, readable from Git alone.
- `STATUS.json`: stage and arm completion, actual timestamps, blockers, no stale running claims, no default promotion.
- `execution_lock.json` and `protocol_addenda.json`: planned/actual budgets, source-only fallback decisions, frozen configs, code hashes and timing inputs.
- `input_manifest.json`, `split_manifest.json`, `exposure_manifest.json`: exact source/target populations, permitted dependencies, parent hashes and unresolved overlap.
- `experiment_matrix.csv`, `scores.csv`, `per_embryo_scores.csv`, `per_clip_scores.csv`: all registered controls/variants with null missing scores and explicit reasons.
- `training_summary.json`, `learning_curves.csv`, `sampling_audit.json`, `calibration_audit.json`: actual optimizer/group/token counts, LR history, convergence limits, support counts, source mining and source-only selection.
- `stage_attribution.json`, `error_transitions.csv`: endpoint, candidate, score, calibration, solver/protection/cap and final-metric decisions; overlapping error-family warning.
- `validation.json`, `fresh_inference.json`, `runtime.json`, `artifact_manifest.json`: tests actually run, clean cold image-to-CSV proof, resource coverage, local artifact paths/hashes and package reconstruction.

Sanitize local paths where they expose personal details; preserve a machine-relative artifact root and SHA256 identity. Results need not embed every image or graph to be verifiable. Write atomic results and immutable freeze receipts.

## Score schema

Each row needs study ID, arm, lane, exposure class, seed, source/target, checkpoint/package hash, scorer revision, exact clip/frame counts, nodes, matched annotations, edge TP/FP/FN, raw and adjusted edge terms, division TP/FP/FN and contribution, official combined score, same-seed control/delta, graph edit counts and status/reason. OP rows must never have clean=true. Historical results must be marked historical until freshly verified. Do not average seeds into a pseudo-official score.

Stage ledgers need stable source/target graph IDs and explicit label availability. Report false-fork changes alongside edge TP survival. A model that abstains must show zero edits and the reason it abstained; it cannot be described as a failed restoration/division intervention that was actually performed.

## REPORT_BACK.md structure

1. **Decision:** 0.95 milestone attained/not attained/not evaluable; promising-clean gate passed/failed; P0 retained. Explain the most important outcome in plain language.
2. **Execution scope:** branch/commit, dates, exact finished/omitted/blocked fits, common schedule choices and why. Compare planned versus actual optimizer updates and compute. State when target results were first opened.
3. **Primary table:** C00/C10 for both target embryos and pooled, separately for both seeds. Include edge/division/count decomposition, no-op status and every clean-exposure gate.
4. **Operational 2x2:** four registered sampler/calibration combinations, or the pre-lock reason the entire lane was omitted. State that this first-seed exposed diagnostic is not clean or replicated causal evidence.
5. **Diagnosis:** source learning curves, negative-only group coverage, hard-negative provenance, actual calibration support and selected-action behavior, candidate-to-final failure funnel and representative failure IDs. Distinguish observations from causal conclusions.
6. **Reproduction:** exact commands using local artifact roots, package/model hashes, environment pins, cold-inference parity, CSV/metric checks and remaining Kaggle-runtime uncertainty.
7. **Next decision:** one primary recommended study, justified by the clean error breakdown, plus what evidence would change it. No automatic new campaign, default switch or Kaggle submission.

## Completion definitions

`planned` means an entry in this handover. `implemented` means executable code exists. `trained` requires final checkpoints and actual optimizer receipts. `evaluated` requires the entire expected image-to-CSV population and official scorer receipts. `clean_evaluated` additionally requires recursive exposure isolation. `cold_verified` requires a fresh process without historical inference artifacts. These states are distinct.

A complete negative result is useful. A promising operational score without clean completion does not satisfy the user's validation goal. Partial execution must identify the precise missing stage and safely resumable commands; do not report a worker as running solely because an archived progress file says so.

No public/private leaderboard result is created by this study. The actual rule/runtime body must be refreshed locally before any later submission plan. No Kaggle account/team changes, new external-data acceptance, uploads, merges or paid compute are authorized here.
