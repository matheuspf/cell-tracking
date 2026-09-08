# Annotation-selection study: final report

Status: [positive_local_replication / promising_but_uncertain /
no_transferable_signal / incomplete_or_invalid]. Date, code revision, metric
revision, data/split hashes, environment, GPU. Link every table to its source.

## Executive answer

State what is measured, what is estimated and what remains unidentifiable.
Answer separately: annotation predictability, deployable local score benefit,
count-adjustment contribution, evidence of annotation-specific rather than
quality-only selection, and plausible limits on hidden-embryo transfer.
Do not turn conditional two-embryo results into an unseen-embryo guarantee.

## Data and labeling

Per embryo and total: number of clips/overlap groups, annotated nodes/edges/
divisions, supplied estimated totals, ratio-of-sums p_ref, candidate P/M/p_match,
GT detection recall, count ratios c0. Count observations versus unique cells.
Show missing metadata and ambiguous candidate rates. Include independently audited
true-cell estimates/intervals only when supported; otherwise mark unavailable.
Describe optional census pack status and sampling design.

## Provenance and validation

Show both directions, inner grouping, overlap checks, upstream predictor training
sets, embargo/lock timestamps and GT-unavailable inference test. Label every lane
clean, contaminated or unknown. List any deviations before/after outer revelation.

## Learnability and operating points

For each locked fraction and model: retained fraction, annotation recall, specificity,
precision, phi/MCC, deleted-group annotation rate/depletion, AUROC, AP and calibration.
Report per embryo; include natural prevalence and cluster uncertainty. Connect
node recall to measured old-TP survival/new TPs instead of assuming recall squared.
Distinguish source-selected primary results from hindsight best sweep/oracle points.

## Actual graph score

Table: lane, direction, policy, model, seed, requested/realized keep, TP/FP/FN,
J0/J1, D0/D1, adjusted edge values, S0/S1/delta, count ratios, matched sample count.
Show pooled official aggregation, baseline/full-identity parity and rare-division
behavior. Include per-sample rows and no-skip assertions. Explain rematching changes.

## Controls, attribution and uncertainty

Random versus quality-only versus membership filters at matched realized budgets.
Alpha=0 diagnostic and count/graph/division attribution. Shuffled-target/random-
annotation controls, quality-audited subsets, oracle feasibility and source-only
selection. Paired supergroup uncertainty, both embryo results and seed variability.
Acknowledge the lack of independent new embryos for unrestricted follow-up tuning.

## Decision and next action

Answer whether worthwhile operating points exist at the measured c0 and baseline
J/D values. Provide exact requirements derived from those values, not the earlier
hypothetical numbers. Give the best source-selected policy, or explain the failure.
Separate lack of signal from poor detection, insufficient labels, contamination or
metric mismatch. No automatic submission or deployment.

## Reproduce and inspect

Provide actual executable commands, resolved configs, runtime/VRAM/RAM/I/O profile,
artifact hashes, resumed/failed stages and a complete status ledger.

Local deliverables (under ignored output root):
`final_report.md`, `dashboard.html`, `sample_inventory.csv`, `retention.csv`,
`score_rows.csv`, `summary.json`, `fold_manifest.json`, `environment.json`,
`artifact_manifest.json`, `status.json`, optional blinded census pack.
The HTML must work offline and clearly label real measurements versus hypothetical
sensitivity curves. Include coverage/count plots, high-recall keep-budget curves,
deleted-label rate, score/edge/division tradeoffs, per-embryo comparison and intervals.
Do not fabricate plots where experiments did not run. No raw patches/GT coordinates
or checkpoints in Git; commit a sanitized report and links to local artifact paths.
