# Multi-dataset training v4: measured transfer report

## Decision

State retained incumbent or selected new model, absolute full-data score, delta
against 0.934802374260586, both embryo deltas and whether external training beats
its matched-compute real-only control. Label the result operational exploratory.
No target-independent or leaderboard claim follows from these reused embryos.

## What actually entered training

For every collection/acquisition: available files, verified hashes, images/labels,
source terms, coordinates/cadence, eligible task, train/validation/test provenance
groups, consumed unique examples/nodes/events, weak/dense/unknown label masks,
actual sampling weights and exclusion reasons. Include all six Zoo and seven
RIKEN entries even if unused. Separate storage totals from unique scientific data.
Record 44b6 simulator calibration and inherited teacher/checkpoint exposure.

## Training and ablations

Table for C0-C6: architecture, initialization, trainable parameters, pretraining
sources, actual updates, unique positive groups, compute, real fine-tuning exposure,
external replay, selected checkpoint and source-only calibration. Include source
losses, generator holdout, Zoo weak-label diagnostic, and real graph results.
A loss curve without target evaluation is not a completed transfer result.
Include the second-seed control where run and state selection timing.

## Mechanism and errors

Report center/localization/duplicate metrics, candidate coverage after gating,
proposals per parent and total, wrong-pair/false-fork rates, recovered/lost GT-edge
identities, division TP/FP/FN, raw/adjusted edge scores and count ratios. Separate
fixed-node G/I improvement, detector-only change, decoder-only change, and final
combination. Did external negatives prevent the v3 false-fork explosion? Did gains
come from more real biological geometry, more simulated images, or merely decoder
changes? An unresolved attribution is acceptable; inventing one is not.

## Validation and deployment

List all sample coverage/hash/metric tests, source-split limitations, no-GT
inference receipts, fresh-image pilots, runtime/VRAM and offline-loading checks.
Report missing resources, excluded sources and bounded optional work. Preserve
prior outputs and declare any protocol changes made after target scores.

## Artifacts

Local output root: `/kaggle/working/cell-tracking/multidata-training-v4/`.
Required: dataset_use.csv, runtime_dataset_index.jsonl, source_manifest.json,
training_receipts.json, learning_curves.csv, calibration.json, ablation_scores.csv,
score_rows.csv, candidate_budget.csv, final_report.md, dashboard.html, status.json,
winning_config.json, checkpoint_manifest.json, inference_receipt.json,
artifact_manifest.json and NEXT_AGENT.md. Reproduction commands must actually run.

Commit sanitized tables/reports/code under `results/multidata-training-v4/` and
maintained tools/docs; keep raw images, coordinates, annotations, predictions and
weights outside Git. A failed data hypothesis still gets a complete report and
an explicit v3 fallback, not a fabricated trained winner.
