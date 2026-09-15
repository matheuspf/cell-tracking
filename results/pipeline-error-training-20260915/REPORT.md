P0 retained

# Pipeline error training — September 15, 2026

Execution status: **executing**. 4 directional fits have completed the locked 178 updates.
The adopted production default remains P0. The recommendation is recorded separately in recommendation.json.

## Exact controls and evidence

All 199 complete clips reproduce P0 **0.934864986413134**, C4_m6 **0.935178370257** and C0 **0.934802374260586**.
Each control has 4,108,943 nodes. The zero-head D00 reproduces P0 exactly.
Both embryos and all recovered/lost TP and removed/introduced FP counts are reported; error families overlap.

- [Experiment status](experiment_matrix.csv)
- [Pooled full metrics](scores.csv) and [each embryo](per_embryo_scores.csv)
- [Every clip](per_clip_scores.csv) and [changed errors](error_transitions.csv)
- [Directional fits](training_fits.csv) and [source-only full-clip screens](source_model_scores.csv)
- [Source feasibility witnesses](source_feasibility_scores.csv)

## Interpretation

These modules use source-only direct fitting and calibration on the exposed P0 proposal pipeline.
Inherited primary/secondary checkpoints and E teachers prevent a clean end-to-end out-of-fold claim.
Exact overlapping frames were unioned before deterministic whole-clip source partitioning, but missing acquisition offsets
prevent independent inner-validation certification. Source calibration and family nomination are exploratory.
Stress diagnostics describe perturbations of these same clips; they do not supply independent biological validation.

Organoid last-block adaptation competes with its frozen backbone control. The native temporal head competes with
the compact representation and the same temporal architecture without identity pretraining. Continuation-only and
closed-bank observation selection run independently. Complete decisions include continuation, birth and other-parent alternatives.

## Execution and limits

The common source-only throughput lock reduced the proposed 16,000-update ceiling to 178 updates,
with equal budgets across matched fits. Models use the fixed final checkpoint, source-only regularized calibration,
and no target threshold selection. Failed implementation attempts remain under the new ignored invalid/ directory.
Cumulative charged GPU lease time is 3.608 hours of 48; detailed memory/runtime evidence is in resource.json.
No measured Kaggle 12-hour runtime or leaderboard improvement is claimed. No weights were published or defaults changed.

See [CONTINUATION.md](CONTINUATION.md) for exact commands and remaining work.
