P0 retained

# Pipeline error training — September 15, 2026

Execution status: **executing**. 16 directional fits have completed the locked 178 updates.
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

## Measured model outcomes

No new model has a complete all-199 target comparison yet. The source screens are reported separately.

## Source-only branching

Source nominees: division **none**; identity/observation **A10**. Replication status: **measured**.

- A10: qualified; 44b6: delta +0.000000000, graph gate pass; 6bba: delta +0.000294383, graph gate pass.
- D10_adapted: not qualified; 44b6: delta -0.018544154, graph gate fail; 6bba: delta -0.012731289, graph gate fail.
- D20_temporal: not qualified; 44b6: delta -0.018544154, graph gate fail; 6bba: delta -0.005245592, graph gate fail.
- O10_swap: not qualified; 44b6: delta +0.000000000, graph gate pass; 6bba: delta +0.000000000, graph gate pass.

These are the predeclared complete source calibration clips, not the all-199 target comparison. The source split is not independently certified. Nominee-only replication does not establish a second-seed advantage over a newly trained matched control.

D10_random was not run because the Organoid family did not qualify in both source directions. No pretrained-advantage claim is made.

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
Equal optimizer-update counts do not imply equal computation. [Encoder operation profiles](encoder_compute_profile.csv)
report fixed 16-node forward/backward costs, boundary support and checkpoint recomputation using PyTorch's registered
FLOP formulas; these exclude unsupported operations, decision heads and variable group sizes, and are not full-training
FLOP totals or hardware throughput. Actual fit and inference times are reported separately.
Performance changes to caches were adopted after exact source parity; fit wall times reflect each job's recorded
implementation and are not a controlled comparison of architecture speed.
The executed learning-rate schedule applies warmup and cosine decay concurrently; the shortened update budget
also shortens its warmup denominator to 178. See [the exact schedule](executed_training_schedule.json).
These short, matched fits do not establish convergence or rule out the architectures after longer source-only training.
Event minibatches are conditioned on groups containing a supported biological positive: the completed temporal
packages record 15 such groups for 44b6 and 74 for 6bba. Their alternatives supply supported identity confusers and
metric-risk negatives; groups containing only negative event hypotheses are not a separate event-minibatch pool.
Identity minibatches sample supported trajectory groups. Calibration expands the supported source sample:
ordinary anchors use one deterministic temporal residue out of nine and receive ninefold weight; event-compatible
anchors are retained. These are fixed expansion weights, not proven randomized row propensities or a complete
source-field census. [The executed sampling audit](calibration_sampling_audit.json) records both raw and expanded
prevalence. These sampling choices limit conclusions about whole-field false-fork rejection.
Cumulative charged GPU lease time is 18.627 hours of 48; detailed memory/runtime evidence is in resource.json.
This includes 2.716 hours conservatively charged for an interrupted
lease through the next host boot. Its exact end was not observed, and the charge can include downtime;
see host_restart_recovery.json. Completed outputs and frozen model hashes were verified before resuming.
No measured Kaggle 12-hour runtime or leaderboard improvement is claimed. No weights were published or defaults changed.

## Executable validation and fresh inference

Tests: **measured** — 54 passed in 5.46s.

The complete primary/secondary/harmonic/DeepCenter/P0 pipeline plus D10_frozen ran from both renamed 100-frame images (specimen_alder: 168.987 s, specimen_birch: 283.788 s). Startup guards denied annotations, historical prediction caches and network access; both reconstructed P0 graphs and CSV/GEFF roundtrips were exact. These two runs are not a measured Kaggle 12-hour bound. A recommended candidate also requires its own cold-image proof.

Identical observation-policy coordinate queries can reuse the verified full-ensemble neural output. A CUDA-disabled replay rebuilt all graph feature arrays exactly at the real changed-coordinate fixture; changed node IDs or coordinates reject reuse. Each ordinary query loader still verifies checkpoints, code and image chunks. This operational cache reuse is separate from the cold-image proof.

[Correctness evidence](validation.json), [fresh-image proof](fresh_image_validation.json), [actual changed-coordinate feature proof](native_refresh_validation.json), [identical native-query reuse proof](native_query_reuse_parity.json), [indexed observation-action proof](observation_edge_parity.json), [resumption proof](resume_validation.json) and [resource measurements](resource.json).

See [CONTINUATION.md](CONTINUATION.md) for exact commands and remaining work.
