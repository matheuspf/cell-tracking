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

No candidate has passed every frozen recommendation gate; P0 is retained. A higher descriptive score alone does not establish a recommendation.

Highest completed new-model pooled result: **A10 0.934939378710** (P0 delta +0.000074392296; C4_m6 delta -0.000238991547).

Complete all-199 results, with embryos ordered 44b6 / 6bba:

- **A10**: pooled 0.934939378710; embryos 0.932236698 (+0.000509442) / 0.935278190 (-0.000006001). Edge TP/FP/FN 123190/5005/5693; division TP/FP/FN 29/92/122; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 18/0, edge FP removed/introduced 0/9; division TP recovered/lost 0/0, division FP removed/introduced 0/0.
- **A10_replication**: pooled 0.934938660898; embryos 0.932187977 (+0.000460720) / 0.935286369 (+0.000002178). Edge TP/FP/FN 123189/5004/5694; division TP/FP/FN 29/92/122; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 17/0, edge FP removed/introduced 0/8; division TP recovered/lost 0/0, division FP removed/introduced 0/0.
- **D10_adapted**: pooled 0.918152447085; embryos 0.918298579 (-0.013428677) / 0.918262675 (-0.017021516). Edge TP/FP/FN 122934/5990/5949; division TP/FP/FN 38/850/113; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 22/260, edge FP removed/introduced 18/1012; division TP recovered/lost 9/0, division FP removed/introduced 0/758.
- **D10_adapted_replacement**: pooled 0.918184567596; embryos 0.918497058 (-0.013230198) / 0.918276236 (-0.017007955). Edge TP/FP/FN 122936/5988/5947; division TP/FP/FN 38/849/113; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 22/258, edge FP removed/introduced 18/1010; division TP recovered/lost 9/0, division FP removed/introduced 0/757.
- **D10_frozen**: pooled 0.918073020654; embryos 0.919079150 (-0.012648106) / 0.918107435 (-0.017176756). Edge TP/FP/FN 122981/6057/5902; division TP/FP/FN 42/949/109; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 30/221, edge FP removed/introduced 25/1086; division TP recovered/lost 13/0, division FP removed/introduced 0/857.
- **D20_compact**: pooled 0.918100978703; embryos 0.917144686 (-0.014582570) / 0.918382547 (-0.016901644). Edge TP/FP/FN 123188/6224/5695; division TP/FP/FN 48/1245/103; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 62/46, edge FP removed/introduced 13/1241; division TP recovered/lost 19/0, division FP removed/introduced 0/1153.
- **D20_no_pretrain**: pooled 0.919262753299; embryos 0.922841374 (-0.008885882) / 0.918911867 (-0.016372324). Edge TP/FP/FN 122913/5856/5970; division TP/FP/FN 36/716/115; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 15/274, edge FP removed/introduced 11/871; division TP recovered/lost 7/0, division FP removed/introduced 0/624.
- **D20_temporal**: pooled 0.919226643837; embryos 0.919484793 (-0.012242464) / 0.919338549 (-0.015945642). Edge TP/FP/FN 123140/6050/5743; division TP/FP/FN 45/1052/106; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 45/77, edge FP removed/introduced 19/1073; division TP recovered/lost 16/0, division FP removed/introduced 0/960.
- **D20_temporal_replacement**: pooled 0.919134189858; embryos 0.919484793 (-0.012242464) / 0.919237362 (-0.016046829). Edge TP/FP/FN 123139/6042/5744; division TP/FP/FN 43/1043/108; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 44/77, edge FP removed/introduced 20/1066; division TP recovered/lost 15/1, division FP removed/introduced 2/953.
- **O10_swap**: pooled 0.934415338644; embryos 0.931542280 (-0.000184977) / 0.934785336 (-0.000498855). Edge TP/FP/FN 123145/5032/5738; division TP/FP/FN 29/92/122; predicted nodes 4108943.
  Relative to P0: edge TP recovered/lost 3/30, edge FP removed/introduced 26/62; division TP recovered/lost 0/0, division FP removed/introduced 0/0.

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
Cumulative charged GPU lease time is 25.235 hours of 48; detailed memory/runtime evidence is in resource.json.
This includes 2.716 hours conservatively charged for an interrupted
lease through the next host boot. Its exact end was not observed, and the charge can include downtime;
see host_restart_recovery.json. Completed outputs and frozen model hashes were verified before resuming.
No measured Kaggle 12-hour runtime or leaderboard improvement is claimed. No weights were published or defaults changed.

## Executable validation and fresh inference

Tests: **measured** — 54 passed in 5.46s.

The complete primary/secondary/harmonic/DeepCenter/P0 pipeline plus D10_frozen ran from both renamed 100-frame images (specimen_alder: 168.987 s, specimen_birch: 283.788 s). Startup guards denied annotations, historical prediction caches and network access; both reconstructed P0 graphs and CSV/GEFF roundtrips were exact. These two runs are not a measured Kaggle 12-hour bound. A recommended candidate also requires its own cold-image proof.

**A10** also completed its own [cold-image pipeline proof](fresh_candidate_A10.json) on both renamed 100-frame images with the frozen source models. Both candidate graphs exactly matched the scored graphs ([parity evidence](fresh_candidate_A10_replay_parity.json)). These correctness checks do not change the metric recommendation gates.

Candidate fresh inference used 337.756 seconds under the GPU lease and 888.284 seconds elapsed, including shared-GPU waits ([timing evidence](fresh_candidate_timing.json)). Lease wall time is not GPU kernel time.

Identical observation-policy coordinate queries can reuse the verified full-ensemble neural output. A CUDA-disabled replay rebuilt all graph feature arrays exactly at the real changed-coordinate fixture; changed node IDs or coordinates reject reuse. Each ordinary query loader still verifies checkpoints, code and image chunks. This operational cache reuse is separate from the cold-image proof.

One observation worker failed when the background monitor could not launch `nvidia-smi` (EFAULT). The identical frozen job passed on retry and produced identical prediction bytes; the failed attempt remains archived ([retry evidence](observation_monitor_retry.json)). Subsequent monitor launches receive an explicit environment snapshot; sampling errors still fail the job ([launch hardening](resource_monitor_subprocess_fix.json)). The exact OS failure cause was not reproduced.

[Correctness evidence](validation.json), [fresh-image proof](fresh_image_validation.json), [actual changed-coordinate feature proof](native_refresh_validation.json), [identical native-query reuse proof](native_query_reuse_parity.json), [indexed observation-action proof](observation_edge_parity.json), [resumption proof](resume_validation.json) and [resource measurements](resource.json).

See [CONTINUATION.md](CONTINUATION.md) for exact commands and remaining work.
