# Multi-dataset training v4: measured transfer

Retain the v3 incumbent.
Selected score **0.934802374260586**, delta **+0.000000000000000** against v3 A_residual_m3.0 = **0.934802374260586**, on all 199 clips.

This study is operational exploratory. Both embryos have been repeatedly reused. The released simulator was calibrated on 44b6, including when 44b6 is the transfer target. The incumbent public checkpoints and historical E teachers retain inherited label exposure. No local delta is a leaderboard gain.

## Measured whole-graph comparisons

| Variant | Pooled score | Delta v3 | Delta C1 (same seed) | 44b6 delta | 6bba delta | Division TP/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| C4_m6 | 0.935178370257 | +0.000375995996 | +0.000375995996 | +0.002040084759 | -0.000034064563 | 30/92/121 |
| C5 | 0.934955680132 | +0.000153305871 | +0.000153305871 | +0.001672144891 | -0.000224792273 | 30/94/121 |
| C4 | 0.934949150386 | +0.000146776125 | +0.000146776125 | +0.001007989027 | -0.000116941379 | 30/95/121 |
| decoder_only | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| C0 | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| C1_seed2 | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| C1 | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| original_objective_C4 | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| C3 | 0.934802374261 | +0.000000000000 | +0.000000000000 | -0.000000000000 | +0.000000000000 | 29/92/122 |
| C7 | 0.934794499038 | -0.000007875223 | -0.000007875223 | +0.000837660173 | -0.000267317079 | 30/97/121 |
| C1short | 0.934746568938 | -0.000055805323 | -0.000055805323 | -0.000341473348 | +0.000000000000 | 29/93/122 |
| C6 | 0.934715204078 | -0.000087170183 | -0.000087170183 | -0.000075298823 | -0.000182098615 | 30/100/121 |
| C2 | 0.934578501259 | -0.000223873002 | -0.000223873002 | -0.000184584337 | -0.000229545800 | 29/95/122 |
| C4_seed2 | 0.934542128420 | -0.000260245840 | -0.000260245840 | -0.000276451744 | -0.000255600022 | 29/95/122 |
| C4_m2 | 0.934039326074 | -0.000763048187 | -0.000763048187 | +0.000559924049 | -0.001092626952 | 30/108/121 |
| C4_Gonly | 0.929719967340 | -0.005082406920 | -0.005082406920 | -0.011957762322 | -0.003563779044 | 31/201/120 |
| C4_relinked | 0.922258418418 | -0.012543955843 | -0.012543955843 | -0.008977270984 | -0.013318623507 | 2/6/149 |
| D_synthetic_C4 | 0.922171819716 | -0.012630554544 | -0.012630554544 | -0.008006957933 | -0.013530273555 | 2/9/149 |
| C1_relinked | 0.907559949709 | -0.027242424551 | -0.027242424551 | -0.021730102725 | -0.028098744283 | 0/0/151 |
| D_synthetic | 0.906137834411 | -0.028664539850 | -0.028664539850 | -0.023426646860 | -0.029470420575 | 0/0/151 |
| D_real | 0.906048374545 | -0.028753999716 | -0.028753999716 | -0.022359251731 | -0.029772496475 | 0/0/151 |

## What the measurements establish

C4 scored 0.934949150386, a pooled delta of +0.000146776125. Its embryo deltas were +0.001007989027 on 44b6 and -0.000116941379 on 6bba. The 6bba regression fails the prespecified adoption rule.
C5 scored 0.934955680132, a pooled delta of +0.000153305871. Its embryo deltas were +0.001672144891 on 44b6 and -0.000224792273 on 6bba. The 6bba regression fails the prespecified adoption rule.
The descriptive margin-6 C4 control scored 0.935178370257, but its 6bba delta was -0.000034064563. A higher pooled score alone does not qualify it. The full real-only C1 and fish-geometry C3 primary policies abstained and preserved the incumbent exactly; the shorter real-only, synthetic-only C2 and appearance-randomized C6 policies regressed in pooled score.
Removing C4 optical evidence reduced the pooled score by 0.005229183045. When continuations were rebuilt, external C4 exceeded matched real-only C1 by 0.014698468708, but remained -0.012543955843 below v3. These controls support a role for image evidence and learned associations inside this pipeline; they do not establish an improvement over the incumbent.
All detector variants regressed relative to v3. detector_attribution.csv compares each coordinate refinement with the identical association policy on unmoved centers, separating refinement from the larger cost of rebuilding continuations.
C1_seed2: pooled 0.934802374261, delta v3 +0.000000000000; 44b6 -0.000000000000, 6bba +0.000000000000; pooled difference from its same-seed C1 control +0.000000000000.
C4_seed2: pooled 0.934542128420, delta v3 -0.000260245840; 44b6 -0.000276451744, 6bba -0.000255600022; pooled difference from its same-seed C1 control -0.000260245840.
C7: pooled 0.934794499038, delta v3 -0.000007875223; 44b6 +0.000837660173, 6bba -0.000267317079; pooled difference from its same-seed C1 control -0.000007875223.

## Actual training and source coverage

Training completed **804,000 optimizer updates** across 69 saved production fits, taking 4.257 summed hours in timed optimizer/validation loops on the RTX 4090. Initial cache loading and separate sanity fits are outside that sum; it is not a measure of continuously saturated GPU compute. Peak training allocation was 1.594 GiB.
D uses native static-image center queries and subvoxel offsets. I uses a shared triplanar frame encoder, explicit masked temporal pooling, and separate parent and daughter evidence. G uses only relative per-axis-normalized point geometry with label-blind nearest-frame context; clean links and clone IDs never enter the geometry input. Primary C1–C6 never use Zoo as image input; optional C7 uses only explicitly rendered Zoo images.
The fixed component budgets are D 12,000, G 20,000, and I 20,000 pretraining updates, then 8,000 adaptation updates per applicable component and source. C1 repeats source Biohub for the full matched budget. C1short receives only G/I adaptation from random initialization. Reused pretraining checkpoints are accounted for once in storage and process totals; see individual histories for each arm.
Adaptation batches contain 75% identically sampled supported source observations and 25% external replay. In real-only controls, the remaining 25% performs source-only consistency without extra ground-truth loss, matching primary real supervision and optimizer steps. Exact sampled rows and visited groups are recorded; alternatives are not counted as independent events.
Non-fork event targets require two recorded single-child transitions on each side of the anchor. This is conditional annotation support: sparse graphs cannot certify absence of every biological daughter. Quiet-chain event labels remain a weak-supervision assumption even after unsupported edge negatives are masked. Event-bank sampling excludes censored/unreachable bags; their usable partial edges are not separately exploited.
An initial real-only detector adaptation diverged under raw-logit consistency and was stopped after its 5,000-update checkpoint. It is preserved under failed_fits and excluded from production comparisons. All real-only adaptations restarted with bounded probability consistency before comparative predictions. Real-only consistency also performs a detached teacher forward; optimizer steps and supervised rows match, but exact FLOPs differ.
A subsequent pre-outcome audit found unsupported edge negatives: a parent with one recorded child incorrectly contradicted unmatched possible second daughters. Those labels were changed to unknown for Biohub and weak Zoo supervision. All affected G/I pretraining and adaptations, including the early second-seed real-only controls, were preserved and restarted from their original initializations for the full budgets. Event targets, input features, partitions and sampling were unchanged. Dense synthetic pretraining and D fits were unaffected. The correction and failed compute accounting are separate receipts.
D-to-I transfer copies only the compatible convolutional encoder. Native D patches and isotropically sampled I patches have different physical support; this is an explicit scale-transfer limitation. The compact D experiment refines existing center proposals; it is not a replacement full-volume U-Net or a new-peak recall experiment.
Real-only detector supervision includes 1,893 center queries and 8,781 image-validated dark-background queries for 44b6, and 9,783 center queries and 17,446 background queries for 6bba; every clip supplies background. Unmatched detections are not labeled negative. Dark background remains heuristic supervision, so very dim unannotated cells cannot be ruled out.

| Source | Partition | Examples/blocks | Event bags | Reachable positive groups |
|---|---|---:|---:|---:|
| biohub_44b6 | source_only_adaptation | 71 | 8311 | 20 |
| biohub_6bba | source_only_adaptation | 128 | 16507 | 84 |
| synthetic_sequence | test | 91 | 11648 | 506 |
| synthetic_sequence | train | 1973 | 94704 | 41485 |
| synthetic_sequence | validation | 110 | 14080 | 607 |
| synthetic_static | test | 72 | 0 | 0 |
| synthetic_static | train | 1386 | 0 | 0 |
| synthetic_static | validation | 81 | 0 | 0 |
| zoo_ascidian | test | 1 | 604 | 20 |
| zoo_ascidian | train | 1 | 3568 | 132 |
| zoo_ascidian | validation | 1 | 90 | 23 |
| zoo_zebrafish | test | 1 | 9344 | 4086 |
| zoo_zebrafish | train | 1 | 45824 | 20083 |
| zoo_zebrafish | validation | 1 | 8448 | 3702 |
| rendered_zoo_44b6 | train | 1 | 45824 | 20083 |
| rendered_zoo_44b6 | validation | 1 | 8448 | 3702 |
| rendered_zoo_44b6 | test | 1 | 9344 | 4086 |
| rendered_zoo_6bba | train | 1 | 45824 | 20083 |
| rendered_zoo_6bba | validation | 1 | 8448 | 3702 |
| rendered_zoo_6bba | test | 1 | 9344 | 4086 |

Synthetic train/holdout identities are preserved. Generator validation and test are engineering diagnostics. Zoo train/validation/test use contiguous acquisition time blocks with six-frame purges and are not independent embryos; their coordinate IQR normalization uses the whole unlabeled acquisition, including held-out blocks. Only the eligible ascidian acquisition supplements zebrafish, capped at 25% of geometry pretraining/replay batches; other species and all RIKEN acquisitions remain excluded as individually recorded in dataset_use.csv.

## Decoder, calibration and attribution

The installed raw solver was tested with the original 1.2 division cost and a parent-specific learned event term: zero gain selected no fork, positive gain selected a true fork, and the nondivision control retained a continuation. Deployment uses the tested local binary event objective, explicit no-op and conflicting-parent/daughter/owner constraints. Existing fork evidence is protected through two generations. A blanket lower division penalty is not used.
Each parent has at most six daughter candidates and 15 unordered pair alternatives. The geometry gate is fixed at 0.5. A pair must beat the log-sum-exp of competing pairs and no-fork by margin 4.0 after temperature scaling. C4_m2 and C4_m6 are the handover-prescribed descriptive margin-2 and margin-6 controls, both wired in before prediction freezing. The original-objective and zero-new-head decoder controls preserve v3.
The 199-clip original-objective negative control is explicitly a structural no-op on the frozen v3 graph. The installed raw solver was executed in the positive/negative fixtures; it was not rerun globally for every identity-control clip.
Temperature is fit on unbalanced observed generator validation bags, using the same calibration access for every arm. Thus “real-only” describes neural weight training; it does not exclude shared external calibration. This is not a biological posterior. Independent source inner groups could not be certified; the fallback primary margin was fixed before new target outcomes. No prior multiplier or target-label threshold search was used.
Primary C1–C6 modify fixed incumbent nodes only. C1_relinked/C4_relinked rebuild continuations from current-point geometry. D_real and D_synthetic refine center coordinates and rebuild all v4 proposals/associations under C1; D_synthetic_C4 is the limited combination. Count and duplicate diagnostics are reported separately from tracking score. No old residual feature array is reused after moving centers.
Final adoption additionally requires a complete qualifying second-seed replication. A positive exploratory variant without that replication is retained in the measured tables but does not replace v3. The second seed is never selected instead of the primary seed.
External-data advantage for the selected result: **False**. delta_vs_C1 uses the control from the same seed; a verified external advantage requires positive differences in both seeds and both embryos. Gains caused by architecture, compute or decoding alone are not credited to external data.

## Validation, scope and artifacts

All 41,468 archived/prepared files passed fresh hashes. All 3,713 synthetic raw/prepared grids were checked. Actual source overfits passed for G/I/D, including encoder gradients, unordered daughter and padding invariance, future censoring, and actual solver fixtures. Thirteen additional contract tests check empty observations, detector displacement units and rounding, sparse unknowns and possible unannotated second daughters, missing future frames, unit scaling, duplicate identity, and conflicting/protected forks, including displaced old children.
The six generator stress conditions alter point observations while retaining the original optical frames. Short history tests missing tracking history, and clipped daughters test missing candidate detections. Stress anchors are known simulated cells; robustness to false-positive parent detections is not established by these diagnostics.
Predictions for both directions and every primary comparison were serialized before comparative official scoring. Every complete graph was validated and scored using the pinned official metric, revision 075fc5f5a52d11077f9dc2b074644618f26939e2; independent aggregation checks passed.
W470 trajectory-backed rendering is conditional on a predeclared external-utility/real-transfer-gap/time test after the required comparisons. The rendering_trial_receipt.json states whether it ran. C7, when measured, replaces half of optical pretraining and replay with Gaussian triplanes rendered from eligible Zoo-fish points using source-only appearance and density statistics. It retains C4 geometry histories and full matched budgets. These are explicitly simulated patches from weak trajectories, not experimental Zoo microscopy; existing acquisition and simulator exposure caveats persist.
Fresh-image/package, second-seed, preservation, and resource receipts state their actual completion independently. The local 4090 runtime is not a verified Kaggle 12-hour guarantee. No submission, notebook publication, gated agreement, new hardware, or paid API was initiated.
The container rejected the Linux user/mount/network namespace probe with Operation not permitted. Offline fresh inference therefore uses process-start Python audit hooks denying annotation and external-training paths and nonlocal sockets. Those datasets remain mounted but are inaccessible through the audited reads; this is dependency-use evidence, not an OS isolation or physical-unmount claim. Local IPC remains allowed.

Local models, detailed predictions, matched identities and image crops remain under /kaggle/working/cell-tracking/multidata-training-v4/. Sanitized score rows, source ledger, learning/transfer curves, calibration caveats, model hashes and this report are exported to results/multidata-training-v4/. Git is not a backup of microscopy or weights.

Reproduction: see docs/multidata-training-v4.md. Inference: see the local inference_package/README.md and inference_receipt.json.

The individual-graph C4 deployment path was replayed against the primary shared-patch path on 199 clips. Exact whole-graph parity passed: **True**. No annotations or new score tuning entered this check.

## Observed candidate attrition

For primary C4, 149/151 observed GT fork parents matched incumbent detections; 113 also had both daughters matched and 104 fit the six-candidate, exact-next-frame proposal set. The fixed gate retained 36 of those exact-ID forks, and 1 exact daughter pairs passed the margin. 0 were newly accepted exact-ID edits after incumbent protection and conflict resolution. This strict immediate-ID diagnostic differs from the official temporal-tolerance division matching, so it is not an alternative official TP count or a biological recall estimate. The full stage counts and control comparisons are in candidate_gt_coverage.csv; they were computed after freezing and were not used to change the gate or threshold.

## Fresh-image delivery measurements

12 fresh image-to-graph executions on 6 fixed clips completed with exact scored-graph parity and integer CSV roundtrips. The external C4 heads were actually loaded; C0 exercised the disabled-head fallback. Additional clips were selected from frozen incumbent density, without labels or scores.
C0: 6 clips, mean 141.7 seconds, maximum 201.0 seconds; a simple 199-clip extrapolation is 7.83 hours on this 4090. Peak reported parent/child RSS was 0.58/3.54 GiB; the new-head GPU allocation peaked at 0.000 GiB. The RSS figures are per-process maxima, and the new-head allocation excludes upstream detector allocations. The small density-selected runtime sample does not certify all-clip or Kaggle runtime.
C4: 6 clips, mean 150.5 seconds, maximum 212.7 seconds; a simple 199-clip extrapolation is 8.32 hours on this 4090. Peak reported parent/child RSS was 1.39/3.62 GiB; the new-head GPU allocation peaked at 0.098 GiB. The RSS figures are per-process maxima, and the new-head allocation excludes upstream detector allocations. The small density-selected runtime sample does not certify all-clip or Kaggle runtime.

Preserved failed/superseded checkpoints additionally account for 431,000 saved optimizer updates and 2.521 summed timed hours. Unsaved work may add to that total; sanity updates are separate. These weights never enter the reported comparisons.

C7 geometry-history parity was checked before its target outcomes: all 2 source fits have configurations identical to same-seed C4 except their names; bitwise weight equality holds for 2 fits. Maximum relative parameter L2 difference was 0. The intended treatment changes the optical data.

An additional fresh C4 run used the same fixed image under the unfamiliar name unseen_v4_pilot, with source model 6bba supplied explicitly. Complete graph parity and CSV renaming passed: **True**, in 163.3 seconds. This is a deployment-routing test, not a new biological sample.

A post-outcome check evaluated the frozen C4/C7 optical towers on the rendered Zoo holdouts at temperature 1, without updates or calibration changes.
Source 44b6: rendered-test correct-pair recall was 0.004 for C4 and 0.648 for C7 on 898 observed positive bags. Wrong-pair fractions among all observed bags were 0.003 for C4 and 0.216 for C7. These are weak, conditionally sampled labels from time blocks of the same acquisition; they do not establish biological generalization.
Source 6bba: rendered-test correct-pair recall was 0.013 for C4 and 0.663 for C7 on 898 observed positive bags. Wrong-pair fractions among all observed bags were 0.003 for C4 and 0.197 for C7. These are weak, conditionally sampled labels from time blocks of the same acquisition; they do not establish biological generalization.
