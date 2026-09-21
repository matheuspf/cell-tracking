# Division reliability v11 — running

Actual status at 2026-09-21T23:43:36.980836+00:00: 4/4 C00 fits, 4/4 C01 fits and 4/4 C11 fits complete; 24/1194 target predictions complete; 0/1194 required target clip/arm scores recorded.

The immutable schedule is U=8,000 and E=4,000 for both embryos and both seeds. Allocation stays 4/34/18/16 GPU lease-hours for pilots/upstream/event/inference. All six affordability candidates and the 25% margin are in allocation_projection.json. No target outcome selected the schedule.

The local v10 work was preserved. Neither completed nor partial v10 weights qualified for reuse because the matching trainer source and recursive ancestry were unavailable. Every retained v11 neural component starts randomly. P0 and the two pre-existing user-modified public946 reports remain unchanged.

Both embryos have historical research exposure. The claim is source_isolated_reused_embryos, not pristine independent generalization. Multiple seeds do not add embryos; calibration clips are not proven acquisition-independent. The original grouped split was retained. One persisted division event occurs in two source44 fit clips and receives one event unit across both.

C01/C11 independently edit their own C00 graph. Their comparison tests practical model-family value, not the isolated causal effect of factorization or one loss. Unknown legal forks remain in deployment denominators and do not become negative biological labels. Source safety and no-op outcomes are reported separately. CSV coordinates and IDs, full populations, official empty-division behavior and the pinned scorer are used; clip scores are never averaged.

The upstream training adapter uses annotation-matched proposal queries for supported incoming groups; complete inference uses dense detections. This leaves a training/inference attention-context difference. Low-intensity background masks are heuristics, not certification that unannotated voxels contain no cells. Full source mask audits and detector-collapse witnesses are retained.

New v11 GPU lease accounting: 12.6041 hours, including measured failures and conservative early-pilot allowances. Historical v10 accounting is separate: 19.3074 hours, including an 8.4-hour unobserved-tail upper bound that may include idle time. Raw telemetry, private logs, checkpoints, arrays and complete predictions stay in work/division-reliability-v11/.

Resource reporting separates original prediction/bank timings, optimizer lease intervals and later cache-reuse process times. Some original controller wall-time receipts were overwritten during the first restart; their exact process durations are unavailable. All GPU lease charges remain accounted for. Optimizer intervals measured from the journal exclude startup and final serialization, so they are reported as observed intervals, not complete process wall times.

Source engineering proofs are actual executions, not retained model scores. They include batch-eight optimizer updates, exact resume, mixed 32-group compact gradients, native crop parity, complete source graphs, and official true/false-fork witnesses. The 250-update pilots produced excessive detections and almost no links; those failures are retained. Early pilot witnesses bypassed the global 2% cap. Later witnesses on retained C00 graphs use the registered solver and cap. Both kinds use labels to choose diagnostic edits and are not learned policies or achievable score bounds. Runtime tests and planning contracts are not evidence of trained accuracy.

The host-memory guard stopped 6bba/20260918 C11 at recorded update 321 when available RAM reached 9.950 GiB, below the registered 10 GiB floor. The checkpoint preserved at the interruption is update 294; 27 recorded updates require replay on recovery. Recovery status: recovered; interruption-specific replay check: passed. All closed GPU leases remain charged, and no unclosed lease required an extra charge. host_memory_interruption.json contains preservation hashes, the resource trigger and recovery evidence. Recovery compared all 27 replayed update records exactly for samples, losses, gradients, learning rates and denominators. Timing and observation-cache occupancy are excluded. No checkpoint existed at the last compared update, so this does not establish model tensor equality there.

At 2026-09-21T19:33:04.165077+00:00, the provisional remaining wall-time estimate was 36–60 hours. It assumes uninterrupted execution and enough host memory for the documented worker limits. Target throughput was not measured for this estimate. runtime_estimate.json records the phase projections and limitations; this is not a model result or a guaranteed deadline.

A midpoint mining implementation failure exposed a TF32 singleton-reference discrepancy. C11 evaluation workers now set NVIDIA_TF32_OVERRIDE=0 before importing numerical libraries, symmetrically for mining, calibration, prediction and cold inference. Fitting settings, checkpoint parameters, bank definitions and the absolute 1e-5 parity tolerance are unchanged. The three tested 4,096-item embedding batches are bit identical; a complete source C00 image-to-CSV control under the override also matches every graph array and CSV byte. This source precision proof does not replace post-freeze target cold validation. Original failures and compute remain accounted for; compact_precision_repair.json records the correction.

An optional bounded source-mining overlap launch omitted CUBLAS_WORKSPACE_CONFIG=:4096:8. Deterministic-mode checks stopped 6 clip workers before any complete mining result. The main fit continued; failed attempts retained 80 closed leases totaling 3.050835 seconds, all charged. Recovery status: recovered. The corrected launcher uses the registered environment with unchanged midpoint, banks and numerical code. mining_overlap_environment_repair.json records the preserved failures, lease status breakdown and subsequent execution checks.

A host restart interrupted the fourth upstream fit and source clip workers. Preserved state resumed at update 5958; 213 replayed update records were compared with their pre-interruption records, with exact agreement True for samples, group selections, learning rates, loss components and gradient norms. There was no checkpoint at the last compared update, so this comparison does not establish model tensor equality there. The unclosed GPU lease was conservatively charged 60 seconds. Complete clip receipts were verified and reused; incomplete clips restarted from unchanged parents. host_restart_resume.json records the preservation hashes and tested recovery.

The host boot at 2026-09-21T17:32:33+00:00 followed a second interruption during source C01 calibration. All eight neural states passed hash verification: four C00 finals, three C11 finals and the final C11 midpoint. The preserved state includes 7 complete calibration caches, 48 complete final-seed mining clips and 7 interrupted CPU jobs. No optimizer updates were lost and no unclosed GPU lease required an additional charge. Recovery status: recovered. Exact CPU runtime after the last resource samples is unknown; host_restart_20260921.json preserves those timing limits, hashes and recovery checks.

4/4 complete upstream artifacts passed full-history audits: every optimizer sample belongs to source-fit clip/time support, all recorded losses/gradients and retained weights are finite, and final weights exactly equal the final resumable model. The resource journal reconciles the retained trajectories plus 213 discarded update attempts. These artifact checks establish execution consistency, not predictive accuracy; validation.json retains the receipts.

- Source safety 44b6/20260918/C01: margin 2, 126 source calibration edits, score change +0.000332. Occurrence support: 4 distinct positive events and 2114 negative groups; insufficient-support fallback False.
- Source safety 44b6/20260918/C11: margin 8, 3 source calibration edits, score change +0.000000. Occurrence support: 4 distinct positive events and 2114 negative groups; insufficient-support fallback False.
- Source safety 44b6/314159/C01: margin 8, 0 source calibration edits, score change +0.000000; selected policy made no calibration edits. Occurrence support: 4 distinct positive events and 2134 negative groups; insufficient-support fallback False.
- Source safety 44b6/314159/C11: margin 8, 10 source calibration edits, score change +0.000000. Occurrence support: 4 distinct positive events and 2134 negative groups; insufficient-support fallback False.
- Source safety 6bba/20260918/C01: margin 8, 0 source calibration edits, score change +0.000000; selected policy made no calibration edits. Occurrence support: 19 distinct positive events and 15779 negative groups; insufficient-support fallback False.
- Source safety 6bba/20260918/C11: margin 8, 0 source calibration edits, score change +0.000000; selected policy made no calibration edits. Occurrence support: 19 distinct positive events and 15779 negative groups; insufficient-support fallback False.
- Source safety 6bba/314159/C01: margin 2, 4 source calibration edits, score change +0.000022. Occurrence support: 21 distinct positive events and 15107 negative groups; insufficient-support fallback False.
- Source safety 6bba/314159/C11: margin 8, 1 source calibration edits, score change +0.000000. Occurrence support: 21 distinct positive events and 15107 negative groups; insufficient-support fallback False.

Comparisons for unfinished arms/populations remain unmeasured. Available complete-population scores are retained; missing comparisons are blank with a reason in the three score CSVs. No 0.95 milestone is claimed from an incomplete matrix.

Global target freeze: False. Complete cold validation: False. Clean provenance for all twelve packages: True.

Resume from the repository root after verifying no existing controller owns the study:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. CUBLAS_WORKSPACE_CONFIG=:4096:8 /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m division_reliability_v11 run --workers 3
```

While a controller is active, use the same command with `report` in place of `run --workers 3` to refresh STATUS. Cell owner locks prevent duplicate training. Durable checkpoint paths, SHA-256 hashes, exact saved update counts, and optimizer/RNG availability are in training_summary.json. Each stage uses atomic output receipts and can be invoked individually with `--source`, `--seed`, and the relevant arm/clip/partition.

The resume proof is a fresh-process 10+10 versus uninterrupted 20-update comparison across source cells, including the transition into proposal queries. Compact mixed-objective replay is also bit exact. A CUDA lazy-initialization RNG reset found during testing was repaired before the lock; failed attempts remain private. Max-pool backward carries a PyTorch determinism warning, so claims of exactness are limited to the tested executions and cold results.

Current next decision: Finish the locked matrix using the existing owners/checkpoints; do not select a new model from partial source or target outcomes.

No merge, Kaggle submission, weight publication, all-data refit or unrelated handover study was launched.
