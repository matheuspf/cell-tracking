# V10 execution plan

## 1. Decision and evidence

Build a clean full-pipeline reference first; test one compact division learner with corrected optimization, supported negative-event sampling and deployment-aware calibration. Keep P0 as the operational reference and preserve all older outputs. Native-resolution detection, observation repair, larger backbones, external-data collection and automatic final all-data training are deferred to the next decision.

The base branch's completed report retains P0 at 0.934864986413134. C4_m6 is 0.935178370257; A10 reaches 0.934939378710 without changing divisions. D20_compact recovers 19 additional divisions but introduces 1,153 additional evaluable division FP. O10_restore makes no graph edits. Sources: `results/pipeline-error-training-20260915/REPORT.md` and `docs/pipeline-errors-20260915.md`.

The latest event fits ran 178 updates, with 89 identity-only updates in standard arms. Warmup and cosine operated concurrently. Positive-containing event groups were sampled, but negative-only ordinary event groups were not their own minibatch pool. The sampling audit also distinguishes deterministic temporal expansion weights from real randomized inclusion probabilities. These motivate controlled tests; they do not prove which issue caused the failures. Sources: the base study's `executed_training_schedule.json`, `calibration_sampling_audit.json`, and `tools/pipeline_error_training/{dataset,train,calibration}.py`.

The dominant released detection component has documented exposure to all 199 clips. The September 16 readiness audit found no final source-only upstream checkpoints. Therefore an exposed score near 0.935 is not a clean starting score. Sources: `docs/incumbent-provenance-20260914.md` and the base study's `clean_upstream_readiness.json`.

## 2. Primary question

Can a fully source-isolated detector/encoder/linker plus a compact division module improve the official graph score on the excluded embryo, without a false-fork explosion, and reach 0.95 in both directions?

The primary comparison is **C10 minus C00**, separately for both seeds. C00 is a new scratch-trained two-frame coarse-grid detector/association model with a declared continuation-only graph decoder. C10 adds the registered compact complete-event module to C00's fixed observations and continuations. C00 deliberately has no learned division mechanism; it is not P0, not the historical native pipeline and not a detector-only score. Record its zero division contribution rather than obscure that design.

Clean means target-excluded fitting of the entire dependency chain. It does not mean the two public embryos are pristine research holdouts; both have been inspected historically. Report that qualification prominently.

## 3. Fixed matrix

Directions are source 44b6 -> target 6bba and source 6bba -> target 44b6. Seeds are 20260918 and 314159. Each target run covers every target clip and frame.

| Lane / arm | Seeds | Definition |
|---|---|---|
| C00 | both | New source-isolated upstream model and continuation graph; no event edits. |
| C10 | both | Same seed/direction C00 plus a freshly fitted N sampler compact event model, deployment calibration and source-fixed margin. |
| OP_P0 / OP_C4 | historical | Verified complete controls only; never clean models. |
| OP_R_ROW | first | P0 observations; R positive-containing event sampler; corrected schedule; row calibration. |
| OP_R_DEPLOY | first | Exactly OP_R_ROW weights; deployment calibration instead. |
| OP_N_ROW | first | P0 observations; N mixed supported-negative sampler; same schedule; row calibration. |
| OP_N_DEPLOY | first | Exactly OP_N_ROW weights; deployment calibration instead. |

Maximum new neural fits: four clean upstream fits, four clean N event fits and four operational R/N event fits = **12**. Calibration variants reuse weights. No OP second-seed comparison is registered; no replicated causal sampling/calibration advantage may be claimed from OP. The primary clean effect is replicated against its own same-seed C00. No architecture or primary arm is selected from OP target results.

OP is an explanatory secondary lane. Its entire four-fit matrix may be omitted before the execution lock for missing artifacts or resource infeasibility. Do not retain only the favorable direction or arm. The clean lane remains mandatory and independent.

## 4. Stages

### S00: preflight and preservation

Inventory actual inputs, local source, runtime and running jobs. Validate expected 71/128 clip counts, Zarr v3 axes and physical scales, GEFF IDs and pairing. Record original graph/model/source hashes without importing exposed inference into clean workers. Recheck scorer revision and local official references. Establish durable free-space reserves and prospective storage needs; do not solve disk shortages by deleting old studies.

Deliver environment, input/provenance manifest, source-only split manifest, preservation receipt and concrete blockers. Do not read target annotations for training, tuning or stage selection. A separate evaluator/data curator can inventory them without exporting label-derived fields to model inputs.

### S10: source-only pilots and feasibility

Implement contracts and tiny source overfits. Profile actual backward, optimizer, source-event enumeration, full 100-frame inference, CSV writing and official scoring. Select small and crowded pilot clips from source image/prediction statistics, never target errors. Execute at least one real supported division and one supported false-fork complete-action witness through the scorer and solver.

Freeze a common upstream update count and a common event update count using only timing, memory and engineering integrity. Produce the matrix cost projection, source feature schemas, decoder configuration and execution lock. Do not lock an infeasible large sweep then silently shrink to a few hundred updates.

### S20: clean upstream fits

Run four independently initialized fits under the new recipe. Do not resume or copy weights from historical experiments, even their apparently source-only intermediate checkpoints. Record complete optimizer/scheduler/RNG/sampler state and source-only learning curves. The old 400-epoch comparison remains an untouched historical study.

Generate source-fit and source-calibration predictions with each fitted upstream package. These are the only observations/features used to train its clean event head. Never train that head on P0 proposals as a convenient substitute.

### S30: compact event fits and calibration

Train clean N in four direction/seed cells. When feasible, run operational R/N at the first seed in both directions. Freeze all checkpoints, source calibrators and margins. Record source gates and calibration fallback, including no-edit outcomes. An unsuccessful source gate does not authorize replacing a registered target model with P0.

### S40: freeze and infer

Before any comparative target scoring, freeze every retained model, recipe, calibration and primary identity. Inference receives images and exactly one explicit model package, not labels or filename-derived policy. Freeze complete image-to-CSV predictions for both directions and seeds. Operational neural outputs may be reused across its two calibrators only with exact provenance/parity.

Target data needed solely for inference must not be used for test-time adaptation or parameter fitting. Do not use OP outcomes to change C10.

### S50: official evaluation

Score exact exported graphs on the complete expected clip set. Export pooled and per-embryo counts, uncertainty limitations, all arm dispositions, error transitions and stage funnels. Compare C10 only to its same-seed C00 for the primary effect. P0 is a separately labeled practical reference, not a clean matched control.

### S60: cold inference and return

Run both primary arms from renamed images in a fresh process with historical caches/annotations/network unavailable. Reproduce scored CSV graphs. Package source/config/model manifests and environment pins locally. Report Kaggle feasibility as measured only after testing the actual runtime; local GPU timings are not a guarantee. Commit code and sanitized results, return REPORT_BACK.md and retain P0.

## 5. Resource decision

One measured local GPU; the historical machine was a 24-GB RTX 4090 with approximately 65 GB RAM, but measure current hardware. Default limits: at most 20 GiB study CUDA reserved with at least 2 GiB device free, 44 GiB total study process RSS with at least 10 GiB system MemAvailable, and 15 GiB durable disk reserve plus measured checkpoint/cache requirements. Reduce admissions for other active applications. Start two loader workers and bounded 4-GiB decoded-frame caching.

Hard cumulative cap: **72 GPU lease-hours** for this study, including pilots, failures, training, leased I/O waits and inference. Other workers' lock-wait time is reported separately, not charged as our compute. This is a resource ceiling, not a wall-clock promise or training-convergence claim. No paid/remote compute is authorized.

Initial allocation: pilots 3, clean upstream 36, event work 18, full inference/evaluation-related GPU work 13, cold packaging 2 hours. Reallocate once from source-only measurements before locking; retain at least 15 hours for full inference/cold proof unless measured projections require more. CPU scorer time is separately reported and does not erase required evaluation.

Upstream final-update choices: [24000, 16000, 8000]. Event choices: [4000, 2000]. Profile both source directions. Choose the largest feasible upstream horizon first, then the largest feasible event horizon, while funding the entire core and reserves. Apply one common horizon per family to every direction, seed and matched arm. Do not equate these finite schedules with convergence.

First fit the mandatory eight-fit clean core into budget. Add the complete OP lane only when it fits without reducing the chosen clean horizons/reserves. If even the minimum clean core cannot fit, report a concrete resource blocker and resumable partial work; do not drop replication, fake full coverage or return an exposed substitute. Resource fallback cannot depend on target score.

Use sequential jobs and bounded multi-update GPU leases rather than moving models every optimizer step. Durable checkpoints every 200 updates and at phase boundaries, with atomic two-generation rotation. Record progress more often without repeatedly writing all weights. Profile before adopting optimizations and verify input/output/gradient parity where claimed. A changed precision or batching recipe is a documented pre-lock scientific change, not an invisible speed fix.
