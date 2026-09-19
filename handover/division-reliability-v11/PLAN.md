# V11 execution plan

## Decision and fixed comparisons

Build or reuse a genuinely source-excluded complete detector/linker, then test fork reliability on its fixed observations. The registered comparison is C11 minus C01, with C00 retained as the no-fork reference. P0 is not a clean comparator. Do not pursue native-resolution sweeps, new datasets, masks, observation replacement, or final all-data training during v11.

| Arm | Definition | New trainable components |
|---|---|---|
| C00 | The inherited v10 clean upstream recipe, continuation-only graph. | Up to four upstream fits, fewer when eligible artifacts exist. |
| C01 | C00 plus factorized, regularized linear occurrence/action heads on image-derived scalar and geometry features. | Small CPU fits; no new image encoder. |
| C11 | C00 plus a compact image encoder and factorized occurrence/action heads using the same scalar features, bank and decoder. | Four compact fits. |

Both source directions (44b6 -> 6bba; 6bba -> 44b6) and seeds (20260918; 314159) are mandatory for complete results. Never pool scores across seeds. C01 and C11 are different model families; their difference tests practical model value, not the isolated causal effect of factorization or any single loss. Maximum new neural fits is eight, not eight plus the old v10 matrix.

## R0: reconcile evidence, preserve work, establish readiness

Before importing training packages, inspect `git status`, worktrees, local refs, v10 result directories, manifests and actual processes. Produce `reconciliation.json`: remote planning base; local code commits; which v10 stages actually ran; checkpoint/input hashes; active jobs; any target metrics already inspected; and missing evidence. Do not kill, reset, stash, overwrite or force-push other work.

If local v10 C00 artifacts exist, check the exact source-fit partition, full recursive ancestry, initialization, schedule, normalization, association objective, candidate/decoder rules and complete model state. Reuse qualified completed artifacts regardless of score. Resume qualified partial artifacts only from copied state in v11-owned storage, with an exact-resume proof and unchanged upstream recipe. Reuse may be per cell, but all four cells must implement the same registered upstream recipe/horizon. Never import v10 C10 or P0 weights into the clean C11 fit.

When v10 has no qualified artifacts, implement its clean upstream prerequisite, not its entire campaign. The retained specification is `handover/clean-validation-division-v10/IMPLEMENTATION.md` section 2 and VALIDATION.md sections 1-4, with v11 resource and reporting rules taking precedence. Existing source-only 400-epoch jobs are separate historical studies; do not silently substitute their intermediate checkpoints.

Newer local results may be described and archived, but must not select the best seed, detector, threshold or source-specific policy for this fixed study. Such results make model-selection history more exposed, not pristine. If an active process owns needed outputs, mark dependent stages pending and continue independent safe work; do not run a duplicate owner. A provenance-ineligible artifact blocks its reuse, not scratch initialization.

## R1: source data, implementation and affordability gates

Verify the full expected image/GEFF inventory and actual spacing. Keep the inherited deterministic duplicate-grouped source fit/calibration split. All upstream and event neural learning and negative mining use fit only; calibration uses calibration only. Neither may read the opposite embryo. Duplicate grouping is not proof of independent acquisition; disclose missing offsets.

Implement `tools/division_reliability_v11/` with explicit paths and the CLI in IMPLEMENTATION.md. First exercise real source crops, nonempty positive and supported-negative labels, sparse unknown gradients, full native coordinates and a complete source image-to-CSV pass. Readiness must include an actual optimizer step, checkpoint/resume, and source true/false-fork examples, not just synthetic tests.

Measure real full-model training, event training, all-frame inference and CPU scoring cost. Freeze the four-cell schedule and resource allocation before production fitting or new target scoring. Source-only implementation repairs are allowed with retained failures and symmetric pre-lock changes; scientific changes outside this registry require a named amended study, not a silent rescue.

## R2: establish C00 in all four cells

Reuse or complete the qualified upstream fits. For fresh fits choose a common horizon from [24000,16000,8000] updates using source timing only; do not shrink below the inherited minimum and call it adequate training. Reused final horizons must match the registered inherited recipe. Generate source-fit and source-calibration observations with each cell's own frozen model. Preserve clean feature schemas; no fabricated teacher votes or inherited embeddings.

Produce complete source-only graph diagnostics and proposal/available-edge coverage. Even a low C00 result remains a measured baseline. If a model collapses or cannot supply positive legal source fork actions, report the bottleneck and complete feasible baseline evaluation rather than inventing positive candidates from GT. Do not claim a failed head comparison is complete.

## R3: establish candidate reachability and meaningful controls

Use prediction-only candidate construction and complete ownership edits. Recompute source-only witness graphs under the actual clean C00 observations; exposed-bank witnesses do not transfer automatically. Trace endpoint, anchor, daughter pair, legal-action, score, safety and solver stages.

Fit C01 before spending the C11 budget. A C01 no-op is a valid negative outcome, not a reason to alter the locked recipe. Confirm at least one positive and supported-negative parent group reaches the deployed bank in each source. If no such support exists, mark the dependent comparison blocked, retain the source evidence and still report complete C00 where possible.

## R4: train C11 and calibrate both fork policies

Run one compact family with mixed supported-negative sampling, one source-fit hard-negative mining pass and a sequential warmup/cosine schedule. C01 and C11 use the same legal action bank and frozen upstream link scores. Select source-only safety margins using the registered rule. Keep raw proposals and disabled-policy outcomes separate; disabled edits are not learned success.

Train every retained direction/seed before unlocking target comparisons. No target-selected retries or architecture expansion. Qualified local v10 event results remain historical diagnostics, not substitutes for C01 or C11.

## R5: freeze, score, cold-test and report

Freeze model/calibration/bank hashes and complete per-frame prediction manifests for all retained cells and arms before evaluation. Evaluate the exact target clip sets, CSV roundtrips and pinned official score. Run cold inference with renamed images in every source/seed cell; no old predictions, GT, network or hidden filename-based routing.

Return the files in VALIDATION_REPORTING.md, commit code and sanitized results on this branch, and push that branch. Never merge, upload weights, submit to Kaggle or change P0 automatically. A concrete partial report and exact resume state are mandatory when a hard blocker prevents completion; a plan-only or training-loss-only report does not satisfy execution.

## Resource envelope, not a completion-time promise

Ceiling: 72 new study GPU lease-hours, including pilots, failures, mining, training and inference. Initial allocations: 4 pilots/integrity; 34 upstream; 18 event/mining; 16 reserved for full inference/cold proofs. Record new and inherited lifetime compute separately. A pre-lock measured allocation may move unused upstream capacity when qualified artifacts are reused, but cannot lower the inference reserve. Do not restart the accounting clock after failure.

Expected host is one RTX 4090; measure actual capacity. Limit total device use to 20 GiB and maintain 2 GiB free; study process-tree RSS <=44 GiB with >=10 GiB host available; maintain >=10 GiB durable free plus measured checkpoint/cache needs. Do not delete other studies. New automatic paid services or external-data campaigns are not authorized.

For C11 choose a common E in [4000,2000], never repeat a 178-update production fit. Choose U and E jointly from measured source costs with a 25% execution margin. Reserve full evaluation first, then choose the largest feasible pair lexicographically by U then E. Existing compatible upstream horizons are fixed. If no minimum pair fits, report an explicit resource blocker and completed pilot measurements; do not secretly underschedule, drop an embryo/seed, or evaluate only the easy clips.

Use bounded GPU leases up to 60 seconds, keeping parameters resident within a lease; checkpoint every 250 updates or 5 minutes and at phase boundaries. Respect existing cooperative ownership. Record transfers/checkpoint/I/O versus training time. Parity is required before caching frozen embeddings; a trainable encoder still needs differentiable forwards. Compact prediction evidence is retained; dense all-field heatmaps need not be permanently stored.
