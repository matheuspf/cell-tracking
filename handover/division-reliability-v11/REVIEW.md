# What was actually available at review

## Repository state, checked September 19, 2026

The GitHub branch list, v10 commit history, and pull-request collection were read. The v10 head is still `08061f0a224387594dd3e915462bafb5d5928fb2`, a September 18 planning commit whose parent is `ab861b5d8c1a37bd920f610b88f552dfc5b5dbb6`. Requests for the v10 result report and implementation directory returned 404 at that ref. The latest measured study available in that ancestry is the pipeline-error study completed September 16. Main remains the older `bd844731c0e93b401cf94e67c92350d4109e12df` snapshot.

This review cannot establish the contents of the user's local disk or running processes. V11 is an iteration over the available measured results plus a critical revision of the unexecuted remote v10 plan. It is not a response to invented v10 scores. Stage R0 explicitly handles newer local evidence.

## Verified measured evidence

Source: [completed report](../../results/pipeline-error-training-20260915/REPORT.md), Git blob `8c52e2098550872d07bc222f16638eb7eaca7bc6` at the pinned base.

| Arm | Official all-199 score | Main result |
|---|---:|---|
| P0 | 0.934864986413134 | Retained; division TP/FP/FN 29/92/122. |
| C4_m6 | 0.935178370257 | Best complete descriptive score; not adopted. |
| A10 | 0.934939378710 | Small edge gain, divisions unchanged. |
| A10 replication | 0.934938660898 | Similar small edge gain. |
| D20_compact | 0.918100978703 | Division counts 48/1245/103: 19 added TP, 1153 added FP versus P0. |
| D20_temporal | 0.919226643837 | Division counts 45/1052/106. |
| O10_restore | 0.934864986413134 | Exact graph identity; no restoration edits. |
| O10_swap | 0.934415338644 | Regressed on both embryos. |

The dominant released detector has training exposure to all 199 clips; these are operational development scores, not clean validation. See [provenance](../../docs/incumbent-provenance-20260914.md) and [clean-readiness receipt](../../results/pipeline-error-training-20260915/clean_upstream_readiness.json).

The [source feasibility receipt](../../results/pipeline-error-training-20260915/source_feasibility.json), Git blob `3df961eb4d390bc3a120db3739c2da8183531daa`, contains label-guided, fully rescored division witnesses: 0.956893956443 on 44b6 and 0.968346931854 on 6bba, retaining 22 and 70 division FP respectively. These are oracle diagnostics on exposed graphs, NOT learned scores, clean ceilings, forecasts, or evidence that v11 can reproduce the repairs. They show useful actions exist in that particular bank. The analogous observation-swap witnesses do not improve P0, so observation replacement is not the primary next experiment.

The [training schedule](../../results/pipeline-error-training-20260915/executed_training_schedule.json) records 178 updates, 89 identity-only prefix updates in standard arms, and concurrent warmup/cosine with peak learning rate about 7.974e-5. The [sampling audit](../../results/pipeline-error-training-20260915/calibration_sampling_audit.json) records positive-containing event groups, no independent negative-only event minibatch pool, and deterministic temporal expansion rather than randomized inclusion probabilities.

## Diagnosis versus hypotheses

Measured: learned fork changes add far too many evaluable FP; reliable continuation changes are tiny; the clean whole-model result is not published. Source inspection of `pipeline_error_training/{dataset,train,models,calibration,actions}.py` supports testing sampling, optimization and selection together, but does not prove which caused the failures.

Hypothesis to test: a parent-level fork-risk decision followed by conditional daughter/action ranking will be more reliable than allowing many individually attractive fork alternatives to compete against weakly learned no-fork scores. This is a proposed model, not a diagnosed mathematical bug in the old solver. V10 already proposed negative sampling and deployment calibration; v11 retains those ideas rather than claiming they are new.

## Changes from the v10 plan

1. Reconcile local execution first; reuse clean C00 only by provenance/recipe, never by score. No duplicate clean campaign.
2. Add a cheap learned fork comparator C01. A gain against continuation-only C00 alone is too weak to attribute value to a new image event model.
3. Replace the event-head campaign with one compact factorized model C11. Freeze upstream association scores; no new global continuation residual, observation swaps, or large backbone survey.
4. Calibrate the occurrence decision, audit the selected high-score tail, deduplicate complete edits, and normalize conditional action scores over the actual deployment bank.
5. Fund full clean evaluation before breadth. There is no new P0 neural ablation matrix. P0 stays historical or freshly reproduced when available, always exposure-labeled.

The public metric description was read on September 19. Keep the pinned scorer `075fc5f5a52d11077f9dc2b074644618f26939e2` and recheck the official repository at local execution. This review does not claim live verification of Kaggle deadlines, GPU availability, or competition terms.
