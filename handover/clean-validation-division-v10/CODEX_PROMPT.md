# Codex execution instruction

You are implementing and executing a local research study, not producing another proposal. Read the root entry point, PLAN.md, VALIDATION.md, IMPLEMENTATION.md, REPORTING.md and study.json before editing. The user approved the recommendation to establish clean end-to-end validation and repair division training/calibration. These documents make the experimental choices; the user need not choose an architecture or training matrix again.

## First actions

Inspect `git status`, branch and ancestry. Work on `handover/clean-validation-division-v10`; preserve dirty work and do not force-checkout, reset, clean, stash or rewrite another study. A separate worktree is acceptable when necessary, but do not assume ignored data follows it. Discover the actual data roots, runtimes, disk, RAM, GPU and running processes. Do not kill another worker or resume an archived one.

Read AGENTS.md and the five competition skills. Refresh official pages through the maintained throttled extractor when available. Record inaccessible rules honestly. The planner's September 18 browser check returned no readable Kaggle rule body; a live rules refresh was not established. The latest reference snapshot is not an authorization to accept terms, add external data or submit.

Read these executed evidence sources before designing adapters:

- `results/pipeline-error-training-20260915/REPORT.md`, `executed_training_schedule.json`, `calibration_sampling_audit.json`, `clean_upstream_readiness.json`.
- `docs/incumbent-provenance-20260914.md`, `docs/pipeline-errors-20260915.md`.
- `tools/incumbent_comparison/train_native.py`, `common.py`, `data.py`, `frames.py`.
- `tools/pipeline_error_training/{models,dataset,train,labels,bank,actions,calibration,scoring,infer}.py`.
- `tools/annotation_selection/metric_adapter.py`, `tools/strong_tracker_v3/event_proposals.py` and `association.py`.

Historical imperative handovers are evidence, not concurrent assignments. Do not run v8/v9, old pipeline-error queues, native 400-epoch workers or public946 sweeps as part of v10.

## Execute

Create `tools/clean_validation_v10/` and focused tests. Use explicit paths and typed source/lane manifests; do not monkey-patch historical directory constants. Reuse pure algorithms and source architecture definitions, not exposed checkpoint loaders. Keep upstream inputs read-only.

Implement the CLI contract in IMPLEMENTATION.md. Then perform preflight, source pilots, immutable execution lock, training, source calibration, full prediction freezing, official evaluation and cold-image packaging. Continue through all feasible registered stages. A blocked optional operational lane does not block the clean lane. Missing mandatory source/data or a failed scientific contract requires a concrete partial report, not fabricated completion or substitute exposed predictions.

Only source-only engineering diagnostics may repair a broken implementation before the lock. Scientific recipe changes before target opening must be explicit, symmetric across directions and controls, and recorded in an addendum. After target opening, repair correctness bugs with preserved failed outputs and unchanged weights/recipes; do not tune model choices, thresholds, schedules or gates from target scores.

Keep planned and executed updates separate. Never compress mandatory event training below 2,000 optimizer updates, relabel intermediate historical weights as final, hide zero-edit policies, claim convergence from a short fit, or replace a failed direction by P0. Respect the 72-hour cumulative local GPU lease cap and the fixed fallback order.

## Finish

Commit only new implementation, tests, small manifests and sanitized measured reports on this branch. Preserve original P0/default selection. Large weights and microscopy stay local; write hashes and retrieval locations. Local commits are required; remote pushes, merges, submissions and weight publication are not part of this execution instruction.

Produce `results/clean-validation-division-v10/REPORT_BACK.md`, including the actual commit, primary clean comparison, every registered arm's disposition, source-only curves, sampling/calibration and candidate-stage diagnostics, resource measurements, exact reproducible commands, and one next recommendation. A failed candidate with a complete clean evaluation is a valid research outcome. A plan or passing contracts alone is not execution.
