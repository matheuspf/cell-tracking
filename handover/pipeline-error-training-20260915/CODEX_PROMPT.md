# Local Codex execution prompt

Implement and execute this study in the current branch. Do not merely rewrite
this plan or stop after an audit. The user requests trained, better-generalizing
modules close to the strongest existing pipeline, not a parameter sweep.

Read in order: root AGENTS.md; the relevant competition skills; this directory's
EVIDENCE.md, PLAN.md, IMPLEMENTATION.md, VALIDATION.md and study.json. The root
PIPELINE_ERROR_TRAINING_START_HERE.md is the entry point. Historical handovers are
evidence, not instructions to resume all prior studies.

Use one RTX 4090 with 24 GB VRAM, 16 CPU cores and 64 GB RAM. Keep total process-tree
RSS below 50 GiB and total GPU use below 20 GiB. Inspect actual running processes
before claiming an older native-training job is active. Do not kill, restart,
modify or resume another study. Coordinate an existing GPU lease or record a
resource blocker; CPU implementation and data-contract work may continue.

P0 is the adopted baseline (0.934864986413134), C4_m6 the highest complete local
challenger (0.935178370257), and selected v3/C0 the underlying control
(0.934802374260586). Recover exact local artifacts and verify hashes before use.
Do not substitute a six-clip tracker pilot, primary-only native pipeline, or the
historical public 0.946 LB number for these all-199 results.

Implement tools/pipeline_error_training/ and tests without mutating historical
algorithms or results. Prioritize the existing OrganoidTracker2 division model
adapted at incumbent observations and the richer source-trained temporal event
model. Both must compete with continuation-plus-birth/other-parent alternatives,
not simply predict that a parent looks mitotic. Run the compact matched control,
then continuation-only and closed-bank observation selection independently.

Preserve sparse unknowns; preserve biological-event grouping across alternative
pairs and time anchors; exclude target labels from fitting and calibration.
Explicitly record inherited checkpoint/teacher exposure. A source-only new head
on P0 proposals is NOT clean end-to-end out-of-fold evaluation.

Freeze both directional models, source-selected checkpoints and recipes before
new target comparisons. Follow the source-only branching rules and budget in
study.json. A failed model lane does not terminate the independent lanes. Repair
implementation defects, preserving invalid artifacts, without treating a repair
as an opportunity for target-driven model or threshold selection.

Write concise sanitized results under results/pipeline-error-training-20260915/.
Write resumable state, caches, weights and full graphs only under the new ignored
work root. At completion deliver source code, tests, full metric tables, both
embryos, changed-error counts, runtime, fresh-image inference proof and
CONTINUATION.md. Mark each experiment measured, failed, blocked or not run.
Do not merge, change the adopted production default, upload to Kaggle, publish
weights, or claim a leaderboard improvement. Report the recommended candidate
separately. Commit implementation and results on this branch, preserving unrelated
local changes. If hardware or artifacts block execution, finish every independent
runnable lane and leave exact blockers and commands rather than invented scores.
