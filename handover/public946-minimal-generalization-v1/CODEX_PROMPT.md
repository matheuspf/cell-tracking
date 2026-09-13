# Execute this minimal public-baseline study

Work on `handover/public946-minimal-generalization-v1`, based on main
`fb5521629eb41c8c485b291a5bcf344944c113ae`. Read repository AGENTS.md, then this
handover's START_HERE.md, BASELINE_AND_EVIDENCE.md, PLAN.md and experiment_lock.json.
This task takes precedence over older segmentation/Ultrack/residual experiment
prompts for the present run; preserve their source and results unchanged.

The user wants a baby step from the public 0.946 LB solution toward 0.95+ LB,
without broad method changes or knob/leaderboard tuning. Implement and execute
P000-P060 locally with the existing runtime and RTX 4090. The only scientific
candidate is original Harmonic Fusion v29 (script version 347965685) with the
actual OUTPUT_MOTION_RELINK flag disabled. The historical no-motion local score
is ~0.934206 versus original ~0.911774; these are not new measurements or an LB
forecast. Do not start from the 0.9348/0.9349 fitted residual selection.

First read the real archived notebook and patched native source. The planner
inspected committed adapters and reports, not those ignored local originals.
Record exact TTA, fusion, preprocessing, checkpoints, graph and repair source.
Freeze their identities. Build B0 and B1 from the same public source; prove that
all other scientific settings match and the flag actually changes at runtime.
Treat the supplied baseline_contract.py as a strict source helper, not a complete
notebook runner or proof of generalization. Repair helper/schema drift by source
audit, never by blind replacements or a different baseline version.

Implement the planned runner/modules, image-only inference boundary, original
baseline reproduction, exact official evaluation, paired replay and fresh full
inference, fixed portability/reflection checks, and standalone Kaggle package.
Read old caches only if their provenance verifies; regenerate from public inputs
when absent. Old work paths, training sample IDs and fitted source-embryo selectors
must not be deployment dependencies. Prevent annotation reads before inference
imports and in workers. Preserve public model/TTA/fusion and downstream repairs.

There is no B2, parameter sweep, new training, external-data expansion, new model,
segmentation integration or adaptive fallback. Do not optimize on per-clip
results or public LB. Local data are already examined/contaminated; do not label
renamed or newly split training clips clean OOF. Lack of clean additional embryos
must be disclosed, but is not itself a reason to avoid preparing the locked trial.
Do not enforce local score >=0.95 or promise an LB target from local deltas.

Do not install/download anything, accept licenses, modify raw inputs, erase
prior artifacts, kill unrelated jobs, overwrite user work, submit to Kaggle or
publish to other external services. Commit and push only this study's code and
sanitized results to this same branch. Discover available disk/runtime first and stream outputs; v6's
volatile scratch may be gone. Implement reusable work rather than stopping at a
missing optional cache. If essential inputs prevent execution, finish independent
checks, record exact blockers and leave partial artifacts and restart commands.

Produce both a faithful baseline-control notebook and the single candidate
notebook locally, scientific/shared-runtime diffs, source/model/runtime hashes,
measured aggregate report, fresh/robustness receipts, and manual LB instructions.
Distinguish planned, tested, measured, blocked and submitted states. Finish with
results/public946-minimal-generalization-v1/START_HERE.md, final_report.md and
status.json, exact artifact paths/checksums, tests run and terminal decision.
Never claim 0.95+, improved generalization or successful Kaggle execution without
corresponding evidence. The expected successful local terminal state is
ready_for_manual_lb_test, not an invented leaderboard score.
