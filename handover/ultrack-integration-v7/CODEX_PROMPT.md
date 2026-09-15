# Execute Ultrack integration v7 locally

Work on `handover/ultrack-integration-v7` in `matheuspf/cell-tracking`.
Read root AGENTS.md, this file, PLAN.md, study.json, and SOURCES.md. Then read the
measured v6 CONTINUATION and final_report cited there. This is an implementation
and execution task: do not return a replacement plan or repeat the point-only
v6 experiment as a substitute for Ultrack.

Implement and run V700-V750 in PLAN.md using the existing local data and one
RTX 4090. Target >=0.95 official local score; preserve both C0 and the stronger
P0 control regardless of the result. The target is not a promised outcome.

First inspect actual git status, active processes, assets, mount capacity and
RAM. Preserve dirty user files and running work; create a separate worktree
when needed. Never reset, clean, kill an old job, or mutate an old environment.
Keep all new runtime output under the selected v7 root.

Provision the specifically requested Ultrack tool and necessary public software
dependencies in a new isolated environment, pinned to the inspected upstream
commit. Prefer available local caches and wheels. Do not download a microscopy
dataset or a new model collection. Do not request FOCUS access, share contact
details, obtain a solver license, assert academic eligibility, or purchase
compute. Existing authorized FOCUS nuclei weights may be reused. Model access
or FOCUS compatibility failures do not block the image-driven Ultrack arms.

Run the supplied synthetic smoke with ACTUAL Ultrack/CBC, then extend it with
split, conflict, empty-frame and window-boundary fixtures. Import success or
passing pure-Python contracts is not successful integration. Implement native
TZYX image/foreground/contour adapters, exact candidate-ID and mask retention,
physical-scale linking, actual Ultrack solving and observation-parent export.
Fix the concrete v6 adapter issues listed in PLAN.md rather than wrapping them.

Run U0, U1 and H1 after source-only calibration; U2 is the bounded learned-mask
comparison when its real provider is available. Use all 199 clips for every
reported complete arm, fresh official matching, exact pooled counts, both
embryos, and the fresh-image offline inference gate. Never replace a missing,
failed or timed-out arm with a control score. Keep unsupported GT events unknown.
Never feed target GT, annotations or GT matching into Ultrack solve.

Observe study.json's resource and search ceilings. Use one admitted clip DB at
a time; keep an 8 GiB free-space reserve plus projected incremental allocation.
Do not allocate a full-dataset dense mask collection. RAM scratch requires a
separate memory/capacity admission and durable small receipts. Do not delete old
artifacts to make room. Stop only the blocked workstream, continue independent
engineering/tests where useful, and report the exact external requirement.

Deliver actual implementation, environment/source locks, tests, measured reports,
an offline diagnostic viewer, and a cold image-to-CSV inference entry point.
Publish sanitized results and CONTINUATION.md to this same branch. Do not merge
or submit to Kaggle. No raw images, detailed GT, masks, checkpoints, credentials
or generated submissions in Git. Say clearly which stages ran and which did not.
