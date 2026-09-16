# Preregistered evaluation, budget and handback

## Fixed controls and splits

C0=0.934802374260586 on all199, per embryo44b6=.931664468721842,
6bba=.935221784097327. Reproduce it from the retained full pipeline before claims.
New v5 results may be catalogued as later evidence but cannot replace C0 silently.
The target is >=.95 complete official local score, not a promised leaderboard gain.

For each direction calibrate/fit on one embryo, freeze, infer on the other. Pilot
windows come from raw-image density/signal/time strata and source-only labels.
Where source clips spatially/temporally overlap, group them and use guarded source
holdout blocks; an overlapping adjacent patch is not an independent validation
sample. Hash exact source/target input membership and inherited checkpoint exposure.
This two-embryo repeatedly evaluated study remains operational/exploratory. Do not
label clip bootstraps population-level biological confidence from 199 independent
embryos. No outcome-based per-clip detector or C0 fallback selection.

## Bounded complete-configuration matrix

At most 12 PRIMARY all199 configurations plus 4 same-recipe learned replicas.
Each complete run and changed recipe consumes a slot, including negative runs;
unchanged checksum-verified resumed shards do not consume a new one. Register the
actual matrix and costs before outer scores. Default allocation:

| Slots | Experiment |
|---|---|
| 1 | Exact C0 control |
| 2–4 | Fixed-C0 center, bbox and mask-aware association, same bank/decoder |
| 5–6 | Independent detector D* center-only and mask-aware local tracking |
| 7–8 | D* Ultrack single labels; uncertainty/foreground-contours |
| 9 | Two-detector/threshold-hypothesis Ultrack |
| 10 | D2 full object tracker or image-motion ablation |
| 11–12 | Conditional adapted detector/scorer and its matched unadapted control |
| 13–16 | Same-recipe learned finalist replicas / identical-C0 control check |

D* is selected from source-only trials, with source-specific settings allowed only
under the frozen two-direction protocol. D1 blocked access means register D2 as D*,
not that an unavailable model was tested. Reallocate an unused slot BEFORE results
of its replacement, with a recorded rationale. Do not require one unsuccessful
fixed-node arm to pass before attempting the independently detected full-object arm.
A replica uses the same recipe and predeclared seed; primary remains export choice.

No more than two full learned detector families. Up to six pilot presets/backend/
source direction, not a large target-score search. Total v6 GPU inference+training
budget <=48 summed GPU-hours, with all attempts, retries and validation counted.
First perform source pilots and throughput projection; do not launch a workload
that cannot fit the remaining budget. Progress/update counts are actual not nominal.
The budget is a ceiling, not a reason to spend unused hours on another point grid.

Start one GPU worker; admit more only after a measured total resource test and
confirmation it does not interfere with v5. Default VRAM cap=min(18GiB,80% currently
free), RAM cap=min(28GiB,70% currently available); record any explicitly justified
larger isolated configuration. Bound aggregate CPU workers by the actual free
physical-core budget (initially <=8); disable nested BLAS/solver oversubscription.
Disk free floor8GiB. Never wait forever on a solver; record timeouts/gaps and fail
unusable graph outputs. Preserve partial completion with exact blockers.

## Scoring and promotion

Invoke the existing official `evaluate_graph`/`aggregate` with new graph hashes,
fresh node matching and correct estimated totals confined to evaluation. Pooled
score is official run-level aggregation, not mean clip Jaccard. Report all199 plus
both embryo results, TP/FP/FN for edges and divisions, predicted node counts and
node-count adjustment, matched/unmatched sparse nodes, missing proposal support
and graph topology errors. Never use matched-only edge diagnostics as full score.

Compare node movement/count change and edge change separately. Report new recovery
and incumbent-correct losses using evaluation-only matches; do not turn that truth
comparison into an inference gate. Proposed robust promotion: pooled score >C0,
no embryo decrease >.002 absolute, all integrity/offline/runtime tests pass. >=.95
and the same per-embryo condition is target success. A pooled winner failing that
robustness condition is exploratory, not discarded or silently promoted. Freeze
these conditions before scores. Learned finals require replica stability; failed
replication is reported, no lucky-seed substitution.

Sparse GEFF supports point coverage and lineage scoring, NOT dense mask IoU/AP or
all-unmatched-negative precision. Dense segmentation metrics require separately
verified fully annotated mask regions. When absent, show descriptive mask size,
fragmentation, temporal consistency, point support and manual image inspection as
PROXIES. Better mask visual quality does not itself establish a better graph.

## Required final artifacts

`results/segmentation-first-tracking-v6/final_report.md`, `summary.json`,
`CONTINUATION.md`, per-clip/embryo/full aggregate tables, exact execution commands,
asset/code/config/weight manifests, resource and failure receipts, and a compact
index to ignored masks/graphs/HTML/checkpoints. Status starts planned and changes
only with evidence; null for missing measurements. Preserve late/in-progress v5
status as historical intake, not a claimed final outcome.

Report detector access/mode/axes/spacing, masks versus boxes retained at each step,
scorer/split/solver design, complete experiment table, learned seeds and exposure,
actual throughput/cold offline validation, graph round trips, uncertain masks and
representative failures. Include a working `scripts/run_segmentation_tracking_v6.sh`
usage for one fresh clip and full inference, plus safe resume. Do not commit raw
images, dense GT, model weights, credentials, giant SQL databases or submissions.

Author-side reference tests do not validate detector quality, speed or official
score. Local Codex must execute the actual integrations; blocked backends stay
explicit even if another detector produces a useful improvement.
