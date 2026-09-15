# Local execution plan — expanded revision 2

## Scope and deliverables

This replaces every one-candidate restriction in the previous handover. Implement
and execute the eight experiments in EXPERIMENTS.md, not a parameter sweep and not
another proposal-only iteration. B0/B1 are anchors. There are three registered
combinations and two conditional B1 transfers; at most 15 scored configurations.
No training, residual classifier, new checkpoint/dataset, architecture replacement,
segmentation or Ultrack integration. Retain the public inter-model harmonic fusion.

Implement under `tools/public946_minimal/`, tests under
`tests/test_public946_minimal*.py`, and a runner
`scripts/run_public946_minimal.sh`. Commands to implement: `preflight`, `audit`,
`controls`, `pilot`, `singles`, `combinations`, `transfers`, `robustness`, `package`,
`report`, `run`. Accept explicit data, artifact, runtime and output paths, `--arm`,
resume and resource-budget arguments. These inference commands do not exist yet;
only the handover's source/registry helpers are implemented by the planner.

Raw outputs: `/kaggle/working/cell-tracking/public946-minimal-generalization-v1/`.
Sanitized reports: `results/public946-minimal-generalization-v1/`. Do not overwrite
an older execution directory; create a revision-specific child when necessary.
Keep predictions, dense maps, images, checkpoints, labels, notebooks and submissions
outside Git; commit code, hashes, concise score tables and failure reports.

## P000 — preserve, inventory and source-audit

Read AGENTS.md and the competition skills. Preserve user changes, earlier studies,
active jobs and raw data. No reset/clean, deletion for space or force push. Discover
the actual runtime; known interpreters include
`/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python` and
`/kaggle/envs/cell-tracking-notebooks/bin/python`. Use the working stack with
`PYTHONNOUSERSITE=1`; inspect `scripts/root_remote_env.sh` for CUDA-library handling.
Do not upgrade packages or install/download additional models for these arms.

Audit original archived v29 and the code it actually patches/materializes. Pin
script version, dependencies, model hashes, input dataset versions and all effective
settings. Read current official metric/rules through the throttled reference tools;
historical metric pin is `075fc5f5a52d11077f9dc2b074644618f26939e2`. Use the same
revision for all arms. If current verification is blocked, label it pending and
continue offline work; do not certify current Kaggle readiness falsely.

Write source-line references for response aggregation, feature sampling, coordinate
centers/units, padding, normalization, time contexts, per-model fusion, candidate
normalization, decoder and all repair stages. Review previous logs for matching
experiments. Mark each E-arm's exact patch seam and evidence requirements in
`applicability.json`; source-driven no-ops need numerical fixtures, not assumptions.
Raw original files are not present in Git, so do not invent source line numbers.

Freeze this plan using `study_contract.py freeze --file <out>/plan_freeze.json`.
Before comparative scoring, write `execution_lock.json` with source/input/model
hashes, source-resolved semantics, arm recipes, metadata-only pilots, safety limits
and evaluator revision. Hash experiment adapter code before scoring its predictions.
Later correctness fixes must preserve the scientific recipe, retain failure receipts,
and rerun every affected ancestor/descendant; no silent new attempt selected by score.

## P010 — build common harness and faithful controls

Start from the public notebook, not v3/v5/v6 inference exports. Some v3 helpers
force motion off; some v2 hooks select AST nodes by old line ranges. Reuse audited
interfaces but not mutable global output paths, hardcoded no-motion defaults or
simplified native-only networks. The existing `baseline_contract.py` is exclusively
an optional B1 patch helper, not a validator that all new arms differ by one flag.

Use the same required deployment adaptation for all arms: explicit paths, offline
inputs and exclusion of the notebook's GT-reading validation block. Preserve the
archived original. Separate `shared_runtime_changes.diff` from each scientific diff.
Instrument response -> detections -> features -> pre-threshold probabilities ->
ILP -> individual repairs -> smoothing -> CSV with stage hashes and minimal summaries.
Assert which stage may first differ for each arm, and detect undeclared differences.

Prediction workers must receive an image-only inventory and no evaluator objects,
annotation graphs or old selected predictions. Enforce denied annotation/evaluation
paths before imports and in child workers; evaluator runs in a separate process.
Never use saved baseline coordinates or labels to force prediction parity.

Reproduce B0/B1 over all 199 available supplied clips using valid read-only evidence.
Known roots: `annotation-selection-v1/public_harmonic_full/`,
`strong-tracker-v2/raw/`, and `strong-tracker-v2/heatmaps/` under the working root.
Audit cache hashes and evidence stage before reuse. Dense-response or feature
experiments cannot be evaluated from repaired graphs alone. Rebuild missing evidence
from original images; no dependency on local residual checkpoints is allowed.

With the historical metric, investigate deviations from B0=0.911774/B1=0.934206
rather than treating them as new gains. Original serialization had two half-tie
compatibility cases and six out-of-bounds nodes. Do not copy old lookup corrections
into inference. Document natural rounding. If mandatory bounds sanitation is needed,
apply it identically to every arm, score the original B0 separately and disclose all
raw/sanitized deltas; no hidden candidate-specific correction or metric clipping.

## P020 — pilots, implementation tests and measured compute allocation

Choose four complete clips before labels/new scores: each embryo's nearest 25th
and 75th percentile of metadata q99-q10 contrast, deterministic stem tie-breaks.
These are smoke/performance pilots, not a holdout or ranking set. Implement all
applicable E-arms with the synthetic tests in EXPERIMENTS.md and fresh image-to-CSV
runs on the pilots. Do not reject a scientific arm merely for a poor pilot score.

Each pilot records changed-stage tensors, valid outputs, GPU wall/active time,
peak VRAM/RSS and output bytes. Fix engineering failures without recipe changes.
Use measurements, not assumed speedups, to project full-199 workloads and Kaggle
runtime. Work on one available RTX 4090; use bounded CPU workers and streaming.
Suggested safety caps to register before execution: GPU budget 96 device-hours,
22 GiB allocated VRAM, 48 GiB new scratch, and at least 10 GiB filesystem reserve.
These are maximum allocations, not completion-time estimates or score thresholds;
reduce them to observed available capacity. Never change them based on which arm
looks promising. If less storage exists, stream/recompute without deleting others'
artifacts. If a true blocker remains, finish independent tests/reports and record it.

Reuse B0 image normalization/features where mathematically legal. E01/E07 can often
use graph/evidence replay; E02 needs local detector-response neighborhoods; E03 needs
features or fresh features; E08 can share B0 detector forwards if per-view responses
were captured. E04/E05 add views and E06 adds valid contexts. Never label reused
neural evidence as fresh or associate a cache from a different recipe with a new arm.
Keep dense tensors clip-local, retain compact candidate scores/patches and hashes,
and resume with source/model/input/module/cache fingerprints.

## P030 — execute the eight singles

Run in this fixed cost-aware order: E01, E07, E02, E03, E08, E05, E06, E04. Every
applicable, feasible arm receives full-199 paired evaluation against B0, even when
earlier arms fail. Do not stop the study at B1 reproduction. Arm-specific blockers
must not terminate independent arms. Report all failed/missing clips explicitly;
no changed sample population, truncated clips or average-of-clip-score substitute.

Use fresh official matching and division evaluation for each graph, exact run-level
weighting, full-cohort and per-embryo scores/counts, TP survival and recovered TP.
Record candidate coverage, altered nodes/edges, predicted divisions, runtime and
memory. Summarize predefined contrast, raw-image crowding, depth and temporal-edge
strata descriptively; no subgroup-specific deployment rule. Full eligibility and
ranking are fixed in VALIDATION.md.

A resource budget stop must retain progress and `NEXT_AGENT.md` with exact resume
commands; do not pretend the unrun arms failed scientifically. Within the current
local execution, continue feasible independent arms rather than asking for a new
plan. Source-proven no-op arms remain visible in the matrix with their evidence.

## P040 — fixed combinations, finalist lock and transfer

Evaluate only C01=E02+E03, C02=E04+E05 and C03=E05+E06, where both constituent singles
pass eligibility. No greedy stacking, arbitrary two-arm search or B1 cross-product.
Measure each full combination against B0 and its constituents, with the same metric
and full population. A harmful combination does not erase successful single results.

Rank eligible, applicable non-E01 recipes as specified in VALIDATION.md; lock up to
two to `finalist_lock.json` with config/code/prediction hashes **before** any X scoring.
Resolve X01/X02 once to those recipes with motion disabled and evaluate against B1
and B0. B1 transfer is an interaction test, not an independent generalization set.
E01 is ineligible for X because disabling the relinker nullifies its mechanism.
No replacement finalist after a transfer failure and no revised numeric settings.

## P050 — fresh replay, robustness and packaging

For all eligible packaging finalists, verify fresh image-to-CSV outputs across the
full cohort using their actual source and public artifacts; control/evidence reuse
is allowed only as a separate comparison. Already-completed fresh passes may be
reused as execution evidence if code/model/input hashes match. Cached replay alone
never establishes standalone notebook execution. At minimum run each actual packaged
notebook on every full pilot; disclose any incomplete full-cohort package validation.

Apply the fixed robustness and portability checks in VALIDATION.md. Any already
available genuinely unexamined, embryo-disjoint in-domain labels may provide one
locked confirmation; otherwise mark it unavailable. Lack of such data does not end
all experiments or force external-data training. Do not rebrand old seen splits.

Produce `baseline_public946.ipynb`, `control_no_motion.ipynb`, and at most two novel
candidate notebooks: the best eligible B0-family recipe and the best eligible B1
transfer, omitting a missing lane honestly. Keep every candidate small: at most two
registered mechanisms, with the optional additional no-motion change only in X.
E01 may occupy the public-family package slot if it wins its registered ranking.
No notebook may import local work directories, training labels, old score caches
or a simplified primary-only adapter. Enumerate arbitrary test stems and actual
shapes. Include licenses/attribution and exact public input dependencies.

Manifests include model/source/config hashes, all scientific and shared-runtime
diffs, applicability, original versus sanitized controls, metric revision, verified
scope, runtime, memory, statuses and SHA256s. Include Python exports, offline run
instructions, local tests and manual Kaggle instructions. Do not invent account
identifiers or authorize downloads/installs/paid services. Do not submit automatically.

## P060 — durable results and terminal conditions

Write `results/public946-minimal-generalization-v1/START_HERE.md`, `REPORT.md`,
`experiment_matrix.csv`, `per_embryo_scores.csv`, `status.json`, `NEXT_AGENT.md`,
source/metric/prediction/packaging receipts, and exact reproduction commands.
Include every registered arm, including no-ops, rejected and unrun arms. No claim
of completed inference from registry or synthetic-helper tests alone.

Valid terminal states: `completed_no_improvement`, `ready_for_manual_lb_test`, or
`partially_executed_with_named_blockers`. Baseline plus B1 alone is insufficient
completion under revision 2. Preserve negative evidence; do not invent another
family or tune until a result appears. Commit implementation and sanitized measured
results to this same branch, without modifying old studies or force-updating history.
