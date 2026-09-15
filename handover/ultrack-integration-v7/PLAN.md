# Ultrack integration v7 — executable local research plan

**Owner:** local Codex. **Status:** planned, not measured. **Date:** 2026-09-12.
**Branch:** `handover/ultrack-integration-v7`, based on merged `main` at
`fb5521629eb41c8c485b291a5bcf344944c113ae`. Source evidence is in [SOURCES.md](SOURCES.md).
The machine-readable arm/resource registry is [study.json](study.json).

## 1. Decision and scope

Integrate **actual Ultrack as a joint segmentation-hypothesis/lineage solver**,
not as another descriptor extracted from a point neighborhood. Keep alternate
merged/split object hypotheses until temporal evidence selects a consistent set.
The primary research question is whether this joint selection, supplied with our
strong image-native association evidence, fixes errors that local point-edge
repairs cannot fix. The secondary question is whether a conservative mapping
back to P0 improves associations without losing its strong detections.

Do not replace the incumbent outright. Deliver independently selectable arms,
measured against **both C0 and P0**. Do not spend this iteration on another
external-data survey, generic neural pretraining, or repeated small P0 tuning.
FOCUS integration is bounded and optional for the core Ultrack study. The
image-derived input arm below is deliberately registered as non-learned
segmentation; it must never be presented as learned masks or a rerun of a
blocked learned-mask experiment.

Target **>=0.95 official local score**. A valid integration without improvement
is still an engineering result; do not promote it as an accuracy gain.

## 2. What the repository actually establishes

The latest measured v6 report is inherited by current main. It reports:

| Control | Pooled / 199 clips | 44b6 / 71 clips | 6bba / 128 clips |
|---|---:|---:|---:|
| C0 | 0.934802374260586 | 0.931664468721842 | 0.935221784097327 |
| P0 | 0.934864986413134 | 0.931727256413358 | 0.935284191136704 |

P0's pooled improvement is only 0.000062612152548. Both controls have 4,108,943
observations. C0 edge TP/FP/FN are 123135/4965/5748; P0 are
123172/4996/5711. Both have division TP/FP/FN 29/92/122. These are inherited
measurements, not v7 results. P0 freezes existing forks, so it has not solved
the division problem. Its source6 fit stopped at its iteration budget; do not
quietly describe it as converged or change that budget for this baseline.

V6 generated **zero learned masks and zero Ultrack competition results**.
Its recorded blockers were missing Ultrack, missing FOCUS Python source and
detectron2 despite local nuclei weights, missing Cellpose/checkpoint, and
6.838 GiB persistent free space below an 8 GiB reserve. Recheck these facts
locally: they are historical, not assertions about today's machine.

Use these maintained integration points:

| Existing path | Reuse / boundary |
|---|---|
| `tools/segmentation_tracking_v6/common.py` | Inventory, C0 paths, metadata validation, resource conventions; do not reuse v6 output constants for v7. |
| `tools/segmentation_tracking_v6/controls.py` | Full-native P0 control, sparse supported labels, official scoring call and source-direction conventions. |
| `tools/segmentation_tracking_v6/regions.py`, `contracts.py` | Retained bbox-local occupancy, hierarchy/ownership checks; inspect and extend rather than duplicate. |
| `tools/segmentation_tracking_v6/ultrack_adapter.py` | Unvalidated boundary to replace/repair within v7, not a completed integration. |
| `tools/segmentation_tracking_v6/fresh.py`, `fresh_point.py`, `network.py` | Fresh-image regeneration and early offline/data-access guards. |
| `tools/strong_tracker_v3/features.py`, `association.py`, `common.py` | Native feature definitions, safe point decoder and graph serialization. |
| `tools/annotation_selection/metric_adapter.py` | The maintained pinned official evaluator; never substitute a homemade metric. |

Current main also contains older parallel handovers. They are historical
references, not simultaneous execution instructions. V7 has one active plan.

## 3. Concrete adapter defects and missing integration

The inspected v6 adapter passes `images=images` into `ultrack.link`. Upstream
expects a **sequence of TZYX channels**, so a single channel must be `(images,)`
or `[images]`, not a bare TZYX array interpreted as a sequence of frames.
It omits `scale`, although physical distance should use ZYX micrometers. It
converts the entire labels input with `np.asarray`, which prevents lazy
large-clip processing. Keep array-like inputs and materialize only bounded
frames/tiles. Validate integer dtype and nonnegative values chunkwise.

It calls `labels_to_contours`, but does not read back candidate masks or complete
the competition export. The returned napari lineage dictionary is not a list
of observation edges. Request node IDs and parent IDs and use the dataframe's
**`id` and `parent_id`**. A track may contain many observations; never use
`track_id` as a detection ID or reconstruct ancestry by nearest centers.

It leaves native link insertion to an unimplemented callback. The inspected
upstream exposes `ultrack.core.linking.processing.add_links` and
`Tracker.add_links`. Use exact database candidate IDs. `add_links` is an insert
operation, not a promised update/upsert: create a fresh arm-local link table,
clear it through the inspected upstream utility, or use a narrowly tested
transactional update. Never append duplicate copies of the same pair.

**Critical weight issue:** upstream's default link transform is power 4.
Negative native logits passed through it can become large positive rewards.
For signed native costs set `link_function="identity"`, set bias explicitly,
and unit-test the effective objective weights. Do not set `solver_name="SCIP"`:
the inspected config supports `CBC`, `GUROBI`, or automatic choice. An installed
PySCIPOpt package does not mean Ultrack has a SCIP backend.

## 4. Architecture and contracts

```text
native Zarr v3 + metadata + existing native checkpoints
    |                          |
    |                          +--> native point/association evidence
    v
per-frame foreground + contours OR authorized learned instance labels
    v
Ultrack segment --> candidate IDs + masks + hierarchy exclusions
    v
candidate links / physical distances / optional image-derived motion
    |                          |
    | U0: region evidence      | U1/U2: native evidence at THESE candidates
    +--------------------------+
    v
Ultrack CPU MIP solve --> selected regions + direct observation ancestry
    |                          |
    | U0/U1/U2                 | H1: one-to-one mapping onto P0 observations
    v                          v
validated integer-voxel graph --> existing CSV export --> official evaluator
```

### Geometry and data interface

Canonical input is `(T,Z,Y,X)` in native voxel coordinates. The reviewed data
snapshot has 100x64x256x256 uint16 clips and spacing
`(1.625,0.40625,0.40625)` micrometers, but read every clip's metadata. Reject
unhandled axes, origin, resampling, units or channel layouts instead of guessing.
When a model uses an isotropic/resized lattice, record the complete transform
and map masks back by nearest-neighbor label resampling. Carry half-open boxes,
voxel occupancy, clipping flags, compartment, provider/model/config hashes and
the native grid identity. Boundaries/contours require the corresponding smooth
resampling rule; they are not integer labels.

Use `max_distance` in micrometers with `scale=spacing_zyx`. Segmentation
`min_area`/`max_area` are voxel **counts** in 3D, not micrometers or square units:
convert source-derived physical volume limits by `sz*sy*sx`. Treat blur sigmas,
image-border sizes and flow vectors as separately documented units.

Store candidate records keyed by `(clip_id, ultrack_node_id)` with `t`, bbox,
retained mask reference/hash, centroid, the chosen export/native-scoring point,
hierarchy/conflict references and provider provenance. Native edge records
carry `(source_id,target_id)`, time, features, raw and transformed scores, and
source-model identity. Keep original IDs integral, not floats through a NumPy
stack; verify no precision loss. IDs are clip-local; CSV row IDs are separate.

### Inputs: remove the dependency deadlock

**U0's image-derived path is mandatory.** Use native fluorescence to estimate
foreground and contours with Ultrack's documented image-processing utilities
(e.g. `detect_foreground`, `robust_invert`) after inspecting their pinned
signatures. Existing model heatmaps may provide supplementary foreground
support, but point impulses or fixed-radius balls are not cell masks. Any
foreground expansion must have image support and a logged recipe. Do not make
the existing point detections an exclusive object list: candidate births and
alternate splits are part of the experiment. Inspect source-only orthogonal
views and coverage before choosing the recipe.

**U2's learned path is separate.** Prefer the existing authorized FOCUS nuclei
checkpoint, if its real headless source/dependencies can be made operational.
Weights without preprocessing/inference source are insufficient. Use a separate
provider process/environment and a small manifest/array interface to avoid
forcing Detectron2 and Ultrack into the incumbent runtime. Only consider the
previously prepared Cellpose provider when its authorized model is already
available; do not launch a new model survey or gated download process. Keep
mask probabilities nullable rather than inventing confidence from labels.

Retain at most two registered segmentation hypotheses per learned provider,
selected using source-only evidence. `labels_to_contours` may combine multiple
label maps, but this does **not** guarantee every original mask survives in its
hierarchy. Measure exact survival and IoU coverage separately. When a necessary
input mask is lost, test the upstream `add_new_node(..., include_overlaps=True)`
path with real conflict constraints or report the limitation. Do not silently
force overlapping incompatible masks into the solution.

### Native evidence and objective

U1 starts from the same U0 hierarchy and **same candidate pairs**. First lock and
hash that bank. Recompute complete primary, secondary and eight-view native
association evidence at the candidate/export geometry; do not copy scores from
nearby C0 centers. Teacher/support features must also have valid mappings or be
explicitly missing, not fabricated. Verify the v6 geometry-fingerprint receipt.
Batch GPU evaluation of crops/edges and reuse native feature maps where exact
parity permits. Persist enough hashes to prove every score refers to the actual
region pair. Do not score only the selected nodes: that discards the alternate
hypotheses Ultrack is meant to resolve.

Start with a small L2 source-trained/calibrated edge model: native log-odds plus
motion-compensated mask IoU, normalized displacement, volume ratio and simple
shape/intensity consistency. Calibrate score scale and appear/disappear/division
penalties jointly using source clips. Record the full raw-to-effective weight
function. Hypothesis-specific node scores are optional, not a prerequisite.
Do not treat all unmatched hypotheses as false cells.

For source supervision, known daughter incoming-parent contradictions support
negatives; unrecorded cells, second daughters and divisions remain **unknown**.
Overlapping hierarchy nodes may match the same annotated object: group/weight
those alternatives so one GT object does not contribute many independent
examples. Do not call `fit_nodes_prob` with defaults that convert sparse
unmatched cells into negatives without a verified sparse-label adapter.

Use Ultrack's real appearance, disappearance and binary-division constraints.
A scalar division penalty is not a daughter-pair morphology model. If adding
union-volume/parent-daughters compatibility, implement explicit pair variables
and consistency constraints in an isolated tested extension; label this as an
extra source diagnostic, not a built-in Ultrack feature or a hidden U1 change.
Do not hold up the basic U0/U1 comparison for that extension.

Motion is optional and bounded: add image-derived flow only after a synthetic
translation establishes direction, temporal index, native voxel units and
backward target-to-source shift. The inspected linker adds target shifts before
search and shifts masks before IoU. Never assume a forward vector is correct.
Rebuilding a motion-expanded candidate bank is a separately logged source
experiment, not a silent change to the U0/U1 same-bank comparison.

### Export and conservative hybrid

Export direct selected observation IDs and parent IDs with actual selected
masks. Choose a deterministic integer native point inside each mask, preferably
nearest to its physical centroid; define tie-breaking. The native scorer must
use that same chosen point. Validate bounds, unique IDs, time, in-degree <=1,
out-degree <=2, parent existence, adjacent-frame edges, hierarchy exclusions,
no duplicate occupancy and no selected child with an unselected parent.
Use the existing CSV exporter and official scorer. Do not feed micrometers to
an exporter expecting voxel indices. Preserve real divisions; do not infer
edges by track-ID equality, nearest-neighbor reconstruction or gap interpolation.

H1 retains every P0 node coordinate and ID. Match selected U1 masks to P0
observations by same-frame containment with strict one-to-one ownership and
explicit ambiguity/multi-center flags. Unmatched or ambiguous observations
remain untouched; never discard a P0 cell just because a mask is missing.
Transfer only supported high-confidence link proposals. The initial H1 uses
the unchanged v3 safe association mechanism on P0, freezes existing forks and
caps changed edges at 2%. It is an association-only hybrid: no division gain
may be attributed to H1. U0/U1/U2 are the full-lineage experiments.

## 5. V700 — reproducible runtime and resource admission

Run the supplied read-only preflight. Inspect actual active jobs, git status,
local package caches, FOCUS source/checkpoints, persistent mounts and available
RAM. Do not confuse stale supervisor JSON with a running process. Preserve all
historical selections, code, queues and weights. Use a separate worktree when
current work is dirty or active; no reset, git clean, or killing another job.

Provision the named Ultrack software in a new Python 3.12 environment, pinned
to `5c94d845eb0a7b78c8dc24492ef00f218a467995`; verify the fetched commit and actual
installed source. The inspected pyproject supports Python >=3.11,<3.14 and
Zarr >=3. Do not downgrade the native Zarr v3 reader based on old Ultrack
installation advice. Record fully resolved versions, pip check, wheel/source
hashes and package provenance after successful installation. The GitHub
`releases/latest` endpoint returned a 2024 publication snapshot, not proof of
the newest package/API. Do not replace this commit pin with that release.

Public software provisioning necessary for this requested tool is a Codex
execution step, using available caches first. No bulk dataset/model downloads,
access applications, contact sharing, license acquisition or paid services.
`gurobipy` may be installed as an upstream dependency, but that neither supplies
nor authorizes a solver license. Select **CBC explicitly** as the portable
baseline. An already authorized local Gurobi installation may be a separately
recorded runtime comparison; do not claim CBC/Gurobi speed parity. The MIP is
CPU work, not a GPU PPO-like update. Reserve the 4090 for native inference and
segmenters; avoid unnecessary multiple CUDA stacks.

The initial storage target is clip-local compressed maps plus one active
SQLite DB; not 199 dense label sets. One native uint16 label volume across all
199 clips is approximately 155.47 GiB even before contours, hypotheses or DB
serialization. Two float32 foreground/contour arrays alone are about 3.125 GiB
per full clip if materialized; avoid silent duplication. SQLite candidate mask
serialization can dominate this and must be measured, not assumed small.

Admission: free bytes must exceed **8 GiB reserve + measured/projected next
allocation**, on each affected filesystem. Recheck before installation and
before every clip. Select another existing approved writable mount when needed;
do not delete old runs or silently reduce the reserve. Bounded RAM scratch is
allowed only when both tmpfs capacity and available RAM cover the peak plus a
separate 8 GiB reserve. Keep durable small receipts outside RAM; document that
RAM data is lost on reboot. If no admitted location exists, emit a precise
storage blocker, continue lightweight implementation/tests, and do not rerun
P0 as a substitute for the missing integration.

## 6. V710 — real Ultrack smoke and integration tests

The supplied `smoke_ultrack.py` is a starting **real synthetic continuation**
probe, not a competition runner. It must execute actual segment, link, CBC solve
and observation export in the local pinned environment. A missing import must
fail, not skip as a pass. Extend it with controlled binary split, exclusive
merged-versus-split hierarchy, two-channel input, anisotropic displacement,
empty middle/first/last frames, near-border cells, label permutation and
nonconsecutive observation IDs.

Mandatory tests: exact `id`/`parent_id` export with a misleading `track_id`;
missing parents; invalid time jumps; >2 daughters; overlap exclusions; signed
logits preserving sign under identity; idempotent link replacement; zero-weight
native augmentation matching U0; real original-mask survival; and rejection of
wrong geometry/native provenance. Unit and synthetic tests are not biological
accuracy claims.

Prefer whole-clip solving initially. If resources require windows, use the
pinned upstream implementation rather than independent windows concatenated
by nearest center. Test overlapping-window ownership, selected-parent
consistency, divisions exactly at boundaries, and all-first/all-last cases.
Compare windowed and full solve on a small feasible instance; require equal
optimal objective within tolerance and graph validity, not necessarily the same
graph under exact ties. Log feasible/optimal/timeout status and MIP gap. A
feasible time-limited graph is not an optimality certificate. An infeasible or
missing result is not an empty prediction or a baseline fallback.

Gate to real data: actual Ultrack solver and export tests pass, including masks
and lineage, and an admitted storage/runtime exists. Save the API signatures,
installed source identity and actual solver backend. No segmentation learning
is required to pass this gate.

## 7. V720-V740 — bounded pilot and complete comparisons

Choose four source diagnostic clips per source embryo using image-only density,
brightness and coverage strata; eight total. Use 8 consecutive frames initially,
then two full clips (one per embryo) for memory/runtime extrapolation. Do not
choose target clips by their errors. Record image/foreground/contour/hypothesis
and selected-mask overlays, foreground support at source annotated centers,
plausible-node count, merge/split ambiguity, candidate-link coverage, DB size,
wall time, CPU RSS, GPU peak and solver gap.

For each direction, calibration uses only that direction's source embryo,
with time-block separation/guard bands inside source clips where possible.
Registered search ceiling: 8 pilot recipes per source, not 8 full-dataset
sweeps; one finalist per arm/direction before new target scoring. Train/calibrate
both source-direction models and freeze their hashes before evaluating either
held-out target direction. Public/inherited checkpoint exposure and repeated
embryo reuse mean this remains **exploratory operational validation**, not a
new independent biological holdout.

| Arm | Required experiment | What its difference can establish |
|---|---|---|
| C0 | Replay preserved incumbent with fresh official scoring | Original reference |
| P0 | Replay retained complete-native v6 point control | Stronger promotion comparator |
| U0 | Image-derived foreground/contours, Ultrack region linking and joint solve | Actual image-driven Ultrack baseline; not learned segmentation |
| U1 | Same U0 hierarchy and candidate bank, recomputed complete-native costs | Value of native evidence inside Ultrack |
| H1 | Conservative one-to-one U1-to-P0 association edits | Mask/lineage support without dropping strong P0 detections |
| U2 | Actual authorized learned instance masks, Ultrack and native costs | Added value of learned object support; optional only when provider unavailable |

These six are the complete-arm ceiling. The calibration/pilot recipes are not
reported as complete scores. If U2 is blocked, leave score/count fields null
and give the exact reason. Do not substitute a second classical arm into its
name. No new detector training until useful masks and tracking benefit are
shown; no additional learned architecture lane in this iteration.

U1/H1 evidence can still be informative if U0 is weak: do not reject native-cost
integration solely on U0's absolute score. But stop scaling when foreground
coverage or candidate-pair coverage cannot represent source GT and report that
bottleneck. Diagnose representation coverage, chosen links, divisions and node
count separately; do not tune exclusively to pooled final score.

Every completed arm must cover **all 199 clips**, with both embryos reported.
Use `annotation_selection.metric_adapter.evaluate_graph` for fresh physical
node matching and official edge/division evaluation. Pool sufficient statistics
using the repository's official aggregation; never average per-clip scores.
Re-evaluate after node geometry/count changes; cached C0 matches are invalid for
Ultrack detections. Report score, delta versus C0/P0, edge/division TP/FP/FN,
node counts, edited edges, graph failures, runtime and memory. Do not derive a
new node-count correction or manipulate the official total estimate.

Initial resource ceiling: one 4090, 24 measured GPU device-hours, 48 CPU-heavy
wall-clock hours, one full-clip DB/solver at a time, at most 8 CPU worker threads,
20 GiB GPU allocation and 24 GiB process-tree RSS, subject to stricter local
availability. These are experiment ceilings, not predicted completion times.
Time is accounted separately for GPU kernels/device activity, wall time and
CPU time. Use a 10-minute initial solver-window cap and recorded feasible gap;
change resource strategy only from source pilot evidence before target scoring.
Fit the full-run forecast to the verified current inference runtime budget with
at least 25% margin; do not extrapolate 4090 performance as Kaggle hardware proof.

## 8. V750 — promotion, cold inference and delivery

A candidate must strictly beat P0 pooled and must not regress either embryo
relative to P0. It must pass every graph/export test and inference/resource
contract. Report gains below 0.001 as small exploratory gains, not a material
advance. >=0.95 is a separate target status. A U0/U1 integration is not a
promotion merely because it ran. Keep C0/P0 available independently.

Rerun a finalist with deterministic graph hashes; if stochastic fitting or
segmentation is used, repeat with a distinct registered seed and report both.
Do not describe two deterministic identical fits as seed robustness. Count the
repeat as a repeat, not a seventh selectable architecture. Nondeterministic
solver ties must be documented with objective and metric differences.

Deliver an annotation-free **cold** CLI that accepts arbitrary new clip names,
loads native images/metadata and explicitly selected source/model artifacts,
regenerates masks/hypotheses/native evidence, runs Ultrack and writes graph/CSV.
Use early filesystem/network guards: deny annotations, GT matching, old
prediction caches and external DNS. First run two full renamed clips, one per
embryo, then cold-run a batch manifest containing more than one clip. Verify
output graph/CSV parity with the selected batch configuration. Do not infer
source-model direction from unknown test filenames or hardcode the two embryo
prefixes as the production routing policy. Define and freeze a deployable
source-agnostic artifact/combination separately; report its own validation and
exposure limitations rather than equating it with cross-fit scores.

Verify the current competition code/runtime/external-model requirements from
the official local references before packaging. The reviewed snapshot says
internet-disabled notebook inference; no license server or model download may
be needed during execution. Do not actually submit or merge anything.

Implementation target is `tools/ultrack_integration_v7/` plus a thin isolated
runtime wrapper such as `scripts/run_ultrack_integration_v7.sh`. Suggested CLI
stages are `preflight`, `smoke`, `pilot`, `freeze`, `evaluate`, `infer`, `report`;
these are **to be implemented**, not existing runnable commands. Support
resumption only after input/config/source checksum validation and unique
per-clip/arm databases; do not replay stale selected flags or parent pointers.

Required durable small reports under `results/ultrack-integration-v7/`:
`status.json`, `final_report.md`, `ablation_scores.csv`, `runtime_summary.json`,
`environment_lock.json`, `source_lock.json`, `selection.json`, and an offline
HTML diagnostic summary. Include local-only overlays/viewer for foreground,
contours, competing masks, chosen lineage, native scores and failure reasons.
Detailed microscopy, annotations, masks, checkpoints and predictions stay in
ignored local output. Commit only sanitized aggregate summaries and software.

Write `handover/ultrack-integration-v7/CONTINUATION.md` with the actual commands,
source/config/artifact hashes, measured stage status, remaining blockers and
next legal experiment. Update the PR with those results; leave it unmerged.
A blocker report must distinguish completed software, real synthetic execution,
real microscopy execution and complete scored comparisons. Never present
passing contracts, a baseline replay, an import, or the plan itself as a
successful Ultrack accuracy experiment.
