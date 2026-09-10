# M600-M690: instance-mask tracking experiment

## Goal, decisions and non-negotiable comparison

Target >=0.95 on all 199 official local clips. Reference C0 is
0.934802374260586, not the old 0.911774 baseline. All v6 output is separate from
v1-v5. The existing v5 snapshot is incomplete; neither assume live processes are
finished nor spend v6 rerunning their historical assignments.

The central deliverable is a real mask-first experiment: image -> instance masks
(and boxes) -> mask-aware associations/divisions -> tracks -> final representative
points. Merely computing masks for a few examples or feeding their centers to the
unchanged point method does not finish the study. Run the controlled hybrid and
standalone Ultrack routes. There is no guarantee that segmentation will improve
point-graph scoring, and the unmodified C0 must remain exportable.

## M600 — freeze baseline, running state and evaluation contract

Read root guidance and v5 CONTINUATION, ablation scores, HOCT unit audit and
observations.py. Detect active local jobs/worktrees and record their state without
killing or restarting them. Use a separate worktree when required. Record the
actual latest validated incumbent, its hashes and exact 199-sample set. Verify
C0 counts and score with the pinned scorer; if local evidence changes the incumbent,
write a new explicit baseline record before new scientific comparisons.

Read the official fluorescent-marker description and inspect raw XY/XZ/YZ views.
Record whether the target's visible object is nucleus, whole cytoplasm, membrane
or unresolved. A pretrained 'cell' model must not silently be interpreted as a
membrane detector. Verify axes, spacing, clipping and normalization using actual
metadata. Do not convert every clip to a dense resampled 256^3 time series.

Record package/weight access and terms. Attempt FOCUS using already authorized
credentials/checkpoint paths. Do not accept new gates/share contact information.
No remote microscopy upload. Cellpose and Ultrack integration proceed regardless
of FOCUS access. Preserve current environments and create separate optional
runtimes, with explicit interpreter paths in the v6 CLI.

## M610 — learned segmentation pilot (first substantive experiment)

Before a large new linking implementation, segment 12 short windows: six per
embryo, selected by image-only density/SNR, depth occupancy and early/mid/late time.
Each window has 8 contiguous full 3D frames, including boundary halos when present.
At least four windows should be fixed by uniform seeded sampling, not handpicked
for visually easy cells. Freeze IDs before mask/GT analysis. Include both current
source roles; do not call this an independent validation set.

Run FOCUS and Cellpose independently; when FOCUS is blocked, replace its actual
comparison with StarDist3D if feasible. At least one learned segmenter must run;
old watershed alone is not acceptable completion. Use at most two initial settings
per model: its supported default and one physically justified size/resolution
setting fixed before target-label plots. Prefer native-grid masks; allow XY2 only
as a separately measured speed/quality condition, with transformed spacing.
Retain patch stitching/seam receipts and compare a whole-volume against tiled
small fixture. Frame-local label IDs must never be treated as persistent cell IDs.

Save selected raw slices, masks, 3D boxes, centers and short track overlays in an
offline viewer. Check mask coverage, boundary alignment proxy, foreground SNR,
fragmentation, empty frames, seam splits, object volume distribution and temporal
flicker. With sparse annotations, report labeled-center coverage, one-to-one center
matching and suspected merges around multiple GT centers; these are NOT mask AP,
Dice, or dense detection precision. Unmatched masks can be true unannotated cells.

Optionally export a blinded 24-ROI source-only boundary audit pack, with sample
probabilities and empty regions included. Missing human labels never blocks the
automated study; without them report boundary quality as unverified. Do not use
evaluation audit corrections for source training. Select at most two usable mask
producers based on valid outputs, resource profile, source diagnostics and full
pilot tracking; retain the ranking as exploratory because these embryos are reused.

M610 outputs include measured seconds/volume and projected full 19,900-frame cost.
No model selected solely because its overlays look attractive. If every model
fails, fix one documented scale/normalization/interface defect and rerun the same
pilot once; otherwise finish with an honest segmentation blocker and all outputs.
Do not replace failed learned segmentation with fabricated masks.

## M620 — object representation and bounded full-data mask cache

Build per-instance records retaining compressed/cropped mask support, exact native
bbox, separate voxel/physical centroids/covariance/volume, intensities, model score,
truncation flags and provenance. Background label 0 is never an instance. Use the
source model's real confidence semantics; absent scores remain missing.

Map masks to C0 points using a one-to-one, label-free geometric/containment assignment
with explicit ambiguous/shared/unmatched states. If two C0 points land in a merged
mask, do not duplicate that mask as two independently measured objects. The fixed-
C0 route retains points lacking valid masks and falls back to its existing evidence.
Do not use the ground-truth matcher to create this inference association.

Preserve support in image coordinates through time. Boxes cheaply prune overlap
calculations and define crops; actual voxel sets decide mask intersection. Compute
raw overlap and overlap after a shared image-derived local motion transform. Use
forward/backward image-flow consistency, not an arbitrary per-pair translation that
can make unrelated shapes align. For nonrigid flow validate source/target direction,
voxel/physical units and lost support near borders. Censored volumes/boxes carry
missingness flags; do not force their mass ratios to look like divisions.

A single uncompressed uint16 native label movie for all 199 clips would occupy
155.47 GiB; compressed size is unknown until measured. Do not write several such
copies. Process one clip or short time block at a time. Keep compact mask crops
or compressed labels, graph features, and manifest hashes. Save dense masks only
for selected audits or a bounded reusable cache; reconstruct reproducibly if needed.
Do not delete old study artifacts to free space.

## M630 — same-detection point / box / full-mask test

This is the most informative low-risk experiment. Keep C0 nodes, native coordinates,
counts and the same source-derived candidate-link universe fixed. Use the same
source-fit procedure, solver and division policy across the following treatments:

- Bpoint: native edge evidence + centroid distance + the existing allowed context.
- Bbox: Bpoint + bbox overlap/extents, occupancy and truncation.
- Bmask: Bbox + actual mask overlap, motion-warped overlap, physical volume/shape
  changes, masked image similarity and parent/daughter-union evidence.
- Bmask_scrambled: deterministic frame/size-stratum mask-feature permutation as
  a negative control; coordinates, labels, candidate set and model size unchanged.

Fit a regularized source-only residual/logistic model first, anchored to native
log odds. This is an attribution test, not the whole requested direction. Preserve
incumbent no-mask behavior and use a maximum one standard alternative capacity
setting if the residual underfits. Avoid another many-model fork sweep. Identity
must be byte-equivalent to C0 before learning. Report all affected links and truth-
identity gains/losses after full rematching, not just predicted edge-ID changes.

Continuation evidence includes shared-motion IoU, source coverage and target coverage
separately, volume log ratio, covariance/eigenvalue change, masked intensity contrast,
local neighborhood consistency and per-model agreement. Raw IoU=0 must not hard-
exclude physically plausible motion. A small object's translational distance can
exceed its diameter while its track remains valid.

Division evidence compares the warped parent to the UNION of two disjoint daughter
masks, their aggregate volume/intensity (soft evidence only), separation/persistence,
and alternative explanations: a parent continuation plus an unrelated nearby birth,
or a segmentation split of one object. Include competing incoming owners and future
path context. No hard universal volume-conservation prior for changing fluorescence.
A shared mask split is not automatically a biological division.

If HOCT embeddings are reused, add the mandatory old-watershed/upstream-unit-correct
control and require exact extractor parity first. Attribute that unit repair
separately; don't credit it to new segmentation. Do not rebuild all C0 associations
with v5 J merely to test a feature.

## M640 — standalone segment-then-track and genuine Ultrack hierarchy

For each selected segmenter, run a standalone hard-instance tracker using centers,
then the SAME hard instances using mask/shape costs. Preserve masks through the
track association. Then run actual Ultrack on the single segmenter's foreground/
contours and on a two-source combination when two usable segmenters exist.

Use `labels_to_contours` only on registered, same-grid, frame-aligned arrays.
It ORs foreground and averages boundaries. Never average integer IDs or assume the
helper imports each original mask as an exact optimizer hypothesis. Remove duplicate
mask sources; failed model outputs are not empty votes. Guard all-empty frames and
inspect hierarchy granularity versus actual candidate masks. This matters where
one model sees a merged object and another sees two: retain the union and plausible
children until temporal evidence chooses, rather than applying irreversible NMS.

Use Ultrack's native hypothesis exclusion and optimization, logging candidate/edge
counts, solution status and time/gap. Run real solver fixtures for continuation,
division, independent birth, crossing masks, missing frames and parent/child
segmentation exclusion. No giant unbounded new global Python MILP. Segment/link/solve
per clip; use supported temporal blocks with overlap and validated seam ownership
when needed. Border appearances are not a global free-birth policy. Do not silently
add gap interpolation as real observations.

Test no flow versus image-derived flow on the pilot. The pinned linker shifts TARGET
regions, so verify sign with a known translation. Keep a native-overlap Ultrack
baseline before supplying learned/native edge scores. If custom weights are used,
record their probability/logit/linear units, insert each edge once through supported
interfaces, and use the same recipe across data/feature ablations. No GT-assisted
Ultrack matching, annotation constraints or built-in GT parameter optimization at
inference. Reproduce candidate-set and objective parity before attributing gains.

Always export the selected masks as well as track tables for local inspection.
Convert database node IDs/parent node IDs to consecutive-frame graph edges; do not
mistake parent-track metadata for an edge at every frame. Preserve forks, ensure
indegree<=1/outdegree<=2, and apply integer coordinate conversion only at final output.
Test fixed C0 coordinates for one-to-one mapped instances separately from mask
centroids/medoids. Unmapped masks use a source-fixed representative rule. A mask
centroid can lie outside a nonconvex mask: retain a deterministic interior medoid
candidate as an ablation, not a GT-optimized coordinate.

## M650 — only if masks are useful: real segmentation adaptation / speed

Do not train another point-event head with the old external mixtures. Fine-tune
or distill segmentation only when M610-M640 identify useful boundary/mask evidence
that is limited by speed or correctable source-domain errors.

Preferred optional route: generate teacher masks on declared source-embryo images,
select stable high-confidence/inter-model-consistent interiors and boundaries,
ignore disagreement/unknown regions, and train a compact foreground+boundary 3D
student. These remain pseudo-labels, and filtering must not simply delete difficult
source cells. Use a teacher-only control and a same-architecture no-finetune control.
Verify gradient/weight changes and learned-mask outputs. Sparse centers can provide
positive localization/seed constraints, never label the whole remaining image as
background. Existing synthetic center-only data have no measured true masks; derived
ellipsoids or renderer geometry must be named synthetic targets, not ground truth.
Zoo/RIKEN without images cannot directly train this appearance segmentation model.

An optional manually curated source-mask pack can improve targets only if it actually
exists; no waiting for human input. Keep evaluation masks separate. Student comparison
uses full graph score and runtime, not pseudo-mask Dice alone. If teacher segmentation
is already fast enough, omit distillation and spend the budget on full tracking.

If dense inference is too slow, preregister a keyframe+image-flow mask propagation
variant with real image checks and periodic resegmentation. Never assign a constant
mask/identity through a division or claim interpolated frames are model detections.
Measure dense-versus-propagated tracking, with identical label-free refresh triggers;
this is optional and cannot silently replace the first dense-mask experiment.

## M660 — combinations, score and adoption

At most 24 complete new configurations, including feature/solver controls and
replicas; avoid a Cartesian product of model, threshold, resolution, motion and
solver grids. The initial comparison budget is allocated in config.json. Pilot
selection can save resources but is explicitly exploratory, not untouched holdout.
Train new learned parameters on source embryo only, freeze both directions before
comparative target scoring, then run the complete 199-sample evaluation.

Candidate evaluation MUST rematch nodes and re-evaluate divisions from scratch.
Enforce the complete sample set, frozen GT/estimated counts, official per-sample
edge denominators/weights and pooled division counts. Report score, adjusted edge,
raw edge J, division TP/FP/FN, node totals, truth-ID gains/losses, mask validity
coverage, proposed/accepted split hypotheses, runtime and no-mask fallback rate.
Mask-only metrics cannot replace the official score. Missing images/predictions
fail the run rather than being omitted. No fake dense ground truth from points.

Primary deployment is one frozen same-recipe candidate, not hindsight per-clip
selection against labels. A deterministic pretrained-only improvement needs exact
rerun/packaging checks; a learned finalist must pass a second-seed full comparison
with the same recipe (primary seed exported). Qualifying delta is positive pooled,
nonnegative in each embryo within numerical tolerance, and passes integrity/runtime
checks. Label target_met only when the adopted full local score >=0.95. Preserve a
smaller repeatable improvement as below_target_gain. Otherwise export C0 unchanged.
Do not infer confidence intervals for unseen embryos from millions of correlated
cells. Source tuning within validated purged blocks is allowed if actually possible;
unknown overlaps mean source engineering calibration and exploratory target sweeps,
not invented independent folds. The legacy public/teacher exposure remains recorded.

## M670-M690 — cold inference, viewer, preservation and report

Run at least four fresh full clips (two per embryo, image-selected density range)
through the chosen image -> masks -> features -> tracking -> CSV path, including
an unfamiliar filename and a shifted/tiled-input fixture. Deny annotations, old
prediction caches, remote image services and training datasets in the deployment
process. Allow only explicitly hashed pretrained weights, actual input images and
new output files. Compare fresh masks/graphs against their scored counterparts;
relabeled instance IDs are acceptable only through a proven canonical mapping.
All production frame nodes/edges must roundtrip through CSV exactly.

Report actual seconds/frame, seconds/clip, total mask+tracking cost, peak VRAM/RSS,
solver runtime, output/cache size and projected hidden-size cost. The old C0 roughly
8-hour 4090 extrapolation is not a universal budget guarantee; additive segmentation
could make a hybrid impractical while a standalone tracker may replace that cost.
Profile both. Do not claim Kaggle hardware/runtime parity from a 4090 sample.

Write final_report.md, scores.csv, mask_quality_proxy.csv, mask_feature_ablation.csv,
source_model_registry.json, runtime.json, mask_manifest.json, prediction_lock.json,
validation_receipt.json, offline dashboard/viewer, inference package and CONTINUATION.md.
The viewer must toggle raw XY/XZ/YZ slices, actual masks, boxes, points, mask overlaps
and forks; include all preselected pilot examples, not only success cases. It uses
local images only and distinguishes model masks, reviewed masks and sparse points.
Git gets sanitized numeric evidence/code; detailed raw crops/masks/GT and weights
stay local. Create a dependency/preservation manifest before ephemeral-host loss.

Finish a meaningful measured report even if one segmenter or solver is blocked or
no recipe wins. Never reinterpret an unfinished study as a negative scientific
result. No paid services, rented machines, submissions, forum posts or PR merge.
