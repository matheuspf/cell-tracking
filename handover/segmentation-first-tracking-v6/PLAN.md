# X600–X690: execute segmentation-first tracking

This plan is implementation plus measured experiments, not a request for another
plan. Keep independent arms running when a detector/solver/adaptation candidate
fails. Never replace the required full-instance experiments with another point-
offset or tiny fork-threshold sweep. Tests under this handover are reference checks,
not an implemented production detector/tracker.

## X600 — Intake, isolation and frozen controls

Read AGENTS.md, REVIEW.md and the latest LOCAL v5 continuation/status. Record live
processes, artifacts, pending configs and their hashes. Never switch the worktree
serving v5, alter its files, terminate jobs, or start competing GPU tasks. Use this
new branch in a separate worktree; share read-only data/weights through explicit
paths. CPU-only preparation may proceed within the actual free resource budget.
Prior work does not need to finish to design/test v6, but GPU admission must respect
current jobs. Record resource-blocked work honestly instead of overwriting it.

Read local official data/rules/evaluation and inspect raw Zarr metadata/channel
semantics. Run the existing exact C0 reconstruction/scorer on all 199 clips and
preserve the verified graph hashes. Do not regenerate C0 using only the primary
native model. Build an input/weight inventory without downloading huge duplicates.
Register two opposite-embryo directions, source-only pilot selection, config budget,
seeds and allowed masks/weights BEFORE scoring v6 candidates. New v5 results can
be reported separately; do not retroactively change the fixed C0 comparator.

## X610 — Persistent object representation and reference fixtures

Create `tools/segmentation_tracking_v6/` with separate configuration, detector
adapters, object store, matching, association, hypothesis graph, inference and
EVALUATION-ONLY modules. Implement CONTRACTS.md. Save original-grid masks as cropped
lossless arrays/RLE or chunked label movies; derive boxes/physical properties and
uncertainty. Store coordinates/units for each field, not one ambiguous `position`.
Add checked full graph serialization supporting moved centers under existing IDs.

Port the small authored reference tests into production tests and add actual 3D
images, tile halos, track/label ID differences and graph export cases. Confirm no
existing input/artifact is modified. Implement cache fingerprints and atomic shard
writes with a resume manifest; incomplete shards are not accepted as completed.

## X620 — Real detector pilots and headless integration

Run D0 native watershed with retained masks as a reference, D1 FOCUS when authorized,
and D2 Cellpose or a real compatible StarDist3D checkpoint. Follow DETECTORS.md.
For each embryo, pick 3 short spatial/time pilot windows by image-only low/median/
high density and signal criteria, without target annotations. Limit each backend
to at most 6 registered preprocessing/size/threshold presets per source direction.
A full-frame discovery pilot must not require existing point prompts.

Inspect orthogonal overlays, mask fragmentation, tile duplication, volume/shape,
border effects, and runtime/memory. Use available source-only supported point/edge
coverage for calibration, never claim dense precision from sparse points. Audit
normalization and input/output transforms. Confirm actual trained weights load;
record source, key coverage, checkpoint hash and a pixel-response test. FOCUS
blocked access does not close D2 or Ultrack. A learned backend's bad initial output
allows one justified source-only repair, not an unbounded outer-score grid.

Choose at most two detector families using source-only evidence before target
scores. Process a complete 100-frame source clip for each selected backend; cache
masks/soft evidence and amortize model load. Project full-study and notebook runtime
with measured per-frame variability and overhead, not a single cropped patch.

## X630 — Fixed-C0 point/box/mask factorial

Use one common full C0 node/candidate population, original coordinates, full native
C0 evidence and the same decoder/edit region. Attach observed segmentation objects
with explicit ambiguity and missing masks. Fit/source-calibrate one small edge/
event scorer family in three modes: center/motion only, center+bbox extents, and
center+bbox+mask/appearance/warped-overlap. Maintain no-op original graph as a valid
alternative. This measures whether object information helps without confounding
new detections, coordinate changes or decoder replacement.

Learn continuation, division and alternate-owner evidence with sparse-supported
positives and censored negatives. Compare continuation+independent birth against
fork explicitly. Do not send a shape model through v4/v5's already restrictive
geometry gate unless that gate is itself a controlled arm. Fixed-C0 failure is
informative but does not stop independent detector-driven reconstruction.

## X640 — Full detector-driven object tracking with Ultrack

Construct a complete movie of independently detected instances, without anchoring
all nodes to C0. Build a center-only control on this same population and a local
mask-aware graph path. Integrate actual Ultrack, first using a single detector's
labels, then foreground/contour hypotheses. Apply original-grid physical scale,
image-only motion/flow and recorded solver settings. One background/contour value
must not be confused with a class probability. Preserve masks through link scoring
and hypothesis selection; only derive integer node coordinates at final export.

Implement selected-parent, ancestor exclusion and biological-fork constraints.
Check solve status, objective/gap, wall and memory for EVERY window; a timed-out
feasible solution is labeled, infeasible/malformed output fails. Use licensed
local Gurobi only when already available, otherwise benchmark an open-source
backend. No required license purchase. If Ultrack cannot execute, retain a
separately named mask-aware local solver experiment and exact blocker evidence.
Full C0 remains a declared reference/fallback, not a target-selected patchwork.

## X650 — Bounded segmentation uncertainty, not union of all detections

Compare two settings from one detector or two genuinely different segmenters via
aligned foreground/boundary hypotheses. Use Ultrack's labels-to-contours facility
where appropriate; audit which source object survives each resegmented hypothesis.
For nonnested cross-model masks, explicitly implement conflicts rather than assume
an ultrametric hierarchy represents every overlap. Parent coarse object and split
children at one frame are alternative SEGMENTATIONS; two children may coexist.

Use mask union versus deformed parent shape and 3–5-frame consistency to distinguish
mitosis from flickering oversegmentation. Do not hard-require equal size or perfect
IoU. Calibrate confidence per backend. Compare single- and multi-segmenter output
with fresh official node matching/count adjustment; more recall is not promotion.
Keep flow-on/off as one bounded ablation when displacement exceeds object extent.

## X660 — Conditional detector/association adaptation

Only after measured pilot domain errors, run actual image-based refinement with
real permitted dense masks or explicit uncertainty-aware weak/pseudo-mask losses.
The downloaded center/lineage datasets are not dense segmentation labels. Optional
external masks require a mask-coverage/license manifest and a source-only split.
Do not require the user to draw new labels before any experiment can run.

Use predicted-mask consistency, unknown-region ignore masks and supported source
points/links; no all-unlabeled-background objective. If using pseudo masks, compare
against the frozen teacher on held-out source regions and inspect collapse/merges.
Train appropriate image layers, not only claim a new detector from a coordinate
regressor. Log actual image coverage, nonzero backbone gradients, changed weights,
steps and calibration receipts. Train each source direction independently; learned
finalists need the same-recipe secondary seed. Do not substitute the lucky replica
or call inherited checkpoint exposure fully independent validation.

A mask-conditioned Trackastra or corrected-unit HOCT scorer is optional within the
same budget; it is not required to replace the primary Ultrack/object experiment.
A blocked adaptation stage does not invalidate working pretrained detection.

## X670 — Complete official evaluation and attribution

Freeze primary weights, transforms, thresholds, graph costs and both source-trained
configurations BEFORE opening target scores. Execute registered configs on all
199 full clips, then use fresh pinned official matching and run-level aggregation.
Follow EVALUATION.md, including count/division effects, two embryo scores, exact
output round trips and known exposure limits. Quarantine truth-assisted diagnostics
from deployment assets. Never promote from a source pilot, six-clip score, improved
mask aesthetics or a truth-assisted feasibility score.

Report attribution: original C0; same-C0 point/box/mask; independent-instance point/
mask; single-segmenter Ultrack; uncertainty/multi-segmenter; optional adaptation.
Separate changing nodes, moving centroids, changing evidence and changing solver.
Do not claim every row isolates one factor when actual code changes several.

## X680 — Fresh-image export, resource reliability and visual review

Build an offline package whose only sample input is an arbitrary-named image Zarr.
Six predeclared full 100-frame clips (3/embryo, image-selected density/signal strata)
must freshly execute every selected detector, mask store, tracker and export with
GT/evaluation/prediction-cache/network denied. Compare exported graphs to the
frozen full-study run and round-trip CSV, including changed-center records. Raw
output sample names must come from input discovery, not training-file allowlists.

Create a local HTML visual report with XY/XZ/YZ slices, time controls, consistent
track colors, masks, bboxes, centroid/edge overlays and selected alternatives.
Include birth/division/merge ambiguity and failures, not just attractive successes.
Overlay GT only in a clearly evaluation-only viewer. No source microscope upload
or hosted demo call is authorized. Local images stay outside Git.

Measure cold model load, per-frame/clip and solver times, VRAM/RSS, disk high water,
cache compression and projected official notebook total. Exercise restart after
partial masks/DB/window outputs and deterministic tie breaking. Reserve disk and
respect actual available GPU/RAM; no all-clips dense uint32 mask cache. Do not reduce
resolution, skip frames or drop candidates silently to meet a timing target.

## X690 — Retain useful code and return measured findings

Commit maintained production modules/tests/wrapper/configs, small asset receipts,
per-clip aggregate metrics and a final report to this new branch. Keep previous
studies and their manifests unchanged. The report must state executed/blocked/
rejected/pending status for every stage, actual backend APIs/versions/weights,
selection gates, runtime, complete pooled/per-embryo scores and failure cases.
Provide `results/segmentation-first-tracking-v6/CONTINUATION.md`, a working raw-image
inference command and a safe resume command. No merge or Kaggle submission.

If >=0.95 is not achieved, preserve C0 and export useful mask/tracker evidence and
negative results; never relabel a partial or oracle-assisted result as success.
The user's requested change is meaningful only if masks/boxes materially enter
association and a complete detector-driven path is actually attempted.
