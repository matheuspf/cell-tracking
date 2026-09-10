# Validation and interpretation

## Included reference tests

The authoring checks cover NumPy mask contracts only. They do not establish that
FOCUS/Cellpose/Ultrack installs, runs correctly, or improves Biohub scores. Local
Codex must build real integrations and run the tests below in their actual
isolated environments. Do not carry forward a historical test count as proof.

## Required integration tests before expensive experiments

1. **Grid/axis/units:** native anisotropic volume, TZYX versus ZYX confusion,
   noninteger resampling origin, independent x/y/z translations, and FOCUS z_ratio.
   Known physical volume/covariance and native/export coordinates must agree.
   Floating dtype must not silently turn T into channels or probability into ID.
2. **Mask evidence:** same centroid, same volume and same bbox can still mean
   different occupancies. Full-mask features must distinguish that fixture.
   Empty/absent masks are missing, not spheres; center-only disablement is identity.
   Label numbers permuted independently per frame must not change tracking.
3. **Tiling:** full-versus-tiled interior masks, duplicated seam instances,
   chunk boundaries, empty frames and one region crossing image/temporal borders.
   Preserve seed/mask collision receipts. Never claim a tile-truncated nucleus
   satisfies full-volume conservation.
4. **Provider interfaces:** actual headless inference on real source volumes,
   nonempty images producing meaningful masks, correct output grids, confidence
   semantics, model hash and CPU/GPU memory. Segmenter runs must occur, not reuse
   old watershed output under a renamed provider.
5. **Ultrack conversion:** one versus multiple label arrays, finite foreground/
   contours on all-empty frames, mask-to-hierarchy coverage, duplicate provider
   invariance, ancestor/descendant exclusion and crossing-hypothesis conflict tests.
   Empty frame handling must not silently drop a required clip from scoring.
6. **Motion:** source-to-target versus target-to-source, no np.roll wraparound,
   no label-ID interpolation, shared-field consistency, validity/occlusion masks,
   and exact distinction between bbox shifts and a true deformable warp.
7. **Scoring features:** same-input official extractor parity, particularly
   HOCT voxel versus physical descriptors; bbox and mask IoU kept separate;
   confidence missingness not conflated with background; no GT feature columns.
8. **Graph/solver:** continuation, true division, nearby unrelated birth,
   duplicate/split hypotheses, displaced owner, no merges, no three-child forks,
   no t+2 edges, deterministic tie handling, timeouts and valid no-op fallback.
   Preserve the full local division evidence window at temporal solve boundaries.
9. **Sparse supervision:** positive centers do not define dense masks; a single
   annotated child leaves another child unknown; no labels/count estimates in
   inference; same-embryo teachers and public checkpoints recorded as exposure.
   Both source directions use the same bounded recipe before target-score reveal.
10. **Fresh run:** raw images with unfamiliar filenames, no old mask/prediction
    caches, no evaluation paths/network access, correct source-model routing,
    exact graph and CSV semantics, measured runtime and whole sample coverage.

## Why evaluation is not dense mask accuracy

No dense masks are supplied in the recorded competition/external-data inventory.
Sparse center containment and one-to-one matching measure partial coverage only.
Mask consistency, temporal overlap and visual inspections are diagnostics. Mask
Dice or average precision requires independent actual mask labels; otherwise mark
it unavailable. A giant merged object can contain many annotated centers, and
unmatched objects may be real unannotated nuclei.

Optional independently reviewed source patches improve calibration but are not a
prerequisite for automatic experiments. Record review sampling, blinding, complete
versus partial annotation, and no target-embryo training. Never invent the review.

## Comparison populations

Use leave-one-embryo-out directions for directly trained additions. Upstream
models and repeatedly examined embryos make these operational exploratory results,
not untouched OOF. Unknown overlap prevents independent clip/node bootstrap claims.
Report each embryo and pooled official score, not a confidence interval made from
millions of correlated cell observations.

A different baseline at S600 may be added only with completed local validation
receipts before v6 outcomes; C0 must stay in every comparison. Existing v5 ongoing
fits need not finish before v6 CPU preparation, but no old job or mutable artifact
may be overwritten. New checkpoints must not accidentally read partially written
v5 outputs.

## Promotion

Validate full official 199-clip aggregation, point/edge schema and fresh inference.
Primary pooled delta must be positive and each embryo nonnegative within 1e-8.
A learned finalist must reproduce the direction of gain in the registered secondary
seed; never replace a failed primary with a lucky replica. Fixed-model deterministic
routes need repeated graph/solver stability, not pretend retraining. >=0.95 is
local target attainment, not hidden-LB prediction. Retain smaller reproducible gains
as below-target outcomes. C0 remains selected if no candidate qualifies.

## Negative or partial completion is useful

Name exact blockers: access, runtime, missing masks, hierarchy attrition, bad
motion, poor association, division precision, model units, or feature transfer.
Do not turn a failed provider into a fabricated result or declare all segmentation
ineffective because one combined model/solver fails. Finish report, dashboard,
status and continuation even when the primary hypothesis is rejected.
