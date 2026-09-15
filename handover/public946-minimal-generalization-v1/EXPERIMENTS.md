# Registered mechanisms — revision 2

## Common contract

Every E-arm starts from **B0, the original public notebook**, not the locally
selected residual model and not B1. Change exactly the named mechanism. Freeze
source-resolved constants and tensor semantics before its first comparative
score. Geometry-derived constants below are algorithm definitions, not search
ranges. All other thresholds, weights, checkpoints, preprocessing, harmonic
inter-model fusion and downstream steps retain public behavior.

A source audit may establish a mechanism is already present, mathematically a
no-op, or cannot be added at the proposed interface. Record `not_applicable`
with source lines, relevant counts and identity tests; do not fabricate a gain,
substitute a different method or tune a fallback after looking at scores. A
missing cache is not non-applicability: rebuild only the needed evidence.

## E01 — native-evidence-gated motion repair

**Hypothesis:** retain useful geometric repairs without allowing the relinker to
overwrite associations the existing model supports more strongly. Unlike B1,
this can accept some original motion proposals. Unlike v3, it fits no residual model.

Hook the original relinker to obtain before/after edge sets. Partition their
symmetric difference into connected conflict components; inspect the proposal as
an atomic edge replacement, not sequential edits whose ordering changes results.
For this arm, consider only components with unchanged nodes, unchanged per-node
in/out degrees, no touched fork and finite native pre-threshold probabilities for
every removed/added edge. Use the exact same target-column probability universe
as the public predictor. Accept only when
`sum(p_added) > sum(p_removed)`; equality or missing evidence means abstain and
retain the pre-relink edges for that component. Original non-conflicting edge
validity checks still run. No probability margin, new candidate pool, threshold
relaxation, new forks, deletion or confidence imputation. Run all later repairs.

This deliberately restricts the action class to score-supported reassignments;
it is not a calibrated posterior guarantee. Capture complete native scores where
necessary: post-threshold-only caches cannot justify absent scores. Report proposed,
eligible, accepted and rejected components, original TP retained/lost, final official
score, division counts and cascades. Test swaps, ties, missing probabilities,
conflicting edits and predicted forks. It may collapse to B1; prove and label that
rather than calling it a separate winner. Never combine E01 with no-motion.

**Cost:** graph replay if native probabilities exist, otherwise fresh evidence capture.

## E02 — analytic sub-voxel output localization

**Hypothesis:** reduce spatial quantization without retraining or changing cell
identities, graph topology, association scores or candidate counts.

Retain each raw detection's ID, generating heatmap, grid center and coordinate
transform. After topology/repair/pruning, immediately before public smoothing,
refine only surviving original detections that have not already been relocated
by a repair. Inserted nodes and relocated originals retain public coordinates.
At the original peak, for each spatial axis use three neighboring values from
the **same scalar detection response used by the public peak finder**:
`delta = 0.5*(h_minus-h_plus)/(h_minus-2*h_zero+h_plus)`.
Apply only to a finite concave local maximum with valid neighbors and
`abs(delta)<=0.5`; otherwise use zero displacement on that axis. Do not try log
responses, centroids, different radii or learned scaling. Map to original voxels
using the audited affine/center-offset transform, not an assumed multiply by four.
Then apply the unchanged public smoother and serializer. No early rounding.

Existing sub-voxel localization means this specific arm may be redundant. Very
few eligible peaks is a measured coverage result, not permission to move synthetic
nodes using an unrelated DeepCenter map. Unit fixtures: shifted parabola, flat or
convex response, border peak, anisotropic axes, transformed coordinates, half ties.
Report eligible fraction, displacement distribution, matched-node localization,
TP losses/gains under fresh rematching and unchanged pre-smoothing topology.
Counter-risk: heatmap peaks may be biased and later smoothing may erase the gain.

**Cost:** small point neighborhoods; fresh detector capture if unavailable.

## E03 — continuous node-feature sampling

**Hypothesis:** truncation makes association evidence discontinuous for fractional
detection coordinates. Interpolation may reduce this discretization sensitivity.

At the installed node-feature lookup only, replace truncation/gather with
trilinear interpolation over the **same encoded feature tensor**, in each public
model branch. Keep detector outputs, positions passed to the transformer,
coordinate normalization, masks, checkpoints and edge-score fusion unchanged.
First count fractional lookup coordinates in the actual public path. All-integer
coordinates or an already-interpolating public extractor yield a documented no-op,
not a claim that the helper's native adapter proves an exploitable defect.

Use the audited feature-grid coordinates. For a spatial axis of size N and index
q, the `align_corners=False` normalized coordinate is `2*(q+0.5)/N-1`.
PyTorch's 5D `grid_sample` grid orders coordinates x,y,z, not array z,y,x; volumetric
`mode='bilinear'` performs trilinear sampling. Preserve valid-node masks and public
boundary behavior; reject unexplained out-of-grid inputs instead of hiding them
with padding. Integer interior coordinates must match gather within measured
numerical precision. Test constants, affine fields, corners, masks, singleton
axes and feature/channel ordering. Do not also change positional embeddings.

Counter-risk: the checkpoints were trained with their original sampling rule, so
interpolation can introduce train/inference mismatch. Score it, do not assume a fix.
Report fractional-coordinate coverage, feature differences, parent ranking changes
and full graph metrics. No retraining is authorized to rescue the arm.

**Cost:** existing features plus new sampling/head execution; stream if not cached.

## E04 — half-grid-phase detector consensus

**Hypothesis:** a cell's alignment to the spatial downsampling lattice affects
whether it is detected; averaging two fixed phases may reduce that sensitivity.

Add exactly one XY image phase to the original detector pipeline. Phases are
`(0,0,0)` and `(0,ds_y/2,ds_x/2)` in original voxels, provided audited XY strides
are even; for `(1,4,4)` this is `(0,2,2)`. This is a translation before downsampling,
not another reflection already covered by the existing TTA. Use non-wrapping
reflection padding and explicit valid-support masks. Invert each phase's scalar
response onto the original detector grid using audited cell-center transforms.
For each model, average the two aligned responses equally where both are supported;
retain the unshifted response where shifted support is absent. Preserve original
within-phase TTA, harmonic inter-model fusion, thresholding and NMS. Run NMS once
on the resulting response, not once per phase followed by coordinate voting.

Association features come from the unshifted public image stream, sampled by the
public rule at the consensus detections. Do not average orientation-dependent
feature channels or change the association head in this arm. Retain original
normalization quantiles for both phases. Audit exact response-to-feature interfaces;
if scalar response cannot be replaced independently, document that interface blocker.

Tests: constant input, impulse and known shifted peaks; sign of inverse mapping;
odd shapes, supported boundaries and duplicate suppression. Report detection births,
losses, link cascades and behavior at borders. Do not search shifts, numbers of phases,
blend weights, downsampling filters or new thresholds. Interpolation of responses
may blur faint peaks; roughly doubled detector work is a cost hypothesis to measure.

**Cost:** one additional full detector phase, existing association architecture.

## E05 — association-only spatial-reflection consensus

**Hypothesis:** detection TTA may leave orientation-sensitive learned associations
unaveraged. Improve those scores without changing which cells were detected.

Freeze B0 detection IDs/coordinates and the complete candidate-parent universe.
Compute association evidence under identity and X reflection, using the same
public checkpoints and inter-model fusion in each view. Transform images and
physical/node positions consistently; map IDs and scores back exactly. Do not
redetect, average learned feature channels, swap unequal-spacing axes or reverse
time. Preserve the original image/decoder contexts and all detection TTA.

Use the existing probability tensor consumed by candidate thresholding. For each
view, retain the public source-column/target-column semantics from the source
(the traced implementation normalizes competing parents in each target column).
Arithmetic-average the two normalized probability tensors on an identical parent
universe, then apply the original threshold and ILP. If public fusion produces an
unnormalized evidence tensor, perform its existing normalization before this new
view average; never silently softmax probabilities again. Record this seam locally.

If the public association path already performs exactly this consensus, identity
proof makes the arm non-applicable. Tests: stable-ID reorder, double reflection,
probability-column sums, masks, unchanged detections and selected synthetic links.
Counter-risk: averaging can dilute a correct strong association. Measure changed
rankings, ambiguity, TP survival and divisions, plus actual memory/runtime.

**Cost:** at most one additional association-view inference; reuse only correctly
provenanced view features. Eight detector views are not automatically eight valid
association views.

## E06 — forward-time context consensus

**Hypothesis:** the same cell transition may receive different scores depending
on its position within a decoder window. Marginalize that context placement.

Keep B0 detections, features, checkpoints and decoder window length. For each
consecutive transition, enumerate all existing-architecture, fully observed,
forward-ordered windows of the public decoder length that contain both endpoints.
Average the resulting normalized parent probability columns equally, using one
common source/target population for that transition. Threshold and solve once.
A window supports a column only if it scores the entire common competing-parent
population; do not average different edges with different denominators. Where no
additional valid context exists, retain the exact baseline column. Preserve
baseline behavior for clips shorter than the native context.

Do not reverse a lineage, concatenate unrelated clips, look outside the input
clip, fabricate future frames, change window length or train a bidirectional model.
If the public predictor already averages all identical contexts with equal weights,
this is a no-op. A claimed five-frame decoder alone does not prove that multiple
context-dependent scores are available; trace calls and tensors before coding.

Tests: identical-window equality, coverage counts, short clips, first/last frames,
window enumeration order and probability normalization. Report edge-score variance
across contexts and final metrics by distance to temporal boundaries. Counter-risk:
edge-window predictions may be systematically worse, and equal weighting may hurt.
No subsequent central-weighting or context-length search in this study.

**Cost:** extra head/window work after reusable features; measure actual limits.

## E07 — fork-anchored smoothing

**Hypothesis:** a generic line fit can distort a real branching trajectory; protect
predicted division neighborhoods without throwing away beneficial smoothing.

Freeze the graph immediately before the public smoother. Split smoothing support
at every predicted fork. Make each fork node and its immediate predecessor and
immediate daughters hard coordinate anchors at their pre-smoothing positions.
For unanchored nodes retain the original fit/window/weights, using only support
from the same nonbranching segment. At insufficient support retain the unsmoothed
coordinate. Do not use GT divisions, re-time an event, introduce a new fork or
select a protection radius. Keep topology, node count and all original rounding.

Audit the original smoother first: if it already implements this exact protection,
record non-applicability. Tests: translating straight track, synthetic Y branch,
short daughter tracks, consecutive forks, gap-inserted nodes and unchanged edges.
Compare changes around *predicted* forks and all other points; the evaluator may
separately analyze GT matches, but inference must not access them. Counter-risk:
anchoring a false fork or a noisy detection may retain error the smoother corrected.

**Cost:** graph/coordinate replay; no new network pass if inputs are available.

## E08 — robust aggregation of existing detection views

**Hypothesis:** an extreme response from one augmented view can create a false
peak. A median across the already-used views may be more robust than the public
within-model aggregation, without adding models or augmentation parameters.

At each model's existing inverse-aligned scalar detection-view stack, replace
only its view aggregation with the voxelwise median. For an even number of views,
use the arithmetic mean of the two central sorted responses. Use exactly the
original views, response scale, inverse maps and valid-support rules. Preserve the
harmonic fusion **between** models, feature stream, threshold and NMS. Do not change
DeepCenter's separate role or median probability tensors for association.

If the actual public stack is already median-aggregated, this arm is redundant.
If only an irreversibly reduced map was cached, regenerate per-view responses.
Tests: all-identical responses, view order invariance, single extreme outlier,
even-count median, and invalid-support masks. Report node additions/removals,
low-response true-cell loss, TP survival and full edge/division scores. A lower
prediction variance alone is not a success criterion. Counter-risk: a genuinely
faint cell may be visible in only a minority of views and disappear under a median.

**Cost:** original detector forwards plus aggregation/storage overhead; stream by
frame, do not retain all clips' dense per-view volumes.

## Fixed combinations and B1 transfer

Only C01=E02+E03, C02=E04+E05 and C03=E05+E06 are registered. Execute each only if
both constituents pass the full B0 eligibility gate and are applicable. Preserve
canonical stage order. C02 computes association scores at its new consensus
coordinates; C03 averages over the fixed joint spatial-view/context product, with
complete equal-support probability columns. Do not average already-thresholded
graphs. Use individual results to report non-additivity; no blend-weight fitting.

Before any X-arm scores, lock up to two eligible non-E01 recipes selected by the
ranking in VALIDATION.md. X01/X02 apply that exact recipe with motion disabled;
they are compared to B1 as well as B0. No new numeric settings, extra components or
transfers chosen after seeing X results. No E01+no-motion transfer: the mechanisms
conflict. Missing/failing components skip a combination, not unrelated experiments.
