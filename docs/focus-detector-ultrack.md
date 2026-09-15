# FOCUS masks, detector centers, and ultrack joint selection

Reviewed 12 September 2026 against ultrack
`5c94d845eb0a7b78c8dc24492ef00f218a467995` and freshly fetched branch
[`handover/ultrack-integration-v7` at d8af4b4](https://github.com/matheuspf/cell-tracking/blob/d8af4b48b9948346e9d2bff78eba0dd13b58f850/handover/ultrack-integration-v7/PLAN.md).
The branch contains a research plan, registry, preflight, contracts, and smoke
scaffolding. It contains no measured v7 competition experiment.

**Recommendation: use FOCUS for object extent, the existing detector for
localization, the complete native association ensemble for identity evidence,
and ultrack for joint mask/lineage selection.** Do not force a mask's geometric
centroid to serve every purpose. A cell hypothesis should carry both its actual
mask centroid and its chosen tracking/scoring point.

Preserving coordinates means that a selected detector-owned hypothesis uses
that detector's point. The full ultrack arm can still accept or reject object
hypotheses. The separate H1 control preserves every P0 observation.

Ultrack accepts foreground and contour maps, generates candidate regions, and
selects compatible regions and temporal links. Instance masks can supply those
maps; multiple label maps can provide alternate segmentations. This supports a
FOCUS-plus-detector proposal bank without replacing ultrack's optimizer.
[Official method](https://www.nature.com/articles/s41592-025-02778-0),
[label conversion source](/home/mpf/code/kaggle/ultrack/ultrack/utils/edge.py:18).

**What the new checks establish.**

On the same 12 full spatial frames used in the center study, a new mask-ownership
audit used predictions and masks only, without GT matching:

- 1,862 of 1,893 detector points (98.4%) fall inside a FOCUS mask.
- 1,610 points (85.1%) are the only detector point in their mask.
- 252 points (13.3%) share 113 FOCUS masks with other detections.
- 31 points (1.6%) lie outside all FOCUS masks. Sixteen of these are within
  2 µm of some foreground; proximity alone is not a valid object assignment.
- 706 of 2,429 FOCUS masks contain no detector point. They can represent real
  missed cells, fragments, or other errors; they are not automatically false
  positives.

Both-embryo breakdowns and full check receipts are in
[design-checks.json](../results/focus-detector-ultrack/design-checks.json).
These observations make preserving the detector's points practical for most
objects, while identifying splits and missed-mask rescue as explicit cases.

I also ran actual ultrack/CBC on a four-frame synthetic movie with eight
selected nodes and six continuation edges. Its masks and bbox hashes stayed
unchanged while linking and exported coordinates used deliberately displaced
in-mask tracking points. Selected mask occupancy remained exact.

A negative control demonstrated an important implementation detail: updating
only `NodeDB.z/y/x` left the linker's serialized `Node.centroid` stale. It still
created a link that should fail the new 1 µm distance gate. Updating both
representations and rebuilding links removed that link. This is a tested
compatibility result, not learned linking or a biological accuracy result.
[Link search](/home/mpf/code/kaggle/ultrack/ultrack/core/linking/processing.py:187),
[observation export](/home/mpf/code/kaggle/ultrack/ultrack/core/export/utils.py:19).

**Construct a small, explicit set of competing masks.**

Run the current detector on its existing image context and run FOCUS nuclei on
the same native volume. Keep the native ZYX lattice and physical scale
`(1.625,0.40625,0.40625)` µm. Restore resized labels by nearest-neighbor
resampling and preserve raw detector coordinates. Start with two label maps:
the original FOCUS segmentation and one detector-conditioned alternative.

Handle ownership locally:

1. **One detector in one mask:** keep the mask and use that detector's original
   point. The mask supplies occupancy, shape, intensity, and overlap; the point
   supplies localization and learned association geometry.
2. **Several detectors in one mask:** create image-supported watershed splits
   inside the FOCUS region, using the detector points as seeds. Retain the
   unsplit region as a competing hypothesis, because two detections may also be
   duplicates. A merged region is still a one-cell hypothesis, with one
   deterministic representative point and evidence describing the competing
   peaks; it must not emit several cells with identical occupancy. Distinct
   seed regions represent the alternative with several cells.
3. **Detector without a mask:** try a bounded, unique physical association to
   nearby foreground; ambiguity remains explicit. Otherwise generate an
   image-supported native-resolution region around the detector, using a
   seeded watershed and local foreground evidence. Do not discard a detection
   solely because FOCUS missed it. A point impulse or fixed ball is not a
   substitute for measured object support. If no defensible mask exists,
   report the unresolved coverage loss in the full ultrack arm and retain the
   point in the P0 control/hybrid.
4. **Mask without a detector:** keep an eligible mask-only hypothesis. Use a
   deterministic in-mask point near its centroid or an independently defined
   image/model peak, and compute new native evidence there. Calibrate how
   these candidates enter the solution; do not force their selection or label
   all of them negative. Adjacent fragments supported by one detector may
   also justify an image-supported union hypothesis.

For a detector just outside a confidently associated mask, keep a distinction
between the detector point and mask geometry. An explicitly bounded association
may retain that point outside the mask; record its distance and provenance.
Do not silently project it to the nearest mask voxel, because that changes the
geometry we intended to preserve. The initial implementation can use strict
containment and route all other cases through rescue/ambiguity handling.

These are proposal rules. The optimizer, using multiple frames, chooses between
the original and corrected regions. Do not commit to every split before tracking.
Bound the bank to the original map plus one registered corrected map initially;
avoid an uncontrolled collection of thresholds, radii, and duplicate masks.

**Keep mask geometry and tracking geometry consistent throughout ultrack.**

Each candidate record should contain: clip and time, integral ultrack node ID,
detector/anchor ID if any, bbox and occupancy hash, true geometric centroid,
chosen native tracking point, center provenance, mask provenance, and exclusion
groups. Multiple mask hypotheses may refer to the same anchor.

After candidate generation, assign the tracking points and update both SQL
coordinate columns and serialized `Node.centroid` in one transaction. Preserve
the original centroid in the sidecar record; leave masks and boxes untouched.
Invalidate and rebuild candidate links after any geometry change. Use the
chosen point for native feature queries, physical candidate distances, and
final node CSV coordinates. Use the mask for overlap and morphology. Merely
snapping coordinates at export would make scoring and optimization disagree.

Enforce two kinds of exclusion: overlapping masks cannot be selected together,
and same-frame alternatives claiming the same detector anchor cannot both be
selected even when their masks are disjoint. This prevents an oversegmented
cell from becoming several observations. Split children with different anchors
remain jointly selectable; the unsplit parent conflicts with each child.

**Preserve masks through the hierarchy, and check every conflict.**

`labels_to_contours` unions foreground and averages boundaries; it does not
guarantee exact recovery of every input region. In actual conversion and DB
readback on one FOCUS frame per embryo, **87/204 original masks survived
exactly**, while **182/204 had a candidate with IoU at least 0.95**. Those are
mask-to-mask preservation diagnostics, not IoU against biological GT. Parameters
were min area 20, max area 100,000 voxels, and min frontier 0.05. Small-mask
pruning and hierarchy reconstruction both belong in the audit.

Retain the registered admissible FOCUS masks and corrected alternatives as
explicit candidate records. Measure exact survival separately from best IoU;
insert necessary missing masks with stable integral IDs and real exclusions.
Keep deliberately rejected tiny/implausible masks in the rejection ledger
rather than reinserting every original label indiscriminately.

The v7 plan suggests `add_new_node(..., include_overlaps=True)`. The pinned
implementation checks only the **10 nearest nodes** for overlap. A new
synthetic probe with a mask overlapping 12 existing candidates inserted only
10 exclusion constraints. Therefore this flag alone is insufficient for a
complete conflict graph. Supplement it with exhaustive relevant bbox/occupancy
intersections and anchor conflicts before solving. Also rebuild links after
insertion: this API creates provisional links using its own initial geometry.
[Insertion/overlap code](/home/mpf/code/kaggle/ultrack/ultrack/core/interactive.py:123),
[solver pair exclusions](/home/mpf/code/kaggle/ultrack/ultrack/core/solve/solver/mip_solver.py:233).

**Use the learned linker inside ultrack.**

Mask IoU is useful evidence but should not replace our native association
models. Build a physical-radius candidate bank that can represent continuations
and both daughters even when their masks have no overlap. The stock linker
first searches a limited neighborhood and then keeps candidates with high IoU;
reweighting only those survivors can never recover a discarded true edge.
Use a bounded union of physical neighbors, native-link proposals, and useful
mask neighbors. Freeze this same bank for matched cost ablations.

Evaluate the full primary/secondary and augmentation recipe at the chosen
points for all hypotheses that can participate in selection. Use a **unique
anchor bank** for the native transformer: several competing masks for one
detector should not appear as repeated copies of that cell in its attention
context. Evaluate each distinct anchor pair once, then reuse that exact score
for its corresponding region alternatives and add their different mask
features. This is an intentional adapter design, not an upstream ultrack API.

The transformer uses cross-attention over other nodes, so equal coordinates
alone do not guarantee equal scores when its surrounding candidate set changes.
Reuse image encodings where exact parity permits; recompute the native context
when new anchor points appear. Never borrow a nearby old detection's score.
Teacher/support features for new points must be recomputed or explicitly
missing in a model trained for that interface; existing P0 coefficients are
not automatically valid on a changed feature distribution.

Start with a small source-calibrated edge score combining native log-odds,
physical displacement, mask IoU, volume ratio, and simple appearance consistency.
Use `link_function="identity"` and explicit bias; the default fourth power
reverses the meaning of negative logits. Calibrate appearance, disappearance,
and division penalties on the same scale. Keep sparse unmatched objects and
unrecorded daughters unknown. Group alternative hypotheses during calibration
so one labeled cell does not become many independent examples.

Optional node rewards need an explicit calibrated interface. The current
`SQLTracking` requires either all node probabilities or none; mixing unset
negative sentinels with valid values raises an error. It also uses the same
configured transform for node and edge rewards. Do not place arbitrary signed
node logits into that field or invent FOCUS confidence from integer labels.

Run real CBC joint selection, retaining its birth/death and binary-division
constraints. A first experiment can use a calibrated scalar division penalty.
A daughter-pair morphology reward requires extra pair variables and constraints;
it is a separate extension, not a built-in consequence of supplying masks.
Export actual selected `id`/`parent_id`, preserving detector-based points, and
score the resulting node/edge graph afresh with the official adapter.

**How this changes the v7 plan.**

Keep its source pin, C0/P0 references, physical units, exact candidate IDs,
complete native evidence, identity transform, CBC, sparse-label treatment,
storage discipline, complete-graph evaluation, and cold-inference requirement.

Revise four decisions for the user's FOCUS-centered experiment:

- Promote learned-mask U2 from optional to the primary experiment. FOCUS now
  works; the old missing-runtime rationale no longer applies. Keep the classical
  U0/U1 recipes as bounded diagnostics initially rather than the central effort.
- Replace centroid-first export with separate mask centroid and detector-based
  scoring/export geometry. This changes the v7 geometry fingerprint and its
  strict in-mask export contract deliberately, not just the final CSV.
- Include detector-conditioned split/merge/rescue hypotheses and complete
  anchor/occupancy conflicts. Do not trust input-mask survival or the limited
  overlap-insertion helper without readback checks.
- Retain H1 as a conservative association-only control. Its frozen forks and
  2% edit cap cannot test the requested full joint segmentation/lineage benefit.

The branch registry is unchanged by this review. Its six-arm ceiling, optional
U2 declaration, mandatory all-199 classical arms, and centroid export rule would
need to be amended together before calling a FOCUS-first run an execution of
that registered plan. The plan's 0.95 target is an objective, not a predicted
result or authorization to claim a gain.

**Concrete end-to-end experiment order.**

1. Preserve C0 and P0 as controls. On a fixed small panel of consecutive real
   frames, run FOCUS masks through stock ultrack to establish a complete
   image-to-mask-to-lineage path.
2. On the same original FOCUS region bank, use detector points and complete
   native association evidence. Within the pilot, separate the coordinate
   change from the edge-weight change so regressions can be attributed.
3. Add the second detector-conditioned mask map and exact missing hypotheses,
   preserving the same native scoring recipe. This is the principal candidate
   for the final experiment: FOCUS extent, detector localization, learned links,
   and joint selection over original/split/merged/rescued regions.
4. Freeze the source-selected recipe before target evaluation. Start with eight
   consecutive frames, then two full 100-frame clips to measure runtime, memory,
   candidate/edge coverage, solver gap, and export validity. Complete the final
   comparison on all 199 clips only after the representation and resource checks
   pass. Use a source-agnostic deployment rule; reused embryos and upstream
   exposure keep this an exploratory evaluation.
5. Deliver a cold CLI that accepts a new clip name and generates detector
   outputs, FOCUS masks, hypotheses, native scores, ultrack solution, and CSV
   without annotations or old prediction caches. Promote only on complete
   official graph scores versus P0, with both embryos reported; center recall
   and mask appearance are diagnostics.

Keep FOCUS, native inference, and ultrack in their working process environments,
with a small versioned array/manifest interface. Load the FOCUS model once and
process frames sequentially or in bounded batches; verify output parity after
that refactor. The measured 15.06 seconds per frame included about 11.99 seconds
inside its logged inference/stitching loop, so removing repeated model loading
alone cannot meet the notebook budget. Batch size, overlap, and inference scale
need measured quality/throughput tradeoffs. Simply downsampling the raw image
while preserving FOCUS's radius rescaling may return to the same internal size.
Keep one clip's maps/DB active and write durable small receipts.

Implementation belongs in an isolated `tools/ultrack_integration_v7/` runner
that reuses maintained v6 region, native, export, and evaluation code. The new
[interface probe](../work/focus-detector-ultrack/interface_probe.py) demonstrates
the coordinate and mask contracts only. This review has not implemented or
scored the complete FOCUS-plus-native-link competition experiment, changed
production policies, or executed the branch's multi-day study.
