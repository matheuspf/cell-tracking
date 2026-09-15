# Implementation contract

These are interfaces for Codex to implement under tools/pipeline_error_training/.
They are not existing executable training commands. Reuse the code below rather
than duplicating a new metric, graph format or whole tracker.

## 1. Integration map

| Existing source | Reuse / change in the NEW namespace |
|---|---|
| strong_tracker_v3/incumbent.py, inference.py | Manifest checks; explicit source package; fresh image-derived baseline |
| segmentation_tracking_v6/controls.py, point_child.py, infer.py | Exact P0 construction and control; maintain full 38-feature evidence |
| strong_tracker_v3/features.py, association.py | Current-coordinate features and supported incoming-parent labels |
| strong_tracker_v3/event_proposals.py | Forward/reverse physical neighbors, daughter paths and anchor expansion; make iteration lazy |
| strong_tracker_v3/decode.py | Action, BoundedActionComponents, solve_actions and legal_edges; retain no-op and owner accounting |
| multidata_training_v4/models.py, proposals.py, decode.py, calibrate.py | Reproduce compact control and instrument old gates; no silent replacement |
| other_trackers/organoid.py | Corrected v3 weights, Keras torch backend, patch rescaling and calibration parity |
| center_comparison/tracking_index.py, best_predictions.py | Exact audit definitions, graph/artifact recovery and official local-window diagnosis |
| annotation_selection/metric_adapter.py | Pinned scorer, explicit persisted-ID mapping and complete-graph scoring |
| biohub_external_data/data_adapter.py | Provenance/schema checks only; primary study does not rerun synthetic pretraining |

Suggested new modules: inventory.py, baseline.py, trace.py, bank.py, labels.py,
splits.py, crops.py, models.py, organoid_adapter.py, train.py, calibration.py,
actions.py, infer.py, evaluate.py and report.py. Keep inference imports separate
from label/oracle/evaluation modules. Add CLI help and small unit fixtures before
long runs. Do not import historical scripts with side effects or work-root globals.

## 2. Label-free bank and trace schema

Every node uses an explicit `(clip, persisted_node_id, t, z, y, x)` identity;
array row indices are not persisted IDs. All spatial distances are physical um
read from actual metadata. Never feed a source/embryo filename, GT match flag,
annotation density, annotation ID, oracle label or target calibration to a model.

Bank entries include parent, unordered daughters, their candidate continuations,
competing owners, alternative donor continuations, raw/final provenance and
boundary masks. Crop centers are predictions. Preserve the no-op alternative.
Use the pinned v3 candidate defaults (6 forward, 2 reverse, 16 um parent gate,
20 um sister gate, 2 paths per daughter) as a declared starting specification,
not a search space. Its actual full union can exceed six daughters. Keep all
incumbent edges/forks even outside normal geometric proposal limits.

The operational bank and its streaming order are frozen before target labels.
Deduplicate equivalent pair permutations and alternate descriptions of a complete
edit; retain timing alternatives but prevent duplicate nearby-frame edits from
claiming the same parent/daughter resources. The source-only oracle runs on this
same bank. Any broader GT-guided feasibility bank is named separately.

Trace records must contain the bank hash, baseline graph hash, node IDs, anchor
frame, raw and calibrated scores, baseline/alternate total decision scores,
removed/added edges, donor alternatives, reason code and solver status/gap.
Official GT/window attribution is written ONLY by the diagnostic/evaluation
process to a separate table. For a missed event summarize the earliest stage
at which ALL compatible local anchors are unavailable, and also retain per-anchor
reasons. Aggregate counts must reconcile with the audit; coverage is not merely
whether a daughter was among exact-frame nearest neighbors.

Memory: stream parent groups and cache embeddings, not pairs. Existing v3's
BoundedActionComponents defaults are the initial resource limits (256 alternatives
per conflict component and 50000 buffered actions). An oversized connected
component abstains in full; a later bridging action must not resurrect part of it.
Do not secretly keep only the most promising oracle alternatives. If a component
limit blocks a source feasible repair, report it rather than doing a limit sweep.

## 3. Supervision and grouping

Create source fit, source calibration and target-evaluation manifests before
training. Unit of sampling is a biological event/trajectory group, not a crop,
daughter pair or frame. Union known overlapping clips, lineage observations,
adjacent event time anchors and duplicate crops into a single group. Use physical
acquisition metadata and frame fingerprints to audit overlaps. Record where
unique biological grouping cannot be established.

For an observed division, the target is the SET of bank hypotheses compatible
with the local parent and both daughter lineages under the pinned timing rules.
Use a masked set/listwise likelihood over compatible hypotheses rather than
forcing the exact annotated split frame or treating other compatible pairs as
negative. Test this against the actual scorer on early/on-time/late fixtures.

For a recorded single-child transition, supervise the observed identity link and
supported contradictory incoming-parent identities. Do not label all possible
unrecorded second daughters as biological nondivision. Continuation-only and
continuation-plus-unobserved-daughter hypotheses can remain compatible with partial
labels. Birth/death labels at crop boundaries, missing future frames and sparse
track truncation are unknown. Quiet chains are not certified biological negatives.

Train the metric-risk/decision-calibration target separately from biological
identity/event targets. An officially evaluable FP fork can be a metric-negative
example, with its exact source of evaluability recorded; this does not prove no
biological division occurred. Unknown/unscored forks are never negative just
because they were unmatched. This separation must be testable in labels.py.

Hard negatives are wrong supported daughter identities, cross-lineage confusers,
competing annotated incoming parents and explicitly labelled source-generated
corruptions. Mine them on SOURCE predictions only. No automatic negative mining
from arbitrary unannotated raw proposals or every low-score whole-field cell.

Sample all supported positive groups across epochs, and log their visitation.
Use group-normalized losses so one event with 100 candidate rows is not 100
independent positives. If balanced/hard-mined batches are used, record exact
sampling probabilities where available. Do not reuse v3's historical parent-group
sampling fraction as a row propensity. Fit final decision calibration on the
unbalanced source candidate distribution, not on a balanced training minibatch.

## 4. Image/temporal architecture and training defaults

New temporal model: shared small anisotropic 3D encoder, channels 16/32/64,
GroupNorm, 128-dimensional node token; ordered temporal GRU or two-layer attention
with explicit relative time and validity masks. Select ONE temporal operator by
implementation simplicity before outcomes and lock it. Symmetric daughter
features use sum/absolute difference, with separate parent, each competing owner
and continuation evidence. Image-only and geometry/native-evidence inputs have
separate missingness masks. Do not treat a absent native edge probability as zero
confidence without its missingness indicator.

Default event context: frames t-2 through t+4, seven frames. For each anchor/node
use prediction-derived positions only; do not recenter a crop on a GT daughter.
Use a two-scale native sampling design: a 16x64x64 ZYX high-resolution patch
(approximately 26 um per side at the documented spacing), plus a two-times-wider
context downsampled to the same tensor shape. Actual metadata overrides these
physical approximations. Parent, daughter and owner tokens are sampled at their
own actual anchors; uncertainty/missing history is explicit, not filled with an
annotated trajectory. Shared ROI extraction/encoding and bounded frame caches
avoid repeating full image reads for every pair. Profile this architecture before
freezing production microbatches; it is a proposed design, not a measured fit.

At clip/image boundaries supply validity masks; never duplicate a nonexistent
future frame and call it observation. Normalize with source-defined robust rules
applied to each image/crop; do not encode global whole-acquisition IQR statistics
containing the target distribution as if they were source-only fitted features.

Continuation pretraining combines supported source same-cell tracklet ranking
and contrastive views of the same source crop. Different annotated cells supply
safe identity negatives. Treat sisters as related lineage, but distinct cells
after the split. Same-crop augmentations teach invariance, not proof of temporal
identity. Avoid a batch construction where every same-clone descendant is a
same-cell positive.

Training augmentations are shared across the compact/temporal controls: XY flips
and right-angle rotations, coherent brightness/contrast/gamma, source-bounded
shot/read noise and blur, mild physical scale variation, center jitter and
history/future masking. Image/point transforms must agree. Do not swap anisotropic
Z with X/Y or time-reverse division sequences into plausible merges. Set noise,
blur and scale ranges once from source images (or fixed conservative defaults),
write them to the lock and do not tune them on target performance. Synthetic
candidate duplication/dropout retains explicit identity/provenance; do not claim
such stress tests reproduce new embryos.

Default optimizer for new networks: AdamW, learning rate 0.0003, weight decay
0.0001, cosine decay, 500-update warm-up, gradient norm clipping at 1, effective
batch 32 parent/trajectory groups. New Organoid competition head uses the same
recipe; adapted pretrained blocks use learning rate 0.00003. Start head-only for
2000 of the registered updates, then unfreeze the final image block. Do not change
normalization/preprocessing from the parity-checked adapter without a named
source-only implementation change.

Loss defaults: event compatible-set loss 1; supported incoming-identity loss 1;
contrastive term 0.1; same-input augmentation consistency 0.1. Normalize each over
supported groups, skipping unavailable terms without fabricating labels.
During continuation pretraining use identity 1 and contrastive 0.1. Registered
8000+8000 steps are ceilings subject to the common source-only resource lock in
PLAN.md. Save intermediate checkpoints for debugging; source-held-out selection
is by the locked supported decision loss, not target score. Never report source
resubstitution performance as source validation.

Do NOT distill the incumbent's no-fork decision as truth in disputed event groups.
A baseline-agreement regularizer may preserve easy supported continuations, but
must be disabled/masked on disputed hypotheses and be identical across controls.
Do not silently add a new loss/regularizer after seeing target outcomes.

## 5. Calibration, alternatives and constrained decoding

Train/fit on source groups only. The deployed decision compares calibrated
complete-hypothesis evidence against keep-P0, accounting for all removed/added
continuations, forks, births and donor outcomes. Use a shared temperature plus
a small regularized decision head fit on source out-of-group predictions, with
no embryo-specific threshold, candidate-count multiplier or global division
penalty sweep. Define the no-op decision at zero residual and choose it on ties.
If clean inner groups cannot be certified, use the fallback in VALIDATION.md
and label source calibration as reused/exploratory.

Preserve the incumbent graph objective; do not add a parent-only optical logit to
every pair and interpret that as a learned pair ranking. Include both daughter
identities and continuation-plus-birth/other-parent alternatives. Log when an
apparently good fork loses because replacing a supported donor continuation is
more costly. Birth and termination costs have consistent units with link/event
scores and are source-fitted or fixed inherited values, never target-search knobs.

Use complete atomic actions with source and target ownership resources. Maximum
indegree 1, outdegree 2, consecutive-frame edges, two distinct daughter branches,
no merges and finite objective values are hard constraints. Baseline is feasible;
nonoptimal/time-limited/oversized components abstain, unless an exact legal solution
with verified objective improvement has a separately predeclared acceptance test.
Initial implementation accepts only optimal bounded-component solutions.

A fork's protection covers the parent predecessor and immediate child/grandchild
support used by the official local window. Editing a downstream support edge can
change division scoring while keeping the two immediate fork edges fixed; v3
already demonstrated this. A10 therefore freezes this full support neighborhood.
D replacement may change it, but must score and own the entire modified local
support. Do not assert that protecting immediate edges guarantees division parity.

Keep the inherited 2% association edge-edit cap for A10. For new division/observation
modules initially use the same 2% changed-edge budget as a fixed preservation
safeguard, with full ledger and abstention counts; do not optimize that cap. O10
also caps changed observation selections at 2% per clip, independent of sparse
GT counts. Counting removed+added edges must be consistent across variants.

## 6. Observation selector specifics

Bank: final P0 points plus original raw candidates, without GT-based filtering.
Use explicit raw-to-final provenance when proven, not nearest-center identity.
Enumerate local tracklet/observation alternatives with keep as a permanent option.
Distinct nearby cells can coexist. Exact duplicate records may be canonicalized
by proven provenance, but a 7 um neighborhood is NOT an exclusion group.

For O10_swap, a complete action replaces one selected observation with one raw
candidate and rewires all incident edges legally. Preserve the count, not the old
coordinate. For O10_restore, allow additional raw trajectory observations with
learned image and incoming/outgoing consistency. It cannot force a cell to have
both a past and a future at boundaries or during legitimate birth/division.
Missing evidence reduces confidence through masks rather than a hard biological rule.

Train ranking with source-supported identity, continuous localization evidence
for known cells, pairwise known-different-cell constraints and explicit duplicate
corruption. Multiple raw proposals near one known cell can remain compatible;
never infer that the unchosen neighbor must be false. Where sparse supervision
cannot identify which real proposal is preferable, mask that loss term.

Rebuild native embeddings and edge logits at every changed point, all affected
candidate/owner relationships and local event evidence. Re-evaluate whole-clip
one-to-one matches from scratch. Separate one-for-one swaps from restore additions
in ledgers and scores, including close annotated cell-pair recall and node counts.

## 7. Minimum tests before production

Test: exact zero-head P0 identity; persisted IDs unrelated to row indices;
physical/voxel axes and crop bounds; daughter-order and padding invariance;
missing future evidence; native feature sampling at replaced coordinates;
unknown branches not negative; single-child incomplete lineage; group sampling;
early/on-time/late divisions with scorer parity; cross-component false fork;
parent owner displacement with a high-confidence competing owner; full fork
window protection; no merges/shared daughters; atomic conflicts; late bridging
of oversized components; solver tie and timeout abstention; count-preserving
swaps; distinct close real cells retained; boundary births; stable rounding;
empty graph/frame; exact serialization; and denial of GT/old cache/network reads
in fresh inference. Test actual solver calls, not only hand-coded expected actions.

Add replay tests for the audit's concrete 44b6_12dfb391/frame-66 event in the
SOURCE direction only. It is a regression fixture, not independent validation.
Do not special-case that filename, event ID or any known target errors in inference.
