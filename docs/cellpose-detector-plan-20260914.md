**Build the next detector around the frozen Cellpose `cpdino-vitb` baseline, first
adapting its center locations, then addressing missing proposals.** Its strong
7 µm result hides a much larger localization gap at 2–3 µm. The incumbent
ensemble is a training-exposed reference; its 99.50% recall is not a clean
generalization ceiling. The [checkpoint-linked audit](incumbent-provenance-20260914.md)
verifies overlap between the secondary model's published training records and
every assessment clip. The [additional-model and native-embedding pilot](cellpose-embedding-results-20260914.md)
tests DINO-L, SAMv2 and proposal unions with explicit detection and association controls.

This is a reviewed development design, informed by the existing benchmark.
The additional measurements below reuse frozen predictions. No adaptation,
training, temporal assignment, merge or submission was performed in this review.

**Execution update — 14 September 2026.** The first stage is now complete:
four source-isolated residual-head fits, locked evaluation, raw-volume replay,
and a separate source-only constant-offset control. Both approaches improve
pooled integer localization but fail our conservative detector-stage guards;
this does not establish a worse final tracking score. Keep the encoder frozen
for the next comparison and retain both unchanged Cellpose and the constant
control. Fixed proposal counts isolated localization in this first experiment,
not a permanent constraint: the next comparison should vary the proposal bank
and use the official full-clip tracking score as its primary endpoint. The
[completed experiment report](cellpose-refinement-results-20260914.md) records
the results, remaining geometric headroom and next comparison. The review below
is the historical basis for that experiment.

**Fetched plans and changes needed**

`git fetch origin` retrieved the new detector handovers. The latest open detector
request is [PR #11, native-resolution v8](https://github.com/matheuspf/cell-tracking/pull/11),
at `e654c50c7543b3486556e2a1cd472f669f07a625`. Its newer
[v9 4090 branch](https://github.com/matheuspf/cell-tracking/tree/01bbcd5bfe232c9364207246afb6a3b1d652860a/handover/native-resolution-detector-v9-4090)
is at `01bbcd5bfe232c9364207246afb6a3b1d652860a`; no pull request for that branch
was listed at inspection. Both are unexecuted plans with reference helpers,
not implemented or validated training systems.

V9 improves on v8: it keeps embryos separate, leaves unannotated regions unknown,
uses physical coordinates, locks both training directions before target scoring,
and measures memory before choosing training schedules. It correctly removes a
default offset-to-nearest-native-voxel target: native integer GEFF coordinates
would make every such target zero. Keep these safeguards.

Replace the large scratch architecture campaign with a focused Cellpose study.
V9 mandates 16 fits and permits 32; it also includes tracking integration. That
scope does not answer the current question efficiently. Four other fixes matter:

- **Enforce all exposure fields.** The v9 ancestry helper accepts an explicit
  declaration of target-embryo calibration because it checks only `fit_embryos`.
  Its registry helper also accepts changing the fixed inner-selection policy
  and removing the low-pass control. These are reproducible validator gaps,
  not evidence that a training run actually leaked.
- **Remove a resolution-control confound.** With sigma 0.8 µm, an ideal Gaussian
  sampled on the stride-4 XY lattice can peak at 0.356 for an integer GT center
  at phase `(2,2)`, below the shared 0.5 threshold. The native target peaks at 1.
  A coarse control needs a lattice-aware target/decoder; native-grid low-pass
  inputs remain the cleaner test of image information.
- **Specify sparse Cellpose supervision.** Stock flow MSE and occupancy BCE
  include every pixel. GEFF points converted into masks with zeros elsewhere
  would teach unannotated cells as background. Stock training also operates on
  2D planes; `run_3D` is an inference wrapper.
- **Budget inference explicitly.** At the measured 9.485 processing seconds per
  native volume, one 19,900-volume Cellpose pass projects to **52.4 processing
  hours**, before adaptation or replication. This is an extrapolation, not a
  measured full-data run or GPU-active total. Do not inherit the v9 eight-hour
  inference reserve. Cache immutable proposals once and measure any new recipe.

All 38 v8 and 26 v9 supplied CPU tests passed. Additional probes reproduced the
three validator gaps and Gaussian lattice effect. See the
[fetch receipt](../results/detector-development-20260914/review-receipt.json),
[test receipt](../results/detector-development-20260914/test_receipt.json) and
[gap reproduction](../results/detector-development-20260914/plan_gap_receipt.json).

**Checkpoint provenance: usable, not certified clean**

The exact benchmarked weights have SHA256
`3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f`.
Hugging Face records their upload on **17 May 2026**, before the competition
began on **29 June**. That chronology supports exclusion of the later public
competition release; it cannot exclude prerelease access or earlier versions of
the underlying acquisitions. No actual overlap evidence was identified.
[Pinned weights](https://huggingface.co/mouseland/cellpose-sam/blob/7c61431b5fbb078f3296754bd15d9f51b320f837/cpdino-vitb),
[release history](https://huggingface.co/mouseland/cellpose-sam/commits/main),
[competition timeline](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/timeline).

The exact model card contains a license header and no image-level training
manifest. The inspected code names DINOv3 ViT-B LVD-1689M as its initializer;
Meta describes that corpus as images curated from public Instagram posts.
Neither record excludes the benchmark acquisitions at image level. The older
Cellpose-SAM paper's training mixture must not be attributed to this exact
cpDINO checkpoint without evidence. Resetting its output head, or fine-tuning
only on one embryo, does not remove inherited encoder exposure.
[Pinned model card](https://huggingface.co/mouseland/cellpose-sam/blob/7c61431b5fbb078f3296754bd15d9f51b320f837/README.md),
[CPDINO initializer](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/vit.py),
[DINOv3 source population](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/MODEL_CARD.md).

The organizer's assurance that public Zebrahub does not overlap competition
test data is useful but narrower than an exclusion of the visible training
embryos or every Biohub source. Consequently, label cpDINO adaptation results
**source-isolated adaptation with unresolved inherited exposure**.
[Organizer reply](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330).

Maintain a separate random-initialized, source-only detector control for the
strongest inherited-exposure check. It must generate its own proposals; a
scratch refiner consuming cpDINO proposals would retain cpDINO ancestry.
Documented satellite-only DINOv3 SAT-493M with a fresh head is a possible later
initialization control, but its ViT-L architecture is not a drop-in ViT-B
replacement and has not been profiled. Raw LVD DINOv3 retains web-corpus
uncertainty. See the [checkpoint evidence receipt](../results/detector-development-20260914/cellpose-provenance.json).

**Measured room for improvement**

These are integer-export results on the same 400 previously examined frames,
40 clips and 2,383 sparse annotated nodes. The official one-to-one matcher was
rerun at each physical radius, with gate and bijection assertions. The previous
5/6/7 µm counts were reproduced exactly for all three analyzed methods.

Cellpose recalls are **21.36% at 1 µm, 63.58% at 2 µm, 71.76% at 3 µm,
87.54% at 4 µm, 90.77% at 5 µm, 94.04% at 6 µm and 95.89% at 7 µm**.
The difference between its 3 and 7 µm recalls is **24.13 percentage points**.
Its mean matched error is 2.07 µm; assigning every 7 µm miss a fixed 7 µm
error gives a censored mean of 2.27 µm. OrganoidTracker reaches a similar
95.80% at 7 µm but only 40.91% at 3 µm. The uncertain incumbent reaches
85.82% at 3 µm. Cellpose is the more attractive of those two external
checkpoints for accurate-center adaptation, not merely gate recovery.

Results differ by embryo: Cellpose reaches **81.41% / 98.16% at 3 / 7 µm on
44b6**, versus **68.53% / 95.13% on 6bba**. There are 597 and 1,786 annotated
nodes respectively. The pooled result is weighted toward 6bba. Neither these
frames nor the embryos are a fresh biological test.

Integer export matters: Cellpose's float-center recall at 3 µm is **77.68%**,
versus 71.76% after rounding, while 7 µm changes by only one match. Native Z
spacing is 1.625 µm: a two-plane Z difference alone is 3.25 µm. Report float
and integer curves together and continuous censored errors so a single
radius's lattice effect does not drive selection. Integer GT reference points
do not establish subvoxel biological accuracy; float scoring is diagnostic.

There are **98 misses at 7 µm**, so perfect recovery has at most **4.11 points
of additional annotated-node recall on this panel**. Nearest-candidate
distances for those misses are: 23 at 7–8 µm, 18 at 8–10 µm, 32 at 10–14 µm,
and 25 beyond 14 µm. None lacks a match solely from a one-to-one conflict at
the original 7 µm gate.

For a refiner whose final emitted centers move at most delta micrometers from
the baseline, maximum-cardinality matching within `7 + delta` supplies an
optimistic geometric bound. At delta 1/2/3 µm, at most **23/33/40 additional
nodes** become recoverable. Thus a 3 µm refiner could reach at most
**2,325 / 2,383 = 97.57%** under this relaxation; at least **58 misses** would
remain. This assumes ideal movements and one distinct candidate per GT,
ignores integer feasibility and the official distance-weight preference, and
does not prove that a nearby proposal belongs to the missed biological cell.
It is not a forecast of learned performance. A new query source is needed to
exceed that bound without allowing larger movements.

Crowding supports measuring tighter errors. **641 / 2,383 annotated cells
(26.9%)** have two or more Cellpose candidates within 7 µm. Among **183
annotated pairs separated by at most 14 µm**, Cellpose makes both distinct
centers available for **176 pairs at 7 µm but 136 at 3 µm**. Only 12 annotated
pairs are within 7 µm, too few for a strong close-pair conclusion. Sparse GT
underestimates real crowding; multiple nearby candidates are not automatically
duplicates or known false positives.

Ranking also matters: capping Cellpose at its highest-scored **400 candidates
per frame** lowers 7 µm recall from 95.89% to **91.19%**; at 200 it is 76.54%.
These are absolute budgets, independent of the uncertain incumbent. Caps do
not force frames with fewer proposals to emit extra points, and actual emitted
counts are retained. The original cell-occupancy-derived score is not a
calibrated center-quality score.

For downstream tracking, Cellpose currently supplies both matched endpoints
for **1,099 / 1,159 sampled GT edges (94.82%)**. The 60 unavailable edges
comprise 25 with one endpoint missing and 35 with both missing: at most **5.18
points of endpoint availability** remain on this panel. On covered GT edges,
the 90th-percentile displacement error is 3.45 µm. This uses GT correspondence
only for diagnosis and does not infer temporal assignments.

Those node and endpoint margins are **not competition-score gains or a bound
on total score improvement**. Better centers can also improve assignments on
already-covered edges. Official scoring additionally depends on predicted
edges, known false-positive edges, node counts and division topology. To
measure score headroom later, hold one linker fixed and separately test oracle
localization, oracle missing-node insertion, and their combination, rebuilding
all node/edge features at changed centers. Those GT-assisted experiments are
diagnostics, never inference inputs. Assignment remains a later stage.
[Official metric](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).

All measured values and per-embryo/count/crowding diagnostics are in
[localization-headroom.json](../results/detector-development-20260914/localization-headroom.json).
The repeatable analysis is [analyze_headroom.py](../tools/detector_screen/analyze_headroom.py).

**Development sequence and training data**

1. **Freeze the experiment boundary.** Preserve the exact checkpoint, eager
   batch-8 inference recipe, normalization, native geometry and original
   scored centers. Record separate fields for inherited exposure, fit labels,
   unlabeled images, calibration, teacher construction and cache ancestry.
   Derive folds from embryo identity: train on 44b6/evaluate 6bba, and reverse.
   Source-side inputs are raw competition volumes and their sparse GEFF
   coordinates; target images enter inference only. Existing native/ensemble
   checkpoints and both-embryo teacher outputs are excluded from adaptation.

2. **Train one small localization intervention first.** Keep CPDINO frozen and
   in evaluation mode. Extract its normalized 768-dimensional patch features
   immediately before `CPDINO.out`; sample query features from aligned
   orthogonal views and optionally a small native image neighborhood. Train a
   zero-initialized residual MLP to predict physical ZYX displacement using
   Smooth-L1 loss. Its first experiment has a 3 µm movement limit and leaves
   candidate counts and ranking fixed, isolating location quality.

   Supervise only supported, unambiguous source-side proposal/GT matches plus
   jittered queries around known points. Record matching distance and
   ambiguity rules before fitting; sparse GT correspondence remains weak
   supervision and should be inspected on source crops. Unmatched real
   proposals receive no negative classification target. Use
   `delta_um = (gt_zyx - query_zyx) * spacing_um`; offsets from fractional or
   displaced queries have meaningful nonzero targets. Enforce the reported
   movement bound on final emitted integer centers too, or report a separate
   continuous-space bound including rounding. Keep the unchanged proposals
   as the paired control; do not concatenate both sets and call that equal
   candidate cost.

   Start with two directions and seeds 20260914/314159: **four small head fits**.
   Choose one training horizon and optimizer recipe using source-only resource
   and numerical checks, then freeze both directions before target scoring.
   The first implementation now fixes these hyperparameters in
   [cellpose-refine-v1.json](../configs/cellpose-refine-v1.json); its resource and
   numerical preflight passed before retained fitting. See the
   [implementation and reproduction commands](../tools/cellpose_refine/README.md).
   The benchmark-guided design is an exploratory iteration, not a blind test.

3. **Adapt the encoder only after the location head is useful.** In a separately
   locked iteration, unfreeze the final one or two transformer blocks with a
   smaller learning rate and compare with the frozen-feature arm. Preserve a
   separate frozen teacher or immutable original proposal cache. Freezing the
   old output head alone does not preserve proposals when its encoder changes.
   Measure actual forward/backward/optimizer memory on the 4090; inference's
   0.676 GB peak allocation is not a training estimate.

4. **Add a center/query head for the remaining misses and ranking.** Use aligned
   planar features or a small native 3D fusion head to predict center heatmaps
   and locally supervised physical displacements. Train positives from source
   GEFF points. Obtain negative center support from verified exhaustive
   synthetic centers or completely reviewed source crops. The prepared
   external inventory contains points, not verified dense instance masks;
   audit completeness and geometry before using synthetic background.
   Keep real unknown pixels ignored and teacher masks explicitly labeled as
   pseudo-labels. A positive-only dense heatmap admits an all-positive solution;
   teacher absence is not a verified negative. Dense instance masks are
   unnecessary for a center head, but stock Cellpose segmentation fine-tuning
   requires complete supervised masks or a custom loss on genuinely known
   regions. Register proposal fusion and ranking before viewing target scores.

5. **Establish a stronger-provenance control and a biological test.** A native
   random-initialized detector trained with the same source isolation and
   justified sparse/synthetic supervision is the independent proposal control.
   Do not initialize it from previous local runs trained on both embryos.
   A later satellite-pretrained control is optional and separately budgeted.
   Ultimately evaluate the locked choice on an untouched embryo/acquisition
   with provenance checked against all training sources. The two already-used
   embryos support source-isolated exploratory evidence; more seeds, shuffled
   clips or unseen frames from them do not create independent subjects.

The Cellpose implementation has three traps to cover in the training preflight:
its returned "style" vector is random noise; its stride-8 modification makes
the default DINO patch-size-16 reshape inappropriate; and its patch projection
uses detached `.data` parameters. Use the actual feature tensor entering the
readout. If the stem is ever unfrozen, fix gradient routing in a separate
wrapper and verify forward parity. Stream query features or bounded crops;
full triplane token caching is approximately 2.72 GB per volume at the
benchmarked padding and BF16, before overhead.
[Pinned model code](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/vit.py#L163-L229),
[training loss](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/train.py#L33-L53),
[3D training documentation](https://cellpose.readthedocs.io/en/latest/do3d.html#training-for-3d-segmentation).

**Evaluation and promotion contract**

Keep 7 µm recall as the competition-facing guard and evaluate localization at
1/2/3/4/5/6/7 µm. For a location intervention, require improvement in recall at
3 µm and in continuous censored error, without a greater than 0.2 percentage
point 7 µm recall drop on either embryo, under the same output budgets. This
is a proposed engineering tolerance, not a confidence interval. Show all
per-embryo and per-seed deltas, including any traded-away matches.

Use fixed operational decoding plus absolute K=100/200/400 curves and actual
counts; show the 3 µm mean over those budgets alongside the 7 µm guard at each
budget. Include close-pair recovery, matched-GT intersections, signed Z/Y/X
error, integer/float differences, faint/deep/border and division-neighborhood
strata. Derive target strata only inside evaluation. Never label every
unmatched prediction false or calculate dense segmentation accuracy from
sparse GEFF. The measured development panel stays fixed; a full-volume,
full-clip final evaluation is separately costed from measured throughput.

Do not tune checkpoints, normalization, thresholds, crop choices or pseudo
labels on the target embryo. Overlapping clips do not provide a certified
inner split. Without verified independent source groups, use a fixed training
horizon instead of target-driven early stopping. Freeze all retained models
and predictions from both directions before new target scores are opened.
Any next change prompted by those scores is another exploratory iteration.

Before model claims, verify unknown-region gradients are zero, human labels
override pseudo-labels, axis/resampling/rounding transforms round-trip, frozen
features are deterministic, only intended parameters receive gradients, and
renamed raw clips run without GT or old prediction paths. Save full code,
environment, source-data manifests, seeds, checkpoint hashes and inference
receipts. Those checks make adaptation reproducible; they cannot retrospectively
certify the released cpDINO training set or replace an independent embryo test.
