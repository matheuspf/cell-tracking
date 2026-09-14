# Native-resolution detector v8 — one executable plan

**Date:** 2026-09-14. **Execution:** local Codex / user-selected GPT 6 Pro.
**Branch:** `handover/native-resolution-detector-v8`, from merged
`main@fb5521629eb41c8c485b291a5bcf344944c113ae`. **Status:** not executed.

## 1. Decision, questions and success

Train a new center detector for THIS nuclear-fluorescence dataset. The primary
input is native **ZYX = (64,256,256)**, with measured spacing expected to be
(1.625,0.40625,0.40625) micrometers. Do not pre-decimate XY in native arms or
upsample Z to 256. Read and validate actual metadata. Learned internal pooling
is allowed; high-resolution features and skips must precede any XY reduction.

Test independently: (a) retention of acquired detail; (b) spacing-aware network
architecture; (c) incomplete-label supervision; (d) subvoxel/reference-point
placement; and (e) whether improved detection helps the unchanged association
system. A pretty segmentation, a training loss decrease, or more predictions
alone is not success. This is detector development, not another Ultrack or
small division-head project.

Required outputs include a trained, independently callable raw-image-to-points
model, complete detector metrics on both embryo directions, architecture/regime
comparisons, and fresh-image inference. The clean detector benchmark is NEW:
0.934864986413134 is a historical FULL-TRACKER score, not a detector baseline or
an embryo-clean target. C0=0.934802374260586 and P0=0.934864986413134 remain
operational graph controls. >=0.95 is the aspirational full-tracker target; a
negative integration result must not be hidden by a detector-only improvement.

## 2. Inspected state and what remains uncertain

The committed v5 adapter hard-codes (1,4,4) downsampling, coarse 64-cubed positional
features, and a primary native checkpoint. Its previous N2 fits changed the image
encoder for ASSOCIATIONS while keeping the detector output head frozen. V4's D
model refined pre-existing center queries, not all-image detection. These are not
negative experiments on the proposed native dense detector. [R1, R2]

The official companion loader uses strided `::dy,::dx` reads. Its point-target
code truncates fractional coarse coordinates and warns about collapsed targets.
Verify the ACTUALLY installed support-pack files and hashes before attributing
those exact behaviors to every current checkpoint. Do not silently fix the old
baseline while measuring a new model. [R3]

Earlier reports describe upstream exposure. The user's current detector identity
and complete training recipe remain to verify. Use per-artifact states:
`verified_source_only`, `verified_external_no_target_exposure`,
`known_target_exposure`, or `unknown`. Never infer clean provenance from a name
such as split_0, an inference guard, a fresh checkout, or frozen weights. The
primary random-initialized detector must not depend on the incumbent's weights,
predictions, FOCUS masks, historical teachers, or target-fitted statistics.

The prior data audit found 73 confirmed overlapping clip pairs and could not
certify independent inner groups. Its 133,318 annotated observations / 4,725,117
estimated total observations (~2.82%) is NOT the real-cell prevalence or a
positive-unlabeled class prior. Repeated observations are not independent cells.
V6's missing libraries/disk are historical constraints to recheck, not proof of
current availability. [R4, R5]

## 3. D800 — inputs, exact controls and split/provenance lock

Read AGENTS.md, docs/competition.md, docs/notebooks.md and the actual local source.
Use existing data under `/kaggle/input/competitions/biohub-cell-tracking-during-development`;
outputs under `/kaggle/working/cell-tracking/native-resolution-detector-v8` and
ignored `work/native-resolution-detector-v8`. Do not restart old studies or modify
active worktrees, raw inputs, original notebook copies or old environments.

Freeze the 199 expected sample IDs and embryo identities (71 44b6 / 128 6bba in
the reviewed inventory), coordinate conventions, input hashes and available model
assets. Resolve raw heatmaps versus final repaired graph centers: compare both as
different baselines. Reproduce the current raw detector's actual detection TTA,
secondary ensemble and postprocessing rather than using a weakened primary-only
checkpoint and calling it the incumbent.

Build a dependency manifest for every trainable model AND every label/feature
teacher: checkpoint hash, initialization, training-image embryos, annotation
embryos, calibration data, teacher ancestors, evidence path, and provenance state.
Check ancestors recursively, including teachers of training examples. A source-only
student distilled from a target-exposed teacher is not source-only. Architecture
source reuse without loading weights is a different dependency from pretrained
weights; record both. Teacher outputs trained on another fold cannot cross back
into that fold through cached features.

**Two outer directions:** train on 44b6, predict 6bba; train on 6bba, predict 44b6.
Target images are read for inference only in the primary lane: no target-image
self-supervised updates, EMA adaptation, joint normalization fitting, pseudo-label
training, or threshold calibration. Image-local intensity normalization at inference
is allowed. Both directions must be frozen before any new outer result is opened.

**Inner selection:** reuse verified crop/time transformations and conservative
overlap groups. Only if a group-disjoint, context-purged source development split
can be certified, tune/early-stop there. Include entire division neighborhoods,
overlapping fields and repeated clips in one group. Never random-split nodes,
frames or overlapping clips. A negative image-fingerprint result is not proof of
non-overlap. If valid inner groups are unavailable, use fixed recipes/checkpoints
and the fallback policy below; source training loss is resubstitution, not a
validation score. Do not invent inner CV to enable a hyperparameter optimizer.

Declare three result namespaces: **primary random/source-only**, **external or
uncertain provenance**, and **operational inherited-linker**. An uncertain external
teacher may be useful experimentally but cannot win the primary clean comparison.
An all-source deployment refit after final locks is a fourth artifact with no
held-out local score, never a replacement for the cross-embryo score.

## 4. D810 — native grid, target audit and executable training skeleton

Implement one reusable loader, target builder and evaluator. A frame is ZYX; T
is never a spatial/channel accident. Normalize using source-fitted rules or each
image's documented quantiles, not pooled target statistics. Preserve native
coordinates, subvoxel predictions, crop origins, spacing and resampling origin.
A strided grid maps q to s*q; average-pool/interpolating grids can carry half-pixel
origins. Document/test the precise convention. Quantization uses a grid index plus
a residual, not silently floored labels. Count collisions explicitly; incompatible
points in one output bin must not average into a fictitious single-cell target.

Produce source-only diagnostics: nearest-neighbor spacing, point collisions after
1x/2x/4x XY reduction, signed coordinate residuals, uncertainty near borders and
mitosis, and native-versus-coarse crop views. Baseline final-coordinate errors and
raw-detector errors stay separate. Inspect the user's FOCUS discrepancy using
unambiguous isolated instances if real masks exist; do not infer annotator intent
from wrong nearest-neighbor matches in crowded regions.

Primary target: a physical-width Gaussian reference-point heatmap plus local
quantization offsets. Center each discrete Gaussian at the nearest assigned bin
with peak value one; its offset retains the full fractional coordinate. Initial
sigma=0.8 um, offset Huber transition=1 um. Targets
are built with actual spacing; the narrow axial support is intentional, not fourfold
upsampled evidence. Supervise local offsets only for an assigned supported point,
not a whole-volume field toward the nearest annotated cell. Decode subvoxel output,
perform NMS in physical space, and round once at the final submission interface.
Use small multi-cell fixtures to verify that native pixel changes affect the
network and that both localization heads receive real gradients.

First memory pilot: batch 1 full `(1,1,64,256,256)` image, base width 16, mixed
precision and checkpointing; use GroupNorm to avoid unstable batch-one statistics.
If full-volume backward does not fit, train native `(64,128,128)` XY tiles with
context halos, explicit valid cores and deterministic all-field coverage. Tiles
CROP native samples; they do not resize them. Compare to full-volume inference
where feasible. Never seed inference tiles only at annotated/old predicted cells.
Loss only on valid cores; every inference output position gets nonzero blend
weight. Deduplicate overlap peaks once globally in physical coordinates.

Do not assert exact full-versus-tiled parity for a nonlocal/group-normalized model:
measure sensitivity and register tiling as part of the recipe. Exact equality is
required for coordinate transforms and replays using the SAME tiling recipe.
Record RF/halo choice, per-axis padding, seam tests, peak memory and throughput.
The primary implementation must run without an optional external library.

## 5. D820 — three architectures and resolution controls

Use one maintained PyTorch implementation; borrow design principles, not an entire
new environment. Keep parameter counts, normalization, target/decoder and training
sample schedule visible. Do not load the old temporal checkpoint by default.

**A_plain:** compact residual 3D U-Net, all 3x3x3 convolutions and ordinary 2x2x2
reductions. This is a native-grid architecture control, not the proposed winner.
Use widths [16,32,64,96] initially and output the reference heatmap/offsets at native
resolution. Padding/cropping must be explicit.

**A_aniso (primary):** residual U-Net with widths [16,32,64,96,128], first two
blocks using 1x3x3 convolutions and reductions (1,2,2), then 3x3x3 processing and
(2,2,2) reductions after feature spacing becomes approximately isotropic. Preserve
native high-resolution skips into the decoder. This follows nnU-Net's spacing-aware
planning principle but uses point losses, not stock dense segmentation losses. [E1]

**A_context:** A_aniso plus a lightweight, image-derived coarse full-field context
branch fused into native skips. This branch can downsample; the native path cannot.
This tests global context without discarding local detail. No incumbent heatmap or
teacher graph is an input to the primary architecture.

Resolution controls use A_plain at 64^3 strided, 64^3 antialiased, and
64x128x128 antialiased, with the same physical target/NMS definitions. They are
controls only, never eligible to fulfill native-resolution delivery. All controls
view the same physical fields; report that kernel RF in um changes with resolution.
To isolate information loss from RF/capacity, also train **A_aniso-lowpass**: lowpass
and reduce native XY fourfold, reconstruct on the original native lattice, then
run the identical A_aniso network. This is explicitly a degraded-input control,
not native-detail training. Verify resampling origins and compare it to A_aniso.
No 256-cubed arm. Do not inflate the model before establishing input/target integrity.

## 6. D830 — incomplete-label training regimes

### R_masked: source labels + explicit unknowns (mandatory)

Known reference points give Gaussian/local-offset targets. All other bright,
plausible or ambiguous objects remain UNKNOWN. Use a compact positive support
(radius at most 1.5 sigma initially) and independently constructed conservative
background supports. Resolve conflicting/nearby positive supports rather than
forcing another cell to be a negative. Do not paint the rest of each patch zero.

Background is a documented image-only heuristic: low intensity AND low local
contrast, excluded near known points, visible local peaks, boundaries and saturated
artifacts. Define the quantile/contrast recipe on source images before outcomes,
inspect random SOURCE background patches, and report how many low-contrast annotated
cells it would wrongly exclude. Without dense annotations this is not certified
background: audit faint-cell sensitivity and include reduced-background-weight
comparisons. Do not use GT absence alone to construct background. Human positives
override every background or pseudo-negative decision.

Normalize the human-supported and background losses separately, initial weights
1 and 0.1. Their masks are disjoint. Unsupported voxels and offset vectors have
EXACTLY zero loss/gradient. Include patches with no annotations, but only their
justified background/consistency supports contribute. Maintain random all-field
patch sampling as well as positive-centered training crops; sample source
lineage/track groups so long bright tracks do not monopolize the positives.
Record unique annotated observations, groups, frames and boundary coverage, not
just optimizer steps. Positive-only training without a collapse check is inadequate.

### R_dense: missing-as-negative diagnostic (mandatory, not primary eligible)

Same architecture, images, optimizer and positive targets, but lightly treat all
nonpositive voxels as background, reproducing the problematic baseline assumption.
Compare to R_masked; it is not claimed to solve incomplete labels. Preserve counts,
faint-cell recall and count-budget curves. A low dense BCE does not validate labels.

### R_ema: source-only self-training/consistency (mandatory)

For the screen, share the identical source R_masked first-500-update warmup;
for final fits regenerate that warmup under the final schedule. Maintain an exponential-moving-
average teacher of only this student (initial decay .99). Both weak/strong views
are of SOURCE images. Accept positive pseudo-peaks only with image support and
agreement after inverse geometric transforms; require spatial consistency over
successive checkpoints, not raw sigmoid >.9 alone. Absence of a teacher peak is
NOT a negative. Births and divisions must not require persistence through every
adjacent frame. Use temporal support only when available and reliable.

Start teacher confidence .9, transformed localization agreement <=1.5 um, pseudo
weight .25, ramp after the warmup. These are provisional recipes, not calibrated
probabilities. If no pseudo points qualify, record it and keep supervised updates;
don't silently lower thresholds on the target. Human annotations override matched
pseudo points, uncertain/duplicate regions are ignored. Use a per-group cap and
separate normalization so millions of pseudo peaks cannot overwhelm true points.
The control receives the same extra updates and image-view compute with pseudo
loss disabled. No cross-fold teacher reuse, target test-time training or incumbent
prediction input. Prediction inflation/constant-positive fields must trigger an
explicit diagnostic; do not "solve" collapse by top-K truncation without reporting it.

### R_object_ref: external object proposals + learned reference point (optional)

Use already functional local FOCUS or LACSS/Spotiflow predictions only when their
coordinate/weight contract can be verified. The user's observation suggests that
a geometric mask center need not equal the approximate human reference location;
this is a hypothesis to test, not an established annotation rule. No manual mask
or new external runtime is needed for the mandatory arms. [E2, E3]

Fit an auxiliary object heatmap and quantization offset to reliable teacher object
centers q. At one-to-one unambiguous SOURCE matches to annotated points p, train a
separate local native-image/mask reference head to predict (p-q) in micrometers.
Initialize reference shifts to zero; bound their norm at 3 um initially and report
cap hits and ambiguous/unreachable matches. The head must use predicted-center
crops (jitter both matched and unmatched inputs consistently), not positives at
GT centers versus negatives at detector centers. Use Huber regression, not an
annotation-membership classifier. Unmatched proposals have no reference-offset
label, not a zero-offset target. Human points missed by the teacher remain strong
training targets; don't discard them to raise apparent precision.

Suppress duplicate teacher/human targets for the SAME object without erasing a
true neighboring daughter. Keep object-presence and reference-placement targets
separate; at inference output q + quantization residual + learned reference shift,
not a union of both heatmap peaks as two cells. Compare fixed teacher proposals
before/after reference correction, then the full native student. A proposal-only
refiner cannot count as the required whole-field detector. Preserve masks for
later tracking but don't add a segmentation objective without actual mask labels.

Unknown/exposed external provenance taints this whole student lane, even if weights
start random. Report its utility separately; a clean primary must finish without
it. Existing synthetic images may be used only as an explicitly scoped future
extension: their measured acquisition/resolution and 44b6-derived calibration must
not silently enter this iteration's primary lane. No fresh Zoo/rendering campaign.

**Positive-unlabeled alternative:** do not add an unbounded nnPU search. The cited
PU work addresses incomplete cell labels, but its histopathology setting does not
establish random annotation selection or the true-class prior here. The 2.82%
annotation/estimated-count ratio is not that prior. R_masked/R_ema are explicit
heuristics, not claims of statistically unbiased PU risk. [E4]

## 7. D840 — bounded parameter and architecture iteration

`study.json` registers nine mandatory first-seed recipes (both source directions),
two optional recipes, and three refinement slots. Run the base native A_aniso first;
external setup is never the critical path. Sanity overfit 2-4 source crops and
validate source-only early, then do the real fits. No synthetic-only pass counts
as training the new detector on Biohub.

Initial screen: 1,500 optimizer updates per source fit, with checkpoints at
500/1,500; final training: 8,000 cumulative updates for up to two source-selected
native finalists plus the native R_masked anchor. For R_ema, its first 500 screen
updates are the identical source warmup and the remaining 1,000 use EMA; its
matched control receives the same total budget/views. Extend final learning curves
at 3,000/8,000 under the same recipe. These are caps, not assertions that convergence
has occurred. Compare equal sampled supervised exposure and measure equal-GPU-time
learning curves; one full-resolution step is not a coarse-model step in compute.

Default AdamW lr=3e-4, wd=1e-4, batch 1, effective accumulation 4, clip norm 1,
200-update warmup then cosine schedule defined separately for screening/final.
Use a fixed final schedule after source selection; do not compare resumed screen
cosine endpoints to fresh fixed-LR training as equivalent. Final recipes train
from their registered initialization with the final schedule (and their own
EMA warmup), unless a resume-exact state checkpoint belongs to that same schedule.
GroupNorm, mixed precision, explicit valid masks. Begin with single-frame detection;
no temporal-attention expansion in this round. Geometry transforms respect axes:
XY flips/90-degree rotations, translations with coordinate updates, modest
source-derived gain/noise; no pretending Z and XY are interchangeable.

With certified source development groups, allocate three refinements in this
order: lr=1e-4 versus default; sigma=1.2 um versus .8; width 24 versus 16. Change
ONE variable per slot, using the same regime and the winning source recipe. Record
parent recipe, hypothesis, all outcomes and source-only selection BEFORE outer
inference. Lower background weight (.03) can REPLACE the sigma slot if source
faint-cell diagnostics identify negative supervision as the bottleneck. It cannot
be an unrecorded fourth search. No full Cartesian grid.

With NO certified source dev groups: run three fixed probes (lr1e-4, sigma1.2,
width24) against the preregistered native anchor, if capacity permits; don't pick
one from target scores and label it source-selected. The prespecified primary is
A_aniso/R_ema with A_aniso/R_masked as the supervised anchor. Only these undergo
confirmatory-style same-recipe replication. Other outer results are descriptive.
Research reuse still makes every local selection exploratory in the broader sense.

At most 14 screen recipes x 2 source directions, plus 3 final recipes x 2 sources
x 2 seeds including secondary seed 314159 (maximum 40 retained fits counting
screens and fresh final fits separately). Failed or resumed
runs have a separate ledger and consume the same actual resource budget. No
third-party recipe is required when its assets are absent. At least the native
R_masked and R_ema comparisons must be completed if raw data/training runtime fit.

## 8. D850 — detector evaluation, not a misleading FP score

Freeze prediction recipes for BOTH directions, then evaluate whole raw videos of
the opposite embryo: all 199 clips / all frames. Full-field inference cannot use
GT coordinates, target annotation density, old graph IDs, or target-specific names.
Do not evaluate only crops centered on annotated cells. Reuse the official
one-to-one time-aware node matcher for authoritative node assignments, with
physical spacing. Verify modified diagnostic-radius matching independently.

Report both floating predictions and the final rounded/bounds-checked integers:

- Annotated-node recall at 1,2,3,5,7 um. Counts and denominators per embryo and
  pooled; show macro-embryo as well as observation-weighted results.
- Recall-versus-emitted-count curves at fixed top-K budgets per FRAME,
  K=[50,100,200,400,800]. Use model-ranked genuine peaks and deterministic ties;
  never fabricate points to fill K. Always report actual emitted count when
  there are fewer peaks. These are diagnostic budgets, not a target-population
  count estimate or deployment rule.
- Main architecture/regime comparison statistic Q = mean recall@3 um over
  K=[100,200,400], accompanied by recall@7 at each budget. This evaluates the
  sparse annotated population only; no claim of all-cell recall/precision.
- Localization medians/p90, signed Z/Y/X residuals, misses, and a censored error
  assigning 7 um to unmatched GT under the fixed 7-um assignment. Also evaluate
  coordinate changes on the intersection of the same matched GT IDs to prevent
  survivorship bias. Signed error among successful matches is conditional.
- Close-pair and division-neighborhood recall, bin collisions, plateau/duplicate
  counts, faint/depth/border strata and per-video prediction-density changes.
  These GT-defined strata are evaluation only, never inference routing features.

Unmatched detections are UNKNOWN, not individual FPs. Do not compute standard
precision/AP/F1 from them. Dense segmentation Dice or all-cell count accuracy also
needs independent dense truth. Optional independently reviewed held-out image
ROIs can supply that truth later but must not block automatic evaluation. State
how teacher misses and apparent false positives remain unidentified.

A clean detector improvement requires Q above the matched scratch/native R_masked
anchor on both embryos, stable same-recipe replication, and no material recall@7
collapse at the SAME budgets (initial noninferiority tolerance .002 absolute).
Also compare all native arms to the scratch coarse controls to answer the original
resolution hypothesis; tie-break source selection within .001 Q toward smaller/
faster and lower censored error. These pragmatic criteria are not guarantees of
an improved graph or hidden embryo. Don't select a winning floating detector that
loses its gain after integer export without flagging it.

## 9. D860 — operational association integration, separately labelled

Lock one unchanged linker/repair recipe and baseline. Compare baseline raw detector,
new reference locations on fixed proposal identities (where mapping is proven),
and the genuinely new full-field candidate population. Baseline coverage near
98% at 7 um does not rule out important precision/close-pair gains. Use the full
incumbent ensemble for its reference; distinguish P0 from C0 as separate graphs.

Re-evaluate native descriptors/association candidates at the NEW centers and
recompute geometry/positional inputs, NMS and link competition. Do not copy the
nearest old edge probability to a newly detected cell or keep stale feature
arrays after a shift. If the linker only supports coarse integer feature sampling,
record that limit; a trilinear sampler is a separate named interface change, not
part of every detector comparison. Do not also retune division penalties, event
heads, motion repair and graph filtering: hold them fixed to isolate detection.

Score at most six complete operational configurations including both unchanged
controls and the selected native detectors, with fresh official graph matching,
divisions, count adjustment and exact per-video weights. Require all expected
samples; errors cannot disappear through directory intersections. Report native
TP/FP/FN and division effects, plus extra-node/count penalties. A learned detector
can be label-isolated while its inherited linker remains uncertain/exposed; this
combined score is NOT the clean detector benchmark. >=0.95 here is only local
operational target attainment, not a validated hidden-LB score.

If no new graph qualifies, preserve C0/P0 and still deliver the best validated
standalone detector, with its distinct status. Never export arbitrary per-clip
best choices determined using target annotations.

## 10. D870-D890 — packaging, resources and completion

Use the installed notebook PyTorch stack; no broad upgrade, mandatory external
weight or new dataset download. Prefer existing local sources; an unavailable
FOCUS backend/Spotiflow package closes only that optional arm. If desired model
code is not local, implement the specified native loss/architecture directly
rather than deferring all work to dependency acquisition. Respect licenses; no
terms acceptance, hosted image upload, paid calls, hardware rental or Kaggle
submission. Author sources below are references, not required external services.

Initial device limits: one 4090 worker; 20 GiB GPU allocation, 28 GiB total host
RSS or less than available RAM, 4 loader workers initially, 8 GiB persistent disk
reserve. Inspect actual current disk BEFORE checkpoints/caches. Store only compact
teacher peak tables, not all-frame full masks/heatmaps. If space is tight, use a
bounded new-study scratch cache and alternate AVAILABLE persistent volume; do not
delete old artifacts or rely on RAM-only final checkpoints. If no durable capacity
exists, report that actual blocker without claiming full trained-model delivery.

Cap at 48 summed GPU-hours INCLUDING teacher inference, screens, failures,
final training, replication and detector validation; CPU evaluator time is logged
separately. This is a budget, not a duration promise. Use source-only pilot
measurements to shrink optional/refinement screens before sacrificing the core
native R_masked/R_ema and their replication. Report incomplete arms explicitly,
checkpoint progress and retain valid negative evidence. No invented wall deadline
or revival of v4's expired budget.

Implement `tools/native_detector_v8/` with focused modules: data/targets,
models/losses, train, evaluate, infer/report; a single CLI/wrapper is enough.
Reuse maintained Zarr/schema readers and `tools/annotation_selection/metric_adapter.py`.
Do not invoke old `load()` functions that silently load uncertain weights.
One command processes a whole raw Zarr at native resolution and returns scored
subvoxel centers; another exports the valid competition graph with the chosen
linker. Keep masks/boxes optional evidence, not required outputs for the detector.

Required tests: unknown heatmap and offset gradients exactly zero; empty-support
loss finite; human-over-pseudo precedence and no double peak; axis/spacing/origin
round trips; no GT-point collisions silently averaged; physical NMS/plateau ties;
all-grid coverage and tile halos; transformed-label equivariance; source teacher
ancestry isolation; no target in preprocessing/statistics/EMA; no all-background
collapse hidden by top-K; actual native first-convolution tensor receipts; and
full all-frame dataset completeness. Included helpers cover only a subset.

Final replay: one complete unseen-filename clip per embryo with old prediction
caches, annotations and network unavailable. Actually load the NEW detector and
run native image inference. Compare identical-recipe points and integer outputs
to saved batch predictions; record runtime/memory and dependency read guards.
A guard test is not training-provenance evidence. Package detector-only inference
without the incumbent or external teacher when the student no longer needs it.
For an external-proposal-dependent model explicitly package that dependency.

Deliver ONE maintained handover plus:
`results/native-resolution-detector-v8/final_report.md`, small `dashboard.html`,
`detection_metrics.csv`, `regime_architecture_comparison.csv`, `learning_curves.csv`,
`graph_comparison.csv`, `provenance.json`, `experiment_ledger.json`, `status.json`
and `handover/native-resolution-detector-v8/CONTINUATION.md`. Large checkpoints,
predictions, detailed GT assignments and image examples remain local with hashes.
Dashboard must show resolution/coordinate comparisons, recall/count curves,
architecture/regime results, label-support proportions and actual error examples;
no fabricated microscopy or metrics for blocked arms. Commit/push sanitized files
on this branch, with an explicit allowlist. Do not merge other research branches.

Conclusions must separately answer: Does native detail help? Does anisotropic
architecture help beyond matched compute? Which incomplete-label regime works?
Can the learned reference point improve FOCUS-like centers? Is the detector
source-isolated, externally uncertain, or exposed? Does the fixed-linker graph
improve? What should a later iteration change? A negative answer is useful;
800,000 updates or a library installation alone is not a result.

## Sources and boundaries (authoring review, not new inference)

**R1:** [current native adapter](https://github.com/matheuspf/cell-tracking/blob/fb5521629eb41c8c485b291a5bcf344944c113ae/tools/image_native_tracking_v5/native_adapter.py),
[architecture audit](https://github.com/matheuspf/cell-tracking/blob/fb5521629eb41c8c485b291a5bcf344944c113ae/results/image-native-tracking-v5/native_architecture.json).
**R2:** [v4 measured report](https://github.com/matheuspf/cell-tracking/blob/fb5521629eb41c8c485b291a5bcf344944c113ae/results/multidata-training-v4/final_report.md).
**R3:** official companion [loader/prediction](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/predict_unet_transformer.py)
and [training/targets](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/train_unet_transformer.py).
**R4:** [v1 audit](https://github.com/matheuspf/cell-tracking/blob/fb5521629eb41c8c485b291a5bcf344944c113ae/results/annotation-selection-v1/report.md).
**R5:** [v6 measured record](https://github.com/matheuspf/cell-tracking/blob/06e2deedf23f5d7108423d14466316e608c93be1/results/segmentation-tracking-v6/final_report.md).

**E1:** [nnU-Net spacing-dependent topology](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/experiment_planners/network_topology.py):
borrow spacing-aware convolution/pooling decisions, not semantic-mask losses.
**E2:** [Spotiflow repository](https://github.com/weigertlab/spotiflow),
[pretrained model domains](https://weigertlab.org/spotiflow/pretrained.html),
[3D spot localization paper](https://doi.org/10.1038/s41592-025-02662-x): borrow
heatmap/subvoxel-flow formulation or run ONE optional local implementation with
both heatmap AND vector targets masked for missing labels. The transcript/synthetic
spot models are not proven Biohub nucleus detectors. Do not run its stock dense
point-list objective unmodified on sparse Biohub and call it sparse-safe.
**E3:** [LACSS](https://github.com/jiyuuchc/lacss): separate localization/segmentation
heads and 2D/3D point-supervised design. It is a conceptual reference or optional
already installed alternative, not a second required JAX environment. Point labels
are not necessarily incomplete point labels. [FOCUS](https://github.com/yu-lab-vt/FOCUS-3D)
can be a local object teacher; code/weights/API/licensing and target exposure must
be checked separately. No unverifiable benchmark transfer is assumed.
**E4:** [positive-unlabeled cell-detection study](https://arxiv.org/abs/2302.08050):
primary research on incomplete annotations in histopathology. Its assumptions
and supervision population do not automatically apply to 3D zebrafish.

Upstream sources may evolve: record exact local commits/licenses when code is
actually reused. Authoring consisted of repository/source inspection and synthetic
helper tests only; no real-image detector experiment ran here.
