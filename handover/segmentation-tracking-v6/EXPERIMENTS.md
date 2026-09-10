# S600-S680: region tracking experiments

## Objective and scope

Target >=0.95 on all 199 local clips, against the retained C0 score
0.934802374260586. This remains operational exploratory, not clean new-embryo
validation or a hidden-LB forecast. The exact current baseline and any validated
later-v5 incumbent are locked at S600 before any v6 score is revealed.

The main intervention is information from IMAGE-DERIVED REGIONS. Run the following
stages even when a preliminary method loses, except when its integrity or capacity
fails. A gate failure blocks the affected arm, not all independent arms. Do not
replace this with another large dataset pretraining campaign or point-only model.

## S600 — Snapshot, resources and baseline

Read the reviewed parent continuation and local status. Record pending/complete
v5 variants truthfully. Do not restart old queues, reuse expired deadlines or kill
active jobs. Use an isolated checkout/output root and avoid competing for GPU/RAM.
Resolve all dataset/checkpoint/cache roots; missing data on a new host is a named
restoration problem, not permission to silently switch sample population.

Read competition data/rules skills and current local official snapshots. Record
source/weight licenses and current scorer hash. Preserve the established metric
pin 075fc5f5a52d11077f9dc2b074644618f26939e2 if still applicable; explain any drift
and rescore comparisons consistently rather than mixing metric versions.

Freeze all expected sample IDs, per-clip count estimates, original C0 graphs,
models, and per-embryo scores. Verify original full ensemble/repair identity on
cached graphs and one fresh image from each embryo. Baseline-only smoke must not
rerun the entire previous study. Introduce a real read-only input namespace for
prediction and ensure old evaluation attributes never enter region tables.

Measure available GPU, RAM and disk. Defaults: one GPU worker; at most four CPU
workers initially; max 20 GiB GPU allocation, 24 GiB total study RSS and 8 GiB disk
reserve. Adjust downward to actual availability. Initial new study budget is 36
GPU-hours INCLUDING segmentation inference/training plus a separate measured CPU
ledger, not 36 free hours per provider. No paid hardware/API use. Prefer one model
resident across frames and clip-at-a-time masks. The old host had low free disk;
never assume dense all-dataset masks fit.

## S610 — Independent segmentation screen

1. Build a label-blind screen manifest: six clips per embryo selected by image
   density/depth/SNR strata, eight consecutive frames per clip, fixed before
   segmentation outcomes. Each direction selects provider settings using its
   SOURCE only; the same other's images cannot tune it. These 96 frames are a
   computational screen, not an independent validation set. Evaluate successful
   finalists on full videos later.
2. Run FOCUS nuclei with authorized access, Cellpose-SAM volumetric masks, and a
   bounded StarDist3D/source-compatible backup. Classical watershed is a control.
   At most two sensible physical-scale recipes per provider. Do not transform
   every frame into another unvalidated grid merely to fit defaults.
3. Store original provider labels and transforms, bbox crops, voxel counts,
   physical volume/covariance, mean/quantile intensity, detector score (nullable),
   border/seam flags, representative positions, and image/model provenance.
   Never reuse frame-local instance labels as track IDs.
4. Audit stitch seams, duplicate masks, splits/merges, foreground leakage, lost
   faint regions and temporal fragmentation. Create local orthogonal overlays and
   animated short volumes. No human judgments may be invented.
5. Sparse GT diagnostics: one-to-one center match recall; containment separately;
   number of known different centers inside one mask; duplicate masks near one
   known center; localization differences. Unmatched masks are UNKNOWN, not FPs.
   A giant foreground mask can have high containment but is a bad instance model.
   Without independent dense masks, Dice/mask AP cannot be measured as ground truth.
6. Select <=2 segmenters per direction by a predeclared source evidence/runtime
   rule; use mask-based tracking on source screen as part of selection, not visual
   attractiveness alone. Ties prefer simpler/faster. Report screen selection as
   exploratory/source-resubstitution where no independent groups are certified.

A blocked FOCUS checkpoint must not stop the screen. At least one genuinely
learned instance-segmentation route must reach graph evaluation; otherwise report
the central task incomplete rather than relabeling watershed as FOCUS.

## S620 — Fixed-observation information test

Preserve every C0 node, coordinate and native candidate edge. Attach masks using
an image-only same-frame correspondence (containment + physical proximity and
one-to-one ownership, with ambiguity flags). Do not use GT for association. One
large mask containing multiple detections is an ambiguous region, not multiple
copies of the same confident object; use an explicit source-selected split or
mark shared support unavailable. Missing masks retain C0 evidence with missingness,
not zero confidence or deletion.

On the same node universe, link pool, solver and source-trained recipe compare:
A: native C0 evidence + centroid/motion geometry only;
B: A + bounding-box overlap/extents;
C: B + true mask overlap, shape and inside-mask appearance;
D: C + image-derived motion alignment and forward/backward reliability.
All feature-dependent models are fitted on the source embryo only. Native scores
must preserve the full ensemble or clearly label a separate weakened control.
Use the same small regularized ranker family and fit budget across A-D so extra
optimizer work is not attributed to masks. Residual score changes are controlled;
no forced deletion of a fraction of cells and no learned label-membership target.

The primary mask features include intersection over union, directional overlap,
physical volume ratio, covariance eigenvalues and orientation ambiguity, masked
appearance similarity and neighboring-region consistency. Boxes are candidate
pruning/cheap evidence; they cannot substitute for 3D occupancy in crowded tissue.
Report outcomes on pairs with indistinguishable point distances but different
region evidence. Add a diagnostic shape-erased/equal-volume control AFTER the
prediction recipe is frozen, never as a new best-threshold search.

Use an image-derived field shared by every candidate pair. Do not optimize a new
translation for each pair until all masks overlap and call that tracking evidence.
Start with global drift registration plus trusted local image motion; test no-flow
and motion-aligned overlap. Cellpose segmentation flows are NOT temporal motion.
Require forward/backward consistency, valid support and border masks. In unreliable
areas reduce motion weight and retain native evidence rather than inventing flow.

C/D can use the existing validated local edit framework, but it is only the
information-isolation arm. The main full-region tracker below must also run.

## S630 — Full segmentation-first Ultrack

For each selected provider, and an eligible two-provider ensemble:
- Produce TZYX instance labels and use pinned labels_to_contours or an explicitly
  equivalent tested implementation. Keep per-provider outputs, foreground union
  and boundary disagreement separate. These are predictions, not training truth.
- Build the Ultrack hierarchy. Record how each original mask maps to candidate
  regions; count lost candidate evidence through filtering/min-max-size thresholds.
  A contour ensemble produces a hierarchy, not necessarily every raw input mask.
- Select consistent regions over time, enforcing same-frame hierarchy exclusions,
  no merges, at most two children, and valid one-step links. Solve complete clips
  if practical; otherwise use overlapping temporal windows with fixed boundaries
  and explicit fork reconciliation. No post-hoc gap edge may skip a frame.
- Run pure Ultrack IoU, Ultrack with its verified image-flow shift, and a region
  tracker with native neural link evidence on the SAME hierarchy. Map native
  detection IDs only when proven; new region centers require new feature/score
  evaluation, not copying a nearest old node's score without disclosure.
- Do not automatically rerun the old motion relinker or smooth across forks.
  Both were behavior-changing transformations. Preserve the newly selected masks
  and use a documented final representative-coordinate rule.

Do not union every segmenter mask into independent simultaneous cells. Hierarchy
ancestor/descendant nodes cannot coexist. Cross-provider crossing hypotheses need
explicit conflict handling; a single hierarchy cannot be assumed to preserve all
arbitrary overlapping alternatives. Use a fixed-label standalone comparison and
hierarchy selection to isolate this loss. Record one-region/two-regions alternatives
until temporal evidence chooses; duplicate model votes are not independent proof.

Source-select sensible birth/death/division penalties with the actual maximization
signs and the same objective across provider comparisons. Run fixtures showing a
supported split can beat independent birth, while an unrelated nearby cell does
not trigger a fork. CBC is a supported no-new-license option. A missing Gurobi
license must not stop the experiment. Exact large-scale optimality is not assumed:
record solver status, gap, timeouts, feasible outputs and deterministic tie policy.

## S640 — Region-based division evidence

Use a warped parent region versus the UNION of two disjoint daughter regions.
Measure union overlap, parent/combined-daughter volume and intensity change,
separation growth, appearance, neck/boundary changes and multi-frame persistence.
These are soft cues: nuclear condensation, bleaching and partially visible objects
can break apparent volume/intensity conservation. Border-censored measurements
must not be treated as full-size observations.

Compare continuation + independent birth versus one-to-two lineage explanations
jointly with competing owners and later paths. Protect/score the full evidence
window, not only immediate fork edges. Use source-supported positive events and
contradictory links; a single annotated child does not certify no second daughter.
Missing annotation cannot supply generic negative masks or divisions.

First use fixed pretrained masks and source-fitted lightweight compatibility, so
segmentation and fitting effects remain distinguishable. An optional final small
mask-crop temporal encoder is warranted only after representation tests establish
useful region signal; include a same-compute center/box control. It must operate on
inside/outside mask appearance and per-frame validity, not repeat v4's unrelated
synthetic fork-mixture study. Repeat a learned finalist with seed 314159.

## S650 — Conditional segmentation refinement

Do not spend the study fine-tuning a segmenter before running its pretrained
outputs through tracking. If source evidence shows a specific failure such as
merged mitotic nuclei or systematic axial fragmentation, run one targeted repair:
image-boundary multi-scale hypotheses, high-agreement source pseudo-mask adaptation,
or fine-tuning on ALREADY available independently curated dense mask patches.

Sparse points are positive localization evidence, not full masks. Pseudo masks
remain weak targets with confidence/ignore regions; disagreement is not background.
The prepared external static/sequence data supplies centers/graphs, not true cell
boundaries; Zoo/RIKEN trajectories likewise do not provide real masks. Do not
fabricate dense ground truth by drawing spheres. Human review packs are optional
and nonblocking; any new annotations require documented source-only provenance.

Separate original pretrained, refined segmenter, and tracker changes. Do not change
all three simultaneously then credit the gain to FOCUS or the external datasets.

## S660 — Bounded comparisons and promotion

Freeze both direction recipes before new target-score reveal. Initial budget:
<=20 complete graph configurations including C0, representation controls, two
provider-specific Ultrack routes, flow/native hybrid, and limited combinations;
reserve up to four additional slots for same-recipe replication. Predeclare
variant IDs and graph hashes; do not hide failed configurations or missing clips.
At least one complete mask-only full-region tracker and one mask/native hybrid
must be measured. Do not stop after fixed-C0 augmentation alone.

Every complete variant is evaluated on all 199 clips using fresh official matching,
division timing/path logic, fixed GT/count estimates and correct denominator-
weighted aggregation. Report edge TP/FP/FN, division TP/FP/FN, node count ratio,
matched GT nodes, old/new GT-edge identities and count/graph/division attribution.
Point recall is not segmentation accuracy. Provider-mask IoU without dense truth
is a consistency measurement, not a label-based score.

Retain C0 unless a valid pipeline improves pooled score and neither embryo falls
by more than 1e-8. For learned additions require the same result direction in a
same-recipe second seed; export the primary seed. Fixed pretrained/deterministic
Ultrack comparisons require repeated solver/package stability rather than inventing
a second training seed. >=0.95 is target success; a smaller repeatable gain is
below-target progress. Preserve valid negative results and explicit incomplete
lanes. Never mix arbitrary per-clip best variants using held-out labels.

## S670 — Fresh inference and compute

Run a single entrypoint from RAW Zarr images on six complete clips spanning both
embryos and low/median/high image density. It must actually run selected segmenter,
mask construction, motion and tracker paths; saved masks/graphs from training are
unavailable. Include an unfamiliar filename and explicit source-model selection.
Disable network, evaluation labels and old prediction caches before imports.
Prediction may read only licensed packaged weights and raw image inputs. Test
C0 fallback and selected default. Hash pre-solver tensors as well as final graphs
so nondeterministic solver differences are diagnosed rather than hidden.

Keep exact semantic graph/CSV parity as the default requirement. Bitwise model-
training resume is not assumed. For solver timeout nondeterminism fix the
implementation or package a stable policy and rerun; do not quietly loosen score
parity after seeing outcomes. Report image-to-graph runtime and actual mask/cache
space, plus an explicitly approximate full-hidden-size projection. Local 4090
pilots do not prove Kaggle runtime compliance. Do not submit anything.

## S680 — Deliver even if unsuccessful

Produce final_report.md, CONTINUATION.md, an offline dashboard, comparison CSVs,
source/model/adapter manifests, semantic graph hashes, resources, package and status.
The dashboard must show native XY/XZ/YZ image overlays, 3D bbox/mask identity,
parent/daughter unions over time, split/merge disagreements, TP/FP/FN tradeoffs,
per-embryo results and actual runtime. The figures come from real local outputs;
do not invent a measured segmentation visualization. Publish only sanitized
aggregate evidence unless data-sharing permission explicitly permits examples.

Answer: Which masks ran? What compartment? Did boxes help? Did full occupancy help
beyond boxes? Did motion help? Did Ultrack hierarchy improve on fixed masks? Did
masks add benefit over the unchanged native evidence? What prevented >=0.95?
A different checkpoint is not an answer without those measured comparisons.
