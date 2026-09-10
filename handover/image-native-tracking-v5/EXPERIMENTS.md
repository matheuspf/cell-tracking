# X500-X580: change the representation and observation space

## Objective and scope

The baseline is C0/v3 A_residual_m3.0: **0.934802374260586** on all 199 provided
clips. Target **>=0.95** complete local score. Preserve C0 and report deltas against
it, not against v1/v2 or a weaker newly trained control. No hidden-test/LB uplift
can be calculated from these local deltas.

Primary hypotheses:

- **H:** a pretrained edge-centric tracker with image-derived object properties
  and sparse real-edge adaptation resolves relationships our fork heads miss.
- **N:** adapting the actual native 3D image/tracking network on dense candidate
  neighborhoods improves links more than training another small feature classifier.
- **P:** full-frame new detections and one-versus-two observation hypotheses recover
  missing objects that coordinate refinement cannot, when selected jointly over time.

An integration failure, inadequate representation, absent candidate evidence and
lack of transfer are different outcomes. Measure all four rather than collapsing
any failed arm to "more data does not work". H and N are independent; complete a
new-observation pilot even when one fails. Do not require a manual audit, FOCUS,
an entirely new embryo, or a synthetic-score gate before running usable experiments.

The first full graph grid has at most 18 configurations. There is no global
annotation-filter arm, no Zoo-renderer extension, no random search over v4 margins,
and no claim that increasing crop-classifier width is a new backbone experiment.

## X500 — preserve evidence; run a short, useful preflight

Read the v4 continuation, actual report, source code and AGENTS.md. Verify the
parent revision, C0 config/selected hashes, pinned evaluator, existing base weights,
DeepCenter dependency and new sources. Distinguish original v4 handover STATUS
from measured completion receipts; v4 is finished. No old seven-hour timer applies.
Use `preflight.py` as a read-only starting check; then validate actual manifests.

Keep v1-v4 immutable. Explicit roots should support both /root and /home layouts.
Reuse existing model/data caches after hash checks, without recomputing 87 GiB of
unchanged input checksums on every stage. Record meaningful fingerprints and which
previous receipts were reused. Never reset, clean, force-checkout or discard user work.

Reproduce C0 scores from complete stored graphs. Independently execute the current
image-to-graph entrypoint on two full pilot clips chosen by image-derived density
(one per embryo) before training. This tests source/grid/preprocessing consistency;
it is not a fresh hidden-test or independent biological validation.

Prepare source-only adaptation directions 44b6->6bba and 6bba->44b6. Labels and
artifacts from the opposite direction must not leak through cache keys. Historical
teacher/upstream checkpoint exposure remains recorded even when direct fitting is
source-only. Public external weights of unknown biological provenance are an
operational diagnostic lane, never silently called clean.

Resource preflight is important: v4 recorded 23.275 GiB free. Set new-output capacity
to min(24 GiB, 60% of currently free storage), retaining at least 8 GiB free; lower
cache sizes as required, not sample coverage. Use streamed frame windows and
compressed graph edits. One RTX 4090, soft GPU allocation ceiling 20 GiB, CPU RSS
ceiling min(28 GiB, half available RAM), <=8 active CPU workers initially. Benchmark
before increasing workers. Use an isolated environment for new incompatible
libraries; do not upgrade the working baseline stack in place.

## X510 — separate candidate evidence from model/decoder failure

Use the C0 node universe and exact scorer, not v2/v4 proposal caches whose centers
or semantics differ. Build a label-free candidate bank containing incumbent links,
available native alternatives, and bounded bidirectional physical-neighbor edges.
Use source-only motion statistics for distance choices, not the evaluator's 7 um
matching tolerance. Start from eight outgoing spatial alternatives, keep every
incumbent link regardless of rank, and record coverage lost at each cap. Both
daughters need independent incoming-parent possibilities; do not row-softmax a
mother to one child.

Compute these evaluation-only diagnostics per embryo:

1. GT node matching, GT edges whose endpoints exist, GT edges in the candidate bank,
   and missed transitions around crowded regions and mitoses.
2. Official locally recoverable division evidence, broken down into missing parent,
   missing daughter, wrong timing, missing downstream path, insufficient candidates,
   unsupported score, and conflict/protection loss. Retain exact-ID metrics only
   as a separate column. Do not equate them to the official timing-window score.
3. Frozen-node, legal truth-assisted association/bifurcation feasibility on the
   candidate bank; then candidate-augmented feasibility after X520. Preserve graph
   constraints and compute the real metric. Heuristic oracles are not global upper
   bounds and must never run inside inference.

No learned source model or operational inference feature reads these target
annotations. Source diagnostics may guide that source's documented engineering;
target results are revealed only after both directional configurations freeze.
Avoid a long census project: reuse prior match/evaluator machinery and compute the
few new distinctions that change which lane is worth scaling.

## X520 — image-derived instances and a genuinely new observation bank (P)

This stage is required, not a conditional repeat of v4 center-offset training.
Two versions must be kept distinct:

**P0 fixed-node morphology.** At each C0 center, derive an image-supported local
object region using existing neural foreground/center maps, anisotropic distance,
and marker watershed (or an equivalent documented local partition). Regions are
inference features, NOT human ground-truth masks. For the fixed-node control,
retain the exact C0 node IDs, coordinates, node count and edges. Region-derived
centroids must not silently replace the scored centers. Query morphology around
an unsupported center with a missing-feature mask or abstain on that neighborhood;
never drop a node because mask painting overwrote its label.

Use the genuine intensity/shape properties required by HOCT. A fixed-radius ball
is an explicit diagnostic fallback only: it destroys morphology and is not a full
HOCT test. If graph construction from a label volume moves centers or loses seeds,
construct a full feature-bearing graph through the supported graph interface and
retain a verified node/region mapping. Handle collocated seeds and duplicate
coordinates explicitly. Do not use the unimplemented `create_graph_from_points`.

**P1 full-frame proposal discovery.** Run the existing native/DeepCenter heatmaps
across full frames at native preprocessing, retaining high-recall local peaks and
alternative watershed splits. Search outside incumbent neighborhoods too; at least
one representative full-clip pilot must be exhaustive over the image field, not
GT-triggered or restricted to annotated cells. New node IDs have explicit origin
and coordinate transforms. A one-region observation and its two-region split are
mutually exclusive; duplicate peaks cannot both count as separate cells simply
because two detectors produced them.

Assemble short temporal support around new proposals using image evidence, native
features, and forward/backward candidate scores. No intensity-only persistence on
flat background. Do not fabricate a daughter trajectory from interpolation alone.
Reference frames out of view, gaps, births at a crop boundary and detections lost
by the model are separate masks, not automatic death/nondivision labels.

**P2 learned dense detector, when P1 evidence justifies it.** Unlike v4's query
refiner, use the existing full-volume detector and actually export novel local
maxima. Start from its existing weights. Supervise annotated positive centers,
plus explicitly tagged low-weight dense teacher positives from agreement/persistence.
Treat unknown real voxels as unknown; background regularization uses audited
image-empty regions and stability to a frozen detector, not a dense zero label
outside sparse GEFF centers. Teacher predictions remain pseudo-labels with stated
failure modes. Use synthetic static centers only as a <=20% controlled replay arm,
not a claim of independent real supervision. No segmentation masks are invented
from sparse center annotations for a dense supervised mask loss.

Hold the linking model/decoder fixed in new-node comparisons, rebuild its candidate
scores at new coordinates, and score node-only/fixed-graph changes separately from
link changes. If added detections cannot be scored by a model's feature interface,
that arm is not a completed end-to-end detector test.

FOCUS is an optional P proposal/teacher comparison only when authorized weights
already exist or can be obtained without accepting new conditions. Its model access
currently requires contact-sharing acceptance. Never upload images to the Space.
Use a cached predictor, explicit ZYX/spacing restoration, and a bounded pilot.
Failure/access absence routes to existing heatmaps, not a new study-wide stop.

## X530 — one temporal explanation objective (J)

Represent short tracklets and candidate links over a 5-frame window initially;
allow a 9-frame context extension only as one registered control. Score actual
consecutive-frame edges; longer context does not authorize output gap edges.

Compare complete explanations of a neighborhood: continuation plus unrelated birth,
one parent with two daughters, alternative owners, temporary nonobservation when
supported by actual candidate images, and no edit. Joint constraints enforce one
incoming edge, at most two outgoing edges, acyclic forward time, and mutually
exclusive observation hypotheses. Birth/termination costs depend on observed
boundary/context, not an assumption that all internal births are divisions.

An implementable formulation uses node selection y, consecutive edges e, births b,
terminations d, and bifurcations s with:

`sum_in(e) + b = y`

`sum_out(e) + d = y + s`, with `s <= y` and `d + s <= y`.

Add observation-exclusion constraints and explicit daughter-pair/path terms as
needed. Never import the old objective unchanged: with free births, second-edge
reward <=1 and split cost1.2, every fork is dominated. Verify real solver fixtures
for a true fork, a continuation plus independent birth, no-cell background, and a
conflicting owner. The no-new-evidence configuration must reproduce C0.

Use mature model link logits/context to score these explanations; do not gate H/N
through v4's independently trained geometry classifier. Do not set the selected
source model's uncalibrated score >0.5 as a universal biological event decision.
Fit small score scales/offsets on permitted source development evidence, with
sparse-label masks and explicit source selection provenance. No target calibration.
If source development support is inadequate, use fixed logged costs and a single
prespecified conservative alternative, not a hidden target sweep.

Timing alternatives for the same inferred division share an event identity. Within
an equivalence class use a fixed max or deduplicated log-mean-exp reduction before
comparison with continuation/birth; do not repeatedly count near-identical timing
hypotheses as independent opposing evidence. The supplied helper tests duplicate
invariance; real event equivalence must be defined from predicted paths only.

Keep C0 fork evidence over the actual scorer context protected unless the complete
replacement explicitly models it. Immediate two-edge protection is insufficient,
as v3 demonstrated. Protected status uses predicted structure, never GT matches.
Score whole regions/windows with no-op fallback. Reconcile overlapping window
solutions with shared boundary variables or disjoint core ownership, rather than
concatenating contradictory decisions. Log component size, timeout, missing-model
features and fallback rate. A 99.99% abstention result must not be described as
successful new tracking.

## X540 — H: pretrained Higher-Order Cell Tracking Transformer

Pin `royerlab/hoct@2ccc5040823bc944ab67790abd1f56eea7cd4f05`. Recheck source terms,
checkpoint provenance, official model hashes and actual runtime. This published
model predicts contextual edge evidence; it is not an image segmentation model.
Its registered weights are JIT inference models. Do not assume a full editable
training backbone is distributed just because a model can run.

Primary model: `general_v1`; `ctc_v0` is a single preregistered alternative.
Use actual object features from P0. Start with official `features.create_graph`
or the feature-complete graph path accepted by `predict(graph=...)`. The convenience
`create_graph_from_points` is a documented `pass` at the inspected pin. Verify
feature order, normalization, dimensionality, scaling and border properties before
looking at scores. Audit morphology versus synthetic spheres; do not fill unavailable
region/intensity fields with zeros and call it a comparable pretrained test.

**H0:** zero-shot fixed-node scores, with J and unchanged candidate bank. Keep the
original HOCT decoder as a separate source-pilot diagnostic, since its default
3-frame gap and 300-distance settings are not the competition output contract.
Explicitly use consecutive candidate edges for export and validated physical units.
Reuse model/feature windows; do not reload weights for each node/frame.

**H1:** freeze the HOCT backbone, extract genuine edge embeddings, fit its linear
probe on supported source transitions. `correction.fit_from_labels`/`ProbedModel`
provides a starting interface. Batch source labels and graph updates rather than
calling its edge-update routine millions of times. Deduplicate repeated edge/window
samples or compensate their weight. Check label propagation is target-parent mutual
exclusion; it must not eliminate a possible unannotated second daughter.

The upstream adaptation consistency target is the current hard ILP solution. Test
one soft/confidence-masked alternative or disable consistency in ambiguous fork
regions; absent edges of C0 are not confirmed negatives. This change has a named
control and its own hashes. Do not suppress every possible fork by distilling the
incumbent's missing-division pattern as truth.

**H2:** combine the adapted HOCT logit and unchanged native logit with one
source-fitted residual calibration (not a free ensemble-weight search). Compare
H1/H2 under the same J objective and P0 nodes. This tests complementary evidence
without replacing every reliable incumbent relationship by an external prediction.
Full HOCT-backbone adaptation is optional only if supported by an audited editable
training path; otherwise H1 is the valid delivered adaptation, not a fake full fit.

A new model that underperforms zero-shot may still justify the sparse probe. Complete
H1 unless its input coverage/runtime/source access fails. Report whether a poor
result is due to region extraction, graph coverage, pretrained domain, or adaptation.

## X550 — N: train the actual native 3D tracker

Use the exact locally installed checkpoint architectures, preprocessing and inference
source recorded by the incumbent package. Do not substitute unpatched upstream
code when the working notebook has patches. First reproduce raw probabilities and
candidate IDs on two clips; document any intentional changed tensors. There is no
license to redo all notebook repairs while calling the experiment backbone training.

**N0:** unchanged native model under J, P0 nodes and the common candidate bank.
This isolates decoding from learning. It need not equal C0, whose legacy repairs
are a separate full control. A J implementation that alone loses substantially
must be fixed using source-only diagnostics before attributing that loss to N/H.

**N1:** keep 3D image weights fixed; train the native association transformer/head
on predicted (not GT-only) neighborhoods. All observed source GT transitions must
be eligible for sampling. Include correct mother->both-daughter targets as two
incoming-parent choices for distinct targets. Matched target with a recorded
predecessor supplies a categorical mother objective only when the correct mother
is represented; a missing mother is censored, not a ground-truth birth. Sparse
unlabeled targets and unsupported second-daughter negatives remain masked.

**N2:** warm-start N1 and unfreeze the native temporal 3D feature extractor at a
smaller learning rate, still evaluating on the same fixed candidate centers. The
small heads from v4 are not reused as the encoder. Log at least one changed encoder
tensor and nonzero gradient as an integrity check; report actual parameter count,
physical receptive field, context frames, candidates/window and memory. Respect
inference downsampling/normalization exactly; a higher-resolution model is a new
registered configuration, not a silent adapter fix.

Initial recipe in config: 8,000 association updates then 12,000 feature-adaptation
updates per direction, effective batch4 windows with accumulation as needed;
AdamW head lr1e-4, encoder lr1e-5, weight decay0.01, gradient norm1.0. They are
starting hyperparameters, not assurances of convergence. Use mixed precision only
after finite-loss/identity tests. All source transitions are sampled over the run;
log missed positive groups. Permit one source-diagnosed optimizer repair when
learning curves diverge or fail the tiny real-source overfit, with the failed fit
preserved. Do not scale a broken optimizer by blindly adding more steps.

Real objectives: supported incoming association likelihood, explicitly positive
branch evidence where both daughters are observed, and low-weight teacher stability
on confidently consistent unlabeled continuations. No quiet-chain hard nondivision
loss and no generic dense background loss. If weak pseudo-labels are used, keep
their targets/weights and source separately recorded. Reweight division windows
without pretending the oversampled batch fraction is a biological event prior.

Train crop halos with complete local competitors, not GT-centered positive crops
versus randomly located negatives. Halo nodes are context; only unique core
transitions contribute to each loss. Candidate edges outside the extracted volume
are masked, and correctly omitted true mothers are censored. Include actual false
parent detections and repaired-node origins, not only perfect synthetic anchors.

Source-only development may use registered, spatial/temporal held-out support blocks;
unknown overlap means those diagnostics are dependent. They are allowed for debugging
but not independent confidence intervals. Checkpoint selection uses source criteria
only. Freeze both direction models and source-derived calibrations before outer
comparisons; do not choose 44b6 hyperparameters from its later target score.

## X560 — combine image observations with the better representation

Test P1 with N0 as a frozen-model control and with source-selected N2 or H2. This is
a real detector/linker factorial: nodes fixed versus expanded, representation frozen
versus adapted. No stale node index, coordinate-based teacher join, or inherited
feature cache after node additions. Existing raw-ID provenance and strict mapping
should be reused where valid; inserted nodes need new features and scores.

Use a small image hierarchy: incumbent instance versus its proposed two-nucleus
split; incumbent plus isolated new peak versus no new cell. In the all-199 run,
select mutually compatible hypotheses using J. Include a proposal-free ablation and
an image-evidence ablation. Do not increase counts globally to chase annotation
recall; report the signed node adjustment and supported false edge/division effects.

Escalate to P2 dense training only when P1's source-only coverage and optical
quality make a real new-object opportunity plausible, or when fixed-node models
have exhausted measured headroom. A constrained candidate oracle is a feasibility
check, not a performance forecast. Full-volume native detector training is a
legitimate new experiment here; another center-only offset head is not.

If the best single pipeline is near the target, allow one source-selected H/N
logit combination and one P combination within the 18-variant cap. Do not run
scores of combinations of failed heads. Learned second-seed replication retains
the same architecture, thresholds, sources and decoder.

## X570 — decision on complete graphs, then real fresh-image inference

Every compared graph receives fresh official node and division matching under the
pinned implementation. Require all expected 199 samples, unchanged GT hashes and
per-clip estimated counts, no skipped exceptions, exact variable-denominator
aggregation, and integer in-bounds consecutive-frame output. Store edge-identity
regret in both predicted-ID and GT-edge space. Report per embryo, not just pooled.

Use the exact additive score budget to interpret changes: S>=0.95 needs +0.0151976
relative to C0; useful changes may come from both edge and division terms. Do not
sum gains from interventions evaluated on different node populations. Re-evaluate
all combinations. Expose the contribution of node count, graph associations,
localization/matching and division evidence, with counterfactuals labeled arithmetic.

Freeze a small batch of configurations before outer scoring. Primary interpretations
are by family H/N/P; the hindsight best is exploratory. For a learned candidate,
require a same-recipe second seed and its matched frozen-model control. Promotion:
positive pooled delta and no embryo below C0 by more than1e-8 in both seeds; primary
seed remains the export, not the better of two lucky seeds. Target success additionally
requires selected complete primary score>=0.95. Improvements below target may be
retained but must be named subtarget gains. A deterministic candidate requires an
independent fresh rerun rather than a meaningless changed training seed.

A failed larger candidate does not justify silently switching to a weaker reference
baseline. If neither seed qualifies, retain C0 and report the best exploratory point
and failure mechanism. Confidence intervals over cells/clips with unknown overlap
are not evidence of biological generalization. Public checkpoint exposure, repeated
embryos and any synthetic 44b6 calibration stay explicit.

For finalists execute image-to-CSV on at least six complete clips spanning both
embryos and density, including all new proposal/model/solver paths. Remove caches
of final graphs from the inference view. Verify image/model-only dependencies and
exact selected graph or documented numeric score parity. Package all new weights
and pinned dependencies; base dependencies may be referenced explicitly with hashes.
Use offline network-denial/read auditing and report its actual strength. Do not
claim Linux namespace isolation when unavailable or 4090 timing as Kaggle runtime.
No submission is authorized.

## X580 — continue the iteration with useful evidence

Write final_report.md, dashboard.html, status.json, all score rows, training curves,
source/model/graph manifests, dependency map, selected package, and CONTINUATION.md.
Track actual data used, source licenses, pretrained graph feature coverage, new-node
recall versus stable-node loss, observed whole-graph edits, abstention, and bottlenecks.
Include a small local optical-review pack for missed/gained divisions selected only
after prediction freezing; it is diagnostic, not new model labels. Do not invent
manual judgments or make execution depend on a user doing them.

The default new compute envelope is <=48 summed GPU training hours, <=18 complete
scored configurations, and the disk/CPU limits in X500. Pilot measurements determine
batch/window sizes and continuation priority. Caps are resource controls, not a
promise about completion time. Do not burn the entire budget on broken integration;
a source/adapter repair is bounded. If capacity is exhausted, finish available
complete arms and report unrun work separately. Smaller valid gains are kept.

Commit/push sanitized results and code in the v5 namespace, plus an updated v5
STATUS and handover CONTINUATION. Check staged files before pushing. Prior branches,
raw microscopy, detailed GT identities, weights, credentials and submissions remain
unchanged/outside public Git. No automatic merge or Kaggle upload.
