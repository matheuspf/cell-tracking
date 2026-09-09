# V2 execution plan: strong baseline, structured repairs, native selection

## Scope and target

Target a **+0.02 pooled local diagnostic gain over the frozen Harmonic Fusion
baseline**, with nonnegative change in both embryo directions. This is a research
target, not a promised result. Smaller valid gains must be retained and reported.
Do not add any local delta to the quoted 0.946 leaderboard score.

Order: **V200 baseline -> V210 failure census -> V220 division repair -> V230
association repair -> V240 native selection -> V250 targeted image/teacher ->
V260 combined evaluation -> V270 report**. V240 can run alongside graph-only
work; V250 is driven by the failure census, not mandatory large-model replacement.

## V200 — Reuse and verify the strong pipeline

Inspect the existing local stores, do not rerun neural inference by default:

- `.../annotation-selection-v1/baseline/public/`: final nodes, edges, features,
  predicted tracklets and confidence missingness.
- `.../annotation-selection-v1/public_harmonic_full/inputs/pre_ilp_*.npz`:
  raw candidate coordinates, node probabilities and candidate edge scores.
- `.../annotation-selection-v1/public_harmonic_full/tracking_repo/predictions/`:
  neural graph exports before the original repair/export pipeline.
- `.../annotation-selection-v1/public_harmonic_full/harmonic_isolated.py`:
  the actual isolated public notebook source, unavailable in the remote aggregate
  review. Read it now; do not infer the 27 repair functions' behavior from names.
- v1 GT/matching stores, per-sample scoring rows and metric source checkout:
  evaluation-only inputs. Preserve all locks and manifests.

Record new v2 source/artifact hashes. Reproduce final graph identity and baseline
counts on two pilots, then all 199 cached graphs. Baseline targets are in
`measured_baseline.json`. Existing scored rows can be reused only with matching
input/evaluator hashes; all changed graphs must be rescored. Restore the exact
public coordinate behavior for baseline parity; do not silently clip the six
recorded z=64 points. A bounds-corrected variant is a separately scored repair.

Native confidence must distinguish original detections from interpolated, inserted,
merged or relocated nodes. Carry stable source node/edge provenance through each
transformation when possible. Keep `missing_confidence` as missingness, not evidence
that a node is a false cell. Validate mapping cardinality and distances; attaching
scores to the nearest center alone can confuse adjacent nuclei or duplicate nodes.

## V210 — Identify the bottleneck, with evaluation-only oracles

Produce a full count/score table at raw neural output, actual major graph-mutating
repair phases, and final output. There may be 27 function bodies but fewer true
transformations. Instrument the call graph and first test up to six coarse phases;
then subdivide only the phases implicated by changes. Preserve identity replay.
Report both cumulative stage changes and selected bypass ablations, because stage
interactions mean a cumulative loss is not proof that deleting one stage helps.

For each of 151 supplied GT division observations, inspect the OFFICIAL local
matching window: predecessor, parent, two daughters and grandchildren. A single
global edge-matching assignment is insufficient for this diagnosis. Assign a
primary failure reason plus secondary flags:

- parent-side evidence absent;
- one daughter lineage lacks any candidate at the allowed times;
- daughters collapse to one prediction or compete in assignment;
- two plausible daughters exist, but only one outgoing branch survives;
- fork exists outside the allowed local timing/topology;
- local merged/shared branch or cross-component evidence invalidates it;
- a correct candidate fork is suppressed by selection/repair;
- division pairing collision or unexplained scorer failure.

Additionally audit all 97 division FPs, 6,682 edge FNs and 6,885 edge FPs by
available candidate evidence and pipeline stage. Cache hard-event windows and
image-derived features, not whole-image copies. Record unique event/lineage groups;
overlapping annotated observations are not 151 independent divisions.

Run four isolated diagnostic interventions: (a) baseline; (b) change legal local
links/forks while keeping node coordinates fixed, using GT only as an oracle;
(c) add missing GT-centered candidates as an explicitly impossible-at-inference
oracle; (d) remove candidates selected using GT as a pruning oracle. Separate
coordinate localization from topology where possible. Rescore all variants.
These are controlled feasibility experiments. A heuristic oracle is not a rigorous
global upper bound; label its search space, constraints and completeness. Do not
train or deploy the GT-only candidate injection, links or thresholds.

**Decision:** if enough daughters already exist, prioritize graph repair. If
missing/merged daughters dominate, V250 becomes a targeted candidate recovery arm.
If original repairs erase useful forks, prioritize their measured ablations.
A low oracle score calls for diagnosing the candidate/matcher limitations, not
blindly increasing selector capacity.

Required outputs: `stage_scores.csv`, `division_failure_census.csv`,
`edge_failure_census.csv`, `candidate_coverage.json`, `oracle_diagnostics.json`.
Detailed event IDs/coordinates and image clips stay local.

## V220 — Division-aware local decoding on frozen nodes

Start with the final strong graph, retaining its nodes and coordinates. Generate
alternative continuation/termination/division hypotheses around *all predicted*
uncertain neighborhoods, not only GT division locations. Use existing pre-ILP
alternatives and image-only physical neighbors. Cap candidates for tractability
while measuring source-side true-division candidate coverage.

First test a small score-based local decoder; then a regularized learned reranker.
Useful features: first/second/third edge probabilities and margins, mutual link
ranks, forward/backward agreement, parent history, daughter separation over the
next few frames, motion-corrected barycenter distance, local density, and changes
in image shape/intensity. Intensity conservation is a soft cue, not a biological
hard constraint. Use several neighboring frames (e.g. t-2 through t+3), explicitly
masking boundaries. Distances are in physical units.

Training positives are known annotated divisions mapped onto predicted candidate
hypotheses. Strong negatives are competing hypotheses at locally annotated
transitions whose truth contradicts them. Mask unknown neighborhoods; do not label
every unannotated fork negative. Avoid declaring a single annotated continuation
negative for every nearby time-shifted division: apply the official local-window
logic and exclude ambiguous/incompletely observed cases. Use all source positive
events, cluster sampling, and a small classifier before any large neural model.

Compare a one-successor continuation hypothesis with the best distinct-daughter
pair, not two independent edge thresholds. Solve small overlapping components
jointly with the existing solver or a bounded ILP: consecutive-time edges,
one incoming parent maximum, two outgoing children maximum, and incompatible
fork/continuation choices mutually exclusive. Keep high-confidence external links
fixed as boundary conditions. Reconcile edits in overlapping windows jointly.
Never insert a second branch simply to score a division without image/temporal
support. The metric rewards correct local lineage reconstruction, not branch count.

Score a bounded family before choosing a configuration: baseline, lower division
penalty with validity constraints, native graph-feature reranker, temporal-feature
reranker. Use source-only selection where groups permit it; otherwise fix settings
before comparative directional outcomes. Track division recall and FP, raw/adjusted
edges and combined score. A division score gain must not hide larger association
losses. Keep actual training label count, calibration and class-sampling weights.

## V230 — Repair ambiguous associations without destroying detections

Use the same frozen node pool and pre-ILP edges. Correct mutually inconsistent
links in crowded neighborhoods, swaps, branches mistaken for births, and short
fragments. Features include motion residual relative to neighboring cells,
pairwise appearance, alternative-edge margin, reverse-time agreement and 3-7-frame
path consistency. Add label-free candidate edges only where needed; record the
candidate-coverage change separately from scoring/decoding gains.

Train on known GT transitions and competing candidate associations at annotated
endpoints, not all unlabeled edges as negatives. Use source-only supervision and
preserve division hypotheses from V220. Start with local reranking and the existing
solver, not a full tracker reimplementation. Maximal predicted tracklets are not
necessarily correct; include uncertainty around the tracklet boundaries.

Report paired link substitutions, deleted FP-only links, restored FN-only links,
newly introduced errors and their effects after full rematching. The earlier
calculation of 1,500 corrected swaps yielding +0.02123 RAW edge Jaccard is a budget
illustration, not a target that the code may manufacture or infer without scoring.

## V240 — Give annotation selection a fair native-candidate test

Build training examples from Harmonic Fusion's own source-embryo candidates,
with the SAME feature extraction and upstream settings used on the other embryo.
Do not load the DoG-trained joblib files as the primary model. Retain the old
transfer model only as a negative control. Public predictor training contamination
remains even when selector labels are held out; this is a conditional diagnostic.

Use all source matched-positive units and a stratified set of unmatched candidates:
high-confidence real-like cells, uncertain detections, duplicates, repaired nodes,
density/depth/time groups. Sample by predicted tracklet/registered biological group
to limit repetition. Record group and node counts and sample probabilities. Correct
training weights and calibration for enriched sampling; evaluate on the natural
full candidate population. Label "matched annotation" separately from biological
cell truth. Mask distance/IDs/GT-derived geometry from all inference features.

Fit native logistic and boosted-tree baselines on geometry-free quality/temporal
features, then with normalized geometry as an ablation. Add node-origin/confidence
missingness and available native linker margins. Always test whether any gain is
ordinary false-detection removal rather than annotation preference.

Replace the fixed 90th-percentile heuristic with a trained tracklet/window risk
estimate: expected lost evaluated TP edges and division evidence per removed node,
with separate accounting for possible FP removal. Generate source training targets
using baseline matching and a bounded sample of actual group-deletion rescoring;
rematching can create new TP or FP, so survivor bookkeeping alone is not truth.
This is an observed-metric-risk target, not a biological background classifier.
Do not sum node probabilities and assume independent endpoints or lineages.

Compare node, whole-tracklet, safe-segment and fork-protected actions. Predict
risk from image/native graph information only. GEFF total estimates and GT match
files may be used for source scoring but are unavailable to deployed policies.
A fixed learned risk threshold may leave some clips unchanged instead of forcing
every clip to lose 10%. If cost normalization uses an estimated total, it must be
an explicitly validated image-only prediction, never test GT metadata.

Prespecified keep-fraction diagnostics: 1,.995,.99,.98,.95,.9,.8,.7,.5. Evaluate
requested and realized budgets, baseline-TP survival/new TP, divisions, and exact
combined score. At 50% retention, the measured-baseline approximation requires
roughly 97.91% TP survival for +0.024 at unchanged FP/divisions. Use this as a
feasibility diagnostic, never a substitute for exact graph scoring. A learned
filter is optional: identity remains a valid decision for a clip or whole model.

## V250 — Temporal image features or targeted FOCUS teacher, only where needed

A native graph model failing does not show that visual information is exhausted.
For annotation-risk or division reranking, train a compact 1-5M-parameter temporal
patch encoder on multiple frames and larger context, with separate local-cell and
neighbor/context representations. Keep triplanar or anisotropic 3D geometry explicit.
Do not perform independent panel flips that imply inconsistent 3D axes; use jointly
transformed coordinates/volumes. Avoid rolling borders into the center as augmentation.

Use all positive groups, diverse negatives and registered overlap deduplication
when available. Report learning curves at 10/30/100% of positive groups and actual
optimization steps. Start with 5,000 steps for the full temporal selector fit,
with an explicit early failure/convergence rule, rather than silently substituting
the v1 6-epoch, ~198-positive source probe. Sparse event rerankers may need far less:
do not force 5,000 steps onto only 26 source division observations without a
regularized/pretrained representation. If a resource limit blocks a full fit,
label it a pilot and finish other valid arms.

FOCUS is a separate, optional branch when V210 finds unresolved daughter candidates
or merged nuclei. Use an already authorized nuclei checkpoint locally, cached once
per worker. Compare masks -> centers on representative *predicted* difficult
windows and uniform control windows; no GT-centered inference selection. Measure
restored centers, division/edge candidate coverage and full local graph gain.
Never equate segmentation-mask quality with tracking score. Direct inference is
allowed only if measured full-pipeline runtime fits the competition constraints;
otherwise distill trusted soft/dense predictions into a small detector using sparse-
label-aware loss. Uncertain teacher negatives remain unknown, not background.
Do not accept gated data-sharing terms or upload images to the Space automatically.

## V260 — Combine only complementary changes; keep validation honest

Evaluate identity, D-only, E-only, F-only, D+E, and D+E+F (D divisions, E associations,
F filtering). Filtering models calibrated on the old graph may not transfer to a
repaired graph: train/infer from the same graph version or disclose the transfer
as its own arm. Fork-protection must use predicted structure, not GT windows.

Use exact official matching, fresh graph objects, local division evaluation and
full sample-set checks for every graph version. Preserve all 199 samples, even
where an arm abstains. Report raw J, adjusted J, division TP/FP/FN, counts, runtime,
and per-embryo/pooled deltas. Changing FP counts changes aggregation weights.
An alpha=0 comparison is a diagnostic only. Include GT-unavailable inference,
identity parity, model/data hash checks and a no-forbidden-graph-edit validator.

Investigate image-registered crop/time transforms to enable purged source-only
space-time blocks when possible. Keep all overlapping image/feature context outside
opposite blocks, and test transform consistency around cycles. Do not pretend that
unmatched image patches establish independence. If transforms remain incomplete,
use fixed limited settings and both directions, with no fabricated bootstrap CIs.
Do not call v2 untouched validation: both embryos and v1 outcomes have been seen.
Any clean OOF claim needs source-only provenance for the upstream predictor too.
Keep operational gains and provenance-clean replications in separate tables.

## V270 — Resources, decision and outputs

Reuse the existing study runtime, cached graphs and patch stores. Start with
bounded CPU local-event processing and deterministic batching. Do not reinstall
PyTorch or recompute neural predictions merely because a new stage exists. Create
an isolated FOCUS environment only when that optional arm is actually selected.
No model downloads are needed for V200-V240.

Suggested soft limits: <=20 GiB GPU allocation, <=32 GiB host memory, <=100 GiB new
cache, <=16 GPU-hours for new non-FOCUS probes. These are experimental resource
caps, not completion-time promises. Log actual utilization, examples, steps and
unique positives; duration alone neither proves undertraining nor adequate training.
Use the cap to prioritize based on measured failure modes, not to terminate the
entire study because one optional arm is expensive.

Local reports: `v2_report.md`, `dashboard.html`, `stage_scores.csv`,
`division_failure_census.csv`, `edge_failure_census.csv`, `oracle_diagnostics.json`,
`native_classifier_metrics.csv`, `operating_points.csv`, `score_rows.csv`,
`learning_curves.csv`, `config_locks.json`, `artifact_manifest.json`, `status.json`.

Decision labels: `significant_local_gain` (+0.02 target, nonnegative both directions),
`smaller_valid_gain`, `no_gain`, or `incomplete`. Add separate provenance and
validation qualification fields; none of these labels implies hidden-test success.
Preserve negative arms and distinguish hindsight-best from prespecified decisions.
Commit only code, configs and sanitized summaries after inspecting staged files;
raw microscopy, GT details, model weights and predictions remain local. No automatic
Kaggle upload, publication, PR merge or paid service call.
