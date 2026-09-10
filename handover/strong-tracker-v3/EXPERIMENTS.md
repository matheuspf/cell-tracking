# V300–V370: incumbent-native, event-aware graph repair

## Objective and experiment discipline

Primary target: +0.02 over 0.9342063149703403, not over the old 0.911774 baseline.
Keep a smaller reproducible gain rather than stopping because the target is missed.
No gain is assumed. The incumbent is always an eligible no-op. Final local promotion
requires a positive pooled delta and no negative embryo delta beyond 1e-10 scoring
noise. >=0.002 is the default useful-gain label; >=0.02 is the stretch-goal label.
All variants remain exploratory given reused embryos and contaminated upstream
models. Passing this gate is not statistical proof or a hidden-test prediction.

Use the same 199 clips, same current verified official metric, same metadata scales
and estimates. Freeze both source-only directions before each bounded round's
comparative results. At most 32 complete scored variants including controls and
combinations initially; no automatic hundred-setting sweep. If an initial round
informs a later decision, timestamp it as adaptive exploratory, not preregistered
confirmation. Save every failure and complete both directional results.

## V300 — Reproduce and freeze the incumbent, not the historical authoring plan

Run the supplied read-only preflight. Reconcile v2 selected_prediction_lock.json,
selected_predictions, winning_config.json and source revision. Independently
rescore the full incumbent, expecting exactly 123023/4930/5860 edge counts and
29/92/122 division counts with score 0.9342063149703403. Match file hashes first,
then numerical parity at 1e-10; never silently accept a new scorer or stale cache.
Score v1 only as a historical secondary comparison. Missing caches: reconstruct
into the v3 namespace from immutable raw neural outputs/known notebook settings;
never mutate sealed v2 or re-use its old global OUT by monkeypatching.

Create a label-free inputs manifest, explicit environment/path config, and per-clip
incumbent feature tables recomputed at CURRENT centers and adjacency. Carry stable
raw native IDs through repairs. Track inserted/relocated nodes and native confidence
missingness separately. V2 data, hypothesis, feature and model caches are old-graph
artifacts, not interchangeable with these tables.

## V310 — A new residual census and conditional repair ablations

Produce a new failure census on the winner. Cover every remaining GT division FN,
evaluable division FP, edge FN and FP, with local matching and endpoint evidence.
Record candidate coverage before each geometric/topological gate and after it.
Group reasons: missing points, ambiguous node assignment, alternative link absent,
alternative available but rejected, fork absent, wrong pair, wrong timing, merged
branches, lost by later phase. Keep GT evidence evaluation-only.

Reconcile old/winner/raw/E disagreements twice: (a) original coordinates, (b)
common incumbent centers and node universe. Compare by both predicted edge IDs
and recovered GT-edge identity. Identify whether old-TP losses represent genuinely
unrecovered truth or equivalent new matches. Do not transplant edges between
unmapped inserted nodes using a guessed nearest-center mapping.

Conditional on motion staying OFF, run a tiny 2x2 safe-division/smoothing grid
(includes incumbent), plus no-pruning and no-gap controls. Do not infer removal
benefit from cumulative phase scores or from ablations where motion was ON.
Stage naming and call order must be extracted from actual notebook code. Trace why
raw neural outputs have zero forks, including candidate generation, score
normalization and final constraints, without changing them yet.

Measure source-only oracle feasibility on the incumbent-native pool: legal
fixed-node rewire, current-pool division assignment and expanded-pool assignment.
Optional target oracles are generated only AFTER frozen non-oracle predictions,
for final diagnostics. They are heuristics, not global bounds. A missing/poor oracle
in one arm does not block the independent arms.

## V320 — Select among disagreements; do not globally relink

Start from incumbent coordinates/nodes. Alternate edges are the union of incumbent,
raw neural, old-final and v2 E-native/E-hgb outputs with proven ID correspondence,
plus modest image-only neighbors. All teacher graphs first get mapped and rescored
at canonical centers; report failed mapping fractions. No candidate is selected
because it matches target GT. Keep common/high-confidence nonconflicting boundaries
fixed. Local problems include every displaced incoming owner and affected source;
a donor must have an explicit legal continuation/termination alternative.

Train two small opposite-embryo rerankers: regularized logistic residual over native
logits and a compact boosted tree. Train on source-supported positive/contradictory
edges only; unlabeled-unlabeled edges remain unknown. Use native logits, alternate
pipeline votes, forward/backward trajectory consistency, displacement, intensity
and density, path continuity, score margins and explicit missingness. Votes are
correlated sources, not independent confidence multipliers. Avoid embryo/file IDs
and GT statistics as features. Use all source positives; no target calibration.

Decode 5-frame disagreement components as complete path alternatives with fixed
external boundaries, maximum indegree 1/outdegree 2 and consecutive-frame edges.
Keep existing fork motifs frozen in the first association-only arm. Compare native
residual + conservative edit margin (primary) with a tree model and a frozen old
v2 reranker transfer control. Keep an explicit no-op option at value zero. Give
new edits a margin and an edit-count cap, so uncertain regions abstain. A whole-
frame unconstrained relinker is not the default.

Initially measure predicted disagreement edits on source groups with exact local
rescoring to identify model bias; any count surrogate is secondary to fresh full
graph evaluation. Correctness evidence at a GT-supported endpoint is enough for a
negative, but arbitrary unlabeled edges must not be penalized. Log old-TP loss/new
TP gain, removed/introduced FPs and affected node/edge counts relative to v2.

## V330 — Expanded temporally coherent division candidates

Generate event centers from IMAGE/PREDICTION evidence: births, terminating tracks,
nearby daughter pairs, persistent bifurcations in candidate trajectories, strong
second-best native edges, or a mitosis appearance score. Include a uniform spatial/
time sample to detect trigger blind spots. Never use GT to choose inference ROIs.

Keep all incumbent fork motifs. Expand top-four to top-six forward alternatives
plus reverse/birth candidates, with physical gates recorded in config. Do not
require the existing continuation to be one daughter. Enumerate split times within
+/-1 of the image-derived anchor, providing early/exact/late alternatives. A
multi-frame model may inspect 9 frames; every exported edge must still have dt=1.
Preserve spatial scale and explicit boundary masks; don't discard clip-edge events
simply because a full centered temporal window is unavailable.

Compute daughter persistence over at least two candidate paths per daughter,
using candidate edges rather than only the current graph's chosen successors.
Do not make persistence an unconditional hard veto; represent missing support
explicitly and use conservative scores. Default parent gate 16um and sister gate
20um are PROPOSAL limits, not biological truths or the scorer's 7um match radius.
Verify source event coverage and pool size; one predeclared fallback expands to
8 neighbors / 20um / 24um when source coverage is below 85%. Report the coverage-
versus-cost curve even if neither setting reaches that diagnostic target. Keep
baseline/no-op candidates always. Bound to 256 alternatives per local component;
oversized components split with fixed boundaries or abstain, never silently drop
challenging samples. Do not use a target-label oracle to decide expansion.

Source supervision has two parts: is this a supported division event, and which
pair/timing/path realizes it? Use the SAME local timing, anchor, branch ownership
and assignment rules as the official scorer when making source labels. All valid
pair/timing alternatives for an event form a positive bag; normalize total positive
weight per event instead of counting every proposal as independent truth. Construct
near-event wrong-pair negatives only when local annotation actually refutes them.
Quiet fully observed continuation windows remain negatives; incomplete/unmatched
contexts are masked. Perturbations and duplicated crops do not create new events.
Test labels with golden early/late/partial/competing/merged examples.

## V340 — Learn real mitosis evidence and score compatible edits

Primary models are an event-balanced tabular logistic model and a small temporal
image+geometry event model. The latter is NOT the v2 annotation-membership network
reused with the same target. Inputs: predicted parent and daughter positions,
triplanar or compact 3D patches over 9 frames, aligned parent-centered context,
local nucleus shape/elongation/separation and intensity changes, native link scores,
competing owners, uncertainty/boundary masks. Use permutation-invariant daughter
features/head or symmetric branch-swap augmentation with a tested invariant output.
No GT-centered positive crops or different positive/negative jitter distributions.
No hard mass-conservation test for fluorescence intensity.

Use all positive event groups, uniformly sampled supported nondivision groups and
hard contradictory daughter pairs. Freeze seed 20260909 primary and 314159 secondary.
Use 10,000 optimizer steps per source/seed initially, at most 8 GPU-hours total for
this lane; log actual steps, unique event groups, source losses and 25/50/100%
positive-group diagnostic fits only if the main fit is valid. Do not consume the
budget just for elapsed time. Fit a compact encoder from scratch or reuse frozen
source-only feature training; existing public image encoders keep their upstream
contamination label. Use source-only calibration with event-blocked controls where
possible, otherwise fixed limited thresholds and explicit uncertainty. No purported
target-probability calibration from the opposite embryo's label frequency.

Score complete continuation/fork/owner-displacement alternatives, not isolated
second edges. Fix the owner-distance guard with individual-owner confidence and
explicit alternative cost. Do not freeze all incumbent forks forever: a separate
arm allows a dubious fork to be suppressed or shifted, comparing all resulting
edges, donor continuations, and daughter paths together. One event cannot be
predicted repeatedly across adjacent frames; branch sharing/merges are disallowed.
Keep unaffected boundaries and no-op. Solve small connected conflict components
with the existing SciPy MILP (or exact enumeration for tiny fixtures), not independent
parent decisions. Unsuccessful solves abstain and record the reason. Record edge
and division changes separately; never accept an event just because TP rises while
FP or edge damage rises faster.

Compare four complete arms at the incumbent: conservative existing-pool division,
expanded-pool tabular, expanded-pool event-image, and expanded-pool event-image with
replacement/suppression. Thresholds/margins use the small source-frozen grid in
experiments.json. A 0.05 threshold from v2 is a negative control, not a universal
posterior threshold. Candidate prevalence, ignored events and calibration must be
reported. Deployment does not read Nhat or annotation frequency from hidden GEFF.

## V350 — Conditional local point rescue / localization

Run only where V310 shows point evidence or localization limits; otherwise document
why omitted and finish other arms. Use the existing DeepCenter/temporal heatmaps
and 3D image volumes before adding a new foundation-model dependency. Reuse cached
heatmaps read-only; query a single cached GPU model for missing frames.

At image-triggered uncertain forks/gaps, test small physical subvoxel refinements,
reappearance of high-confidence pre-pruning native detections, and a capped set of
secondary local maxima/watershed splits. Do not insert points at GT centers during
inference. Union the old unchanged candidate and the local alternatives; rerun
only the affected region's associations/divisions and preserve all external
boundaries. Require time-persistent image evidence for a new object; duplicated
centers must not artificially inflate count or matching. Report count penalty,
localization, event coverage and end-to-end score, not only sparse node recall.

FOCUS is optional only if an already authorized local checkpoint is available and
these existing models lack needed separation. No gated terms acceptance, remote
Space inference, weight download or external upload is required for v3. A missing
FOCUS model must not block the study. Treat any external teacher's data provenance
as unknown until audited. Benchmark warmed inference and memory on representative
image-derived regions before scaling; no entire-dataset mask generation by default.

## V360 — Combination, regret audit and promotion

Freeze source-trained component models before combinations. Evaluate incumbent,
A-only, D-only, R-only when eligible, A+D, and A+D+R. Regenerate graph-dependent
features after association changes; a stale-feature combination is a named transfer
control, never the primary. A stage winner may inform this round but that selection
is explicitly exploratory. Do not add standalone improvements numerically.

Use an edit ledger: for each local accepted action, list immutable IDs, canonical
coordinates, removed/added edges, native support, uncertainty, fork/owner changes,
solver status, and post-hoc scored cost. Group regret by step and event type; never
allow the evaluation ledger into the inference package. Recompute whole-graph
assignments on every scored sample, not fixed-match bookkeeping. Distinguish GT-edge
recovery from predicted-pair changes. Count estimates remain fixed per clip.

If no source-frozen configuration improves both embryos, retain v2 and report
any one-embryo/hindsight result as unpromoted. Do not choose an embryo-specific
policy by its name. A genuine image/graph quality gating rule must be trained
source-only and tested in both directions. Record worst affected samples without
claiming 199 independent biological replicates; the two embryo results and unknown
overlap are the uncertainty statement, not a node bootstrap.

## V370 — Fresh-image delivery, reporting, and next iteration

Rebuild selected inference in an isolated copy of the actual notebook/CLI entry.
Verify source hashes and preserve originals. Run fresh neural inference on at least
two preselected image-only pilot clips (one per embryo, fixed by image statistics,
not favorable scores), then apply v2 motion bypass and v3 additions. Compare the
cache-derived and fresh graph outputs; differences require explanation/rescoring,
not copying scored GT-dependent coordinates. Run all 199 from fresh images if the
measured budget fits; otherwise report fresh-image pilot versus cached full-run
separately. Report startup, GPU, CPU decoding, I/O and full-set runtime projections
with assumptions; a 4090 pilot does not prove Kaggle GPU performance.

Prove GT-unavailable inference from process start. Do not let a permissive audit
hook import a GT-loaded inventory first. Validate integer in-bounds coordinates,
unique IDs/edges, consecutive time, single parents and <=2 daughters; export no
artificial duplicates, temporal shortcuts or disconnected division decorations.

Write final_report.md, self-contained dashboard.html, summary.json, score_rows.csv,
per-event aggregate diagnostics, model/candidate coverage, runtime and artifact
manifests, winning_config.json and selected_prediction_lock.json. The portable
report must carry per-embryo and pooled DELTAS AGAINST V2, source versions, failed
variants, limits and clean reproduction instructions. Update current STATUS.json
and a NEXT_AGENT.md so the next planner cannot mistake authoring status for an
unfinished experiment. Commit/push only reviewed sanitized outputs to v3. Preserve
raw files locally and list backup/restore dependencies without claiming persistence.
