# Learning and implementation specification

Implement in `tools/division_generalization_v2/`. These are requested interfaces,
not already-working commands. Existing pipeline_error_training modules and outputs
remain immutable reference implementations.

## Reuse map

- `pipeline_error_training/common.py`, artifacts.py: verified inputs, IDs and graph IO.
- bank.py, actions.py: candidate union, complete donor alternatives, bounded conflict
  closure and literal application. Do not import a global work path into new output.
- labels.py, feasibility.py: sparse support and source-only overlays. Strengthen
  counterfactual validation; their output is training data, not deployable features.
- crops.py, fast_crops.py, frame_crops.py, training_images.py: native coordinates,
  safe reads and normalization. New joint scenes do not follow a selected wrong
  incumbent tracklet as the only optical evidence.
- train.py/dataset.py/calibration.py/scoring.py: failure controls to compare, not
  scripts to restart. Replace the positive-only sampler, budget shrink, concurrent
  schedule and native-logit-sum objective only in the new namespace.
- annotation_selection/metric_adapter.py and center_comparison/tracking_index.py:
  exact scorer, sparse errors, independent division-window assignment.
- existing fresh/native validation and resource lease utilities: preserve security,
  model hash, coordinate, serialization and shared-hardware contracts.

## Candidate identities and scene encoding

Keep P0 integer ZYX coordinates, times and persisted IDs exactly. Read real Zarr
metadata (documented spacing ZYX 1.625/0.40625/0.40625 um); array indices are not
persisted node IDs. Model source is an explicit package argument, never an embryo
filename dispatch. No GT coordinate, ID, match flag, annotation density or sparse
count estimate becomes a model input.

A bank group contains one parent/time anchor, candidate daughters, legal daughter
continuations, competing owners, their alternatives, and keep-P0. Preserve the
existing bank defaults: 6 forward/2 reverse neighbors, parent 16 um, sisters 20 um,
2 paths per daughter, with incumbent edges and timing alternatives retained.
Stream legal alternatives rather than materializing millions of pair/path rows.
Do not shortlist true events using annotations. Log every resource abstention.

The trainable encoder receives ordered frames t-3..t+3 in a fixed physical scene
around the predicted parent, not seven independent scenes for every owner/daughter.
Use fine 16x64x64 native crops and a two-times-wider field resampled to the same
tensor size; batch the two scales through a shared small 3D encoder (16/32/64
channels, GroupNorm). Keep spatial feature maps for query sampling, not only
one global mean per frame. Use explicit spatial/time validity masks and physical
query coordinates. If a candidate lies outside a valid fine field, the coarse
field/mask must represent that honestly, not clamp it onto another cell.

Query parent, daughter-pair and owner evidence from these SAME maps, with a small
ordered temporal module (two attention layers, 128 hidden channels, four heads)
and symmetric daughter interaction (sum/absolute difference or equivalent tested
permutation invariance). Aim below five million trainable parameters. No architecture
sweep. Encode a scene once, then score all candidates in vectorized chunks.
Actual crop adequacy for 16-um candidates and moving objects is a source diagnostic;
missing coverage is not a reason to inject annotation-centered crops.

Native link probabilities, displacement, missingness and incumbent ownership are
auxiliary features. Raw-scene evidence must still exist when incumbent ownership
is wrong. Candidate-marker channels are generated from predictions only. Do not
use ground-truth trajectories to align the movie. The model learns early-two-cell
versus late-two-cell evidence; no hard rule assumes daughter separation, intensity
conservation, a minimum duration or an unannotated branch is biologically absent.

Use source-derived coherent XY flips/rotations, brightness/gamma, modest noise/blur
and center jitter, shared across frames/points. Do not exchange Z with XY, reverse
division videos into merges, or paste arbitrary nuclei with invented biology.
Use the same augmentations in both main arms. Synthetic stress fixtures stay
separate from real supervised biological examples.

## Sparse supervision and complete-action targets

Three masks are separate: biological division evidence, supported identity, and
metric-evaluable action evidence. A recorded single child does not certify no
biological division. Unmatched cells and quiet sparse regions are not negatives.

For true divisions, compatible timing/daughter alternatives are positive as a
SET. Match the official local-window allowance; do not punish another legal split
frame as negative. For supported false hypotheses, train rejection of those
specific alternatives. An anchor with unobserved possible daughters remains
partially labelled: mask unknown alternatives rather than assigning a blanket
no-division label. Keep-P0 is the comparator, not universal ground truth.

Canonicalize alternatives by their FULL resulting edit (removed/added edges and
relevant support). Deduplicate no-op once per decision group, and unite equivalent
representations across pairs. Do not weight a group by how many descriptions it
happens to generate. Timing alternatives that yield different graphs remain
alternatives but cannot be simultaneously selected to claim one event twice.

For a bounded set of source candidates, actually apply each complete action and
compute the changed official counts against the original P0 graph. Validate the
fast labeler against full-graph scorer replay on source fixtures and randomly
sampled source edits, including early/late forks, competing forks, lost true links
and global one-to-one division assignment. A per-fork local compatibility test
alone cannot prove recovery when another fork already owns that GT event.
Cache fixed-node matches; recompute local topology and any affected one-to-one
fork assignment. Approximations are training heuristics until parity is measured.

Record (edge TP/FP/FN delta, division TP/FP/FN delta, known/unknown support).
Compute the exact score change using source-only aggregate denominators when
labelling utility, without averaging per-clip scores. GT denominators/utility
are LABELS/weights only, never inference features or target-derived thresholds.
A division-compatible edit can lose valuable edges; retain that tradeoff instead
of setting its action target automatically positive. Unknown-only zero changes
are censored, not declared harmful; supported zero-utility actions prefer keep.

Train a masked listwise complete-action score with a learned keep token, supported
identity auxiliary loss (weight 0.25), and utility-sign/risk auxiliary loss (weight
1). Use one weight per biological/trajectory group; average supported masks,
not candidate counts. Risk loss uses representative sampling weights. Pair ranking
may oversample positives but must not be interpreted as a biological posterior.
For partially labelled groups, compute likelihood only on known compatible and
contradictory alternatives; do not silently include unknowns in the negative
softmax denominator. A negative-only group must produce nonzero rejection gradient.

## Training mixture and mining

Effective batch 32 groups in both main arms:
8 positive-event groups, 16 random supported negative/ordinary groups, and
8 supported hard identity/event confuser groups. Before mining exists, the last
8 come from independently sampled supported ordinary/confuser groups. Sample
with replacement where needed; report unique biological-group counts and repeated
visits, not artificial new data volume. In small sources, expose every supported
positive group during each fixed sampling cycle. Never duplicate one group into
fit and calibration because there are few divisions.

`J_uniform` keeps all 24 nonpositive/confuser slots on the seeded representative
sampler. `J_mined` replaces only the final 8 slots with source hard-negative replay
at the predeclared refreshes. All 16 random-background slots remain. Positives and
augmentation seeds are aligned where practical. Mining includes high-scoring
supported false forks from both pre-decoder and post-decoder rankings. Cap at one
representative per source conflict/event group per refresh before the next pass,
so one pathological clip does not dominate. Do not rank unknowns as negatives.

Keep exact source sampling probabilities for the random stream. Correct the
representative risk loss for the declared sampling design. The mined ranking
stream is a separate auxiliary stream, not an alleged unbiased prevalence sample.
Report raw strata sizes, weights, effective sample size and post-maximization
false-positive rate. Do not recycle the old deterministic residue times nine.

## Scoring and calibration at deployment

For each complete action a, score `s(a) - s(keep)` from the SAME model and context.
Native edge logits are features; do not add their raw sum outside the learned
comparison. Do not reward edge count or fork count. Initial output heads must
produce a literal zero difference so an untrained/zero model keeps P0 through the
real solver; calibration disabled for this identity fixture. A generic disable
switch is tested separately.

Fit a scalar temperature/intercept for EDIT-GAIN evidence on source calibration
PREDICTIONS at the inference decision unit: canonical competing complete actions,
including the max-selected candidate and null, not independently duplicated rows.
Use unbalanced source support with an unpenalized intercept and documented finite
separation handling; fit regularization/temperature bounds once, not a target grid.
When fitting on oversampled data, use only valid inclusion weights, not guessed
population priors. Never call sparse-supported calibration a whole-biological-field
posterior. Test held-source reliability and error after maximization/decoding.

Keep zero as the fixed calibrated gain decision boundary, with keep on ties.
This is not a sweep for the threshold that happens to score 0.95. Source-only
checkpoint and protected/replacement application selection is allowed under the
fixed validation rule, with all tried source results recorded. No embryo-specific
acceptance threshold at deployment and no known-error exception table.

Decoder takes complete actions and positive learned gains, using existing atomic
ownership, no merges, indegree <=1, outdegree <=2 and consecutive-frame edges.
Existing forks' full parent/predecessor/child/grandchild support is frozen in the
protected arm. The replacement arm may alter it only as one explicitly scored
complete action. Oversized/nonoptimal components abstain unchanged. Same 2% edge
edit cap and existing solver resource limits in all comparisons; no cap search.
Report pre-score/score/calibration/owner/conflict/export rejection reasons.

## Required implementation tests

Literal zero scorer through real solver; disable parity separately; large positive
native logits cannot create a zero-evidence fork; duplicate no-op/action invariance;
daughter permutation; distinct close cells preserved; physically correct crops;
missing frames and masks; unknown-only loss zero; negative-only rejection gradients;
all event groups visited; random sampling independent of GT component IDs; hard
miner excludes unknowns; resumed scheduler/data/RNG match; trainable caches invalidate;
train/infer action scores agree; complete counterfactual score parity; competing
fork one-to-one assignment; owner displacement; existing fork support; no illegal
merges; timeout/oversized-component no-op; native CSV/GEFF/ID parity; and fresh
annotation/cache/network denial. Run real solver and microscopy fixtures locally.
The supplied stdlib tests check planning arithmetic only and do not replace these.
