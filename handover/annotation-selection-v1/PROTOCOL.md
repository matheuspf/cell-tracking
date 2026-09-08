# Experimental protocol

## 1. Questions and nonclaims

H1: image/geometry/temporal features predict membership in the supplied sparse
annotation graph on an unseen embryo, beyond basic detection-quality selection.
H2: a filter learned without the target embryo's labels preserves enough evaluated
links/divisions to improve the official adjusted score after full rematching.
H3: a substantial part of any gain comes from the signed node-count adjustment,
not only removal of genuinely poor detections or associations.

These are separate hypotheses. High classifier accuracy is not evidence for H2;
a score gain is not by itself evidence for H1 or H3. This is a local study using
provided training annotations, not leaderboard probing or recovery of hidden labels.

A graph node is a cell observation at a timepoint, not a unique biological cell.
Repeated observations and overlapping clips are not independent examples. Exact
annotation counts are identifiable; exact all-cell denominators generally are not
identifiable from sparse labels alone. No fixed 10% prevalence or 99% recall target
is assumed. The earlier 0.946-to-0.970 calculation is an illustrative sensitivity
check, never an observed baseline or a hard promotion gate.

## 2. Estimands and denominators

For every sample i, embryo, and pooled evaluation split, record:

- `A_i`: exact supplied GT nodes after ID/schema integrity checks; `G_i`: GT edges;
  GT division count and annotation-component lengths/gaps.
- `Nhat_i`: `estimated_number_of_nodes` read from GEFF metadata, with source path,
  type and hash. `p_ref_i=A_i/Nhat_i` is coverage relative to a coarse estimate.
  Keep values above one visible; do not clamp or silently replace the estimate.
- `P_i`: all frozen candidate nodes; `M_i`: distinct candidates matched one-to-one
  to GT by the pinned matcher. `p_match_i=M_i/P_i` is candidate-conditional
  annotation prevalence. `R_det_i=M_i/A_i` is GT node detection recall. Under
  valid one-to-one matching, verify `M_i <= min(A_i,P_i)`.
- `c0_i=P_i/Nhat_i`; original edge/division TP, FP, FN; sample score weights.
- Estimated real-cell counts from an independent quality audit, if available,
  with uncertainty and assumptions. Do not relabel unmatched detections as
  confirmed real unannotated cells.

Do not report the ratio of sums as a mean of sample percentages. Show both when
useful and name them. Dataset-weighted cell observations can count the same
biological observation twice where crops overlap; report that separately from
an overlap-deduplicated census, which requires validated global transforms.
Per-frame reference coverage is unknown if only a video-wide Nhat is provided:
do not distribute Nhat uniformly over frames and call it measured.

Candidate annotation membership is an observable supervised target *conditional
on the chosen candidate generator and matcher*. It is not necessarily the latent
human-selection probability among all real cells. Distinguish unmatched real cells,
false detections, localization misses, and duplicate competitors near GT. Report
ambiguous near-GT unmatched candidates as a separate audit stratum. Do not inject
GT-centered positive candidates into the deployment population.

## 3. Data split, contamination and embargo

Confirm embryo identities from actual naming/metadata and host references.
The repository snapshot reports two embryos and 199 clips; rediscover rather than
hard-code those counts. Use leave-one-embryo-out in both directions. With two
embryos, A->B and B->A are two replications, not hundreds of independent subjects.

Before fitting, lock a sample manifest. Detect exact duplicates using existing
archive identities plus content checks as needed. Inspect crop/time offsets,
image transforms and overlap; where metadata is insufficient, use image-only
fingerprints/registration on suspected neighbors. Form conservative overlap
supergroups. Within a source embryo, use three grouped inner folds if possible,
keeping complete clips, overlapping windows and connected lineage observations
together. Never split random nodes or adjacent frames. Purge the complete feature
context at boundaries. If too few independent groups exist, use fixed settings
rather than tuning on the outer embryo; mark weak inner validation explicitly.

Freeze feature families, policies, models, thresholds and hashes for BOTH outer
directions before revealing either outer result. Each direction's source labels
may be used for its own training/inner selection. Global schema and count discovery
are allowed up front; detailed outer annotation-versus-feature maps and comparative
outer metrics are generated only after locks. Do not choose a new feature or a
threshold from the held-out maps and still call its score confirmatory OOF.
Follow-up experiments need a new version and must be labeled exploratory on reused
embryos. Report the entire preregistered sweep, not only its best point.

A clean learned detector, image encoder, normalizer, linker and classifier must
not have trained on the outer embryo's annotations. Public notebook checkpoints
may have used both training embryos. Unknown provenance means diagnostic-only,
not clean OOF. Known contamination is not repaired merely by freezing weights.
For the clean learned lane, generate classifier-training features using inner
cross-fitting of supervised detection/linking components, or explicitly separate
that mismatch as a sensitivity lane. Start with fixed image-only candidate
extraction to avoid this expensive dependency for the initial study.

No model feature may contain GT match IDs/distances/degrees, GEFF counts, annotation
masks, manual evaluation answers, target embryo identity, file paths, serialized
GT-derived IDs, or evaluator attributes left on a matched graph. Image-derived
normalized coordinates/time and quality features are allowed. Sample IDs are join
keys only. Nhat belongs to scoring, not to a filter that would require unknown
hidden-test metadata. Policies use fixed fractions/thresholds available at inference.

## 4. S000-S030: establish trustworthy inputs

**S000 — Environment and source lock.** Read local skills/reference snapshots;
refresh relevant overview and metric-patch threads with existing throttles if
needed. Record repository revision, current metric revision, installed dependency
versions and hardware. Run the supplied synthetic tests and existing lightweight
repo checks. Verify data with the existing `--verify-only` path; do not bootstrap
or redownload. Keep notebook runtime separate. Audit public checkpoint training
provenance from support-pack configs/scripts and metadata, not notebook titles.
Check current competition rules before preparing any public artifact; no submission
or external publication is authorized by this plan.

**S010 — Census and integrity.** Run `audit_metadata.py`, then implement the full
streaming GEFF reader: validate IDs, coordinates, endpoints, frame ranges, units,
shapes, duplicate edges, divisions, and expected time direction. Read actual voxel
scale; the repository's scale is a prior, not a silent fallback. Produce exact A/G
and estimated p_ref tables. Inventory candidate input support packs and mirrored
notebooks. Record unexpected schema or metadata conditions, not fabricated values.
Create the split/overlap manifest and complete the preregistration before fitting.

**S020 — Freeze baseline graphs.** Two explicitly separate lanes:

1. **Clean primary feasibility lane:** fixed image-only multiscale blob/local-peak
   candidates, deduplication/NMS in physical units, then a simple deterministic
   temporal linker. Choose thresholds from source-only inner data or use settings
   fixed before either embryo is scored. Include forks only through a source-tuned,
   valid division rule. Classical candidates may be a weak tracker; that limits
   extrapolation to a 0.946 pipeline but not the initial predictability measurement.
2. **Operational diagnostic lane:** adapt one mirrored 0.946 notebook (Harmonic
   Fusion first) to run on training clips and export all candidate/node/link
   scores before filtering. Preserve original notebook/code/checkpoints; use
   isolated outputs. Reproduce its unfiltered graph exactly. Do not claim clean
   generalization unless provenance and split isolation actually support it.

Persist immutable, label-free baseline nodes/edges and hashes. Do not run all three
notebook mirrors by default: they share inputs and may provide little independence.
If available, add a provenance-clean source-trained detector after the cheap study,
not a broad tracker optimization campaign. Keep its frozen graph separate.

**S030 — Ground-truth matching and candidate tables.** Use official time-aware,
anisotropic, one-to-one matching at the verified tolerance. `evaluate` mutates its
input graph; always evaluate copies and never send its attributes into training or
inference. Singleton/no-edge candidate sets need the actual node-matching adapter,
not a nearest-neighbor substitute or an early-return score. Persist matching tables
only in the evaluation area. Log collisions, localization-distance distributions,
missed GT nodes and division neighborhoods. This stage creates source training
labels and sealed outer evaluation labels, not inference features.

## 5. S040-S060: learnability without a misleading denominator

**S040 — Quality and selection audit.** Compare high- and low-confidence candidate
populations and two candidate thresholds. Measure detection recall against annotated
nodes, but do not assume it applies to unlabeled cells. Stratify plots by embryo,
depth, time, normalized position, density, intensity/SNR, boundary distance and
predicted temporal persistence. Outer-label feature plots remain sealed until S090.

Create an optional blinded census pack: initially 24 random 3D ROIs per embryo,
stratified across depth and early/middle/late frames, with known sampling probabilities
and non-overlapping evaluation cores plus image halos. Sample ROIs independently
of detections. Hide sparse GT markers and model keep decisions. Count every visible
center in each core, adjudicate ambiguous objects, and double-review a subset when
human annotations are available. Independently audit a stratified candidate sample
for real-cell precision. Keep human labels out of selector training in this study.
Use design-weighted totals and cluster uncertainty; a candidate-precision audit
alone cannot count cells missed by the detector. A full automatic detector count
is a proxy, not ground truth. Capture-recapture across correlated detectors is only
sensitivity analysis, never an identified all-cell denominator. No human response
is required to continue: export the pack and report exact true-cell prevalence as
unresolved when independent census labels are absent.

**S050 — Tabular probes.** Fit a constant baseline, regularized logistic regression,
and one small boosted-tree family (fixed grid in experiments.json). Feature groups:
geometry/time; appearance/SNR/density; predicted temporal/link confidence; all groups.
Compare to a quality-only model. Fit transforms on inner training groups only.
Use all candidates for the observable match-membership target; additionally report
a real-like/high-confidence subset and a sensitivity fit excluding ambiguous near-GT
duplicates. Apply those eligibility rules from label-blind features at inference;
GT-based exclusion is diagnostic, not a deployable population change.
Train with class weighting or recorded negative subsampling if necessary, but
calibrate and evaluate on the natural candidate prevalence, with sampling correction.
Report AUROC, average precision relative to prevalence, calibration/Brier score,
recall at the preregistered keep fractions, phi/MCC at each threshold and deleted-group
annotation rate. Main outcome: preservation of existing TP links after actual filtering.

**S060 — Small 4090 image probe.** Run one bounded image-only and one image-plus-tabular
probe if candidate/schema integrity holds; CPU failure alone does not rule out visual
selection cues. Start with triplanar 2.5D or compact 3D patches centered on PREDICTED
centroids (e.g. 32x48x48 voxels, using actual physical scale), optionally previous/
current/next images. Use a small encoder, train-only normalization, light symmetric
jitter applied to all classes, and group-disjoint inner early stopping. Do not crop
positives at GT centers and negatives at detector centers. Record all receptive-field
and augmentation parameters before training. Two seeds (20260908 primary; 314159 robustness only), at most 12 epochs per fit,
no whole-volume network sweep. Adjust batch size to the measured device memory;
record peak memory, GPU time, data-loader throughput and checkpoint choice.

## 6. S070-S090: test the real score, not a classifier surrogate

**S070 — Preregistered filters.** Keep fractions: 1,.9,.8,.7,.5,.3,.1, independently
within each video. Report realized node retention after graph closure/rounding.
Unfiltered r=1 must preserve the graph exactly. Compare:

- Random node selection and random coherent tracklet selection (20 fixed seeds).
- Ordinary detection-confidence selection, with the same graph policy/budget.
- Annotation-score node selection.
- Annotation-score coherent selection of maximal predicted nonbranching tracklets.
  Use the preregistered upper-quantile score; rank units and count their node costs.
- Coherent selection with protected predicted-fork context (parent/predecessor,
  children/grandchildren), using ONLY predicted graph structure. Closure can exceed
  the requested budget; do not hide this or use GT divisions for protection.

Remove incident edges of removed nodes; never bridge missing frames, duplicate
edges, fabricate GT-guided links, or change matching tolerance. Keep other graph
postprocessing fixed. Isolated-node cleanup is a separate named ablation, not a
hidden behavior at r=1. Apply the same policy to quality-only controls. Existing
fragmentation and division failures must remain visible.

**S080 — Source-only selection and lock.** Inner evaluation chooses one configuration
per direction using exact official score, including the no-filter baseline. Tie
within 0.001 score favors less deletion, then simpler model. The full fixed sweep
is descriptive; only the source-selected rule is the primary outer test. Freeze all
models, policies, source manifests, inference feature schemas and predicted outer
graphs before opening outer score files. A separate process must reproduce predictions
without loading GEFF/annotation artifacts. Verify identical output hashes when
annotation paths are unavailable. A model trained on A may naturally use A labels;
it must never read B labels to produce A->B predictions, and conversely.

**S090 — Full official evaluation.** Recreate fresh graph objects and rematch every
variant against each outer sample, including local division matching. Require the
locked expected sample list, not the prediction/GT directory intersection. Fail
on unreadable/missing predictions, unknown positive total estimates, nonfinite
required metrics, or changed GT edges/count estimates between comparisons. Report
zero-event cases explicitly and reproduce the official division-term behavior.
Use `evaluate` -> `per_sample_metrics` -> `summarise`; `evaluate_datasets` alone
omits the count adjustment. Upstream's evaluate.py can skip samples: wrap it or
call the functions with strict checks. Match against the current Kaggle scorer
contract as well as the public repo; unresolved drift makes results diagnostic.

## 7. Mathematics, controls and attribution

For one sample, with J edge Jaccard, D division Jaccard, c0 original predicted/
estimated node ratio, and realized retention r:

`S0=J0*(1.1-.1*c0)+.1*D0`

`S1=J1*(1.1-.1*c0*r)+.1*D1`

These forms assume a positive multiplier; actual scoring clips the adjusted edge
term at zero. With fixed matching and b=FP0/GT_edges, u=TP1/TP0, v=FP1/FP0:

`J1/J0 = u*(1+b)/(1+v*b)`.

Independent endpoint selection suggests u approximately t^2; coherent tracklet
selection need not. Measure endpoint keep correlation and actual link survival;
never use the squared approximation in place of graph evaluation. Rematching can
recover new TPs: report old-TP survival and newly recovered TPs separately.

For classifier annotation prevalence p, keep rate r, recall t:

`phi=sqrt(p/(1-p))*(t-r)/sqrt(r*(1-r))`

`annotation_rate_in_deleted = p*(1-t)/(1-r)`.

Degenerate denominators are undefined, not zero. Under independent random annotation
selection in expectation t=r and deleting coherent units lowers the count-adjusted
edge contribution in the simple no-FP, c0=1 setting. Synthetic random-selection controls
should show no repeatable predictive advantage.

**Run-level scoring matters:** sample weights are TP+FP+FN and can change when FPs
change. Aggregate current per-sample adjusted values using current weights; micro-
aggregate division counts. Do not pool all node counts into one global ratio or
macro-average sample Jaccards. `analysis.py` checks this arithmetic after official
counts are available; require parity tests before trusting it on real outputs.

**S100 — Controls and uncertainty.** Show the official score and a diagnostic alpha=0
score; the latter is not the competition score. Decompose count versus graph effects
using two-factor Shapley attribution on the exact aggregator: mix baseline/filtered
node counts with baseline/filtered edge counts in the two counterfactual orders,
and average marginal changes; isolate the division term separately. Counterfactuals
are arithmetic attribution, not physically realizable predicted graphs.

Compare selection against quality filtering at matched REALIZED budgets and against
random filters. Use groupwise shuffled targets and a synthetic independent random
annotation-membership task on candidate tracklets as leakage controls; preserve
within-unit label correlation and report the randomization unit. Source-only oracle
filtering is a diagnostic feasibility test; after final locks, outer oracles can
show hindsight headroom but are never eligible for selection/promotion. A heuristic
oracle is not a proof of a global upper bound, especially after rematching.

Bootstrap paired complete non-overlap sample/supergroup contributions within each
embryo (2,000 draws), recomputing the complete aggregator including divisions for
each draw. Do not resample independent nodes or drop rare-division cases. Report
both direction-specific deltas and seed variability. These conditional intervals
do not quantify generalization to a population of unseen embryos; there are only
two observed embryos. Sparse positives and correlated tracks can make 99% recall
estimates much less certain than their node count suggests.

## 8. S110 — Decision and output

Finish even if there is no gain. Use one of:

- `positive_local_replication`: the source-selected rule gains in both directions,
  pooled gain >=0.002, paired within-embryo evidence is stable, the GT-free inference
  test passes, and required provenance/metric checks are clean. The 0.002 is a
  pragmatic local promotion threshold, not a promised gain or generalization bound.
- `promising_but_uncertain`: smaller gain, inconsistent embryos/seeds/intervals,
  missing independent candidate census, or only diagnostic public-checkpoint gains.
- `no_transferable_signal`: valid completed clean experiments do not improve;
  quantify remaining oracle headroom and detection-versus-selection limitations.
- `incomplete_or_invalid`: name the blocked lanes and still report valid audits.

A positive score result may support H2 while H1/H3 remain unresolved. Claim
annotation-specific benefit only with the corresponding controls, not merely a
positive combined score. No automatic Kaggle upload or deployment promotion.

Produce the report/dashboard/tables in REPORT_TEMPLATE.md; link every numerical
claim to a result artifact and command. Commit only sanitized summaries and code.
Keep predictions, raw image patches, original references, checkpoints and detailed
GT matching tables in ignored local storage. Reused-embryo follow-ups require a
new experiment version and must not be presented as untouched validation.
