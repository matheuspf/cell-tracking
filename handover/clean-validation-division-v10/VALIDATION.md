# Validation, isolation and decision rules

## 1. Dependency isolation is recursive

For each clean direction and seed, the detector, encoder, association model, event model, pseudo-labels, normalizers, calibration and any cached neural features must trace only to the permitted source-fit/source-calibration partition. Initialization is random for every neural component. No incumbent checkpoint, DeepCenter, E teacher, all-embryo refiner, target-calibrated simulator, external pretrained encoder or exposed logits may enter C00/C10. Architecture source reuse is allowed after import-side-effect inspection; checkpoint loading before overwriting a state is not.

Use explicit `lane`, `source_embryo`, `seed`, `training_clip_ids`, `calibration_clip_ids`, parent artifact hashes and exposure classification in each artifact. Unknown provenance fails the clean gate. A detector-only clean fit with an inherited linker is not a clean full pipeline.

The operational lane may consume the pinned P0 graph and native evidence, but all its new direct fitting/calibration remains source-only. Its exposed upstream status must survive every report/export. Clean workers must reject operational caches even when filenames have been changed.

Prefer explicit manifest-based data interfaces and allowlisted artifact parents. Install read guards before importing numerical/model dependencies; inspect path aliases, symlinks, directory-FD opens and delegated child processes. Python audit guards are useful regression controls, not a complete security sandbox. Add semantic dependency checks and fresh-process negative tests. No remote telemetry/API dependency is permitted during training or inference.

Never feed annotation density, GT division support, coarse GT total-node estimates, biological component IDs or matched IDs into inference features or model routing. Source labels may define training supervision and source-only sampling; those decisions are not deployment inputs. Model packages route folds explicitly, not by parsing hidden filenames.

## 2. Source partitions and historical reuse

Validate 44b6 has 71 clips and 6bba has 128. Build exact duplicate-frame/clip groups before splitting. Group all connected duplicate clips together. Do not treat a hash mismatch as proof of spatial/acquisition independence: crop offsets and overlapping acquisitions can remain unknown. Cross-embryo exact duplication is a clean-validation blocker to investigate; do not silently remove target clips to manufacture independence.

Within each source, assign connected clip groups once using a fixed hash(seed=20260918, group identity): 80% fit, 20% calibration. All neural upstream and head supervision uses fit only; source-calibration labels are reserved for the specified calibration/gating. Source-calibration images are inferred by fixed models, not used for adaptive encoder training. Freeze exact memberships before fitting and keep them the same across seeds and lanes. Report actual counts rather than assuming a precise 80/20 count.

Calibrating on whole clips prevents a known same-frame split but does not certify an independent acquisition. If grouping leaves too few usable groups, do not invent a node/random-frame split or move target labels into calibration. Use the fixed calibration fallback and report the limitation. Retain a feasible fit partition selected by the fixed hash; a source with no fit clips is a blocker.

Sparse division counts can be small. Do not revise partitions after model outcomes to increase apparent calibration performance. Record source-positive event counts and complete supported negative groups. Seeds replicate optimization, not biological specimens. Both embryos are historically reused; label the result `source_isolated_reused_embryos`, never pristine independent biological validation.

## 3. Freeze protocol

A source diagnostic is a training/engineering observation, not a clean target score. All primary recipes, budgets, two directional fits and seeds, calibrators and target prediction hashes must be fixed before comparative target results are opened. Final checkpoints use the registered update horizon; no target-based early stopping.

Operational target results are also withheld until the clean recipe/models are fixed. They cannot choose the primary architecture, thresholds or calibration. Required source-only margin selection is an explicitly registered inner selection, with uncertified acquisition independence disclosed.

After the freeze, report every retained run. No best-seed selection, per-embryo threshold routing, replacing failed target graphs by P0, post-hoc score-dependent omissions or averaging only the intersection of completed clips. A legitimate empty frame is distinct from missing inference. An incomplete matrix has null missing scores and an explicit partial status.

## 4. Scorer and export parity

Pinned reference: `royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`, including `metrics.md` and the full scorer dependency versions. Recheck upstream at execution. If it changed, retain the historical scorer for historical controls and add a separately named current-metric column; never compare cross-version deltas as one metric or silently replace old numbers.

Native images are T/Z/Y/X Zarr v3; expected spacing is z=1.625 and y=x=0.40625 micrometers. Verify real metadata. Node matching uses 7-micrometer optimal one-to-one assignment per timepoint. Division matching uses its independent local windows/timing allowance; immediate-edge topology is not the verdict. Node count adjustment uses the provided coarse total, not sparse annotation count. Unmatched predictions/edges are not automatically false positives.

Map persisted node IDs to library-internal indices explicitly. Preserve physical coordinate origins through striding, interpolation, augmentation and export. Round once using a declared policy, clamp to native bounds, export integer node coordinates and the required node/edge CSV rows. Validate unique IDs, dataset coverage, edge ownership, allowed degree and temporal adjacency. Score the CSV roundtrip, not a higher-precision internal graph instead.

For clip i, w_i = edge_TP_i + edge_FP_i + edge_FN_i. Aggregate adjusted edge Jaccard as sum(w_i * adjusted_i)/sum(w_i), and division Jaccard from summed division counts. Overall score is adjusted edge contribution + 0.1 * division Jaccard. Preserve the official empty-denominator behavior. Do not average per-clip scores or pools across seeds; compute the official pooled score independently for each seed.

Use the existing metric adapter and an independent aggregation implementation. Reproduce P0/C4 controls only when the exact historical inputs are available. Failure to retrieve them blocks the operational comparison, not clean training. Historical controls remain labeled historical until freshly verified.

## 5. Required diagnostics

Report C00/C10 detector matching at 3 and 7 micrometers, final node counts, source background heuristics, close-cell recovery and graph score. Detector localization is secondary; the official graph score is primary.

For each GT division and supported edge, separate: endpoint availability; deployment-anchor inclusion; daughter/continuation candidate coverage; geometry filtering; complete legal action availability; model rank; calibrated margin; protected-fork/ownership rejection; oversized/time-limited solver abstention; global edit cap; final official verdict. Save a joinable reason ledger. A final no-fork count alone does not identify the causal stage.

Quantify proposal rate, evaluable false-fork rate, selected-event calibration and no-op fraction on full source-calibration clips and targets. Unknown-support events need their own count, not a zero FP label. Report exact recovered/lost TP and introduced/removed FP for both edges and divisions. Do not add overlapping flags as independent biological errors.

Any oracle witness/repair uses source labels only during development and is labeled diagnostic, never a model result. Later target oracle analysis may diagnose after freeze but cannot be used to change this study's selected model.

Report each target embryo, official pooled result per seed, worst clips and distributions. Clip bootstraps, if provided, are descriptive conditional uncertainty under correlated crops, not confidence intervals over unseen embryos. Do not claim biological significance from two specimens.

## 6. Fixed decision thresholds

The **0.95 milestone** requires complete C10 image-to-CSV evaluation, recursive clean provenance, no target-based selection, and score >=0.95 separately in each target embryo and the pooled result in BOTH seeds. Also require no per-embryo regression versus its same-seed C00. Report pooled-only achievement separately; it is not this stricter milestone.

A **promising next-stage candidate** requires official pooled delta C10-C00 >=0.002 in each seed; every direction/seed delta >=-0.001; full correctness/cold proof; and no evidence that the gain exists only through a count-adjustment artifact. Report exact edge/division/count decomposition rather than applying a hidden extra selection rule. These are prespecified practical gates, not statistical significance claims. A nonzero result smaller than the gate remains measured but does not justify claiming the problem solved.

For recommending an operational variant for later testing, require delta versus P0 >=0.001 pooled and no negative per-embryo delta. OP has no registered second seed, so such a variant is only a replication nominee, not an adopted model or verified sampling advantage. P0 is not automatically replaced by any v10 outcome.

Never add local deltas to the historical 0.946 leaderboard score. No private-score forecast or public/private correlation is established. These two-direction models each use only a source subset for training; a future final model trained on both embryos is a different package and needs its own registered fit and runtime verification.

## 7. Correctness gates

Tests must cover target-read/renamed-artifact rejection, unknown upstream exposure, source/target reversal, no target dependence of gradients/checkpoint selection, sparse unknown gradients, possible unrecorded second daughter, negative-only groups, all-unknown groups, duplicate compatible alternatives, source mining separation, warmup/cosine endpoints, exact resume, integer/physical mapping, one-to-one IDs, daughter permutation, padding, zero-head graph identity, protected forks, donor ownership, solver timeout/cap abstention, group-sampling weights, complete population accounting and CSV rescoring.

Real checks must include source overfit and gradients, full-frame/cached crop parity, a supported true and false fork, complete source calibration graphs, and cold inference on a renamed small and crowded target clip for both seeds after prediction freeze. Cold inference receives only each clean package and images. Dataset name changes may affect exported names, never geometry/topology/model policy. Changes to target GEFF in a separate negative-test sandbox must not change inference outputs.
