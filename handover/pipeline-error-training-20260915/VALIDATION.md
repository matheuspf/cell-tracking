# Validation, exposure and acceptance

## 1. Three different claims

**Operational improvement:** new module on the exact exposed P0 pipeline, compared
on all 199 clips and both embryos. This is the main practical result. It does not
establish hidden-LB or independent biological generalization.

**Module transfer:** direct fitting/calibration uses only source embryo 44b6 and
is evaluated on 6bba, and vice versa. Report both directions. If P0 candidates,
released embeddings, native logits or historical E teachers are used, label the
result `source-only head; inherited upstream exposure`. Do not call it clean OOF.

**Clean end-to-end transfer:** requires target exclusion from every learned
upstream component, pretraining/simulator calibration, teacher and calibration
choice. Reuse the existing 400-epoch source-only native fits ONLY if completed,
validated and provenance-clean. Do not treat progress snapshots or intermediate
weights as the promised completed comparison, restart these runs, or wait for
them as a prerequisite to all other work. When available, test the source-selected
module and its matched control on the same clean upstream banks without retuning.
If unavailable, explicitly leave this claim unestablished. Stress tests and
source-held-out patches are not replacements for this claim.

Both public embryos have already influenced project development. Even completed
source-only training now gives limited two-embryo transfer evidence, not a pristine
unseen biological test set. No target refit or favorable subset may repair this
limitation rhetorically.

## 2. Split and checkpoint discipline

Record source acquisition/time/space overlap metadata. Union known overlapping
clips and lineage observations before creating source inner groups, and purge
at least the complete temporal/crop support on either side of a boundary (minimum
nine source frames; increase if an inherited window reads farther). Never split
random event rows or daughter alternatives from the same group across partitions.
Whole-clip assignment alone is not proof of independence when crops overlap.

Prefer source inner training/calibration groups that can be independently
certified. If independence cannot be certified, use deterministic whole-clip
source groups for operational calibration with known overlaps unioned, state the
remaining uncertainty, and report this as exploratory rather than independent.
When that leaves insufficient event groups, use a fixed final checkpoint and
source-only regularized calibration, explicitly resubstitution; do not inspect
target outcomes to choose a fallback threshold. Continue other useful lanes.

Train both directions and freeze their manifests before comparing new target
scores. Each source checkpoint depends only on its permitted source, declared
external assets and immutable exposed upstream evidence. All target labels are
available only to a separate scorer AFTER prediction serialization. Rename copies
of images during fresh inference; source model choice is an explicit argument,
never filename/embryo dispatch. Prediction processes must fail on GT access.

The already published aggregate audit can guide this predeclared design. New
target per-case outcomes cannot guide new training samples, thresholds, per-clip
exceptions, checkpoint selection or candidate-bank revision. Trace target reasons
post-freeze. Any later target-informed study requires a new dated plan and an
honest exploratory label, not a retroactive rewrite of this lock.

## 3. Source-only screening and finite matrix

Use source grouped validation, supported identity/event losses, catastrophic-FP
checks and actual source graph replays to screen. Candidate AP alone does not
qualify a family. Bank source feasibility is a diagnostic, not a claim that an
oracle is learnable. No arbitrary 0.95 source-score floor aborts every lane.

If an adapted Organoid or temporal family is source-qualified, train its primary
matched controls, then nominate at most ONE division family and ONE identity/
observation family before target scoring. Replicate nominees with seed 314159
for both directions. The original seed 20260915 remains primary. Both seeds'
models and one preregistered composition are frozen before their new target
comparison. Do not replace the primary seed with whichever scores best.

Source qualification means: contracts pass, supported held-group decision loss
beats its matched compact/frozen control or exposes a useful distinct tradeoff,
source full-graph replay is not materially worse than P0, and FP/edit counts are
finite and explained. Persist the exact selection inputs. When no independent
inner source grouping exists, any nomination is operational/exploratory and must
be described that way. Run all independent primary lanes within the registered
budget even when another family fails.

Existing-fork protection versus replacement is a fixed decoder ablation using
identical scores. It is not an extra neural model or a target-chosen policy.
Default nomination uses the conservative add/redirect application; nominate
replacement only from source results. No global threshold/margin/penalty grid,
per-embryo acceptance setting, target-loss tuning or ensemble search is authorized.

## 4. Full metric and error accounting

Pinned scorer revision: 075fc5f5a52d11077f9dc2b074644618f26939e2.
Freshly evaluate every complete 100-frame clip (use actual image metadata if it
differs). Aggregate adjusted edge terms with the exact official per-clip weights
and division TP/FP/FN by micro counts. Never average per-clip scores or use the
upstream accuracy-times-recall proxy for promotion. Persist independent aggregation
checks and actual per-clip estimates.

Report baseline/candidate score, delta versus P0 and versus C4_m6, raw/adjusted
edge terms, node adjustment, division TP/FP/FN, edge TP/FP/FN, matched nodes and
selected node counts. Report pooled and each embryo separately, and every primary
and replicated fit. Include old C0 as a construction reference.

Recompute the September 15 error partitions. Report TP recovered AND lost by GT
identity, FP removed AND introduced, division recovered AND lost, selection-gap
versus raw-gap changes, near competing-center FP, and matched-endpoint FN.
Changes to both metrics can concern the same event: no additive opportunity sum.
Keep exact unchanged-node identity checks for D/A and fresh matching for O.

Small source repaired-graph probes must report their actual graphs and full score.
Count-only scenarios are labelled accounting. Do not claim perfect divisions or
perfect center matching predict an achievable training result. Descriptive per-clip
bootstrap, if used, is not an independent-embryo confidence interval; two embryos
are insufficient for a persuasive broad-generalization CI.

## 5. Recommendation gates

Hard correctness: complete expected clip inventory; valid graph/CSV contracts;
zero-head identity; no target-label reads; baseline preservation; no changes outside
logged components; deterministic candidate generation; no hidden graph/count
manipulation; and runnable annotation-denied fresh-image inference.

For a **recommended replacement candidate**, require BOTH training seeds to have
positive pooled delta versus P0 and nonnegative full-score delta on EACH embryo.
The primary seed must also exceed C4_m6 in pooled score. Require the same matched
control advantage in both seeds for claims about a specific learning change.
A deterministic frozen arm uses repeatability instead of artificial seed claims.
Record uncertainty and seed spread even if these conditions pass.

Do not enforce zero individual TP losses inside explicitly edited components:
that would forbid some legitimate division/identity corrections. Instead report
all regressions, full-metric tradeoffs and their concentration. Outside the
permitted components preservation IS exact. Structural failures or one-embryo
regression prevent a replacement recommendation regardless of pooled gain.

Label the effect size rather than inflate it: <0.001 pooled gain is small;
>=0.001 is a useful local gain; >=0.003 is material for this study; >=0.95 combined
is the stretch target. These labels do not change thresholds or authorize more
trials. A small reliable gain remains reportable, not proof that the main problem
has been solved. Always retain unsuccessful candidates in the table.

Even a passing operational recommendation must say whether clean transfer was
measured, blocked or failed. A source-only head on P0 cannot satisfy clean transfer
by definition. Do not promote a new default or make a Kaggle submission here.

## 6. Generalization-oriented diagnostics

Use the same predeclared source-bounded brightness/contrast/noise/blur perturbations,
small center jitter and missing-context masks across models and controls. Report
paired degradation in full graph score on a fixed representative subset, plus
module supported losses. Partition by density, depth, signal strength, boundary
status, local crowding and event support; strata thresholds use source/image-only
rules frozen before target labels. Report close-cell pairs separately.

Do not optimize against this stress suite repeatedly. Perturbed graphs are
auxiliary diagnostics, not clean new embryos. Do not implement undocumented
transductive normalization or test-time parameter updates.

## 7. Inference, resources and artifacts

Measure complete crowded and ordinary clips, then fresh pipeline reconstruction
for two full clips with unfamiliar filenames and explicit source-model arguments.
Deny GT, old prediction/feature caches and network access from process startup.
Only caches generated in the current fresh run may be read, with image/model/code
hashes. Check serialized GEFF/CSV parity and integer coordinates. New observation
selections must trigger fresh native feature extraction.

Measure total GPU peak (not just allocator), process-tree RSS, CPU threads, disk
use, training and full inference wall time, and solver optimality/abstentions.
A local 4090 measurement is not a measured Kaggle runtime. Read the current official
notebook rules before packaging; no new external network dependency at inference.
Do not claim a 12-hour Kaggle pass without representative execution evidence.

Commit code, configs, environment pins and concise numeric evidence. Keep images,
raw references, large candidate banks, weights, label tables, full predictions,
logs, submissions and credentials under ignored local roots.

Required result files: STATUS.json, input_manifest.json, exposure_manifest.json,
split_manifest.json, execution_lock.json, experiment_matrix.csv, scores.csv,
per_embryo_scores.csv, error_transitions.csv, stage_attribution.json,
source_feasibility.json, training_summary.json, replication.json, resource.json,
validation.json, REPORT.md and CONTINUATION.md. Explicit nulls plus reasons mark
blocked/unrun metrics; never zero-fill them. Keep model/package hashes in Git,
not private payloads. Any public links in the report must be portable, not localhost.
