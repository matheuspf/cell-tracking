# W400-W490: external pretraining -> sparse adaptation -> real graph evaluation

## Objective and fixed baseline

Beat v3 A_residual_m3.0 = **0.934802374260586** on the same 199 local clips,
while retaining nonnegative per-embryo deltas. A +0.01 incremental gain is a
meaningful target and +0.02 a stretch target, not a promise or terminal condition.
Smaller verified gains are retained; no winner means the v3 incumbent remains.
Actual external-data parameter training and real transfer evaluation are required.

All comparisons are operational exploratory: repeatedly inspected embryos,
public checkpoint exposure and unknown overlap remain. Run source44b6->6bba
and source6bba->44b6. Freeze both directions before new target outcomes. Do not
pool target labels into training, use visible test copies, or add local deltas
to historical leaderboard scores. The simulator's 44b6 dependence must be shown.

## W400: source lock and integrity

Read completed v3 results and external data guide, then run the supplied tests
and preflight. Resolve actual paths, source-use eligibility, archived checksums,
preparation version, source metadata, dependencies and GPU. Verify the selected
v3 graph hashes and exact fresh official score before experiments.

Create a dataset-use ledger with one row per collection/acquisition: available,
verified, label/modality capabilities, provenance/aliases, calibration status,
permitted use, selected task and reason for exclusion. Include every downloaded
collection, even unused RIKEN/Mouse. No blanket 'all external data added' claim.
Recheck generator source and the published/direct-parent Zoo schema, not just
aggregate counts. Synthetic CC0 evidence comes from the archived author statement;
Zoo source terms/organizer clearance are separate checks. Bounded RIKEN audit is
optional and must not delay eligible training.

Lock source partitions, teacher/checkpoint provenance and training configurations.
Record the generator sanity holdout as such. If raw/prepared roots are missing,
search only documented/mounted paths and restore available files using existing
instructions; do not force a complete re-download or user-side overlay.

## W410: adapters, source diagnostics and sanity fitting

Implement explicit-grid/intensity/censoring adapters. Make source-only sanity
panels from static native, pooled sequence, Zoo graph windows and Biohub; verify
numeric coordinates against source and prepared metadata, with no OCR.
Count unique examples/events, not candidate alternatives. Reconcile dense
synthetic counts and observed-future masks; audit Zoo graph quality and duplicates.

Generate frozen corrupted-observation fixtures: localization noise, dropout,
duplicates, clipped daughter visibility and shortened history. All corruption
seeds stay in the same partition as their clean source. Tiny overfit tests must
show encoder gradients and learn both true divisions and difficult continuations.
Run actual fork-objective positive/negative tests. If a specific adapter fails,
fix it before training that source; independent eligible sources continue.

## W420: real-trajectory geometry pretraining

Train G on the synthetic sequence graphs and Zoo zebrafish with domain-first,
example/event-balanced sampling. Zoo's known links/forks are weak supervision;
train robust weights and mask uncertain/censored labels. Displacements are either
verified physical units or explicitly unitless features, never assumed Biohub units.
The primary representation must not read clean target parenthood as input topology.

Train a synthetic-only G, a zebrafish-only G and a combined G with identical
architecture and the same optimizer-step budget. Synthetic-only is not a control
for computational effort unless its budget matches. Use fixed exposure cycles
covering the eligible training examples and a broad set of unique event windows;
log actual coverage. Evaluate on untouched generator examples and the declared
Zoo held-out acquisition/time diagnostics, preserving their limitations.

A fourth G adds eligible fly/ascidian/worm/beetle sources; cap these at 25% of
geometry batches, evenly by source, rather than proportional to node count.
Mouse may join continuity batches only, with event loss disabled. Exact eligible
sources and actual weights are frozen before Biohub transfer evaluation. Report
zebrafish-only versus multispecies; do not assume more species is better.

## W430: paired-image and center pretraining

Train D on native static synthetic centers with dense simulated negatives and
localization. Compare with the same architecture/initialization trained only on
source Biohub under sparse-aware losses. Use synthetic sequences for temporal
features on their true grid; do not manufacture native sequence detail.

Train I on actual six-frame paired images and dense event/edge targets, initialized
with compatible D features and optionally G. Include continuation/birth-observation
corruptions and hard same-neighborhood wrong-pair examples. Use class balance for
representation learning but retain unbalanced validation for discrimination and
calibration diagnostics. Use parent plus daughter image evidence, not an unchanged
parent-only score duplicated across millions of alternatives.

A fixed source-derived appearance-randomization control tests the simulator gap.
The downloaded release has no internal appearance/disappearance; observation
corruptions must supply that inference stress without inventing biological labels.
No target-specific appearance calibration in a source-only comparison.

## W440: sparse Biohub adaptation and data ablations

Fine-tune the external representations with source-embryo Biohub labels and a
small external replay mixture to prevent catastrophic forgetting. Start with
75% source Biohub / 25% eligible external batches, domain-normalized losses,
small encoder learning rate and a higher new-head rate. Keep dense synthetic
background supervision; unknown real voxels remain masked. Use a fixed warmup
then limited encoder unfreeze. Do not fine-tune an old cross-exposed E-teacher and
label it clean; direct and inherited exposure is always in the manifest.

Required controls (same new architecture, candidate pipeline and decoder):

| Arm | Pretraining | Adaptation | Question |
|---|---|---|---|
| C0 | None | None: unchanged v3 | Does anything improve the incumbent? |
| C1 | Source Biohub only for matched steps | Source Biohub | Does extra architecture/compute alone explain the gain? |
| C2 | Synthetic paired images/graphs | Source Biohub + replay | Do dense simulated labels transfer? |
| C3 | Zoo zebrafish geometry | Source Biohub | Do experimental trajectories transfer without fake image features? |
| C4 | Synthetic image tower + synthetic/Zoo-fish geometry tower | Source Biohub + replay | Are modalities complementary? |
| C5 | C4 plus eligible other-species geometry | Same adaptation | Do other species help or hurt? |
| C6 | C4 with fixed observation/appearance randomization | Same adaptation | Does reducing simulator shortcuts help? |

Share identical frozen pretraining checkpoints where appropriate, but do not
reuse weights between incompatible control histories. Match optimizer updates,
initialization, trainable parameter counts and final real-adaptation exposure;
report example-level compute differences. C1 uses the same total update schedule
as the compared external arm, repeating eligible source data rather than stopping
early. Also report a real-only short control to separate longer training effects.
Use the primary seed for initial comparisons; repeat the strongest qualifying
external arm and C1 with a second seed before final adoption. Do not pick a lucky
seed or ensemble all arms to conceal a failed data contribution.

Default component budgets: D 12,000 pretraining updates, G 20,000, I 20,000;
Biohub adaptation 8,000 updates per applicable source/component. These are
maximum fixed pilot budgets, not wall-time promises. Record full source examples
and unique event groups consumed; aim to expose all eligible training sequence
examples and >=20,000 unique simulated division parents per full event pretrain
when the partition contains that many. A sanity fit cannot stand in for a full
arm. If actual convergence/coverage is inadequate, one predeclared 2x extension
for C1 and its paired external comparison is allowed before target evaluation,
within the overall cap. No target-score-driven indefinite training.

## W450: bounded inference and honest calibration

Use the same proposal cap and decoder for C1-C6. Evaluate learned event scores
against no-fork and competing pair/path options, not independent low thresholds
on every candidate. Explicitly repair the raw fork-objective incompatibility in
this isolated module and retain the original decoder as a negative control.
A score-gain claim from changing that objective alone does not establish a dataset
benefit; decoder-only control and real-only training controls are mandatory.

Calibrate using held-out external examples and source-only supported contexts.
Report reliability conditional on the observed labels; never claim a biological
posterior from sparse real negatives or apply the forum prior multiplier. If no
independent source calibration groups can be certified, use the fixed primary
margin 4.0 and descriptive margins 2.0/6.0 after external score scaling, clearly
stating this fallback. Choose one candidate by external/source evidence before
opening new target results; serialize all planned comparisons first.

The event gate starts at a conservative operating point and may abstain. Record
coverage lost to gate, daughter cap and pair cap. One bounded cap expansion can
be justified by external validation or source-only coverage, never by target
misses. Initial full real evaluation cap: 24 configurations, including controls,
detector-only changes and final combinations. Never recreate 190M alternatives.

## W460: separate detection from association/division gains

First score G/I changes on the frozen incumbent nodes to isolate graph learning.
Then evaluate D-real-only and D-synthetic variants with the same downstream
weights/policy. Compare matched-center recall, localization, count ratio, duplicate
competition and division-specific daughter coverage; these are diagnostics, not
sufficient promotion criteria. A different detector requires all graph-dependent
features and associations to be rebuilt. Existing v3 A-residual applied unchanged
to a new detection population is an explicitly labeled transfer control.

Only after isolated tests, combine trained detector and best G/I model. Rescore
every complete graph from scratch with the pinned official implementation and
fixed sample list/estimates. Protect local daughter evidence when editing links;
immediate fork edges alone do not guarantee division preservation.

## W470: optional source-gap experiment

Only if C4/C6 show external validation utility but poor real transfer, run one
bounded trajectory-backed rendering arm using eligible Zoo graph windows, an
explicit synthetic renderer and source-only imaging statistics. Keep rendered
image provenance distinct from experimental measurements. RIKEN is not an event
source. This arm is optional and within the same resource/configuration caps;
it must not replace completed C1-C4 comparisons with another data-preparation plan.

## W480-W490: promotion, inference package and reports

Compare every result to the current v3 incumbent, not v2. Promote only complete
199-clip variants with positive pooled delta, each embryo nonnegative within
1e-10, graph/schema integrity, GT-free inference, and no unexplained runtime or
provenance violation. Substantial gain means >=.01; smaller qualifying gains are
still reported and retained. A claimed external-data advantage also requires
comparison against its matched-compute real-only control. If that control is
better, report the result but do not credit external data. All choices after
reading reused target results remain exploratory.

Run fresh-image pilots on fixed clips from both embryos, verify selected weights
actually load, and compare complete image->nodes->edges outputs to the scored
configuration. Test offline dependency loading and integer in-bounds serialization.
Measure end-to-end runtime/memory on representative density; extrapolation from
4090 to Kaggle is an estimate, not a verified 12-hour guarantee. Prepare a local
inference package but do not submit it. Identity fallback must work with all new
heads disabled and no external dataset files mounted at inference.

Produce report/dashboard, dataset-use ledger, source manifests, matched-compute
results, learning/transfer curves, candidate-budget metrics, complete per-sample
score rows, calibration caveats, selected config and model hashes, inference
receipt, status and NEXT_AGENT. Report failed and excluded sources explicitly.
Commit/push sanitized findings to v4; never commit raw arrays, weights or labels.

## Resource and stop rules

Single 4090; soft peak 20 GiB VRAM, available host RAM capped initially at 32 GiB,
new derived cache/output budget 100 GiB; lower limits to host capacity. Pretraining
checkpoints and source files stream from disk. Overall initial cap 36 GPU-hours;
profile before launching and reserve adaptation/evaluation budget. No hardware
rental, background service or paid API. Long local jobs may use the user's usual
terminal/tmux, with real checkpoints and deterministic resume, not promises of
future remote execution. If a cap blocks an arm, finish available measurements,
report its coverage/steps and keep the incumbent. Do not silently shorten a full
arm to one batch or skip target samples to produce a score.
