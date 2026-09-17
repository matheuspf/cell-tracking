# Division generalization v2: fewer models, adequate training, representative errors

## Objective and scope

Target >=0.95 on complete scored graphs. Keep P0's detector, all 4,108,943 final
observations, normal continuation links and inference preprocessing unchanged
outside explicitly scored local event edits. C4_m6 is a mandatory challenger.
The goal is a materially better learned module, not a new threshold or another
A10-sized residual. There is no promise that available data supports the target.

The primary question is whether a trained **joint-scene, complete-action division
verifier** can recover correct forks while rejecting ordinary false ones. The
previous trial did not give that question adequate or representative training.
Do not run another tracker survey, native detector campaign, external-data collection,
raw-proposal union, O10/A10 rerun or large architecture matrix in this study.

## G00 — Reuse the completed evidence; validate only what needs validating

Read AGENTS.md and applicable skills. Inspect current processes, environment,
space, actual image metadata and artifact manifests. Resolve P0/C4_m6 using the
previous study's verified inputs. Keep work/pipeline-error-training-20260915
read-only; create work/division-generalization-v2 and results/division-generalization-v2.
No stale job metadata authorizes resuming or terminating an older run.

Verify all-199 baseline graph hashes and per-clip metric receipts. Freshly rescore
one crowded and one ordinary clip per embryo and check independent aggregation
of the existing full receipts. Reuse valid pinned baseline scores instead of
rerunning every unchanged pipeline for every arm. Any mismatch requires explanation
and full revalidation of the affected artifacts. New candidates always get fresh
complete-graph evaluation. Preserve P0 as the production reference.

Run the actual zero-valued event scorer through candidate generation, complete
alternatives, conflict solving and export; require exact P0 identity. The old
`zero=True` shortcut is an additional disable test, not this proof. Test a free
second daughter with a large positive native edge logit: without new learned
event evidence it must not force a fork. Existing native logits become input
features, not an unconditional extra-edge reward in the new module.

## G10 — Fix the training population and supervision before changing model size

Build a prediction-only candidate bank on the SAME P0 points, using the existing
physical forward/reverse union, timing/path alternatives and ownership closure.
Keep the existing default geometry limits and 2% changed-edge safety cap. No
radius, top-k, margin or penalty search. All model comparisons use one bank.

Enumerate source fit/calibration candidates with explicit support labels, including
ordinary negative-only event groups, difficult competing parents, apparent duplicate
splits and real division-compatible groups. Unknown hypotheses stay masked. Do
not reuse `event_keys` as the only event sampler or the temporal residue weighting.
Use a full source census where possible; otherwise use a fixed seeded probability
sample independent of GT component indices, with known inclusion probabilities
and cluster-level bookkeeping. Positive oversampling and hard-example mining are
separate training streams, never prevalence estimates.

Mine difficult cases from SOURCE predictions only, including the model's selected
false forks after decoding. Retain random supported backgrounds so the miner does
not define the entire training distribution. Archive unknown high-scoring cases
for diagnosis without assigning them fake negative labels. All alternative pairs
and timing descriptions of one biological event remain in one split/group.

Label complete legal actions relative to P0: recovered/lost supported edges,
recovered/lost divisions, new/removed evaluable false forks and neutral/unknown
changes. A true-looking split that destroys better supported links is not
unconditionally a positive action. Use source-only counterfactual graph scoring
to validate training labels as specified in SPEC.md. No oracle rows enter inference.

Deliver sampling_audit.json BEFORE production fits, including negative-only group
counts and visits, unknown masks, candidate coverage, and stage-wise true/false
retention. Finish these audits, but do not stop at auditing: proceed to training.

## G20 — Make adequate learning fit on the 4090

Replace per-node repeated seven-frame encoding with a shared scene per parent/time
anchor, reused across pairs/owners. Batch groups by clip/frame; normalize/read a
frame once; shard uint8 crops and compact frozen features. Cache trainable features
only within the same forward/backward, never across optimizer updates.

Profile cold/warm loader, encoder, backward, optimizer, transfer, lease wait and
checkpoint writes separately. Test AMP with FP32 reductions/losses, finite gradients
and a short source parity comparison. If unsafe, retain FP32; do not change the
scientific model to hide numerical problems. Keep model/optimizer resident during
a permitted lease block, with cooperative yield at the existing lease's allowed
boundaries. Never hold a longer lease than other users/jobs permit. Save resume
state every 128 updates, on cooperative interruption, and at milestones, rather
than forcing disk writes after every update. Test interruption/replay reproducibility.

Lock effective batch 32 groups, 4096 JOINT updates, AdamW lr 3e-4, weight decay
1e-4, gradient clipping 1. Warmup 128 updates, plateau through step 3276, then
cosine decay to 10% of nominal. `contracts.py` defines zero-based indexing and
checks it. This is a training recipe, not claimed optimal tuning. Maintain the
same examples, budget and schedule across the two main comparison arms.

A slow pilot may change microbatch/accumulation, cache/chunk layout and scheduling;
it may NOT lower 4096 to 178. Profile after caches are warm and exclude other-job
waits from compute projections. Minimum steps do not establish convergence: also
report fixed source loss curves, gradient norms, unique group exposure and full
source graph screens. If the budget expires, preserve resumable state and mark
training incomplete; do not record an undertrained architecture as disproven.

## G30 — Cheap, trained complete-action control

Train a small shared event/action head on frozen available evidence: P0 native
association/geometry, existing image features or newly cached frozen image tokens
at actual P0 points. Do not use a failed old D20 target prediction as ground truth.
Include the same representative supported negatives and complete-action labels.
This isolates how much improvement comes from corrected learning/decision framing
before expensive spatial-temporal adaptation. New native encoder features inherit
checkpoint exposure and must be described accordingly.

Use 4096 vectorized head updates per source with the same optimizer schedule;
CPU is acceptable when faster. This control is not required to pass a score gate
before the primary raw-scene model receives its training attempt. Do not claim
its compute equals a trainable image encoder's compute.

## G40 — Main model: learn the visual transition, not just a parent score

Implement the shared-scene model in SPEC.md. It jointly observes the parent,
both proposed daughters and competing owners in ordered raw image context.
It learns whether one observed object becomes two, whether two objects already
existed, and whether ownership/continuation evidence favors a non-division graph.
This is learned image evidence, not a hand-coded split-duration or intensity rule.

Train with real source positives, random supported negative-only event groups,
supported identity confusers and coherent source-only augmentations. No new
synthetic embryo generator or mouse-pretrained model is required. Train event
and action/risk heads from update one, with a small supported identity auxiliary
loss. Do not spend half the very short fit before activating the division loss.

Per source and seed, train a shared prefix for 2048 joint updates. Save exact
model/optimizer/RNG/data-order state. Branch it into:

- `J_uniform`: 2048 more updates with the same random supported-background stream.
- `J_mined`: 2048 more updates with an added hard-negative stream from this source,
  preserving the same batch size, positive visitation, random background coverage,
  model architecture, decoder and total updates.

Thus each arm has 4096 joint updates; the first 2048 are shared and counted once
in compute accounting. The mining difference is the primary controlled test.
Use two predeclared mining refreshes, at 2048 and 3072; no target-driven retraining.
The uniform branch runs equal source diagnostics but does not train on their mined
cases. Preserve source event diversity and forbid a single large lineage from
dominating the negative replay. Do not broaden into more encoders/loss grids.

## G50 — Source validation, adequate-training diagnosis, and the one extension

Run full source graph screens over the source calibration partition, not four
convenient clips. Evaluate corrected action confidence AFTER pair maximization
and conflict resolution, including total/evaluable false forks, per-embryo future
routing, candidate availability, and lost correct links. Fit calibration on
held-group source data only; SPEC.md defines the sparse/low-count fallback.

Make one predeclared 4096-to-8192 continuation available ONLY when held-source
supported action loss is still improving by at least 1% between checkpoints 3072
and 4096, and the source full graph has not regressed versus its own 3072 checkpoint.
The schedule for that continuation is constant 0.1 times the original LR, no
warmup restart. Apply the same extended budget to both matched branches in that
direction. Record this source-only decision before target access. No second
extension or target-derived rescue. It is permissible for the two source directions
to select different durations under this same rule; compare like-for-like per direction.

If the fixed-source loss cannot overfit a tiny fully supported fixture, debug
implementation instead of enlarging the model. If fit improves but source held
performance does not, record the data/generalization limitation. If scoring is good
before decoding but fails after it, compare the same learned scores using exact
local action ownership fixtures and full graph replay; no new target knob sweep.

## G60 — Replication and false-fork replacement application

Nominate at most one main family from source evidence, with P0 fallback and no
favorable-embryo model selection. Train the second seed for the nominated family
and its matched uniform/mined control in both directions, reusing that seed's
own common prefix only. Seeds are 20260916 and 314159. Freeze choices and both
seed packages before new target comparisons.

Train on existing P0 forks as well as missing-fork proposals. Apply the same
learned model first with existing forks protected, then with a separately named
complete replacement/suppression application. It may remove false forks or move
split timing, but must own and score the full affected parent/daughter/owner
neighborhood. This is a decoder application ablation, not another neural model.
Choose its deployable application from source screens before target outcomes.
Do not run O10/A10 composition or a pipeline ensemble in this campaign.

## G70 — Full comparison and delivery

Predict both target directions before reading new target labels; score every clip
in each nominated full-199 comparison with the unchanged official metric.
Compare P0, C4_m6, cached-feature control and nominated matched image arms/replications.
Do not spend most GPU time exporting many source-failed models; retain their
complete source results and optionally one preregistered failed diagnostic only.
No score on a pilot subset qualifies as 0.95.

Require fresh image-to-P0-to-candidate reconstruction on two renamed complete
clips, annotation/old-cache/network denial and exact scored-graph parity for the
recommended package. Report detector/runtime overhead, parameter count, total
GPU memory/RSS, training compute and full inference time separately.

First report line: `P0 retained` or `Candidate recommended: <name>`; then measured
score, delta vs BOTH P0 and C4_m6, both embryo scores and seed spread, and whether
>=0.95 is achieved. Include full edge/division error transitions and rejected
hypotheses. No LB improvement is claimed without an actual submission result.

## Budget and scope control

One RTX 4090, 24 GB VRAM; 16 CPU cores; 64 GB RAM. Limits: total GPU 20 GiB,
study process-tree RSS 50 GiB, total CPU threads 16 (at most 8 loader workers or
4 scorers, with per-worker BLAS threads one; avoid nested oversubscription).
Use at most 80 GiB new cache when space permits and preserve 20 GiB disk free.
Reuse/recompute only this study's regenerable caches; do not delete old studies.

Recommended initial cap: 72 measured GPU lease-hours: 8 preparation/profiling,
48 focused training including replication, 16 inference/calibration/fresh proof.
This is a planning budget, not a completion-time promise or permission to disrupt
other jobs. CPU waits and model compute are recorded separately; charge the full
actual exclusive lease interval, never subtract inconvenient preprocessing from
resource accounting. Drop optional extensions/decoder exports before reducing
main training. If adequate training cannot fit, deliver resumable incomplete work
with measurements and exact continuation commands rather than another tiny-fit
architecture verdict. Do not consume the cap merely because it is available.
