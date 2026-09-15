# Improve the largest pipeline errors through learning

## 1. Decision

Keep the public TemporalUNet-based detector, the incumbent observations and P0's
strong ordinary tracking. Replace or adapt ONE local decision module at a time.
Start with **division versus continuation plus birth/another owner**. Then learn
**continuation identity** and **which existing raw observation should survive**.
Do not begin with full detector retraining, a new segmenter, wholesale relinking,
a global threshold sweep, arbitrary deduplication or more synthetic data alone.

The baseline is P0 = 0.934864986413134. C4_m6 = 0.935178370257 is a required
challenger, not the adopted parent. Selected v3/C0 = 0.934802374260586 supplies
an additional construction control. This study aims for a reproducible material
local improvement, with 0.95 a stretch result rather than an expected outcome.
See EVIDENCE.md for numbers, previous failures and the source code inspected.

The new contribution must be better training or a narrowly adapted module.
Candidate tracing, a zero-head identity decoder and calibration are supporting
controls, not sufficient final deliverables.

## 2. What remains fixed

For division and association experiments, freeze node IDs, timestamps, integer
coordinates and counts; the detector weights, preprocessing, peak selection,
ensemble, ILP/reconstruction and baseline P0 parameters; and the pinned official
scorer. Never infer a source-model choice from a target filename at deployment.
Pass model packages explicitly. Edits outside the selected local conflict
components must be exactly absent.

Observation experiments alone may change the selected observations, using a
frozen union of P0 and existing raw candidates. They must rebuild features and
candidate associations at the actual new coordinates. They do not add Cellpose,
FOCUS, a new heatmap network or hand-written detection threshold variants.

Preserve all original code, artifacts and selected-policy locks. Implement a new
namespace rather than silently modifying historical v3/v4/v6 behavior.

## 3. Execution order

### S00 — Recover and reproduce the actual parent

Read root AGENTS.md, relevant competition skills and the code-only planning
handover. Check the current checkout, paths, installed runtimes, actual GPU/CPU/RAM,
disk capacity, active jobs and artifact hashes. Historical absolute paths are
hints, not proof of existence. Resolve P0/C4_m6 through
`tools/center_comparison/best_predictions.py`, existing receipts and manifests.
Recover missing generated files using existing documented inference; never
replace missing results with a weaker convenient pilot.

Freshly score complete P0, C4_m6 and C0 graphs on all 199 clips using the pinned
metric. Check per-clip as well as pooled counts and exact node identity. A source
hash or graph mismatch is a correctness blocker until understood. Unavailable
artifacts block only the dependent lane; finish independent implementation/tests.

Create an immutable input manifest and new work/output roots. Inspect, but do
not resume or terminate, the source-only native fits described in
`docs/incumbent-comparison-20260914.md`. Their old progress files are not live
process evidence.

### S10 — Trace the bottleneck and establish feasible local edits

Implement tracing in the EXACT P0 and C4_m6 paths, not only a surrogate candidate
builder. For every audited division and edge, separate: observation availability,
anchor activation, candidate bank, geometric rejection, score, calibration/gate,
owner/protection rule, conflict/solver result, and official final-window result.
For each missed division test every allowed local anchor; do not classify an
exact-frame failure as complete window failure. Store stable IDs and reason codes.

The historical all-199 diagnosis is exploratory. Do not use new target-embryo
trace outcomes to choose architecture, candidate limits or acceptance parameters.
Build the proposed bank with fixed prediction-only rules first. Source-only
trace/counterfactual results may drive the prespecified lane eligibility checks;
new target diagnostics are post-freeze reporting.

Reuse the v3 forward/reverse union and alternative paths as the initial bank,
with its pinned defaults. Preserve every incumbent edge and fork. Do not reapply
v4's learned 0.5 geometry gate as a hard front-end exclusion. Score complete
alternatives downstream instead. This is a declared structural difference, not
a gate sweep. Controls get the same bank. No GT-injected proposals may enter
training deployment banks or inference.

Use lazy pair/path enumeration, cached per-node embeddings and bounded conflict
components. Do not allocate the old 190-million-alternative table. Any resource
pruning/abstention must be deterministic, shared by controls, logged and included
in candidate-coverage denominators. Test spill/streaming equivalence.

Run source-label-only feasible-edit probes on current P0: divisions, then
continuations, then observation selection. Actually serialize and rescore
repaired graphs; do not substitute count arithmetic. These are heuristic
feasibility witnesses, not upper bounds, trained results or model inputs.
First run a fixed-bank probe; an explicit GT-injected observation diagnostic may
be reported separately but never qualify a deployable lane.

Do not terminate the study because one bank has poor coverage. If source tracing
shows the supported repair is outside that bank, record it and progress to the
independent observation/identity lane. No radius/top-k search follows a bad score.

### D00 — Shared local decision/decoder contract

Implement complete alternatives: keep P0; continuation to a daughter with the
other daughter born independently; two continuations owned by distinct parents;
a true parent-to-two-daughters fork; and, only in the declared replacement arm,
existing-fork-to-continuation or fork timing/pair replacement. Include displaced
owners, their legal alternate continuations and termination/birth consequences.

Reuse v3 Action/solve_actions semantics. Do not apply a positive mitosis bonus to
a second edge while ignoring the competing graph. No-op must always be feasible
and win exact ties. A zero residual must produce P0 exactly, even with a larger
bank. Retain the ordinary tracker outside the edited components.

Initially add/redirect previously absent forks without editing an existing fork's
local evidence. Then test a separately named full event-replacement application
of the SAME learned scores, which can repair false forks. Count that as a decoder
ablation; never attribute its entire delta to representation learning. See the
full local-window protection and ownership tests in IMPLEMENTATION.md.

### D10 — First practical transfer: adapt OrganoidTracker2's division module

Use the already integrated, pinned, corrected v3 OrganoidTracker division model
at the actual incumbent centers. Keep P0 observations and normal association;
do not call OrganoidTracker's candidate linker, graph cleanup or whole tracker.
Run its official patch/preprocessing/scaling parity checks first. Its parent
mitosis score is evidence for a local pair decision, not sufficient daughter
identity evidence on its own.

Train the same small symmetric daughter/owner competition head with two versions:
`D10_frozen` uses a frozen pretrained image backbone; `D10_adapted` fine-tunes the
last image block and head using supported SOURCE-embryo events plus abundant
source continuation/identity examples. Include feature masks and native P0
association evidence. Never treat the source-only head as cleaning P0 exposure.

Use raw and released-calibrated scores as logged inputs, but fit Biohub decision
calibration only on source groups. No imported mouse event-prevalence prior is
assumed correct. Save a same-architecture random-initialization control
`D10_random` when this family is source-qualified; this is required before
attributing an advantage specifically to pretrained weights. Frozen versus
adapted alone establishes adaptation benefit, not the isolated effect of pretraining.

### D20 — Train a temporal event/identity representation rather than a tiny event classifier

This is the second substantive route, not a reuse of C4 logits. Train a small
native-resolution 3D crop encoder, retaining ordered seven-frame context,
separate parent and daughter tracklet features, relative physical geometry and
competing owner evidence. Use a shared frame encoder and lightweight temporal
attention/GRU, not a full-volume network or giant model. Crop and memory contracts
are specified in IMPLEMENTATION.md.

Exploit the many supported continuations to learn identity BEFORE fitting on
rare division events. The primary recipe uses 8000 continuation/contrastive
updates followed by 8000 joint event/identity updates. Sample event groups, not
millions of near-identical pair rows. All known compatible time/pair alternatives
are positive as a set; unsupported branches remain masked.

Three matched arms use identical source rows, candidate banks, decoder and total
update count:

| Arm | Representation/training | Question |
|---|---|---|
| D20_compact | v4-sized 3-frame/12x12 image tower, newly trained under the NEW objective | Does the new decision objective help without new image capacity? |
| D20_temporal | native 3D/ordered 7-frame tower, continuation pretraining then joint training | Does richer learned temporal identity repair divisions without extra FP? |
| D20_no_pretrain | same temporal architecture; 8000 supervised warm-up + 8000 joint updates | Does continuation representation pretraining help versus equal update count? |

Model widths, loss weights, augmentations, optimizer and final selection rule are
fixed in the execution lock before target evaluation. Equal optimizer steps are
not equal FLOPs: measure both, report them separately, and make no equal-compute
claim without evidence. Do not repeat a synthetic-only or geometry-only run as
the main new proposal. The existing synthetic release is not in primary training.

### A10 — Independent continuation-only identity training

Run this even if division training fails. Learn a local incoming-parent/tracklet
ranking head on the many supported real continuations, using source-only temporal
features and hard competing annotated identities. Keep P0 points fixed. Compare
against P0's existing 38-feature logistic residual on the same candidates and
bounded association decoder, with the inherited margin/edit cap unchanged.

The new prediction is a calibrated association residual at current coordinates,
not a replacement from-scratch tracker. Cover two-sided swaps/displaced owners;
score all edges removed and added. Preserve existing fork evidence over the
full official local support window, not only the two immediate fork edges.
Report candidate-available FN recovered and unmatched-competing-center FP changed.

Do not infer success from pair AP or edge accuracy: replay full graphs. Do not
rebuild all continuations as C1_relinked/C4_relinked did.

### O10 — Independent learned observation selection on a closed raw bank

Run after the fixed-node lanes, not only after a division success. Freeze a
prediction-only union of P0 observations and the original raw proposals. Keep
stable proposal provenance; nearby does not mean identical. Train a temporal
selector/ranker on source-supported tracklet hypotheses, with separate states
for keep, substitute and retain both. One-to-one candidate assignment and complete
incoming/outgoing edges belong to the decision; no global deduplication radius.

First test `O10_swap`: one-for-one selection changes, exact total node-count
preservation, new features at chosen coordinates. This isolates identity/selection
from simply adding cells. Then test `O10_restore`: allow a previously discarded raw
trajectory to survive alongside the original when learned image/temporal evidence
supports two objects. It may change counts and must pay the full official count
adjustment. Both use the same frozen bank and source-trained selector.

Training uses supported GT trajectories, known different annotated identities,
source-synthetic duplicate/jitter corruption with explicit identity provenance,
and masked consistency. Unmatched real whole-field proposals are not blanket
negatives. Positive-label sampling may identify training groups but never generate
inference anchors at GT positions. Avoid harmful teacher distillation on the
exact disputed ownership/selection decisions being repaired.

No unconditional union, count inflation, coordinate-only tuning or Cellpose
augmentation arm is authorized. When centers/nodes change, all affected native
features, proposal links, ownership and official assignments are recomputed.
The 670 raw-proposal-limited annotations are deliberately not the first detector
retraining target of this study.

### C10/R10 — Replication and limited composition

Use source-only grouped validation to choose at most one division family and one
identity/selection family for replication and composition. Selection occurs
before new target results. Retrain BOTH source directions with seed 314159,
keeping the primary seed 20260915 and all recipes. Do not choose the lucky seed.
Deterministic frozen-model controls get exact repeat tests rather than a fake
second training seed.

The single composition is the source-selected event module plus the source-selected
identity/observation module, applied with fresh evidence after any point changes.
Do not search combinations or reorder stages after target scores. Independently
score each module and the composition against P0 and C4_m6. A failed component
must not be concealed inside a blend or ensemble. No ensembling is needed for
this study; two-seed comparisons are reliability tests, not ensemble members.

## 4. Resource envelope and bounded execution

Use the existing compatible Python runtimes, not an in-place upgrade of shared
environments. Keras/Organoid imports may be isolated in their existing runtime.
Use one GPU training/inference process, AMP where checked, and no GPU contention.
Default limits: GPU total 20 GiB; process-tree RSS 50 GiB; 8 data workers; at most
4 scoring workers; 16 total CPU threads. Do not run the full loader and scoring
worker allocation concurrently if it exceeds 16 cores. Limit BLAS/OpenMP inside
workers to one thread. Use memmapped/sharded features, not giant RAM arrays.

Profile one full crowded clip and a training step before scaling. Lower only
microbatch size and increase accumulation to preserve effective batch 32 when
needed; no data-driven architecture change may be disguised as OOM handling.
Native crops are streamed, embeddings cached per unique node/frame/model, event
pairs scored in chunks, and raw frame caches kept bounded. Preserve output hashes
across cache reuse. Preserve input data and other studies' files.

The first campaign has a **48 measured GPU-hour planning cap**, not a finish-time
promise: 4 for preflight/adapters/pilot inference, 24 for first-seed training,
12 reserved for replications, 8 for final inference/package checks. Inspect pilot
throughput and write a source-only execution lock before production. If the
registered update counts do not fit, choose one common lower budget for matched
arms BEFORE target access and retain the replication reservation. Do not shorten
only an inconvenient control. Log actual completed updates and unfinished arms.

Use at most 80 GiB of NEW persistent cache when available, always leaving at least
20 GiB free; inspect current space rather than assuming historical free space.
Stream/recompute within this study when storage is tight. Do not put persistent
checkpoints in /dev/shm or delete old studies to make room. Resource blockers
are explicit, not permission to substitute unmeasured evidence.

## 5. Deliverables

New source namespace and tests; immutable data/exposure/split/model/candidate
manifests; source training and calibration histories; stage attribution;
actual feasible-edit probes; all-199 full-score tables; per-embryo counts;
error transitions by GT identity; two-seed comparisons; unchanged-region checks;
resource and Kaggle-runtime measurements; and an annotation-denied fresh-image
inference package for any recommended candidate.

Write results/pipeline-error-training-20260915/REPORT.md with the first line
`P0 retained` or `Candidate recommended: <id>` and the actual delta versus BOTH
P0 and C4_m6. Include failed/blocked/unrun rows and CONTINUATION.md. Recommendation
does not change the production default. See VALIDATION.md for promotion rules.
