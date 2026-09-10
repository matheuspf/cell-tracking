# P500–P580: pretrained full-lineage reconstruction

## Objective and decisions

Primary target: official local score >=0.95 on the same 199 clips. Incumbent:
0.934802374260586, with embryo scores 0.931664468721842 / 0.935221784097327.
New score gains must be relative to this incumbent. Keep it immutable.

Hypotheses:
H1: pretrained region/edge representations select associations and divisions
better than the scratch-trained v4 event towers on this small real dataset.
H2: new candidate centers and competing instance supports recover truth that
fixed-node repair cannot, without excessive false links or count inflation.
H3: a complete lineage objective uses these alternatives better than gated,
add-only edits that largely abstain. These are separate tests, not assumed gains.

Initial cap: 24 complete new graph configurations (identity is not counted),
36 GPU-hours including pilots/adaptation/new inference, one RTX 4090. Resource
and scientific caps are separate. The old seven-hour deadline is not inherited.
Use early engineering tests and source-only diagnostics; do not burn the whole
budget on infrastructure, all-model grids, or 190 million pair combinations.

## P500 — Restore and freeze the actual incumbent

Read completed v4 continuation, source receipts and AGENTS.md. Discover explicit
input/output/interpreter paths; verify required model/cache hashes without
redownloading the input collection. Read current official metric/rules snapshots
and compare the installed evaluator with the pinned one. Keep comparisons separate
if there is verified metric drift. Reproduce the full C0 score, all sample names,
per-embryo counts and strict CSV contract before reporting any new gain.

Freeze a manifest of expected samples, target/source directions and v5 configuration
before training. Reused targets remain exploratory. Use 12 image-only pilot clips,
six per embryo, spread over incumbent density and depth/SNR; select by fixed seed
without event labels. They are engineering/development clips, not an independent
holdout or substitute for the full population. Reserve four other image-only clips
for runtime/serialization checks. Read detailed prior censuses as development
information; do not relabel these embryos unseen.

The common primary runtime is the existing study environment. Create small isolated
third-party environments only when necessary; preserve torch/CUDA in the incumbent.
Implement explicit worker bridges using NPZ/Parquet, not imports that overwrite
shared packages. Test scientific unit/graph contracts before model benchmarking.

## P510 — Native evidence and an actual region interface

Build two label-free input populations:
A0: exactly the incumbent node IDs/coordinates. Build foreground-limited, native-
image seeded watershed regions with one distinct seed per represented node.
These are proxy instance supports, not biological segmentation GT. Keep a mask-
label to candidate-ID map. A region centroid must not silently replace the original
point. Evaluate fixed-center outputs to isolate linking from localization.

A1: a genuinely new candidate population from full-frame native DeepCenter and
available U-Net heatmaps, not v4's center-query head. Use full-frame or tiled scans,
physical-space peaks, image foreground and duplicate exclusion. Include an image-
watershed partition and a broader split hypothesis where distinct maxima support
it. Preserve incumbent and raw detections as provenance-bearing alternatives;
the union is NOT automatically all selected output nodes. Crop halos, coordinates,
resolution changes and duplicate candidates must be explicit.

Export available native transition logits BEFORE probability threshold/top-k
pruning and BEFORE ILP. Confirm the actual patched predictor's softmax direction,
normalization, threshold, window fusion and ID mapping from its source. Do not
pretend old pre-ILP caches contain scores discarded by the old export. Recompute
neural windows where necessary; bound candidate edges spatially, not by hidden GT
or the evaluator's 7-um matching tolerance. Use 8 spatial neighbors in each direction,
source-image-supported alternatives and all incumbent edges, with window context
5/7 frames. Default final candidate links are consecutive-frame only.

Measure on source development data: GT center recall, localization, duplicate
competition, correct-edge coverage and division-window coverage at A0 versus A1.
Include candidate counts and timing. Event oracles are evaluation-only heuristic
feasibility diagnostics. A low fixed-node ceiling triggers A1 rather than ending
all work. No frame/sample may disappear because there are no proposals.

## P520 — Two genuinely different pretrained linkers

Implement and execute BOTH bounded pilot routes:
- HOCT `general_v1`, pinned in model_registry.json, with images plus instance
  supports through implemented create_graph/predict. Its point-only helper is a
  stub and must not be called. Explicit max_delta_t=1 for competition edges;
  choose physical candidate radius from source motion/context, not the default
  300 or the scoring tolerance. Keep pretrained region-feature conventions.
- Trackastra `ctc` (supports 3D), not general_2d or the SAM2 2D model. Use its
  mask/image route, inspect anisotropy/coordinate scaling in the actual version,
  and compare division-enabled decoding. Assert final edges are consecutive.

Run upstream synthetic/small example checks first, then A0 and A1 pilot tracking.
Keep raw learned edge scores, birth/orphan evidence and features before decoding.
Test whole direct tracking solutions, not only tiny edits around the incumbent.
If proxy regions hurt, compare real image-derived A1 regions before concluding
that pretrained features fail. Do not use zero-filled region descriptors.

Advance both working routes to a full 199-clip baseline if within budget. A broken
adapter blocks only that route: allow one bounded dependency/API repair pass and
continue the other and P550. Mark missing model/provenance/runtime accurately.
Do not manufacture a zero-shot score from a subset or tutorial benchmark.

## P530 — Adapt the new representation, not another event MLP

For every viable new linker, compare unchanged pretrained weights with a source-
only sparse adaptation. HOCT's documented sparse edge probe is an initial route;
set hard ILP consistency on unknown edges OFF in the primary arm. Otherwise the
old errors and zero-fork solution can become fake ground truth. A soft frozen-
model distillation term is a named optional control, never GT.

For the most viable trainable backbone, also adapt at least the final transformer
blocks, not only the final scalar threshold. Trackastra has maintained training
support. HOCT may expose only scripted inference modules: require a verified
trainable parameter/gradient path before claiming backbone adaptation. A valid
head-only fit is reported as such, not full fine-tuning. Use a controlled native
model fork if needed; do not promise absent upstream training APIs.

Training inputs are P510 prediction-derived candidates/regions with source-only
matching labels. Known incoming predecessors support competition among source
candidates. One recorded child does not label every other outgoing edge negative;
missing parents or final frames do not establish birth/death truth. Respect the
unknown masks in IMPLEMENTATION.md. Include all source positives across windows
and hard supported negatives. Deduplicate or reweight repeated windows/groups.
Do not discard rare divisions via uniform negative sampling.

Initial optimizer recipe: backbone AdamW 1e-5, new heads 1e-4, mixed precision
only after parity checks, gradient norm cap 1, 8,000 effective updates per source
with fixed recorded checkpoints 0/2k/8k. Use source-only independent groups for
selection only if actually certified; otherwise fixed final checkpoint is primary.
Keep a zero-shot control and a source-only head adaptation control. Log changed
parameters, feature loss, actual positive/negative exposures and source fit
saturation, not only update totals. A source support failure triggers sampling/
loss repair and a new config hash, not more epochs on a broken target.

The downloaded synthetic sequences may supply a small dense-supervision control
(up to 20% of adapted batches) only after verifying mask/feature compatibility.
Zoo graphs without real images do not train region descriptors. Do not repeat v4's
multi-dataset factorial. Real pretrained morphology/association transfer is the
new experiment; preserve external pretrained-data provenance and weak-label limits.

## P540 — Genuine image/detection diversity

Evaluate A1 with the adapted linkers, and separate detection-only changes under
an identical decoder from full linking changes. Run full-volume DeepCenter
proposal extraction even if offset refinement failed in v4: those are different
hypotheses. Measure new matched centers, newly recoverable daughter windows,
point competition and actual graph score, not only center recall.

Optional independent masks: use FOCUS nuclei weights only when already authorized
and locally available, or lawfully accessible without accepting new conditions.
Benchmark one cached predictor; do not reload it every frame. If full inference
would violate the measured runtime budget, test it offline as an actual teacher
on source images, not in hosted inference. No FOCUS availability is required for
P510–P580. Dense teacher outputs remain noisy labels with uncertainty masks.

If full-frame learned evidence still misses source daughters, allow one meaningful
fine-tune of the EXISTING full-frame U-Net/DeepCenter detector using source sparse
positive centers plus validated dense synthetic/teacher targets. This must scan
whole images to create new centers at inference; a center-offset MLP does not
satisfy this stage. Use sparse masks, fixed teacher safeguards and native-grid
supervision; unknown voxels are not confirmed background. This conditional fit
has at most 8 GPU-hours within the overall cap, and must be compared to unchanged
weights using the same linker. Do not perform an unfocused detector architecture
search. If the source census finds little detection headroom, prioritize P550.

## P550 — Joint full-lineage decoding before repairs

Build a common sparse candidate graph from A0/A1 and exported native/pretrained
scores. Compare direct HOCT/Trackastra decoding and a joint objective with explicit
node selection, continuation, two-child division, birth, death and conflicting-
instance choices. Use model likelihood costs; do not add independent probability
rewards to an impossible fixed fork penalty. No merges; at most two successors;
all final links consecutive. A fork must remain a selectable state in exact tiny
fixtures, and no-fork must remain feasible. Do not import v4's 0.5 hard gate and
margin 4 as universal calibration.

Score both daughters jointly with source-trained or pretrained evidence and the
costs of displaced owners. Link consistency across 5–7-frame windows and instance
conflicts belongs inside the optimization, not independent overlapping edits.
Aggregate window evidence with normalized weights and merge exact alternative
identities before normalization; duplicate windows must not multiply certainty.
Stitch with overlap constraints so the same cell cannot acquire two parents.

Run two new-evidence arms: new-linker-only and a calibrated soft combination of
native neural plus new-linker scores. The incumbent is a soft prior or complete
fallback, not a set of untouchable associations that makes change impossible.
First establish nonzero useful differences, then use confidence/solver fallback
only for invalid, timed-out or explicitly uncertain components. Log all fallback
reasons and rates; near-total fallback is not a successful model experiment.

For any path-edit arm retaining old divisions, protect or jointly replace the
complete local evidence window used by the official division scorer, not just
immediate fork edges. A full replacement tracker need not preserve each old
predicted fork; it must be scored for net graph/division performance.

The inexpensive native-only raw-graph redecoding control is mandatory: use fresh
pre-threshold neural evidence and the same new global objective, followed by an
explicit selected cleanup policy. Do not silently run all old 27 helpers on a new
tracker; that can erase divisions and invalidate attribution. A raw new graph,
a minimally validated graph and one documented smoothing/cleanup variant are
separate, budgeted conditions. Ground-truth-guided local choices are forbidden.

## P560 — Freeze and evaluate the bounded comparisons

Suggested slots (maximum 24 complete new configurations): zero-shot HOCT and
Trackastra on A0/A1 (4); their source-adapted counterparts (4); native-only fresh
redecode A0/A1 (2); new/native joint evidence on A0/A1 (2); two pretrained versus
backbone-adapted controls (2); at most two real detection/teacher changes (2);
second-seed repeats of at most two promising learned families on A0/A1 (4);
at most four declared fallback/cleanup/combination controls (4).

Reuse score-equivalent identities but do not count an abstention as a new trial.
Lock the actual executable slot table and both source directions before new
comparative outer scoring. Source-only stages may allocate unused conditional
slots. Choices made after outer revelation are a separate exploratory block,
not the original primary test; maintain experiment lineage and reserve replication.
A zero-shot deterministic model needs repeatability, not a fabricated training seed.

Every complete variant gets fresh official matching/division assignment on all
199 samples, exact denominator weighting and per-embryo results. Partial pilots
are explicitly partial and cannot replace missing samples with old scored rows.
Include raw edge Jaccard, adjusted edge contribution, divisions, node counts,
GT-edge identity gains/losses, fork-window regret and full inference cost.
A model may exceed 1 under the published metric; do not clip score to one.

## P570 — Promotion and target status

Use VALIDATION.md. Target reached requires pooled >=0.95, no embryo regression
against incumbent within 1e-10, real fresh-image/package parity, and independent
second-seed confirmation for learned selected additions. No hidden-score claim.
Smaller qualifying gains are saved as `improved_below_target`, not thrown away.
If the threshold is reached only in one seed, retain it as promising/unreplicated.

Do not impose a per-edge never-change rule: new complete trackers will trade
some correct and incorrect links. Measure aggregate improvement, division regret,
per-embryo safety, and deployment correctness. Do not lower the 0.95 target or
change denominator weighting to make an experimental result look successful.

If no replacement qualifies, keep C0 and conclude which limitation remains:
region interface, new-node recall, score quality, calibration, objective, runtime,
or domain transfer. Report conditional oracle headroom separately. Finish real
experiments and the next-agent handover rather than returning only a new proposal.

## P580 — Actual image-to-graph delivery

Package one runner with explicit --images/--output/--models/--source-model and
no dependence on training labels, caches named after train clips, evaluation
metadata or external training datasets. Verify all scored cached graph outputs,
then at least six representative full image-to-graph runs and an unfamiliar clip
name. These are deployment checks, not new biological samples. Evaluate runtime
on the actual CPU/GPU allocation including decoding and mask extraction.

Report full model/download/feature-cache/segmentation/decoder costs separately.
Provide the offline dashboard, final report, score CSVs, model and source manifests,
selected config, archive checks, failures, stage ledger and NEXT_AGENT.md.
Commit only code and sanitized aggregates to this v5 branch; no PR merge or Kaggle
submission. Preserve prior artifacts and explicitly list local dependencies.
