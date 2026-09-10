# Implementation and model contracts

Create additive `tools/multidata_training_v4/`, tests, and a root-independent
`scripts/run_multidata_training_v4.sh`. These training modules are to implement
locally; the handover only supplies tested reference contracts and path preflight.
Never redirect v1/v2/v3 hard-coded output globals into a new study and assume it
is safe. Use explicit input/output roots and read-only access to old caches.

## Three compatible learned components

**G: geometry encoder.** Start with a small relative-coordinate set/temporal
encoder (roughly 1-5M parameters) and link, no-link and unordered daughter-pair
heads. Point/time windows, candidate masks and dimensionless geometry are its
complete transferable inputs. Train with synthetic truth and eligible Zoo weak
tracks. On Biohub, additional native scores may enter a separate zero-initialized
residual head; they are never fabricated for Zoo. Initialize that head identically
in every arm. Geometry-only inference must run with image/logit features absent.

**I: image event model.** Use a shared frame encoder plus masked temporal pooling
or attention rather than flattening a fixed nine-frame tensor. Start with the
v3 parent/context crop support, but add daughter-centered evidence; the two
daughter branches share weights and combine symmetrically. Six- and nine-frame
windows use explicit validity and relative frame-offset inputs. A six-frame clip
provides at most six observations, regardless of padding. A learned geometry
representation from G can fuse with image features. No optical appearance is
learned directly from image-free Zoo. Detector/heatmap weights may initialize the
image encoder only where parameter shape/preprocessing compatibility is verified.

**D: center detector/encoder.** Adapt the existing local TemporalUNet3D/DeepCenter
training interface if compatible with the pinned checkpoint and source. Use
native static synthetic images and source Biohub sparse centers, with a separate
sequence-grid adapter. Compare real-only versus synthetic pretraining with the
same architecture/initial checkpoint. This is an actual weight-training experiment,
not just using the old detector on more inputs. First evaluate centers/embeddings;
then integrate changed detections on a separately rebuilt tracking graph. Do not
silently merge two candidate populations or reuse old residual features/IDs.

Reuse inspected `scripts/train_unet_transformer.py` as a reference for temporal
U-Net + node transformer, not an unexamined executable import. Its sparse active
row/column edge mask and source-axis softmax preserve the possibility of two
outgoing daughters. Do not replace it with source-normalized successor softmax
that forces exactly one child. Add an explicit no-parent/null alternative for
births/unmatched detections when needed. Static data must not be fabricated into
motion sequences merely to satisfy a tensor interface.

## Targets and losses

Every training record carries `label_kind`, `task_mask`, `frame_valid`,
`future_observed`, `candidate_visibility`, `source_id`, `provenance_group`,
`image_representation`, `grid`, `voxel_spacing` and `spacing_verified`.

Dense simulated detection: center heatmaps/subvoxel localization, not instance
masks. Dense simulated edges: candidate positives and genuine alternative-parent
negatives; preserve crop visibility and unmatched/distractor handling. Dense
simulated fork targets: complete-future parents only, distinguishing no-fork from
unobserved future. Weak Zoo: known continuity/forks can supervise weighted teacher
targets; unknown absence of a recorded fork is not universally confirmed nondivision.
Mouse contributes no event-class loss. Real Biohub: positive centers and supported
positive/contradictory edges/events only; unknown objects are masked, not negative.

For real detection, do not optimize only positive voxels (all-positive collapse).
Retain supervised dense-synthetic negatives during adaptation; use image-validated
source background or conservative teacher-consistency as separate weighted terms.
Keep uncertain teacher absences masked. Teacher confidence is not ground truth.

A parent event bag has a no-fork hypothesis and bounded pair/path hypotheses.
Optimize bag-normalized rank/classification losses, with one unit of weight per
observed event rather than per alternative. On dense data the target includes the
correct pair (or no-fork); measure candidate coverage and mask unreachable targets
rather than inserting a GT-guided candidate in deployable validation. For weak or
partial real labels, compare only the alternatives the annotation can distinguish.
Predicted geometry/features come from the corrupted/detected input graph. Target
adjacency is never a feature. Correct clean-GT input is allowed only as a named
teacher-forcing pretraining control; the primary transfer model sees realistic
observations and label-blind candidates.

For sampled losses, log population counts, exact sampling procedure and the
objective actually optimized. A balanced event objective is acceptable for
representation/ranking, but its sigmoid is not a deployment posterior. Do not
use the v3 parent-group sampling fraction as row inverse-probability weight.
Implement proper stratum probabilities or keep the loss explicitly group-balanced.

## Bounded proposals and decoding

Keep the v3 incumbent topology for initial scoring. Limit primary event generation
to six daughter candidates and at most 32 pair/path alternatives per parent-time
anchor, plus no-op; stream candidates rather than materializing all combinations.
Use a cheap learned eventness gate trained on external nondivision examples,
then a richer pair scorer. Record coverage after EACH pruning stage, all false
positive events, and proposals per million parent observations. Candidate caps
are frozen across training-data arms and must not depend on target annotations.
Do not prune to top events using a GT-derived expected total.

The original raw ILP is a negative control: second-edge reward <=1, division cost
1.2 and free birth suppress all forks. Add a calibrated/rank-based learned event
term or implement a unified continuation/fork/no-op local objective. Test this
on the real solver with positive and negative toy graphs; a penalty change alone
is not a model gain. Keep decoder/objective choices IDENTICAL across the external
and real-only controls. Freeze the incumbent outside bounded edited components;
include displaced owners and protect full daughter evidence windows, not just
immediate fork edges. New-event selection must replace or reconcile overlapping
safe-division proposals, not add duplicate events in two pipeline stages.

Prediction scores after class balancing/domain mixing require new calibration.
Use held-out external validation and source-only supported event groups for score
scaling/margin selection. The latter is conditional on annotation support, not
all-cell biological prevalence. If source blocks cannot be made independent,
use the fixed prespecified margin protocol and label source diagnostics as reused;
never use target outcomes for calibration. Carry no-fork priors as sensitivity
parameters, not the forum's 15.7x correction. Always include abstention/no-op.

## Code, artifacts and tests

Modules: `sources`, `index`, `adapters`, `corruptions`, `sampling`, `models`,
`losses`, `train`, `calibrate`, `proposals`, `decode`, `infer`, `evaluate`, `report`.
A runtime dataset index stores canonical group/partition, source hashes, task
availability and source-use status. Never mix license eligibility with biological
label quality: they are separate fields. Weight manifests list direct and inherited
training/calibration sources, including 44b6-dependent simulation and old teachers.

Required local tests beyond the 40 authoring tests: actual NPZ/native-image overlay
parity; source scale and dtype checks; direct edge/ID conversion on each eligible
Zoo source; duplicate-alias split rejection; temporal/visibility censoring after
corruption; no GT input fields; daughter permutation invariance; padding invariance;
gradient flow into external-trained encoders; tiny dense-set overfit; source-only
training reads; actual ILP fork/no-fork fixtures; zero-weight head identity parity;
complete sample sets and official aggregation parity; GT-unavailable inference;
fresh-image run and submission-schema validation. The reference helpers do not
replace any of these real-data or official-integration tests.

Use NumPy-only prepared arrays in the training runtime where possible. Existing
Zarr-v2/HDF5 preparation dependencies stay isolated from the working Zarr-v3/
PyTorch runtime. Locate the GPU environment rather than assuming `/root` or
`/home/mpf`; use explicit interpreter arguments. No broad reinstall or driver
change. Stream source files; bound decompression workers, cache by content and
transform hash, and reserve GPU headroom. Log actual steps, parameter counts,
unique source samples/event groups, optimization loss, wall time, peak VRAM/RAM
and data-loader time. Loss-only success is not transfer success.
