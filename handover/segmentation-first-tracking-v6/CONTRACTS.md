# Instance, association and inference contracts

## Coordinate systems and object store

Canonical image coordinates are original `(t,z,y,x)` voxel-center indices. Typical
raw spacing is `(1.625,.40625,.40625)` micrometers, but metadata decides. Every
resampling/crop carries an explicit affine index transform and axis order. Check
round trips on landmarks and anisotropic synthetic ellipsoids BEFORE score access.
Never apply both an anisotropy correction and the same correction in the affine.
Labels resample nearest-neighbor; probability/intensity interpolation is separate.

Each instance/hypothesis record contains:

- `(dataset,t,proposal_id,detector_id,model_hash,grid_id)`; stable IDs are not inferred
  from a label integer reused in another frame. No GT IDs enter this namespace.
- Raw-grid half-open bbox `[z0,y0,x0,z1,y1,x1]`, owned cropped binary mask or lossless
  chunk/RLE reference, raw floating centroid, physical centroid, voxel count and
  physical volume. Keep intensity-weighted center separately, never overwrite one
  center meaning with another. Record clipping/tile-boundary flags.
- Physical shape covariance/eigenvalues, aspect ratios, image intensities/background
  contrast, and mask-pooled image embeddings when a model provides them. Variable
  shapes require real masks, not fixed spheres around predicted or annotated points.
- Detector confidence if defined, separate quality features and explicit missing
  values; no invented common probability calibration between different detectors.
- Source tile/transform, preprocessing hash, object support, alternative-family
  and conflict relationships, optional uncertainty and motion reliability.

Frame-level label maps are mutually disjoint selections. Alternative overlapping
masks can live in a separate hypothesis store. All arrays reference the same raw
field and time; label IDs alone are not track IDs. Cache keys include image hash,
model/transform/normalization/threshold/tiling and code hashes. Do not key by a
training filename and secretly substitute cached target graphs during inference.

One raw uint32 label movie of the typical shape is 1,677,721,600 bytes (1.5625 GiB).
Across 199 clips that is 310.9375 GiB PER detector before other maps/databases.
Stream complete clips or short windows, retain compressed sparse masks and graph
summaries, and account for simultaneous temporary maps, DBs, weights and cold run
output. A compression ratio is measured, not assumed. Reserve >=8 GiB free disk;
never delete prior study data to make this study fit. Use uint32 or checked wider
labels; do not overflow uint16 IDs in long sequences.

## C0 attachment and fair ablation

Preserve C0 node IDs AND coordinates in enrichment arms. Match points to real masks
by containment and physical distance with a bounded candidate list, not GT matching.
Ambiguous one-mask/multiple-point or one-point/multiple-mask cases remain explicit.
Do not clone one physical mask into multiple independently countable observations.
A missing mask is unknown, not proof the C0 node is false. Use a learned/calibrated
missing-evidence representation or exact original evidence fallback.

Use the same complete C0 evidence and candidate bank across point/box/mask arms.
The v5 primary-only N0/J path is not equivalent to C0. Hold decoder and permitted
edit set fixed to isolate added information. The full-detector arms intentionally
change node population, but get their own center-only versus mask-aware control.
No outcome-based per-clip switching, no target-dependent detector choice, and no
truth-assisted no-regression selection are permitted.

## Association must use the object, not just its centroid

Candidate gates combine physical motion radius and expanded bbox extent. Do not
require unwarped mask IoU >0: a real moving object may have disjoint footprints.
Compute source-mask-to-target-mask overlap after image-only rigid/flow displacement,
intersection over union, directional coverage, shape/volume/intensity changes,
mask-pooled appearance similarity and neighborhood context. Record uncertainty and
image-registration failure rather than trusting every flow vector. Keep centroid
motion as one feature, not the sole tracker. Compare point/box/mask features with
the same source-trained scorer family and candidate set.

A calibrated edge score may combine native/C0 association with these object terms.
Do not hardcode object mass constancy, equal daughter volume, constant intensity,
or strict brightness ordering. Photobleaching, boundary clipping and mitosis can
violate them. Source losses/calibration must handle unknown negative support.

A division is a temporal event with one selected parent and TWO distinct selected
children. Compare it against continuation+birth and two competing parent assignments.
Use the deformed union of daughter masks, separation, temporal appearance and soft
volume/shape consistency. A single frame splitting one object is a segmentation
alternative, not automatically a biological division. Re-merging in later frames
or a parent alternative can refute it. Never score two duplicate masks as daughters.

## Graph and hypothesis constraints

Selected node incoming degree <=1, outgoing degree <=2, directed forward in time;
ordinary exported edges link adjacent frames unless the actual official contract
explicitly permits another form. Birth/death costs are not free interior-window
slack. Test a continuation+independent birth versus a fork to prevent the v5-style
fork-heavy degenerate objective. Boundary visibility is a censored condition.

For a coarse object versus two split objects at the SAME time, conflict the coarse
hypothesis with either child. The two disjoint split children may coexist; putting
all three in a single at-most-one group is wrong. Distinguish hierarchical ancestor
exclusion from nonnested cross-detector overlaps. Bbox intersection alone is not an
exclusion constraint for adjacent cells. Tile duplicate masks require deterministic
stitching/deduplication independent of GT. Parent-child links only connect selected
hypotheses and must survive window/clip assembly.

Do not insert unsupported interpolated points to repair gaps. A gap proposal needs
image-supported intermediate object hypotheses and the permitted graph topology.
Every output centroid comes from the selected object representation or the frozen
C0 coordinate in that arm. Round at export with a documented rule, check bounds,
and freshly match/scored integer output; subvoxel internal values are not the CSV.

The existing v5 delta serializer loses coordinate-only edits under unchanged IDs.
Write full immutable graphs or explicit updated-node records. Require exact
node/edge/coordinate round-trip through NPZ/GEFF/CSV and the scorer's graph hash.
Preserve original node IDs within C0 controls; newly generated instance IDs must
be collision-free and deterministic across tile/chunk/window processing.

## Sparse supervision and held-out information

Unannotated voxels/objects/edges are UNKNOWN unless coverage proves otherwise.
Use supported positive links, censored endpoint-aware association losses, and
well-defined competing assignments inside annotation-supported neighborhoods.
For splits supervise both supported daughters, not just the easier one. Audit the
negative-sampling policy; neither dense BCE over all image background nor generic
Ultrack GT-matching defaults may silently label all absent sparse points negative.
Do not train a node filter to delete unmatched proposals solely from sparse GEFF.

Detector fine-tuning needs real dense-mask coverage, or explicitly weak supervision:
trusted pseudo-mask cores/boundaries with uncertainty regions ignored, consistency
across image augmentations, and no target-label-derived masks. Synthetic centers
and externally generated shapes are separate synthetic evidence, not measured cell
boundaries. False pseudo masks need a quality/abstention mechanism. No ground-truth
count, estimated_total, GEFF attributes, evaluation match or future test information
may enter deployed detections, thresholds, likelihoods or solver costs.

## Execution and acceptance fixtures

Test transforms/axes/anisotropy, bbox off-by-one, mask ownership, same bbox/different
mask, moving nonoverlap, tile and temporal seams, touching cells, coarse/split
conflicts, undersegmentation/remerging versus actual division, births/boundary
censoring, missing masks, graph integer serialization and same-ID moved centers.
Test FOCUS key mismatches/blocked assets and each dependency's actual API in isolated
processes. No silent pass/empty-mask fallback after a model or solver exception.

Held-out fresh inference must run selected detector(s), mask processing, association,
solver and export from images and authorized packaged weights only. Deny GT/GEFF
input, evaluation directories, previously generated predictions and network before
imports; audit allowed files and fail on unexpected access. Dependencies may have
legitimate pinned local caches, but no broad allowlist that permits target answers.
A deterministic failed backend policy, if shipped, must be fixed before evaluation
and timed including its cost. Never manually select C0 only on known failing clips.
