# Implementation contracts

The new production package is to be implemented locally under
`tools/instance_tracking_v6/`, with an absolute-path-safe wrapper
`scripts/run_instance_tracking_v6.sh`. Neither exists in this handover yet.
Keep the original v5 code/results unchanged; reuse functions only through explicit
inputs/outputs rather than changing module-global old study roots.

## Modules

- `preflight/inventory`: active process/worktree state, current baseline and sample
  set, source pins/weights, marker, metadata, resources and environment receipts.
- `segmenters/focus`, `segmenters/cellpose`, optional `segmenters/stardist`:
  actual version-tested inference in isolated runtimes; native transform receipt;
  no GT input; one persistent model per GPU worker; cached atomic per-frame results.
- `instances`: mask store, frame-scoped IDs, bbox, center/medoid, voxel and physical
  feature schemas, frame/model alignment, collisions, border truncation and mapping
  to fixed C0 points without labels.
- `motion`: common image-derived forward/backward fields and interpolation contract;
  observed/raw support overlap versus registered overlap; no pairwise perfect alignment.
- `features`: point/box/mask/appearance and union-of-daughters costs, masks remaining
  accessible for overlap calculations. Missing descriptors flagged, not replaced by
  confident zero or synthetic spheres.
- `hybrid`: source-only feature-model fits and C0-preserving no-mask fallback.
  Same nodes/candidates/decoder across point-box-mask comparisons. Truth labels
  live under evaluation, never on serialized inference instances.
- `ultrack_adapter`: proper foreground/contours, one sqlite/database path per clip
  and recipe, native hypothesis exclusions, tested flow convention, supported
  custom edge insertion, selected-mask and node-edge export.
- `evaluate`: current pinned scorer, fresh graph copies, exact all-sample checks,
  unchanged reference estimates, direct GT-truth identity accounting.
- `student` (conditional): real foreground/boundary model, teacher-mask ignore
  regions, source-only input provenance, measured model/gradient/segmentation outputs.
- `viewer/report/package`: local offline masks/boxes/points playback, scores and
  cold image-to-graph command with no labels or old predictions.

## Mask table and artifact identities

The primary key is `(dataset, frame, producer, local_instance_id)`. A label integer
may be reused in every frame and by every model and MUST NOT denote a track ID.
Training examples and mask encoders must not take arbitrary label IDs as features.

Every instance record includes:
`key`, `mask_path`/compressed support, `bbox_lo_zyx`, `bbox_hi_exclusive_zyx`,
`grid_shape`, `grid_to_native_affine`, `native_shape`, `spacing_um_zyx`,
`centroid_grid`, `centroid_native`, `centroid_um`, `voxel_count`, `volume_um3`,
`covariance_vox2`, `covariance_um2`, `bbox_extents_vox/um`, masked intensity stats,
`score` and score semantics, `touches_image_boundary`, `touches_tile_boundary`,
`mask_observed/propagated`, `crop_missing_fraction`, producer/weights/config hashes.

Half-open bbox hi is not the maximum included voxel index. A mask lives on a
registered grid: a global ZYX integer label volume is not a channel-first image.
Keep fractional centers until final export. Record voxel-center offsets when
resampling: block-mean XY2/XY4 and stride-sampling do not have the same center
convention. A scale alone may omit the half-pixel offset. The reference helper
accepts an explicit affine; it does not decide the correct affine for each model.
Never use trilinear interpolation directly on integer IDs. Resample binary/probability
support or labels with nearest neighbor and then remeasure masks.

The global mask store is immutable and chunked per time. Cropped masks and bboxes
must roundtrip to the same full-volume support. Multiple disconnected components
under one label are detected, not silently joined into a giant box. Tile stitching
uses overlap support, not only bounding-box NMS; boxes alone can overlap for truly
distinct cells. The optional multiple-segmenter route deduplicates identical mask
partitions before contour voting and retains agreement as a feature, not truth.

## API facts verified in authoring

FOCUS notebook: `infer_volume(image_path=..., config_file=..., weights_path=...,
output_dir=..., z_ratio=..., ...)`; output `instance_map_path`. Actual retained
confidence arrays may require adapter instrumentation; verify from source.

Ultrack: functional `segment(foreground, contours, config)`, `link(config, ...)`,
`solve(config, ...)`; `labels_to_contours` and selected node/track exporters. Inspect
signatures in the installed pin because the README's OO example and detailed docs
have naming differences. Do not copy argument names from a different version.
`add_links` accepts database node IDs and weights, not per-frame model labels.
`add_nodes_prob` expects genuine model probabilities or explicitly calibrated scores;
do not insert human-selected/GT-matched nodes. All annotation/GT constraint flags false.

Ultrack uses voxel counts for segmentation size controls unless source explicitly
states otherwise; convert desired physical volumes using actual voxel volume. The
link `scale` affects distance calculations; it does not transform masks/feature tensors
or repair a morphology-model unit mismatch. The inspected target-shift operation is
translation of target bboxes before IoU; retain separate forward-warp feature code.

HOCT optional: require numerical feature parity with its pinned official extraction
on real identical image/mask fixtures BEFORE target scores. Maintain a documented
model-input unit schema and a separate physical feature schema. A dimension match
alone does not validate units/intensity normalization.

## Sparse labels and optional training

Known positive transitions can reject competing parents for the same target because
merges are forbidden. A single recorded child does NOT establish that the parent
has no second unannotated child. Only exhaustive neighborhoods justify outgoing
negatives. Last frames and boundary/cropped daughters are censored. The provided
`sparse_link_label` helper illustrates conservative support, not official FP scoring.
Use the official scorer's distinct sparse-aware FP definition for evaluation.

Mask supervision requires real curated boundaries or explicitly weak/pseudo labels.
Do not train a dense mask loss on a point-rasterization and call it segmentation GT.
For pseudo masks, target mask confidence/consistency is separate from unknown real
background; mixed teacher disagreement may be ignored rather than forced negative.
External point trajectories alone contain no image boundaries. No dedicated annotation-
membership classifier is part of the primary v6 design.

## Integration tests local Codex must add and run

1. Actual one-volume model forward; output nonnegative integer ZYX masks and real
   nonempty components; explicit marker/model match; tiling/native-grid registration.
2. Exact-ID-independent mask comparison, 3D bbox extents, native/physical volume and
   covariance, crop roundtrip, anisotropic and shifted-grid fixtures.
3. No array wrapping in translations; target-versus-source Ultrack shift sign;
   subpixel warp convention; outside-field truncation; zero-overlap true continuations.
4. Duplicate model partitions do not change ensemble voting; blank frames never
   create NaN contours; each model's missing frame is not a background vote.
5. Same-centroid/different-shape fixture changes mask costs but not point-only costs;
   same-box/different-support fixture distinguishes bbox from mask; mask-feature
   permutation negative control; unit-correct HOCT extractor parity if used.
6. Parent-union versus independent birth/split fixtures; children disjoint; merged
   segmentation parent cannot be selected alongside children; no duplicated tracks
   from one detector-mask alias; conflicting owners and division context handled.
7. Real Ultrack solver fixture and selected-label-to-CSV reconstruction; actual
   immediate parent edges distinct from parent_track_id; no forbidden merges,
   nonconsecutive links or missing timepoint observations invented by gap filling.
8. Identity C0 and no-mask fallback exact; one-to-one mapping never drops unmatched
   C0 nodes; shared mask ambiguity cannot become duplicate confident morphology.
9. GT-unavailable cold process, unfamiliar filename, rerun mask/graph parity,
   failed/missing sample rejection, strict output scope and staged-file audit.
10. Official scoring aggregation parity under changed node counts and FP weights;
    dense-mask metrics absent unless independent real mask labels are available.

The included tests exercise NumPy reference math and integrity only. They are not
proof of model inference, official evaluator integration or an improvement.

## Runtime and persistence

Set `PYTHONNOUSERSITE=1`; use the existing tested 4090 environment for incumbent
inference and separate segmenter/tracker runtimes where package versions conflict.
Do not guess a Cellpose v3/v4 API or add a GPU toolkit globally. No auto-upgrade of
PyTorch, drivers, v5 package pins or Zarr codecs in shared environments.

Initial one-GPU-heavy-process scheduling avoids repeating v5 reboot/recovery risk;
the cause of those reboots is unknown, so do not claim they were OOM. Lower batch/
workers to measured available memory. Every completed frame/clip is hash-stamped
and atomically renamed after fsync; source identities/configs join cache keys.
A seed or mask threshold change creates another namespace rather than silently
reusing masks. Check completed stage receipts to avoid repeating model inference.

Measure cost of segmentation + feature extraction + native base if used + linking
+ optimization + export, including model load and caching. Pilot estimates must
cover density and object-count tails. If 19,900 full-frame calls cannot fit the
research cap, use the documented smaller screening stage and a measured student/
propagation alternative; do not silently label a subset full OOF. Before disk
reserve is breached, stop creating disposable v6 caches and publish valid partial
measurements; never delete unrelated data or predecessor masks/checkpoints.
