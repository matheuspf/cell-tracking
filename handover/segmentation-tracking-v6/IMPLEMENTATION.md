# Local implementation contracts

Create an additive `tools/segmentation_tracking_v6/` package and
`scripts/run_segmentation_tracking_v6.sh`. These do NOT exist in the initial
handover. Existing study modules may be imported only where their semantics are
verified; no new output may be written through hard-coded old study roots.

## Modules

| Module | Required behavior |
|---|---|
| snapshot | Read current status, freeze baseline/source IDs, inspect active jobs/resources, resolve paths, record model/license/metric pins. |
| segmenters | Headless FOCUS, Cellpose and optional StarDist adapters; preserve output grid, preprocessing, label IDs and optional scores. |
| regions | Instance masks/boxes, immutable source lineage, physical descriptors, ambiguity/missing/border/seam flags, compressed persistence. |
| image_motion | Shared image-based drift/local fields, reliability, coordinate conventions, no pair-specific mask-fitting shortcut. |
| associations | Same-node point/box/mask/warped-mask feature ablations; sparse source supervision; no zero-confidence imputation. |
| ultrack_adapter | Labels-to-contours, hierarchy mapping, region links, native-evidence injection, real solver/config receipts, mask/lineage export. |
| division | Warped parent versus daughter union, competing ownership/path evidence and temporal-censor masks. |
| fit | Matched-compute source-only link/optional event fits; immutable feature/model manifests and seed replication. |
| evaluate | Strict sample completeness; fresh official graph/node/division matching; expected-count and graph identity checks. |
| infer | Raw images and packaged models only; no training masks/graphs, labels, counts, or target-specific filename routing. |
| report | Actual tables/overlays/dashboard, runtime, hypothesis decisions and CONTINUATION.md. |

## Region identity and geometry

One row represents `(dataset, frame, provider, provider_instance_id, hypothesis_id)`.
Do not use only a provider integer across frames or across models. Store:

- image path/hash, model/checkpoint/config hash and prediction-only provenance;
- `observed_compartment`, source grid axes/shape, native shape, spacing, affine/
  resampling transform, voxel-center convention and preprocessing version;
- half-open bbox `[z0,y0,x0,z1,y1,x1)`, local boolean mask reference/shape/checksum,
  original instance label, count of foreground voxels, physical volume;
- geometric and intensity-weighted centroids, a deterministic in-mask representative,
  covariance/eigenvalues, extents, intensity quantiles and boundary-touch fractions;
- real model confidence if available, otherwise null with a missing flag;
- support/mapping ambiguity, shared-mask ownership, seam and truncated-object flags.

Keep physical and model-specific features in separate namespaces. Metric distances
use physical units; HOCT checkpoint inputs use its proven reference extractor
semantics. Scaling coordinates is not sufficient: volume scales by sx*sy*sz and
second moments by squared lengths. Do not turn bbox IoU into mask IoU. Native
point coordinates locate voxel centers at indices; a resizer's half-voxel transform
must be explicit and tested. Do not assume dividing/multiplying indices is always
the correct inverse of an interpolating image resize.

Integer submission coordinates are a FINAL representation. Preserve subvoxel
centers and mask shape during tracking. A concave mask's center can be outside
its support; compare a fixed source-selected centroid rule with an in-mask rule
only as registered variants. Never choose a representative from held-out GT.

## Storage and streaming

A single full native uint32 mask volume per frame is about 16 MiB. At the recorded
199 x 100 frames, a dense one-provider series would occupy approximately 311 GiB
uncompressed. This is a calculation, not measured compression/runtime. Do not
materialize multiple such series. Use frame chunks or bbox-local packed masks/
RLE, one clip at a time, and bounded temporal context. Retain selected masks and
small diagnostics; bulky new intermediates may be reclaimed ONLY from v6 scratch
after their reproducibility/hash records are saved. Never delete old studies or
raw images to create space. Use explicit provider/grid/dtype schema metadata.

Ultrack stores hypothesis masks in its own database; budget that storage separately
from labels/contours. Pilot candidate counts, SQL size and solve RSS on the densest
SOURCE clip before scaling. Avoid one shared mutable SQLite across unrelated
workers or configurations. A provider ensemble can greatly increase hypothesis
count; cap source-controlled hierarchy levels rather than silently truncating
individual clips to meet memory. Log every cutoff and resulting candidate coverage.

## Motion and overlap

`region_contracts.translate` is an integer TRANSLATION fixture, not optical flow.
For a forward field u(x), a source voxel moves to x+u(x). For inverse-sampling of
an output target image, coordinates instead refer back to the source. Document
which is used; never flip the sign by intuition. Test all three axes.

Pinned Ultrack shifts target bboxes toward the source with rounded voxel shifts.
Its native add_flow route does not deform a mask nonrigidly. A custom dense warp
must handle interpolation, occlusion and validity explicitly, avoid label-ID
interpolation, and quantify voxel/volume change. Use nearest-neighbor labels or
warped probabilities with explicit threshold; never linear-interpolate integer IDs.
Pair features use the same field, not a separate registration chosen per pair.

For division features, align parent evidence into the daughter time/grid, union
actual disjoint daughter voxels, and subtract their intersection where necessary.
Do not count an overlapping duplicate as a second daughter. Volume/intensity
conservation is a learned soft descriptor, not an enforced biological law.

## Ultrack integration details

Call the actual pinned API only after signature tests. `labels_to_contours`
expects complete equally shaped time-series label arrays, not a list of individual
ZYX frames interpreted as segmentation methods. Convert provider outputs into a
common grid with preserved transforms. Model consensus in boundary maps is not
independent evidence; record provenance and avoid counting duplicate masks twice.

Keep separate label-to-hierarchy and hierarchy-to-export ID maps. Validate exclusion
between ancestors and descendants and arbitrary cross-provider overlapping choices.
If exact supplied masks are not preserved by the hierarchy, report attrition and
keep a fixed-instance tracking control. An Ultrack output can differ from every
original segmenter's mask; do not attribute that exact mask to one provider.

Native graph logits belong to specific nodes and frame contexts. Recompute native
features at new region centers when needed; nearest old-score transfer is an
explicit diagnostic only. Normalize repeated window evidence by observation support.
`Tracker.add_links` integration must remove/replace duplicate link entries in a
verified new database, and calibrate the link function's utility sign/scale. An
old negative-cost solver formula cannot be pasted into a positive-utility solver.

Test root/birth/continuation/division/death flow constraints, mutually exclusive
observations, empty frames, and image/temporal-border censoring. Solver use of
annotations and ground_truth_match must remain false. Do not silently use
fit_nodes_prob/match_to_ground_truth on sparse GEFF points as dense mask truth.

Export one observation node per selected region/frame and real parent-child
links, not one graph node for an entire track ID. Tracklet parent dictionaries
and observation graph edges have different semantics. Consecutive same-track
observations link one-to-one; a dividing final parent connects to the first
observations of both daughters. An absent frame needs image-supported recovery
or an explicit break; no t-to-t+2 submission edge is fabricated.

## Tables to persist locally

`run_lock.json`, `source_manifest.json`, `provider_registry.json`, `screen.csv`,
`regions.parquet`, `mask_manifest.json`, `region_matches.parquet` (evaluation only),
`hierarchy_mapping.parquet`, `motion_audit.json`, `feature_parity.json`,
`variant_lock.json`, `score_rows.csv`, `representation_ablation.csv`,
`division_regret.csv`, `segmentation_diagnostics.csv`, `runtime.csv`,
`fresh_inference_receipt.json`, `artifact_manifest.json`, `status.json`.

Every numerical report row names its population, provider, candidate bank,
feature/decoder/model hash, source direction and actual sample count. Null denotes
unavailable supervision or confidence. Never replace missing data with an invented
zero or drop a failed sample from the official aggregate.

The final package needs raw-image segmenter inference, transforms, region tracker,
weights, exact configs, frozen routing and CSV validation. It needs no external
training data, raw GT or old selected graph. Track source-code and weight licenses
separately. Keep the whole C0 package available as a verified fallback.
