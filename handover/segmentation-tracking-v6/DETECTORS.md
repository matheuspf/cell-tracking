# Segmenter and tracker decisions

## 1. FOCUS-3D: first-choice volumetric mask source when authorized

Pinned source: yu-lab-vt/FOCUS-3D at
`5c4b53f743a0fbbae056e2c1a139895ae819f069`.
The current model card lists `model_final.pth`, `model_final_membrane.pth`, and
`model_final_nuclei.pth`. Prefer the nuclei specialization for verified nuclear
images; compare the general model only within the bounded screen. Do not use a
membrane model simply because the task calls objects cells.

The card declares Apache-2.0 for weights and currently requires agreeing to share
contact information. Source code and model rights are separate records. Check
existing local authorized files first; record path/hash and provenance. This plan
does not authorize accepting a gated agreement or sharing contact information.
No microscopy may be uploaded to the Hugging Face demo.

Use `notebooks/01_inference.ipynb` as the headless interface reference:
`focus3d.segmentation.FOCUS3D.inference_win.infer_volume` accepts image_path,
config_file, weights_path, output_dir, z_ratio and documented inference settings;
its result exposes `instance_map_path`. Extract/reuse the backend, not the GUI.
Process ONE ZYX volume at a time, never pass TZYX as if T were Z. Avoid repeatedly
loading the model per frame when a verified equivalent persistent worker is possible.

At the native Biohub spacing, z_ratio is 4, not the notebook demo's 5. Derive it
from actual metadata. A resampled grid has a different ratio. Radius/size/stride
parameters are provider-grid quantities; record their physical equivalents.
Defaults such as background threshold and 2D min-edge-area are not universal.
Audit output shape, scaling back, tile stitching, seam objects and removed-instance
counts. Keep foreground, instance scores/mask logits only when actually exposed;
missing scores must be explicit rather than fabricated.

## 2. Cellpose-SAM: accessible independent comparison

Pinned source: MouseLand/cellpose at
`a54cb48849b7e225a81e8e43dcb042d42427f543`; resolve compatible package/model hashes.
Use the installed version's actual `CellposeModel` signature, grayscale input,
explicit z_axis/channel handling and `do_3D=True` with measured anisotropy.
The documented volumetric route combines orthogonal 2D flows then runs 3D dynamics;
it is not a native 3D convolutional backbone. Keep the resulting 3D instances.

Preserve spatial flows as segmentation outputs, but DO NOT treat Cellpose's
intra-volume flows toward cell centers as motion between timepoints. Temporal
flow must be estimated independently from successive images.

Cellpose docs state flow_threshold is ignored in this 3D route. Do not waste a
sweep on it. A slice-stitching alternative is allowed as one named diagnostic
when orthogonal views fail; its per-slice size threshold is not a 3D volume cutoff.
Check actual weight/license identity and isolate dependencies from shared runtimes.

## 3. StarDist3D: bounded alternative, not a promised foundation model

The official registry exposes StarDist3D `3D_demo` with archive SHA-256
`ea05831eb5acc8a2fd31eaa23f4460a196a9af53b14f40affb9d80885f699f90`.
There is no claim here of a universal pretrained 3D nuclei model. Inspect its
configuration, training-domain notes and physical object scale before use.
Run it as a cheap third-source diagnostic, or the second learned source when
FOCUS is unavailable. Native 3D star-convex masks can be useful for nuclei, while
non-star-convex/mitotic regions may violate the representation.

Do not pass a 2D versatile model over slices and call it native StarDist3D. If
2D stitching is attempted, name that different treatment. TensorFlow dependencies
belong in a separate environment. Do not train from the sparse point annotations
as though they were fully labelled instance masks.

## 4. Classical image foreground/watershed: control and complementary contours

Use a raw-image-derived foreground/contour baseline without dense annotation
assumptions. Preserve v5's shared-marker watershed as a named historical control,
but do not feed new markers into the fixed-marker comparison. Multi-scale
foreground/gradient or inverted distance maps can supply hierarchy alternatives.
A fixed ball or bbox around a center is only a negative control, not evidence that
segmentation was performed.

## 5. Ultrack: primary region-selection/lineage engine

Pinned source: royerlab/ultrack at
`5c94d845eb0a7b78c8dc24492ef00f218a467995`.
Its documented inputs are TZYX foreground and contour arrays, or one/more instance
label time series via `labels_to_contours`. It constructs alternative segmentation
hypotheses, links adjacent-frame regions and chooses compatible lineages.

`labels_to_contours` forms the union foreground and averages OUTER boundaries
across providers; it does not guarantee every original instance survives as a
selectable hierarchy node. Measure mapping/coverage through the hierarchy. Use
finite empty-frame handling, since smoothing followed by normalization by a zero
maximum would otherwise generate invalid contours. Start sigma=None; add only
physical-scale-aware smoothing in an explicit source-controlled arm.

Use `Tracker.segment`, optional `add_flow`, `link`, `solve`, and exports or their
pinned functional equivalents. `add_links` supports custom link evidence; prove
whether existing links are appended/replaced and prevent duplicate weighting.
Use `use_annotations=False` and `use_ground_truth_match=False` in inference. Do
not run its auto-GT fitting helpers on sparse points treated as dense masks.

Pinned linking defaults use region IoU minus a distance term. Its flow integration
shifts TARGET bounding boxes toward the SOURCE using rounded voxel shifts. This
is a region translation, not a dense deformable mask warp. Test sign/axis/units on
a known moving object. A custom nonrigid warp needs a separate implementation and
comparison; do not attribute it to add_flow.

The solver maximizes link/node utilities and its config uses negative appearance,
disappearance and division penalties. Old solver weights used a different sign/
scale. Use source-only sensible calibration and test competing fork versus birth
fixtures before full solves. CBC is supported without buying a license; Gurobi
is permitted only with an already valid suitable license. Limit threads and per-
clip SQL/database storage. Validate hierarchy conflicts, output track-parent
semantics and window-boundary divisions rather than trusting a pretty overlay.

## 6. HOCT / Trackastra: secondary mask-feature consumers

These are optional comparisons AFTER valid masks exist, not the core v6 model
search. Use official feature extraction and exact checkpoint units. For HOCT,
retain separate physical descriptors for analysis; compare its model tensor to
upstream voxel-valued regionprops on the same labels. Missing masks remain
missing. Do not replace them with sphere features or simply repeat v5's adapter.

## Selection policy

Screen the authorized FOCUS nuclei/general candidates, Cellpose-SAM and optional
StarDist3D on a fixed source-only schedule. Bound to two parameter recipes per
provider. Select at most two providers per direction, using source-supported node/
edge evidence and measured runtime; also retain provider-specific standalone
results. Run at least one learned-mask route even when another is blocked. Never
claim all sources were used when a backend failed or produced no real masks.
