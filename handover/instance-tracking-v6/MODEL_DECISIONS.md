# Segmenters and tracker: selected roles

All versioned implementation links and verification sources are in [SOURCES.md](SOURCES.md).
Capture actual package/checkpoint hashes locally. Do not assume current main will
remain unchanged. No model inference or package installation ran in authoring.

## FOCUS-3D — primary requested segmentation arm

Inspected Git commit `5c4b53f743a0fbbae056e2c1a139895ae819f069`.
The official inference notebook uses
`focus3d.segmentation.FOCUS3D.inference_win.infer_volume` with explicit image,
config, weights and output paths. It returns `instance_map_path`; keep that
instance map, not merely its centers. Extract notebook code without executing
install/display cells. Build a persistent-model runner only after one-call parity
so per-frame model reload does not dominate the benchmark.

The authors' Hugging Face card lists `model_final.pth`, `model_final_nuclei.pth`
and `model_final_membrane.pth`. Prefer the model matching the confirmed fluorescent
compartment. The card declares Apache-2.0 for weights; GitHub code is BSD-3-Clause.
Access still requires accepting conditions/sharing contact information. Existing
authorized local weights are usable; absent authorization blocks only this arm.
Do not bypass the gate or upload data to the public demo. This plan is not a
legal eligibility ruling; archive source terms and verify competition compatibility.

Use native ZYX volumes and actual z/xy ratio (reported Biohub spacing implies 4,
not the notebook's example 5). Do not apply anisotropy twice. Cell-radius units,
raw background threshold, normalization percentiles, patch overlap and stitched
output transform must be recorded. Defaults such as top-k and size filtering
are per-volume/patch implementation choices, not universal biology. Count truncated
patch outputs and seam splits. Preserve detector scores if exposed; unavailable
confidence stays unavailable, not invented. Verify all axes and native-grid output.

## Cellpose-SAM — independent, nonblocking segmentation arm

Use the official `CellposeModel` API in an isolated current pinned runtime. Inspect
its actual call signature rather than mixing Cellpose 3 and 4 examples. Explicit
ZYX/z-axis/channel-axis input, `do_3D=True`, and native anisotropy are required.
If XY is downsampled twofold, its new z/xy ratio is 2, not 4. The 3D algorithm
combines orthogonal-plane flows and runs 3D dynamics; do not label it a trained
3D convolutional backbone. Cellpose's documentation states `flow_threshold` is
ignored in 3D, so do not waste a grid on that parameter. Slice stitching is a
separate fallback, not silently equivalent volumetric inference.

Retain masks and flow/cell-probability outputs, with confidence semantics documented.
A built-in nuclei model from an older Cellpose release is an optional separately
pinned control, never silently selected by passing a deprecated name to CPSAM.
Fine-tuning requires actual curated/pseudo mask targets and proper ignore regions;
sparse center annotations are not dense mask training labels.

## StarDist3D — bounded fallback/comparator

The official registry includes `StarDist3D.from_pretrained('3D_demo')`, archive
`python_3D_demo.zip`, registry checksum
`ea05831eb5acc8a2fd31eaa23f4460a196a9af53b14f40affb9d80885f699f90`.
This is a demo 3D model, not evidence of a universal zebrafish segmenter. Screen it
only if FOCUS is unavailable, the first mask producer fails, or its pilot adds
complementary masks. It uses star-convex polyhedra and can be a distinct shape
prior. TensorFlow stays in a separate environment. No fictitious '3D versatile'
checkpoint. MicroSAM is a further fallback only after these paths fail; no extra
foundation-model integration campaign by default.

## Ultrack — central mask-aware tracker, not a pretrained detector

Inspected Git commit `5c94d845eb0a7b78c8dc24492ef00f218a467995`.
Use public core functions `segment`, `link`, `solve` and supported exporters,
or the version-matched Tracker interface. It builds hierarchical segmentation
hypotheses from foreground and contours, links regions and selects a compatible
temporal solution. It is not a requirement to retrain a new tracker first.

`ultrack.utils.labels_to_contours([label_volume_A, label_volume_B])` expects
same-shaped TZYX label arrays. Its inspected implementation ORs foreground and
averages **boundary maps**, not integer IDs or instance probabilities. First test
one source; then test two. A model's label 7 has no relation to another's label 7.
Drop exact duplicate representations before averaging; missing model output is
unknown, not a vote for background. An all-empty frame with smoothing needs a
max=0 guard: the inspected helper otherwise normalizes by its maximum. Test this
in the local version; use sigma=None plus safe external smoothing if necessary.

The helper does not promise to preserve every original non-nested instance as an
exact optimizer candidate. Audit generated hypothesis coverage and overlaps.
Do not describe a merged label image as a faithful union of arbitrary hypotheses.
Standard hierarchy/exclusion must prohibit selecting a region and its subregions
at once. Track masks through solving and export the selected per-frame node IDs;
tracklet IDs are not node IDs and parent-track metadata are not adjacent edges.

The inspected default link weight is `Node.IoU`. Motion changes are applied to
**target** boxes before linking, so its shift convention is not automatically the
same as a source-forward warp. Verify direction and units using a translated-mask
fixture before using `add_flow`. Inspect actual overlap weighting and zero-IoU
candidate behavior; do not interpret an API scale argument as scaling every
feature. Production deformation needs image-derived flow; optimizing a translation
separately for every candidate would artificially inflate overlap.

Run the installed backend's small division/no-division/exclusion fixtures. Gurobi
is optional, and no license purchase or new academic declaration is authorized.
Use a supported available fallback when it exists and measure it. If the selected
version cannot solve without an unavailable license, record that blocker and run
the mask-aware hybrid; do not claim a homegrown solver is Ultrack. The task still
requires an actual Ultrack attempt, not only source reading.

## HOCT reuse — controlled auxiliary, not the main direction

Optional mask-derived region embeddings can reuse the already installed HOCT.
Before rescoring, require upstream extractor parity on the same labels/images:
model inputs in expected voxel/intensity convention, separate physical features
for our own model, exact column order/normalization. New masks must be real support,
not spheres. A unit-correct old-watershed control isolates adapter repair from
mask improvements. An HOCT dependency failure must not block Ultrack or basic
mask-overlap linking. No new broad HOCT training grid.
