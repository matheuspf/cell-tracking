# Detector and segmentation choices

The objects must follow the fluorescence channel. For a nuclear marker, track the
entire visible nucleus as the cell proxy; do not claim cytoplasm/membrane boundaries
that the image does not show. Inspect raw XY/XZ/YZ views and local official channel
metadata before selecting a nuclei/general/membrane checkpoint. A bounding box is
a useful extent and inexpensive association gate; a mask preserves shape, touching
boundaries and overlap. Keep both. Final competition output remains a point graph.

## D1: FOCUS-3D, priority when weights are authorized

Inspected code commit: `5c4b53f743a0fbbae056e2c1a139895ae819f069`.
The repository supplies a real 3D instance model and a headless inference notebook,
not only a napari GUI. Its notebook imports:

```python
from focus3d.segmentation.FOCUS3D.inference_win import infer_volume
# Existing upstream interface; adapt paths/parameters after local inspection.
result = infer_volume(image_path=path, config_file=config_path,
                      weights_path=weights_path, output_dir=new_output, **params)
```

The official model card lists general, membrane and nuclei `.pth` checkpoints,
Apache-2.0 weights, and a contact-sharing access gate. Public source does not imply
ungated weight access. Use an existing authorized local checkpoint or preauthorized
account access; do not accept terms, disclose contact information, contact authors
or upload images to the hosted demo. If inaccessible, record D1=blocked_access and
continue the complete D2/object/Ultrack study. Do not substitute invented weights.
Only download the selected checkpoint initially, not all multi-gigabyte variants.

The checked inference backend constructs the model without Detectron2. Detectron2
is a separate fine-tuning concern; do not make it an inference prerequisite simply
because the Linux training guide mentions it. Use an isolated environment, without
upgrading the working v5 runtime. The notebook's z_ratio=5 is an example, NOT our
raw-grid ratio=4. Derive ratio and all transforms from actual metadata; native
[1,4,4]-downsampled input has ratio=1. Preserve native resolution where memory allows.

Audit normalization, radius in XY pixels, score/mask thresholds, volume filtering,
patch halo/stitching and output grid. Run headless full-frame detection independent
of C0 seeds. Retain instance confidence and soft masks/foreground where actually
available; a hard label map must not be relabeled a calibrated probability map.
The public loader uses `weights_only=False` and `strict=False`, with missing-key
logs commented out. Require checkpoint provenance/hash, safe loading where supported,
and an explicit approved key/shape compatibility report. Do not silently continue
with random/unloaded layers. Do not deserialize an untrusted pickle as a workaround.
If wrapper calls reload the model per frame, implement a persistent model adapter
and compare its output against the audited upstream path on fixed source volumes.

## D2: independent learned segmentation, required when available

First screen a permitted pinned **Cellpose** model on the same source-only volumes.
The documented `do_3D=True` path computes orthogonal 2D flows and combines them into
3D dynamics: it is a 2.5D-derived segmentation, not a native 3D image backbone.
Set z/channel axes explicitly and anisotropy from metadata. Check volumetric minimum
size and flow smoothing; `flow_threshold` is ignored in documented 3D mode. A 2D
slice/stitch result must be tested for axial fragmentation and distinguished from
that mode. Pin the selected package/checkpoint, inspect actual returned fields,
and audit code, weight and training-data terms separately against competition rules.
Do not blindly upgrade to a newer family in the shared environment.

**StarDist3D** is the bounded alternative if Cellpose access, modality or runtime
fails. Its star-convex volumes are particularly worth testing for compact nuclear
objects. It expects dense instance annotations for ordinary supervised training;
the sparse GEFF points are not those labels. The advertised 2D fluorescent model
is not a 3D pretrained model. Inspect an actual available 3D checkpoint's training
modality, spacing, rays and normalization; a demo checkpoint is only a diagnostic,
not established Biohub coverage. Isolate TensorFlow/CUDA dependencies from PyTorch.
Select at most two learned detector families for full production evaluation.

D0 is the existing image-supported watershed rebuilt with masks retained. It is a
useful reference/availability fallback, but is not a substitute for attempting a
real external instance segmenter. C0 points may prompt a SEPARATE refinement arm;
they may not secretly define every candidate in the full-detector arm.

## T1: Ultrack, the primary segmentation-aware tracking backend

Inspected commit: `5c94d845eb0a7b78c8dc24492ef00f218a467995`.
Ultrack chooses mutually compatible segmentation hypotheses and temporal links;
it is not itself the neural instance detector. The inspected core API accepts
`labels` OR `foreground` plus `contours`, with image evidence, physical scale and
optional vector field. The canonical keyword is `contours`; old `edges` is an alias.
Use the installed pinned API, not an assumed README signature.

Start with one detector's full label movie. Then retain richer foreground/boundary
uncertainty or use `labels_to_contours` with two aligned segmentation label movies.
Never average integer instance IDs or union every detector's nodes into a lineage.
Label-to-contour conversion is a surrogate uncertainty representation, not raw
network boundary confidence; record this distinction. The foreground union must
cover missing-cell alternatives because a tracker cannot select an absent segment.
Map Ultrack hypothesis IDs back to source masks/provenance. If resegmentation changes
instances, export those selected masks, not only the original detector masks.

Use physical scale for distance, voxel counts for documented area thresholds,
image-only registration/flow and explicit pixel/physical displacement conversions.
Create a separate local database per clip/configuration. Never use ground-truth
matching/annotation solver flags in inference. Use the existing lawful offline
solver or test the open-source backend; an academic/commercial Gurobi license is
not presumed. Do not purchase/register a license or rely on online activation.
If no practical Ultrack solver is available, execute the local mask-aware graph
backend as a separately named alternative and mark actual Ultrack blocked.

The pinned commit fixes window-boundary phantom selections and dangling parents.
Verify that behavior before trusting a different installed release. Test division
near window seams, selected-parent existence, anchored boundary birth/death costs,
and consistent single-window/multi-window solutions on unambiguous fixtures.
A mask-aware edge scorer can use Ultrack's explicit link weights, but must preserve
its candidate/exclusion constraints and source-only calibration.

## Optional challenger, not an installation sweep

A mask-conditioned learned tracker such as Trackastra can be tested only after
D1/D2 and T1 run, within the same budget. Audit actual 3D checkpoint support and
feature units before scores. Correcting the existing HOCT unit adapter is likewise
a separate source-only control, not permission to overwrite v5 results. Neither
optional route may replace the required mask-overlap and detector-driven tests.

Before any production run, write `assets.json`: URL/revision, local path, SHA-256,
code/weight/data license evidence, access status, model mode, expected axes/units,
normalization and dependencies. Missing permission is blocked, not implicitly granted.
