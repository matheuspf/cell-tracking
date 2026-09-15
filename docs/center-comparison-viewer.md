# Cell-center and FOCUS-3D mask viewer

The local viewer compares real microscopy, current center predictions, and
FOCUS-3D nuclei masks with synchronized time, depth, brightness, zoom and pan.
Open **http://localhost:8767** while the local server is running.

## Pipeline report and tracking review

Open [Pipeline report](http://localhost:8767/#tab=report) for the current best
complete solution, score decomposition, main failure groups, idealized score
impact, recommended next experiments, and embryo/clip breakdowns. The
[written report](pipeline-errors-20260915.md) explains the evidence and its
limits. C4_m6 has the highest pooled local score; P0 remains the adopted pipeline.

The [Tracking & divisions](http://localhost:8767/#tab=tracking) tab covers every
official edge FN/FP and division FN/FP for both pipelines: **10,931 C4_m6** and
**10,921 P0** flags across all 199 full clips. Filter by error, evidence, embryo
or clip. Expected and predicted graphs are shown alongside exact native images
from the event window, with matching physical crops, XY/XZ/YZ views, depth and
brightness controls. Every center-check row links to its Detection review.
Copy case links or save/export your notes. Related flags can describe the same
biological mistake; their counts are not independent events.

The missed-division scenes use the official division-window matching. Their
center-check table uses whole-clip matching and can therefore differ. The UI
labels both. A predicted edge outside the sparse annotation's evaluable region
stays **Unevaluated**.

Build the tracking index after the detection index, using the existing scoring
environment (about nine minutes for both pipelines on this machine):

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m center_comparison index-tracking
```

The command verifies all 398 graph evaluations and writes the report and
read-only SQLite catalog under the existing output root. It does not rerun
inference. Refreshing the source detection index requires rebuilding this audit.
Run `python -m center_comparison.verify_tracking` with `PYTHONPATH=tools` in the
main `cell-tracking` environment to verify data and browser behavior.

## Browse all errors

The homepage opens **Detection review**, with **50,546 flagged examples for C4_m6**
from all **199 training clips**. P0 has 50,535 examples. The old default opened a
single 16-frame movie and showed only 28 cases; those old sequence links now open
the complete review as well.

Each example has two views: **Full image** and **Zoom**. Both overlay **gold GT
diamonds** and **cyan prediction circles** on the microscopy. A dashed gold
**7 µm radius** surrounds the selected GT; a white box marks the zoom in the
full image. Exact overlaps retain both colors and shapes.

Choose **Final output · C4_m6**, **Final output · P0**, or **Detector candidates ·
before tracking** in the visible Predictions selector. The last option replaces
the old “Raw” label: these are detector centers before the tracking pipeline.
Click a cyan prediction in either image, a nearby P1/P2 button, or use **Select
prediction** to choose any center in the frame. The gold GT selector changes the
comparison target. Switching prediction sources preserves that GT and frame.

The selected pair shows its **3D distance**, **Inside / Outside 7 µm**, and
**assignment** separately. A nearby prediction can be assigned to another GT.
The dashed line measures the selected pair; solid lines show actual assignments.
The circle is a projection of the 7 µm sphere: hidden-axis separation still counts
in the 3D distance. **Fit selected pair** includes the GT circle, prediction and
its assigned GT in the crop. Automatic depth also includes every candidate within
7 µm. Only the current frame's centers appear on its images.

Use **Next / Previous** or **Go to** an example number. Next advances across
pages and reaches the last indexed example. **All issues**, **Detection**, and
**Association** always search all clips. Ambiguous centers and connections are
included. **Unlabeled predictions** keeps every unassigned prediction available;
sparse labels cannot establish whether these are false detections.

Click empty space in the full image to move the zoom; drag or scroll the zoom to
pan or resize it. Use the plane, brightness, or playback controls when needed.
**Options** contains clip/frame filters and depth controls. **Details & notes**
contains source IDs, assignment tables and saved judgments.
**Export notes** saves the browser's assessments as JSON. New copied links restore
the example and view; notes stay local to the browser's hostname and port.

The **FOCUS comparison · 42 frames** tab retains the earlier segmentation and
tracking viewer. It is explicitly limited to exported FOCUS frames. Use
`#tab=comparison` for a direct link to that tab.

The default is now **C4_m6, the highest recorded total score among the complete
non-oracle pipelines: 0.935178370257 on all 199 clips**. **P0, the retained
pipeline at 0.934864986413134**, and **C0 / selected v3 at 0.934802374261** remain
selectable. These are the actual final predicted centers and tracking edges.
The [v4 result](../results/multidata-training-v4/final_report.md) did not adopt
C4_m6 because its 6bba score regressed slightly and replication did not qualify;
the [v6 result](../results/segmentation-tracking-v6/final_report.md) retained P0
after its checks passed. The highest pooled score and adopted status are labelled
separately. The newer two-clip ultrack pilots do not replace these 199-clip results.

The initial export contains **42 unique full 3D frames**:

- `6bba_e16ffc58`, frames 20–35, continuous.
- `44b6_8f5ab931`, frames 20–35, continuous.
- All 12 original pilot snapshots, frames 25 and 75 of six clips. Two frames
  overlap the movies. The 50-frame snapshot gap is explicit; playback is disabled
  for these pairs.

These windows were selected from the frozen pilot before new FOCUS inference.
They are exploratory views of two reused training embryos. They do not establish
performance on an independent embryo.

## Inspect cells

In the FOCUS comparison tab, choose a category under
**Find the cell center**, **Connect cells across frames**, or **Recover the
division**, then use **Next case** or the list. **Search in → All 42 exported
frames** includes all six displayed clips. C4_m6 and P0 remain selectable.

Center and temporal assignments are separate concepts:

- **Missing center:** no prediction is within the chosen 3D radius of an
  annotation. The location and nearest prediction are shown separately.
- **Center assignment conflict:** a prediction is nearby, but the optimal
  one-to-one assignment does not assign it to this annotation. This is distinct
  from absence of any nearby prediction.
- **Center offset ≥3 µm:** the annotation has a match, but its center is displaced.
  This is a localization diagnostic, not an extra competition score penalty.
- **Missing connection:** the two relevant centers are matched, but their
  expected temporal edge is absent.
- **Wrong / extra connection:** an actual predicted edge contradicts an evaluable
  annotated connection. Related missing-edge and incorrect-edge flags are grouped
  into the same case.
- **Center failure breaks link:** an annotated connection cannot be recovered
  because at least one endpoint has no match at 7 µm. Inspect detection first;
  this does not isolate a linking-model failure.
- **Missed or incorrect division:** the official division scorer did not recover
  an annotated split or rejected a predicted split.

Each case has a plain-language explanation, explicit **Center matched / No
matched center** checks, and a connection verdict. Annotated cells have short
labels **A, B, C**; **Pred A** is the model's assigned center for A. Exact source
IDs and related metric flags are available in an expandable section.

For a temporal error, the earlier and following frames appear **side by side**,
with the same physical crop. Every image contains only its own timepoint's
centers. **Expected from annotations** and **Actually predicted** rows show the
connections by name, marking missing, correct, and incorrect edges. A division
with one correctly linked daughter retains that correct branch in the display.
The parent-to-other-daughter omission is identified explicitly.

For a center error, both images show **the same timepoint**: annotated location
and model result. Axis-specific offsets are shown in µm. When depth dominates a
center error, the initial plane switches to XZ or YZ so the displacement is
visible. A dashed ring marks the matching radius in the displayed plane;
matching itself always uses all three spatial axes.

**Only this case** hides unrelated prediction markers by default. **Around these
cells** projects a small depth range containing the relevant cells in each
frame, with the exact range printed below each image. Use the case's plane,
brightness and depth controls to inspect it. An unavailable source image is
explicitly marked as not exported; no neighbouring image is substituted.

**Return to full comparison** opens the original microscopy / predictions /
FOCUS panels, movie controls, mask overlays and annotation table. The copied
view link preserves the pipeline, case, category, radius and main plane.
Old links to grouped edge flags still open their corresponding case.

Tracking and division decisions always use the official 7 µm evaluation on the
**complete 100-frame clip** before selecting the viewer's image windows. Center
cases use the selected radius. Full-clip totals and overall 199-clip scores are
under **Full evaluation scores and scoring details**.

At 7 µm, each pipeline has **70 browsable cases** in these 42 images: 5 missing
centers, 48 displaced centers, 1 missing connection, 6 wrong-connection cases,
7 connections affected by a missing endpoint, and 1 missed division. These
explain all **84 original scoring/diagnostic flags**; related edge flags are
grouped. Center, linking and division cases can still describe related failures
in the same cells, so their counts are not independent biological events.

Choose **Center assignments** to compare pre-ILP detector proposals, selected v3
tracker centers, or V5 seeded-watershed centroids against FOCUS geometric,
intensity-weighted, or smoothed-peak centers. Match radius choices are 1, 2, 3, 5,
and 7 µm. The exporter runs the installed official `tracksdata.DistanceMatching`
with optimal assignment independently for every method and radius. It does not
filter a previously computed 7 µm assignment.

White diamonds are sparse annotated centers, cyan circles are current
predictions, and orange crosses are FOCUS centers. Red diamonds have no match
at the selected radius. Click an annotation row to jump to its depth and inspect
the matched IDs, exact coordinates, physical errors and associated mask. Masks
are also directly clickable. The inspector counts annotated and detector centers
inside a mask, including masks containing multiple sparse annotations.

Choose **Temporal links** for the saved C4_m6, P0 or v3 tracker graph and the saved
annotation graph. Dashed FOCUS trails are **display-only centroid associations**, computed
without GT: maximum-cardinality, minimum-distance one-to-one matching within
7 µm between consecutive frames, using exact centers. They have no division
model or gap closing and are not a trained FOCUS tracker. FOCUS instance IDs are
local to a frame; colors follow these preview associations in temporal mode.

The three panels support XY, XZ and YZ views, individual slices, five-plane slabs,
and full-depth maximum projections. Projection masks take the label at the
brightest **displayed** voxel on each ray; overlapping cells can be hidden, so
use slices to inspect boundaries. Scale bars and aspect ratios use actual
physical voxel spacing. All centers are shown and matched after NumPy integer
rounding and native-grid clipping; the inspector retains exact center coordinates.

Scroll over a panel to zoom; drag to pan. Space plays or pauses, left/right steps
time, and up/down steps depth when the page or a canvas has keyboard focus.
Copy view link preserves sequence, frame, plane, depth, methods, radius,
selection, zoom, brightness and overlays. Playback speed is display fps, not an
assertion about biological sampling time. A slow frame buffers instead of
pairing one timepoint's image with another timepoint's masks.

Recall and assignment tables always cover the **entire 3D frame**, including
centers outside the currently displayed slice. Unmatched predictions are not
confirmed false positives because annotations are sparse. Masks have no dense
ground truth. These center statistics are not the competition tracking score.

## Run and rebuild

Run from the repository root. Serve the existing export with the main environment:

```sh
conda activate cell-tracking
PYTHONNOUSERSITE=1 PYTHONPATH=tools python -m center_comparison serve
```

The server binds only to `127.0.0.1`, serves the generated `site` assets and a
read-only detection API for the indexed graphs and canonical image frames,
and makes no external requests. It refreshes the viewer's HTML, JavaScript and CSS
from the repository on startup. If accessing this machine over SSH, forward
port 8767 and open the forwarded localhost URL. A current Chromium, Firefox or
Safari supporting `DecompressionStream` is required. Open through HTTP, since
the viewer loads local binary assets.

The generated output root is `/kaggle/working/cell-tracking/center-comparison`.
It contains source-derived frames and point caches, a manifest, inference
receipts, graph excerpts, the static `site/`, and validation receipts. The site
is approximately 115 MB; it streams compressed native volumes into a bounded
six-frame browser cache. Images are exported as 8-bit display volumes with fixed
pooled 1st–99.9th-percentile limits per clip. Native instance IDs are losslessly
exported as little-endian uint32 volumes. Neither display normalization nor
visualization edits change model inputs, masks, or selected tracking outputs.

Build or refresh the complete detection index after changing predictions:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m center_comparison index-detection
```

Then start `serve` as above, or restart an already running viewer server. This
stage reads raw proposals from
`/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/inputs`,
P0 and C4_m6 predictions, and canonical GEFF annotations. It checks every final
prediction against its evaluation receipt, reproduces official scoring counts
for all 398 model/clip pairs, and matches raw proposals independently. An atomic
SQLite index and frozen graph snapshots live under `detection-review/` (about
254 MB). Canonical microscopy is read on demand, with bounded frame caches; no
full microscopy copy or FOCUS inference is needed. Missing native chunks cause
an explicit error. `--audit-limit` is for partial development checks only.

Rebuild using the existing isolated runtimes:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m center_comparison prepare
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-focus-v6/bin/python -m center_comparison focus
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m center_comparison export
```

To add or refresh the complete pipelines and error locations on an existing
export, without rerunning segmentation or rewriting microscopy volumes:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m center_comparison add-best
```

`export` includes this stage. It reads P0 from
`/kaggle/working/cell-tracking/segmentation-tracking-v6-local/prior-v6/ram-outputs/predictions/P0`
and C4_m6 from `/kaggle/working/cell-tracking/multidata-training-v4/predictions/C4_m6`.
It checks all 398 full-population prediction file hashes against their evaluation
receipts, reaggregates both 199-clip scores, and runs fresh official diagnostics
on the six viewed clips per method. Per-frame assignments at all five radii are
recomputed, with exact equality checked against the full-clip assignment at 7 µm.
The native images and masks, older comparisons, and source predictions stay intact.
`--best-predictions` and `--best-evaluation` override the restored P0 paths.

`prepare` reads canonical Zarr metadata and GEFF arrays, checks physical spacing
agreement, rejects missing image chunks, and loads the existing saved prediction
artifacts. The original pilot masks are reused only after checking pixel-exact
image agreement and checkpoint identity. `focus` resumes completed frames and
runs the same nuclei recipe for new frames: source
`5c4b53f743a0fbbae056e2c1a139895ae819f069`, nuclei checkpoint SHA-256
`b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a`,
physical radius 4.5 µm, confidence and mask thresholds 0.5, native-grid restoration.
The 30 additional masks were inferred on the RTX 4090.

For another window use `prepare --datasets NAME --start 0 --stop 99 --no-pilot
--output /kaggle/working/cell-tracking/another-comparison`, then use that same
`--output` in subsequent stages. `--checkpoint`, `--focus-source`, `--pilot`, and
`--data-root` override restored-machine paths. This exporter requires the saved
detector, v3 tracker, and V5 region artifacts for every selected clip; it does not
train or regenerate those methods.

Current method context and the original experiment:
[segmentation and center comparison](segmentation-center-comparison.md).
For contracts available in a Git-only checkout, see the
[competition notes](competition.md) and [scoring audit](pipeline-errors-20260915.md).
The locally synced official snapshots are `reference/overview/data-description.md`
and `reference/overview/evaluation.md`; these ignored files require reference sync.

## Verification

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m pytest tests/test_center_comparison.py -q
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m center_comparison.verify_data
PYTHONNOUSERSITE=1 PYTHONPATH=tools python -m center_comparison.verify_browser
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m pytest tests/test_detection_review.py -q
PYTHONNOUSERSITE=1 PYTHONPATH=tools python -m center_comparison.verify_detection_data
PYTHONNOUSERSITE=1 PYTHONPATH=tools python -m center_comparison.verify_detection_browser
PYTHONNOUSERSITE=1 PYTHONPATH=tools python -m center_comparison.verify_detection_selection
```

The detection checks independently verify physical distances and candidate counts,
complete annotation/edge-error coverage, and the last pages of both unknown pools.
Browser checks cover the default full-population view, old 28-case links,
first/last examples, page-boundary navigation, full/zoom pixels and crop geometry,
both pipelines, playback, filter reset, saved judgments, and desktop/mobile layouts.
Selection checks inspect rendered colors at the actual selected coordinates,
including exact overlaps; capture the actual 7 µm arcs in all planes; and exercise
canvas clicks, every-prediction selection, source switching, copied views and
real assignment conflicts. They check 3D distances independently and ensure
selection does not change assignments. Detection receipts and screenshots are
under `detection-review/`; the current record is
`results/center-comparison-viewer/overlay-selection-validation.json`. Earlier
navigation and full-data audits remain in `simple-review-validation.json` and
`detection-validation.json` beside it.

The eight unit tests cover mask IDs, centroid fallback, rounding, anisotropic
distance, non-greedy one-to-one association, division lineage identity, sparse
edge validity, official division misses, nearest-versus-assigned centers, root-cause
classification, grouped edge flags and preservation of a correctly linked daughter.
Artifact verification checks exact exported mask/display voxels, assignment
distances and uniqueness, source prediction coordinates, error-context graph
membership, consecutive FOCUS links, and 360 original pilot method/radius
comparisons. Browser verification exercises all 75 method/radius combinations,
both complete pipelines, category navigation, isolated before/after images,
independent native-pixel checks for each timepoint in all three planes, shared
case links, correct daughter branches, mobile case layouts, full-population
score labels, synchronized pixels, native plane indexing,
play/pause, rapid seeking, snapshot gaps, mask picking, selected-cell links,
and desktop/mobile layouts. Receipts and screenshots stay with the generated
output; concise validation is tracked in `results/center-comparison-viewer/`.
