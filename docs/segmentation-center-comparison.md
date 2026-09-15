# Segmentation and center localization for Biohub

Reviewed 12 September 2026. **Keep the existing center detector as the reference.
FOCUS-3D nuclei is the strongest mask-based candidate tested here, but it has not
earned a replacement of the detector.** For this competition, the immediate
objective is to find the annotated centers reliably; a more detailed cell mask
does not necessarily improve that objective.

This review includes a new experiment on 12 full spatial frames from six clips,
covering both supplied embryos and 91 annotated nodes. It measures centers before
new tracking or optimization. The panel is small and exploratory, and the existing
checkpoints have inherited exposure to these embryos. It cannot establish an
unseen-embryo winner.

**What is already available.**

- The main imported model is a `TemporalUNet3D` with a detection heatmap head,
  used inside `UNetNodeTransformer`. Detection is trained against point locations,
  not instance masks. It uses two image frames and a spatial grid downsampled by
  `(1,4,4)`, corresponding to 1.625 µm sampling on each axis. The upstream loader
  uses strided sampling. Its training target truncates coordinates to grid bins.
  This makes a native-resolution offset or local refinement a plausible avenue,
  but is not evidence that a constant coordinate shift will help.
  [Runtime adapter](../tools/image_native_tracking_v5/native_adapter.py),
  [installed detector source](/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/tracking_repo/scripts/train_unet_transformer.py:473).
- `DeepCenter` supplies a separate pretrained center heatmap used for rescue,
  veto, and image evidence. It is not a dense instance segmenter.
  [Adapter](../tools/image_native_tracking_v5/deepcenter.py).
- V4 includes compact image-patch center refiners with subvoxel offsets, trained
  with real and synthetic point supervision. These refine existing proposals;
  they do not discover an entire new set of cells. The prior matched-association
  comparisons did not establish a replacement: `D_real` and `D_synthetic`
  decreased tracking score by about 0.00151 and 0.00142 even against their own
  unchanged-center association control. Those are graph results, not a direct
  center-localization ranking. Reuse these controls before proposing another
  generic refiner. [Measured study](../results/multidata-training-v4/final_report.md),
  [attribution](../results/multidata-training-v4/detector_attribution.csv).
- V5 creates image-supported watershed regions seeded by current detections on
  the downsampled grid. It calculates geometric centroids and shape/intensity
  features; the named `optical_centroids` are not intensity-weighted centers.
  These regions are experimental evidence, and do not automatically replace the
  retained center coordinates. [Implementation](../tools/image_native_tracking_v5/observations.py:34).
- V6 contains FOCUS and Cellpose adapter scaffolding. Its historical report
  generated no learned masks. The restored FOCUS runtime and nuclei checkpoint
  now work: this review actually ran them. Cellpose was not benchmarked here.
  [Adapters](../tools/segmentation_tracking_v6/providers.py:31),
  [historical scope](../results/segmentation-tracking-v6/final_report.md).

**The two external choices.**

FOCUS-3D predicts 3D instance masks using a transformer-based model and combines
overlapping patches. Its release includes general, nuclei, and membrane weights;
this experiment used the nuclei checkpoint. The useful output is a nucleus
region from which we can calculate several candidate centers. Its scale,
thresholds, resampling, and patch stitching can all affect those centers.
[Source checkout](/home/mpf/code/kaggle/FOCUS-3D/src/focus3d/segmentation/FOCUS3D/inference.py:227),
[model release](https://huggingface.co/Qinghua-thu/FOCUS-3D).

Ultrack has no single required pretrained segmenter. Its zebrahub example uses
background-subtracted foreground detection, robust inverted intensity as an
approximate boundary image, and hierarchical watershed. The tracker normally
chooses among overlapping segmentation hypotheses using temporal evidence.
For an independent segmentation comparison, this experiment uses fixed cuts of
that hierarchy; those cuts are not final ultrack tracking outputs.
[Example](/home/mpf/code/kaggle/ultrack/examples/zebrahub/zebrahub.ipynb),
[foreground](/home/mpf/code/kaggle/ultrack/ultrack/imgproc/segmentation.py:50),
[hierarchy](/home/mpf/code/kaggle/ultrack/ultrack/core/segmentation/hierarchy.py).

**What the pilot measured.**

The same native `64 × 256 × 256` spatial frames were used for all methods.
The existing detector outputs were read before ILP selection; FOCUS and ultrack
were run afresh. The existing detector has temporal image context, whereas the
new mask runs see one timepoint. This compares the available pipelines, not
architectures under matched training and input context.

The following are pooled annotated-node recalls at 2, 3, and 7 µm, respectively:

- **Existing detector:** 67.0%, 89.0%, **100.0%**; 1,893 predictions over 12 frames.
- **FOCUS geometric centroid:** 71.4%, 78.0%, **97.8%**; 2,429 predictions.
- **FOCUS intensity-weighted centroid:** 69.2%, 76.9%, **96.7%**; 2,429 predictions.
- **Existing seeded-watershed centroid:** 47.3%, 68.1%, **89.0%**; 1,507 valid
  regions. This includes the effect of rejecting seeds without valid regions.
- **Ultrack fixed cut 500, geometric centroid:** 30.8%, 35.2%, **70.3%**;
  939 predictions. Cut 1000 reached 67.0% at 7 µm with 810 predictions.

At 1 µm, FOCUS geometric centroids found 30/91 points versus the detector's
25/91. At 7 µm, the detector found 91/91 and FOCUS 89/91. FOCUS therefore shows
promise for fine localization on some cells, but has a worse tail despite
producing 28% more candidates. Unmatched predictions cannot be called false
positives because the real labels are sparse.

Both methods have about 1.72 µm median error among their 7 µm matches. Their
conditional 90th-percentile errors are 3.07 µm for the detector and 4.91 µm for
FOCUS geometric centroids. Counting each unmatched GT point as 7 µm gives mean
errors of **1.77 versus 2.07 µm**. On the same 89 matched GT points, FOCUS improves
46, worsens 39, and ties 4; its mean error is nevertheless 0.18 µm higher. This
is why a median or a count of individual wins alone is inadequate.

The center extraction rule is also empirical. Intensity weighting made almost
no pooled mean difference versus FOCUS's geometric centroid on their 88 shared
matches, and its direction changed between embryos. A smoothed brightest voxel
was substantially worse overall: 85.7% recall at 7 µm, including only 82.1% on
6bba. For the existing watershed regions, even the 81 GT points matched by both
methods had mean error 0.53 µm worse than the original detector. Blindly moving
every detection to a mask centroid is not supported.

All retained ultrack hierarchy centroids together reached 80.2% at 7 µm with
2,297 overlapping candidates. That is an optimistic candidate-coverage
diagnostic, not an admissible segmentation or fair detector score. The tested
classical recipe needs substantial improvement before testing it as the main
center source; this result does not rank all possible ultrack segmentations.

The complete thresholds, embryo breakdowns, center rules, paired comparisons,
and timing receipts are in [the summary](../results/segmentation-center-review/summary.json).

**What our labels can establish.**

The competition provides 133,318 sparse GEFF nodes across 199 training clips,
with links but no dense cell masks. Native spacing is
`(z,y,x) = (1.625,0.40625,0.40625)` µm. The official per-frame matching uses
optimal bipartite assignment with a 7 µm distance limit. Tracking score also
depends on edges and a node-count penalty, so center recall alone is not the
competition score. [Official evaluation](../reference/overview/evaluation.md),
[pinned metric](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md).

Our prepared external inventory also supplies no dense masks. It includes
1,539 synthetic static images with 423,853 centers and 2,174 six-frame synthetic
sequences with 4,056,226 nodes. These paired images support center pretraining
and controlled detection benchmarks. Their complete synthetic object lists
also permit precision measurements within the simulation. They do not establish
precision on real Biohub images; the simulator's calibration on 44b6 further
limits claims of independent transfer. Zoo tracks lack paired microscopy, and
the inspected RIKEN measurements do not provide mask truth.
[Inventory](external-data-guide/README.md),
[training caveats](../results/multidata-training-v4/final_report.md).

Measure each candidate as follows:

1. Freeze predictions without consulting evaluation annotations. Convert all
   centers to the native ZYX coordinate system, then apply the intended integer
   submission rounding. Match in physical units using the official
   `tracksdata.DistanceMatching(optimal=True)` implementation. Rerun matching at
   each threshold; a 7 µm assignment filtered afterward is not identical.
2. Report annotated-node recall at 1, 2, 3, 5, and 7 µm; matched median/p90 error;
   error with misses capped at 7 µm; and signed Z/Y/X residuals. For recentering,
   also compare error on the same GT identities. Show both embryos separately.
3. Report predictions per frame and recall versus prediction budget while
   varying confidence or size thresholds. Extra candidates can improve sparse
   recall without improving usefulness. Keep overlapping hierarchy hypotheses
   in a separate coverage diagnostic.
4. Inspect dim cells, crowding, borders, and divisions. GT points in background
   reveal foreground failures; multiple labeled cells inside one instance can
   reveal merges. Sparse points cannot provide valid real-image Dice, mask IoU,
   segmentation AP, or ordinary whole-field precision/F1. Boundary evaluation
   would require a small, exhaustively labeled mask panel.
5. For the next confirmation, freeze a larger panel spanning 30–40 clips and
   early/middle/late times, targeting at least 2,000 annotated nodes. Separate
   development and assessment clip groups before tuning. Repeat any trained
   adaptation across embryo directions with every learned component respecting
   the split. Existing upstream exposure still prevents calling the current
   frozen-checkpoint comparison independent validation. The four visible test
   examples are training copies.

**Recommended next segmentation experiment.**

Keep the detector's centers as control. Compare a native-resolution,
detector-seeded watershed and FOCUS nuclei masks, first as optional center
refinements with an unchanged candidate count. Associate masks to detections
without GT, preserve the original center on missing or ambiguous assignments,
and separately test geometric and intensity-weighted centers. Only after that
test should FOCUS introduce additional detections. This separates localization,
new-cell recall, and the consequences of deleting existing detections.

For a learned replacement, point-supervised heatmaps with a native-resolution
offset head align with the labels we actually have. That is a proposal, not a
measured improvement; the unsuccessful V4 refiners are essential controls.
Among the tested methods that produce full masks, FOCUS yielded the strongest
center recovery. The existing point detector is the stronger reference for
robust center recovery overall; mask-boundary quality remains unmeasured.

Runtime remains a practical selection criterion. This unoptimized FOCUS script
averaged 15.06 seconds per frame on the RTX 4090, including model setup, output,
and center extraction; it reloads the model each frame. The CPU ultrack
preprocessing/hierarchy/two-cut experiment averaged 4.87 seconds. These are not
optimized throughput or matched-device benchmarks. At roughly 19,900 hidden
frames, even the full 12-hour notebook budget averages about 2.17 seconds per
frame for every stage combined. The current FOCUS invocation needs a substantial
throughput improvement before full-competition use.

**Experiment recipe and receipts.**

The panel selected three clips per embryo at the 35th, 65th, and 90th percentiles
of annotated-node count among clips labeled at both frames 25 and 75. Both
frames were used. Selection preceded model results; the panel is annotation
enriched rather than a random independent sample. It contains 24 GT nodes from
44b6 and 67 from 6bba. Cached GT coordinates and IDs were checked against actual
GEFF arrays before evaluation.

FOCUS source: `5c4b53f743a0fbbae056e2c1a139895ae819f069`. Nuclei checkpoint
SHA-256: `b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a`.
Parameters: `z_ratio=4`, physical radius 4.5 µm converted to 11.076923 XY pixels,
1st/99th-percentile normalization, confidence/mask thresholds 0.5, no final
minimum-size filter, batch 1, FP16, and restored native-grid labels. The model
resampled XY to 347 × 347 and evaluated 147 overlapping patches per frame.
Only this nuclei checkpoint and scale were tested.

Ultrack source: `5c94d845eb0a7b78c8dc24492ef00f218a467995`. The zebrahub recipe
uses `detect_foreground(sigma=25 µm)`, `robust_invert(sigma=1 µm)`, and an area
hierarchy bounded by 500–10,000 native voxels. Fixed horizontal cuts at 500 and
1000 retain only regions within those size bounds. Raw cuts can expose pixel
leaves; those are pruned before counting cells. Neither a temporal solver nor
GT-based hypothesis selection was used.

FOCUS and ultrack centers use geometric centroids, intensity-weighted centroids
after subtracting each region's 10th-percentile intensity, or the brightest
voxel after Gaussian smoothing with native-voxel sigma `(0.5,1,1)`. The existing
watershed cache uses its own earlier processing and only valid incumbent-seeded
regions. `repo_final` is included in the full summary as a tracker-conditioned
control, and is excluded from the main detector ranking.

Local reproduction and detailed outputs: [benchmark script](../work/segmentation-center-review/benchmark.py),
[frozen panel](../work/segmentation-center-review/manifest.json),
[full results](../work/segmentation-center-review/results.json).
Stages are `prepare`, `focus`, `ultrack`, and `evaluate`; the Python runtimes are
respectively `/kaggle/envs/cell-tracking-notebooks/bin/python`,
`/kaggle/envs/cell-tracking-focus-v6/bin/python`,
`/kaggle/envs/cell-tracking-ultrack-v6/bin/python`, and the notebook runtime.
Use `PYTHONNOUSERSITE=1`. Detailed images, masks, annotations, and checkpoints
remain outside Git. This review changed no production model or tracking policy.
