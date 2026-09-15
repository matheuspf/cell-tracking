**Adapting ultrack to Biohub — checked 12 September 2026**

Ultrack is technically compatible with this competition. The best next experiment is to combine image-derived segmentation hypotheses with the existing image-based link models, then let ultrack select a consistent lineage. The main uncertainty is whether the new cell centers and hypotheses improve the competition metric. The database, solver, and CSV bridge are tractable integration work.

The local checkout is `/home/mpf/code/kaggle/ultrack`, version **0.7.3**, commit `5c94d845eb0a7b78c8dc24492ef00f218a467995`. A live upstream check found the same main-branch commit. The official pages were refreshed with the project extractor: eight pages, no content changes. The forum index now has 102 topics; relevant September 10–12 threads were fetched.

**What was verified locally**

- All 199 training images and four visible test images have an array at `0`, shape `(100,64,256,256)`, dtype `uint16`, Zarr v3, and axis order `T,Z,Y,X`. The inspected root transform is `(1,1.625,0.40625,0.40625)`; spatial units are micrometers. Read metadata for every input rather than hardcoding hidden-test dimensions.
- The isolated runtime `/kaggle/envs/cell-tracking-ultrack-v6/bin/python` exists and successfully imported this checkout. It has Python 3.11.16, Zarr 3.1.6, Higra 0.6.13, python-mip 2.0.0, and CPU PyTorch 2.8.0. The main `cell-tracking` environment does not contain ultrack; the notebook environment is separate and has GPU PyTorch.
- CBC initialized and solved actual ultrack models. A synthetic eight-frame division produced **12 selected observations, 11 edges, one division**. Full-video solving and `window_size=3, overlap_size=1` produced identical observation/parent graphs.
- Three complete spatial frames from each embryo passed `labels_to_contours → segment → link → solve → to_tracks_layer → CSV`. For `44b6_0113de3b`, 385 candidates yielded 352 nodes and 234 edges. For `6bba_05b6850b`, 125 candidates yielded 122 nodes and 81 edges. Parent existence, adjacent-frame edges, out-degree ≤2, voxel bounds, integer fields, and CSV round trips passed.
- Those real-image probes used Gaussian filtering plus a top-1%-intensity connected-component mask. They establish execution and export compatibility only. They do **not** establish useful detection recall, correct divisions, a competition score, or full-video runtime.

Probe code and detailed receipts are in [work/ultrack-adaptation](../work/ultrack-adaptation/), especially [smoke-results.json](../work/ultrack-adaptation/smoke-results.json). These are ignored local artifacts. The ultrack checkout and production tracker code were left unchanged. The old [v6 report](../results/segmentation-tracking-v6/final_report.md) recorded ultrack as absent; that historical runtime blocker no longer describes this isolated environment.

**The required input adaptation**

Ultrack consumes a foreground map plus a contour map, or integer instance labels that it converts to those maps. It constructs alternative regions and selects compatible regions and temporal links jointly. A sparse GEFF graph supplies centers and edges, not foreground masks. A center heatmap also has a different meaning from a cell-occupancy or boundary map.

Start with two proposal sources: (1) ultrack's `detect_foreground` and `robust_invert` image preprocessing as a cheap control; (2) the existing detector's centers as seeds for a watershed on the actual image, or an available volumetric instance segmenter. Use physical spacing for smoothing and watershed distances. Preserve real occupancy masks, bounding boxes, and native-grid transforms. Synthetic balls around points can test plumbing, but cannot substantiate a benefit from cell shape.

To retain uncertainty, combine several plausible segmentations or foreground/contour thresholds before the hierarchy is solved. `labels_to_contours([labels_a, labels_b])` accepts multiple full `TZYX` label arrays; its union foreground and averaged boundaries are resegmented into a hierarchy. It does not promise to retain every original instance unchanged. Measure exact input-mask survival and the recall of competing split/merge proposals. Excessively large connected foreground regions can make the hierarchy expensive.

The project’s external-data inventory explicitly says its prepared collections contain no dense cell masks. Those points and trajectories cannot directly supervise a dense segmentation target. See [the external-data handover](external-data-guide/README.md) and the inspected [conversion implementation](/home/mpf/code/kaggle/ultrack/ultrack/utils/edge.py:18).

**Fix linking units and carry image evidence into the new candidates**

Call `link(config, images=(image,), scale=(1.625,0.40625,0.40625))`. With that scale, `max_distance` is in micrometers. Keep all stored/exported positions in native voxels. The `images` argument is a sequence of whole image channels, each `TZYX`; a bare `TZYX` array is iterated as time slices and is incorrect here. Supplying images enables intensity-based candidate filtering; it does not invoke the existing learned link model. A baseline can use `images=()` to isolate the effect of that filter. See [link](/home/mpf/code/kaggle/ultrack/ultrack/core/linking/processing.py:294).

The competition's 7 µm threshold is a prediction-to-GT matching tolerance, not a maximum movement per frame. An audit of all 199 actual GEFFs found 133,318 labeled nodes and 128,883 edges, all spanning exactly one frame. The 99th-percentile labeled step is 7.20 µm for `44b6` and 8.49 µm for `6bba`; division-edge 99th percentiles are 11.10 and 12.89 µm. There are 151 annotated division parents. The maximum observed edge is 60.76 µm; that is an outlier to inspect, not a justified global linking radius. These are sparse-label observations, not unbiased biological motion statistics. [Audit receipt](../work/ultrack-adaptation/gt-motion.json).

For a pilot, test candidate radii around 12–18 µm with a bounded neighbor count, measuring candidate-edge coverage at **predicted** centers. Do not hardcode 7 µm, and do not assume the labeled-step quantiles include detector localization errors. Tune separately for continuation and division coverage where appropriate.

Stock links use mask IoU minus an optional distance penalty and retain only a small number of neighbors. This can discard a true link before the learned model sees it, especially during fast motion or division. For the hybrid arm, build a sufficiently broad adjacent-frame candidate set from actual ultrack node IDs, then recompute native image features and scores at those exact regions/centers. Reusing nearest-neighbor scores from the original detector changes the meaning of the learned evidence.

Use the inspected `NodeDB` schema for hypothesis IDs/masks and [add_links](/home/mpf/code/kaggle/ultrack/ultrack/core/linking/processing.py:341) for a newly scored candidate set. That function appends; it is not an update operation. Either skip stock `link` for that arm or clear that experiment's old links before inserting replacements. Avoid parallel duplicate edges. Preserve the full primary/secondary/augmentation recipe when comparing with the current ensemble.

If custom edge scores can be negative, use `tracking_config.link_function="identity"` with deliberately calibrated weights. The default fourth power can turn negative costs into positive rewards. Birth, death, division, and edge weights must share a sensible objective scale. Optional `NodeDB.node_prob` values must be populated for all hypotheses or left unset for all; the current SQL solver rejects mixed availability. See [weight transforms](/home/mpf/code/kaggle/ultrack/ultrack/config/trackingconfig.py:84).

**Preserve divisions and export observation ancestry**

Ultrack's ILP already supports one incoming edge, up to two outgoing edges, appearance/disappearance, and incompatible segmentation hypotheses. It is a suitable structural model for this task. A one-to-one tracker would discard the division term. Stock ultrack uses one global `division_weight`; a dedicated learned score for each parent/two-daughter event would need an additional solver objective/constraint integration. Establish the stock and learned-edge baselines before adding that extension.

Export `to_tracks_layer(config, include_parents=True, include_node_ids=True)`. Its dataframe exposes `id` and `parent_id` for **observations**, alongside `track_id` and `parent_track_id`. Write each observation as one node; write `parent_id → id` for every non-root observation. The second returned object is a track-lineage mapping; it is not the complete edge list for submission. Do not reconstruct edges by equal track IDs, dataframe row numbers, or nearest centroids. See [export source](/home/mpf/code/kaggle/ultrack/ultrack/core/export/tracks_layer.py:10).

Round the chosen centers to integer native-grid `z,y,x`; undo any preprocessing crop/resampling transforms first. Keep `t` as the clip's frame index. The tested direct graph query already supplies the needed centroids and parents, so a full dense output-label volume is unnecessary for CSV generation. Retain selected masks separately only where geometry experiments need them.

Use the exact header `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`. Fill unused fields with `-1`. Use the full test-directory stem as `dataset`, validate IDs within each dataset, and assign a consecutive global throwaway `id` after combining datasets. Discover every hidden-test directory dynamically. The tested CSVs are partial probe graphs with diagnostic dataset names; they are not submissions. [Official contract](../reference/overview/evaluation.md).

The current scorer evaluates local directed division structure, two distinct daughter branches, and matching within a small window around a split. It is not a graph-wide weak-connectivity test. Use the unchanged official scorer at commit `075fc5f5a52d11077f9dc2b074644618f26939e2`, which still matches upstream main. The existing project metric adapter already pins it. [Organizer metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md), [host patch announcement](../reference/forum/topics/727154-division-metric-exploit-and-patch.md).

**Changes to the existing v6 adapter**

The integration entry point is [tools/segmentation_tracking_v6/ultrack_adapter.py](../tools/segmentation_tracking_v6/ultrack_adapter.py:60). Its API names match this checkout. The remaining work is concrete:

1. Run its preflight and import checks in the isolated ultrack interpreter; finding the source directory or checking the main environment alone is insufficient.
2. Replace `images=images` with `images=(images,)` and explicitly pass physical `scale`.
3. Remove eager `np.asarray(labels)` over whole videos; validate shape/dtype metadata and stream per-frame checks. Accept Zarr-backed arrays and explicit per-dataset cache paths.
4. Read actual candidate masks/IDs after `segment`, then implement the learned-link writer and exact-center feature extraction for the hybrid arm.
5. Convert returned observation parents to node/edge arrays and complete the retained-mask export. The adapter currently returns `export_validated=False`; it is still scaffolding.
6. Make the CLI, reports, and preflight reflect measured execution rather than the previous machine's fixed missing-runtime status.

For the mask-bearing v6 route, selected `NodeDB.pickle` objects supply mask/bbox/centroid data. Read only the locally generated database with the matching ultrack version. Preserve hypothesis identity rather than mapping back by proximity to the original instance labels.

**A minimal call sequence**

This illustrates the integration boundary once a segmenter has supplied native-grid `labels`. The numerical settings are pilot values, not a validated competition configuration. Each `run_dir` must belong to one new experiment and one dataset.

```python
from pathlib import Path
import zarr
from ultrack import MainConfig, segment, link, solve, to_tracks_layer
from ultrack.utils import labels_to_contours

def track_clip(dataset_dir, labels, run_dir):
    image = zarr.open_group(str(dataset_dir), mode="r")["0"]
    assert labels.shape == image.shape and labels.ndim == 4
    cfg = MainConfig(data={"working_dir": Path(run_dir)})
    cfg.segmentation_config.min_area = 100       # voxels; tune from real masks
    cfg.segmentation_config.max_area = 10_000
    cfg.linking_config.max_distance = 15.0      # µm because scale is supplied
    cfg.linking_config.max_neighbors = 8
    cfg.tracking_config.solver_name = "CBC"
    cfg.tracking_config.window_size = 25
    cfg.tracking_config.overlap_size = 4
    cfg.tracking_config.n_threads = 4
    cfg.tracking_config.time_limit = 20         # seconds PER solve window
    cfg.tracking_config.solution_gap = 0.01
    foreground, contours = labels_to_contours(
        labels,
        foreground_store_or_path=Path(run_dir) / "foreground.zarr",
        contours_store_or_path=Path(run_dir) / "contours.zarr",
    )
    segment(foreground, contours, cfg)
    link(cfg, images=(image,), scale=(1.625, 0.40625, 0.40625))
    solve(cfg)
    observations, _ = to_tracks_layer(cfg, include_parents=True,
                                     include_node_ids=True)
    # observations.id is node_id; each non-root parent_id -> id is an edge.
    return observations
```

Use the explicit stage calls for custom linking. In this checkout the high-level `track()` signature accepts `link_kwargs`, but its body does not forward them to `link()`.

**Runtime and packaging**

Force CBC for the first offline submission path. Installing `pyscipopt` does not give this checkout a SCIP backend: its configuration supports CBC, Gurobi, or auto selection. Porting the solver to SCIP is optional additional engineering. Gurobi availability in a local environment is not evidence that a licensed full-sized solve will work in Kaggle.

Process one clip at a time, use separate SQLite databases, cap hypothesis counts, and clean that clip's intermediates after export. At the observed shape, a full uncompressed raw clip is 800 MiB, bool foreground is 400 MiB, float32 contours are 1,600 MiB, and int32 instance labels are another 1,600 MiB. Together those buffers are about 4.30 GiB before hierarchy objects, database serialization, model activations, and worker copies. Streaming and compressed caches matter more than the tiny-probe RSS.

The notebook limit remains 12 hours with internet disabled. If the hidden set has approximately 199 clips, that is only about **217 seconds per clip**, including initialization, inference, segmentation, solving, and writing. A planning target around 150–180 seconds leaves some margin, but hidden size is approximate. The stock solver's 36,000-second time limit is per solve and unsuitable as a competition default. Measure complete 100-frame clips and impose a total remaining-time budget, not just a per-window timeout. [Current notebook requirements](../reference/overview/code-requirements.md).

Bundle wheels and authorized model assets before notebook execution. Match Python ABI and installed dependencies to the actual Kaggle image: this local ultrack interpreter is Python 3.11, while the downloaded-notebook runtime is Python 3.12. Current source requires Zarr ≥3; do not apply old advice to downgrade it to Zarr 2. A successful local CPU probe is not a Kaggle package or GPU-model validation.

**The first useful comparison**

Keep the present graph/decoder as a fixed control, then compare three arms on the same clips:

- **Same new centers, current linker/decoder:** reveals whether proposal geometry helps before changing the optimizer.
- **Same masks, stock ultrack IoU/CBC:** measures the value of joint hypothesis and lineage selection with the standard objective.
- **Same hierarchy, recomputed native learned links/CBC:** tests the recommended hybrid without changing proposals simultaneously.

Start with a balanced 10–20-clip development panel from both embryos, including crowded areas, dim cells, image borders, and annotated divisions; keep the final evaluation distribution representative. Log matched-node recall, candidate-link coverage, raw and adjusted edge Jaccard, division TP/FP/FN, total predicted nodes, per-stage wall time, peak RSS, database size, and feasible-solution status. Gate expansion on a verified native-grid export, no material recall loss, and plausible full-clip runtime.

Use the official aggregation: weighted per-sample adjusted edge Jaccard plus 0.1 times micro-averaged division Jaccard. Sparse unmatched cells are not ordinary negative labels; optimize density with the full score and recall, not sparse classification accuracy. Do not cap scores at 1 or substitute generic segmentation/CTC metrics. Train-time estimated node totals are evaluation information; they are not available as an inference oracle for hidden clips.

Then evaluate by embryo. Training on one embryo and evaluating on the other is the useful transfer test only if **all** learned components respect that split. The existing recorded C0 = 0.934802 and P0 = 0.934865 are exploratory scores with documented upstream label exposure, not independent validation or leaderboard results. The four visible test clips are training copies. [Embryo statement from the host](../reference/forum/topics/716793-only-2-groups-of-embryo-id.md), [existing experiment record](../handover/segmentation-tracking-v6/README.md).

**What the current forum adds**

In the refreshed [FOCUS3D discussion](../reference/forum/topics/738217-focus3d-one-of-the-best-3d-cell-segmentation.md), participants report center mismatch, ineffective oversegmentation, and dense FOCUS3D/ultrack tracks that did not improve their competition pipelines. One participant reports roughly 0.7 edge Jaccard from FOCUS3D plus ultrack. These are participant experiences with different configurations, not organizer benchmarks or a performance bound. They support measuring proposal geometry and re-evaluating the learned link model at the new centers before attempting pseudo-label distillation.

The [division-motion thread](../reference/forum/topics/740573-division-steps-are-not-long-steps-base-rates-from-the-t.md) also cautions against identifying divisions from displacement alone and against treating sparse annotated events as an unbiased sample. Our separate all-199-clip audit supports a broad division candidate gate, while leaving appearance and branch consistency to decide which pairs to accept.

Some forum examples use **ultrack-td**, a separate C++/tracksdata implementation. Its v1 compatibility layer documents missing features including temporal windowing, image features, and flow. It may be worth a later segmentation-throughput comparison; it should not be silently substituted for the checked SQLite/CBC API. [Upstream ultrack-td documentation](https://github.com/royerlab/ultrack-td), [zebrahub example](https://github.com/royerlab/ultrack-td/blob/main/examples/zebrahub.py).
