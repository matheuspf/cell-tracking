**Cellpose cpDINO-ViT-B + ultrack scored 0.79986 on six complete 100-frame clips.**

This uses the full official competition metric, including temporal edges, divisions and the GEFF total-node adjustment.

The panel contains three clips per embryo and was inherited from the existing detector pilot. It is an exploratory local result on reused public embryos; hidden-test performance is unmeasured.

- Full score: **0.79986197**.

- Adjusted edge Jaccard: **0.79941554**.

- Unadjusted edge Jaccard: **0.80464301**; TP / FP / FN = 3,674 / 540 / 352.

- Division Jaccard: **0.00446429**; TP / FP / FN = 1 / 219 / 4.

Scores by embryo use the organizer's count-weighted aggregation:

- **44b6: 0.79940621**; edge Jaccard 0.81385979; division TP / FP / FN 0 / 58 / 2.

- **6bba: 0.80002877**; edge Jaccard 0.80120301; division TP / FP / FN 1 / 161 / 2.

Complete-clip scores:

- `44b6_81c256f0`: **0.94418851**; 10,795 selected nodes; division TP / FP / FN 0 / 6 / 0.

- `44b6_8f5ab931`: **0.73724441**; 36,100 selected nodes; division TP / FP / FN 0 / 45 / 0.

- `44b6_a21120c2`: **0.84676886**; 22,674 selected nodes; division TP / FP / FN 0 / 7 / 2.

- `6bba_af149c94`: **0.79427238**; 9,424 selected nodes; division TP / FP / FN 0 / 95 / 0.

- `6bba_e16ffc58`: **0.86113573**; 5,985 selected nodes; division TP / FP / FN 0 / 31 / 1.

- `6bba_fe670320`: **0.73703438**; 11,631 selected nodes; division TP / FP / FN 1 / 35 / 1.

Raw Cellpose recovered **4,010/4,141 annotated nodes (96.84%)**. After ultrack hypothesis selection, this was **3,980/4,141 (96.11%)**. Both rates are pooled by annotated-node count, distinct from the official summary's unweighted mean per-clip recall.

The selected graphs make both endpoints available for **3,830 GT edges**; **3,674 (95.93%)** of those are linked correctly. There are **540 evaluable false-positive edges**. Sparse unmatched cells and forks are not automatically known false positives.

Only five annotated divisions are present in this panel. Its division result is too small a sample to establish broad division performance.

The checkpoint is `cpdino-vitb`, the strongest new pretrained detector in the repo's earlier screen, SHA256 `3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f`. Its original eager BF16, batch-8, native-geometry 3D inference recipe was retained. There was no fine-tuning. Inherited pretraining exposure remains unresolved, as documented in the prior detector review.

Ultrack uses Cellpose labels to construct foreground/contours and hierarchical mask hypotheses, stock IoU link weights with a 15 micrometer physical gate, and CBC joint selection. Final runs all use 20-frame windows with overlap 5. Selected observation `id`/`parent_id` yields complete 100-frame graphs; the score is not an average of window scores. Masks, SQL coordinates and serialized linker centroids pass an agreement audit, and integer CSV export passes round-trip checks.

Two execution issues were resolved explicitly. A six-voxel disconnected component crashed hierarchy construction, so its existing minimum-component filter was set to exclude regions below eight voxels. Next, global CBC solved the first clip at 0.9488359062 but found no feasible solution on the dense second clip within 180 seconds. The final windowed recipe was applied uniformly, including rerunning the first clip. Those initial results/failures are retained separately; no mask or association weights were tuned against labels.

All **30 window solves** completed. Recorded statuses: {'OPTIMAL': 30}; largest relative gap **0.000278**. Per-window solver optimality does not establish optimality for a global 100-frame objective.

Cellpose used **92.2 processing minutes**, averaging **9.22 seconds per volume**, plus **17.3 minutes waiting for the shared GPU**. Tracking stages totalled **13.0 minutes** and overlapped later segmentation. These processing sums exclude failed diagnostic solves and are not end-to-end wall time.

A simple 19,900-volume extrapolation gives **51.0 hours** of local Cellpose processing before tracking. Full-199 and Kaggle notebook runtime were not measured.

Validation passed for all six complete graphs, the official edge/division accounting, CSV integrity, and exact agreement with **12 previously benchmarked Cellpose masks and center arrays**. Eight graph/metric tests and the additional real-ultrack tiny-component regression passed in their isolated runtimes.

No matched complete-clip FOCUS3D run was performed, so this study does not establish whether Cellpose + ultrack beats FOCUS3D + ultrack on the same cohort.

[Metric evidence](../results/cellpose-ultrack-20260914/summary.json), [validation and runtime receipts](../results/cellpose-ultrack-20260914/validation.json), [per-window solver receipts](../results/cellpose-ultrack-20260914/tracking-receipts.json), [final configuration](../configs/cellpose-ultrack-windowed-v1.json), and [reproduction commands](../tools/cellpose_ultrack/README.md).

The unmodified [organizer metric](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md) is pinned to `075fc5f5a52d11077f9dc2b074644618f26939e2`, verified against upstream main on 14 September 2026.

An independent reconstruction of the dense clip `44b6_8f5ab931` from identical masks and settings scored **0.73827398**, versus the original **0.73724441**: a change of **0.00102957**. The original graph remains in the aggregate. This exposes small run-to-run tracking variation; one repeat does not give a confidence interval. [Stability receipt](../results/cellpose-ultrack-20260914/stability.json).

Official local division-window matching finds parent-side and both daughter-side evidence after selection for **4/5 annotated divisions**. Of those available events, **3 are still not recovered** as valid divisions. This diagnostic separates local node availability from division topology; it does not replace the official division score. [Division diagnostics](../results/cellpose-ultrack-20260914/division-diagnostics.json).
