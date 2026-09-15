**Most remaining error is ordinary frame-to-frame tracking.** This audit concerns the latest temporal-image + one-frame-persistence arm, score **0.82415820**, on the same six full 100-frame clips. Its matched control scores 0.82813536; selected v3 scores 0.97178168.

Ordinary temporal tracking dominates the remaining gap: 86.1% of the 0.14762 score difference to v3 comes from false and missed edges. The division-bonus difference contributes 6.0%. Current edge TP/FP/FN: 3635/376/391.

Of the 391 missed edges, 197 have both endpoints matched: 155 correct candidate links are rejected and 42 are pruned before optimization. Another 194 lack a matched endpoint: 165 already lack a raw Cellpose endpoint match, and 29 lose available raw matches during ultrack selection. No missing candidate between matched endpoints exceeds the 15 µm distance gate.

374 of 376 false edges touch an unmatched prediction. In 186 cases that point is still within 7 µm of the expected annotated cell, which was assigned to another prediction. This supports investigating competing detections and which observation the link chooses; the matching result alone cannot distinguish fragmentation, nearby real cells or duplicate centers.

Endpoint failures warrant a localization check. Among 160 unmatched annotated nodes after selection, 101 have the nearest predicted center between 7 and 10 µm away, while 59 have none within 10 µm. Even before ultrack, 80 of the 131 raw Cellpose misses fall in that 7–10 µm band. Proximity alone does not distinguish an off-center detection from a different nearby cell.

The 6bba_af149c94 and 6bba_fe670320 clips contain 442 of 767 edge FP+FN counts (57.6%). Division errors cluster elsewhere: 13 of 22 false divisions are in 44b6_a21120c2. Counts reflect different annotation coverage across clips.

The image model’s incremental regression has a different pattern from the overall error burden: versus the matched control it adds 22 false edges from forks and removes one false edge from single-child links. Thus the remaining total error is mostly ordinary tracking, while the image term’s extra FP cost is concentrated in branching.

The 155 rejected correct candidates face conflicting selected links: 80 have the source linked elsewhere and the target assigned another parent; 36 have the source linked elsewhere while the target starts a new track; 39 have the source end while the target has another parent. Thus every one of these misses competes with an alternative selected connection. This does not prove that all conflicts are local decisions; ultrack optimizes a joint graph.

**The remaining 0.14762348 score gap to v3 breaks down as follows:**

- Missing true edges: 0.06768933 (45.9% of the gap).

- Excess false edges: 0.05942796 (40.3% of the gap).

- Node-count adjustment: 0.01170989 (7.9% of the gap).

- Division bonus: 0.00879630 (6.0% of the gap).

This is exact descriptive metric accounting, with FP/FN contributions averaging both substitution orders. It is not a causal detector/linker decomposition. Division errors can also create false temporal edges; the division-bonus term does not capture every consequence of a bad fork.

Current node matching recovers 3,981/4,141 annotated nodes (96.14%), versus v3’s 4,096 (98.91%). Current and matched-control graphs both have 3,832 annotated edges with matched endpoints; v3 has 3,972. Relative to v3’s 103 edge FN, the current 391 comprise 140 additional misses with unavailable matched endpoints and 148 additional misses despite available endpoints.

**What the image model changes:** both persistence arms have the same aggregate matched-node and available-edge counts. The image arm increases rejected correct candidates from 139 to 155, and adds 21 net edge FP. The 44b6_a21120c2 clip alone adds 24 FP and six FN versus the control, while recovering the frame-52 division. Its 13 division FP account for most of the image arm’s 22. This points to the added image reward favoring wrong associations in that clip. Solver tolerances can also change individual edges; this comparison does not establish the biological identity of unmatched detections.

For that incremental comparison, false edges from forks increase from 20 to 42, while false edges from single-child sources decrease from 335 to 334. The net extra 21 FP therefore consists of 22 extra fork-associated FP minus one single-child FP. Overall error burden and the optical term’s specific regression must not be conflated. The image arm loses 31 previously correct edges and gains 15 different correct edges, producing the net loss of 16.

All four missed annotated divisions have raw Cellpose matches on their parent and both daughter sides under the official division-window matcher. Three retain matches on all three sides after selection; the frame-26 event in 6bba_e16ffc58 loses one daughter-side match. Side coverage is diagnostic and does not guarantee an available compatible daughter pair or a recoverable connected lineage.

Core-boundary missed-edge rates are 17/166 (10.24%) versus 374/3860 (9.69%) in the interior. This descriptive comparison gives little indication of a large boundary-specific concentration, but is not a windowing ablation.

**Priority suggested by the evidence:** improve localization at Cellpose centers and score continuation identity at those actual proposals; test candidate-link coverage before widening a distance threshold. A larger global division penalty addresses only a small part of the remaining errors. Train and calibrate any new model by embryo, and retain this reused panel as development evidence.

Fresh official scores for all 24 frozen current/control/stock/v3 graphs agree with their saved receipts. Graph hashes, the full candidate banks, actual annotation scales and the pinned unmodified scorer were checked. No model, prediction, link bank or tracking parameter changed.

[Machine-readable audit](../results/cellpose-ultrack-persistent-divisions-20260914/error-analysis.json), [all experiments](cellpose-ultrack-error-analysis-20260914.md), [division-window evidence](../results/cellpose-ultrack-persistent-divisions-20260914-pair-image/division-diagnostics.json), [official metric description](../reference/overview/evaluation.md).
