# Additional Cellpose models and native embeddings

The checkpoint-linked provenance audit confirms overlap between the incumbent secondary model’s published training list and every local assessment clip. This establishes training exposure in our benchmark. No hidden competition-test label leak was established. See the [full audit](incumbent-provenance-20260914.md).

**Evaluation:** 24 volumes, 12 adjacent pairs (25→26 and 75→76), six clips, two embryos, 180 annotated nodes. Predictions, proposal banks and native scores were frozen before this evaluation. This is an exploratory pilot on previously inspected public training clips. **No complete-clip competition score is measured here.**

Open [the standalone HTML report](../results/cellpose-embedding-probe-20260914/report.html) in a browser. It includes embryo filters, detector curves, association controls, counts and downloadable evidence. It needs no TSX or JavaScript build.

## Findings and decision

- **DINO-L is a modest detector improvement in this pilot:** R3 rises from 67.22% to 70.00% and R7 from 98.33% to 98.89%. Candidates fall from 4,520 to 4,377. Native-image parent retrieval rises by only one edge, 81/88 to 82/88. This is not convincing evidence of a tracking-score gain.
- **Naive unions improve tight recall but hurt association ranking:** the three-Cellpose union reaches R3 73.33%, yet image parent retrieval falls to 73/88, versus 81/88 with DINO-B. Incumbent plus all Cellpose raises R3 from 88.89% to 93.89%, but image parent retrieval falls from 85/88 to 71/88. R7 stays at 100% and candidates rise 3,798 → 6,878.
- **The native model can embed the new detections, but its image features do not consistently beat geometry.** On DINO-B, geometry gets 82/88 parents correct and the image ensemble gets 81/88; on DINO-L the figures are 80/88 and 82/88. On SAMv2 they are 83/88 and 80/88. These controls provide no broad association improvement from simply swapping proposal generators.
- **Keep DINO-L as a candidate for a wider detector test; do not promote these unions from this pilot.** More proposals are allowed and can help. They need learned selection or compatible instance-level association, followed by a fixed complete-clip scorer comparison. This experiment does not establish that increasing proposals must worsen the final metric.
- **There are no annotated close pairs or divisions in these selected frames.** Small-radius recall and nearby proposal competition are measured, but close-cell separation and division accuracy remain untested here. The earlier 400-frame panel and a division-focused, independently frozen panel are needed to cover these failure modes.

## Detector comparison on the identical 24 frames

Recall uses integer submission coordinates and the official distance matcher at each radius. The counts below are matched annotations / total annotations. Unannotated proposals remain competitors; sparse labels do not supply a conventional false-positive rate.

- **Incumbent**, 3,798 candidates: 1 µm: **27.78%** (50/180); 2 µm: **70.00%** (126/180); 3 µm: **88.89%** (160/180); 5 µm: **98.33%** (177/180); 7 µm: **100.00%** (180/180).
- **Cellpose DINO-B**, 4,520 candidates: 1 µm: **19.44%** (35/180); 2 µm: **59.44%** (107/180); 3 µm: **67.22%** (121/180); 5 µm: **93.89%** (169/180); 7 µm: **98.33%** (177/180).
- **Cellpose DINO-L**, 4,377 candidates: 1 µm: **27.22%** (49/180); 2 µm: **61.67%** (111/180); 3 µm: **70.00%** (126/180); 5 µm: **95.56%** (172/180); 7 µm: **98.89%** (178/180).
- **Cellpose SAMv2**, 3,990 candidates: 1 µm: **23.89%** (43/180); 2 µm: **60.00%** (108/180); 3 µm: **67.78%** (122/180); 5 µm: **94.44%** (170/180); 7 µm: **98.33%** (177/180).
- **DINO-B + DINO-L**, 5,329 candidates: 1 µm: **21.67%** (39/180); 2 µm: **61.67%** (111/180); 3 µm: **69.44%** (125/180); 5 µm: **95.56%** (172/180); 7 µm: **98.33%** (177/180).
- **DINO-B + SAMv2**, 5,169 candidates: 1 µm: **22.22%** (40/180); 2 µm: **63.89%** (115/180); 3 µm: **73.33%** (132/180); 5 µm: **95.00%** (171/180); 7 µm: **98.33%** (177/180).
- **All three Cellpose**, 5,790 candidates: 1 µm: **23.33%** (42/180); 2 µm: **65.00%** (117/180); 3 µm: **73.33%** (132/180); 5 µm: **95.56%** (172/180); 7 µm: **98.33%** (177/180).
- **Incumbent + DINO-B**, 5,690 candidates: 1 µm: **32.78%** (59/180); 2 µm: **82.22%** (148/180); 3 µm: **93.33%** (168/180); 5 µm: **99.44%** (179/180); 7 µm: **100.00%** (180/180).
- **Incumbent + all Cellpose**, 6,878 candidates: 1 µm: **35.56%** (64/180); 2 µm: **84.44%** (152/180); 3 µm: **93.89%** (169/180); 5 µm: **99.44%** (179/180); 7 µm: **100.00%** (180/180).

The union rule preserves the base bank and appends alternate-model points farther than 2 µm from every retained center. Within an added bank, original confidence determines order. No cross-model confidence calibration or GT-based filtering is used. Counts are deliberately allowed to change.

## Association with the same proposal banks

The existing TemporalUNet encoder produces 32-dimensional image features at every new center. Each bank is passed separately through the trained node transformer with its own candidate context. Geometry-only scores use negative physical distance. The zero-image control preserves the trained transformer, positions and coordinates. Its distribution shift means it is an ablation, not a retrained geometry model.

The primary result below is **correct top-1 parent / all eligible GT edges**: missed endpoints count as failures. Conditional parent accuracy is also shown among edges whose two endpoints were matched within 7 µm. This is a ranking diagnostic, not a globally consistent tracking graph or a competition score. A fixed 15 µm motion gate applies to all controls. The ensemble uses the notebook’s calibrated low-margin blend: secondary weight at most 0.15, margin 0.35, temperature 1.

- **Incumbent** — geometry: **84/88 (95.45%)**; conditional 95.45% on 88 available edges. zero image: **84/88 (95.45%)**; conditional 95.45% on 88 available edges. image embeddings: **85/88 (96.59%)**; conditional 96.59% on 88 available edges.
- **Cellpose DINO-B** — geometry: **82/88 (93.18%)**; conditional 95.35% on 86 available edges. zero image: **82/88 (93.18%)**; conditional 95.35% on 86 available edges. image embeddings: **81/88 (92.05%)**; conditional 94.19% on 86 available edges.
- **Cellpose DINO-L** — geometry: **80/88 (90.91%)**; conditional 91.95% on 87 available edges. zero image: **80/88 (90.91%)**; conditional 91.95% on 87 available edges. image embeddings: **82/88 (93.18%)**; conditional 94.25% on 87 available edges.
- **Cellpose SAMv2** — geometry: **83/88 (94.32%)**; conditional 96.51% on 86 available edges. zero image: **83/88 (94.32%)**; conditional 96.51% on 86 available edges. image embeddings: **80/88 (90.91%)**; conditional 93.02% on 86 available edges.
- **DINO-B + DINO-L** — geometry: **74/88 (84.09%)**; conditional 86.05% on 86 available edges. zero image: **77/88 (87.50%)**; conditional 89.53% on 86 available edges. image embeddings: **76/88 (86.36%)**; conditional 88.37% on 86 available edges.
- **DINO-B + SAMv2** — geometry: **77/88 (87.50%)**; conditional 89.53% on 86 available edges. zero image: **78/88 (88.64%)**; conditional 90.70% on 86 available edges. image embeddings: **75/88 (85.23%)**; conditional 87.21% on 86 available edges.
- **All three Cellpose** — geometry: **73/88 (82.95%)**; conditional 84.88% on 86 available edges. zero image: **75/88 (85.23%)**; conditional 87.21% on 86 available edges. image embeddings: **73/88 (82.95%)**; conditional 84.88% on 86 available edges.
- **Incumbent + DINO-B** — geometry: **76/88 (86.36%)**; conditional 86.36% on 88 available edges. zero image: **72/88 (81.82%)**; conditional 81.82% on 88 available edges. image embeddings: **75/88 (85.23%)**; conditional 85.23% on 88 available edges.
- **Incumbent + all Cellpose** — geometry: **71/88 (80.68%)**; conditional 80.68% on 88 available edges. zero image: **70/88 (79.55%)**; conditional 79.55% on 88 available edges. image embeddings: **71/88 (80.68%)**; conditional 80.68% on 88 available edges.

## Tight localization and ambiguous neighborhoods

A close annotated pair has centers separated by at most 7 µm. Both must receive distinct detections under the official matching. These annotations are sparse, so this does not enumerate every crowded region.

- **Incumbent**: 20/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **Cellpose DINO-B**: 39/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **Cellpose DINO-L**: 47/180 annotations have multiple proposals within 7 µm; 1 duplicate integer points.
- **Cellpose SAMv2**: 37/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **DINO-B + DINO-L**: 63/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **DINO-B + SAMv2**: 66/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **All three Cellpose**: 74/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **Incumbent + DINO-B**: 106/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.
- **Incumbent + all Cellpose**: 119/180 annotations have multiple proposals within 7 µm; 0 duplicate integer points.

The selected pairs contain **0 annotated division events**. That is too little evidence to establish division performance.

## Per-embryo results

- **44b6 / Incumbent**: 2,668 proposals; R3 89.36% (42/47); R7 100.00% (47/47); image parent retrieval 22/23 (95.65%) versus geometry 22/23.
- **44b6 / Cellpose DINO-B**: 3,238 proposals; R3 74.47% (35/47); R7 100.00% (47/47); image parent retrieval 21/23 (91.30%) versus geometry 23/23.
- **44b6 / Cellpose DINO-L**: 2,981 proposals; R3 76.60% (36/47); R7 100.00% (47/47); image parent retrieval 22/23 (95.65%) versus geometry 23/23.
- **44b6 / Cellpose SAMv2**: 2,710 proposals; R3 80.85% (38/47); R7 100.00% (47/47); image parent retrieval 21/23 (91.30%) versus geometry 23/23.
- **44b6 / DINO-B + DINO-L**: 3,638 proposals; R3 76.60% (36/47); R7 100.00% (47/47); image parent retrieval 19/23 (82.61%) versus geometry 20/23.
- **44b6 / DINO-B + SAMv2**: 3,553 proposals; R3 78.72% (37/47); R7 100.00% (47/47); image parent retrieval 20/23 (86.96%) versus geometry 22/23.
- **44b6 / All three Cellpose**: 3,868 proposals; R3 76.60% (36/47); R7 100.00% (47/47); image parent retrieval 19/23 (82.61%) versus geometry 20/23.
- **44b6 / Incumbent + DINO-B**: 3,861 proposals; R3 93.62% (44/47); R7 100.00% (47/47); image parent retrieval 20/23 (86.96%) versus geometry 23/23.
- **44b6 / Incumbent + all Cellpose**: 4,439 proposals; R3 93.62% (44/47); R7 100.00% (47/47); image parent retrieval 21/23 (91.30%) versus geometry 23/23.
- **6bba / Incumbent**: 1,130 proposals; R3 88.72% (118/133); R7 100.00% (133/133); image parent retrieval 63/65 (96.92%) versus geometry 62/65.
- **6bba / Cellpose DINO-B**: 1,282 proposals; R3 64.66% (86/133); R7 97.74% (130/133); image parent retrieval 60/65 (92.31%) versus geometry 59/65.
- **6bba / Cellpose DINO-L**: 1,396 proposals; R3 67.67% (90/133); R7 98.50% (131/133); image parent retrieval 60/65 (92.31%) versus geometry 57/65.
- **6bba / Cellpose SAMv2**: 1,280 proposals; R3 63.16% (84/133); R7 97.74% (130/133); image parent retrieval 59/65 (90.77%) versus geometry 60/65.
- **6bba / DINO-B + DINO-L**: 1,691 proposals; R3 66.92% (89/133); R7 97.74% (130/133); image parent retrieval 57/65 (87.69%) versus geometry 54/65.
- **6bba / DINO-B + SAMv2**: 1,616 proposals; R3 71.43% (95/133); R7 97.74% (130/133); image parent retrieval 55/65 (84.62%) versus geometry 55/65.
- **6bba / All three Cellpose**: 1,922 proposals; R3 72.18% (96/133); R7 97.74% (130/133); image parent retrieval 54/65 (83.08%) versus geometry 53/65.
- **6bba / Incumbent + DINO-B**: 1,829 proposals; R3 93.23% (124/133); R7 100.00% (133/133); image parent retrieval 55/65 (84.62%) versus geometry 53/65.
- **6bba / Incumbent + all Cellpose**: 2,439 proposals; R3 93.98% (125/133); R7 100.00% (133/133); image parent retrieval 50/65 (76.92%) versus geometry 48/65.

## Runtime and provenance

- **Cellpose DINO-L**: 19.04 seconds/volume, 1.12 GiB peak CUDA reservation across 24 volumes. Includes inference, reconstruction and writing outputs; excludes the shared GPU queue. Runtime is machine-specific.
- **Cellpose SAMv2**: 47.88 seconds/volume, 1.07 GiB peak CUDA reservation across 24 volumes. Includes inference, reconstruction and writing outputs; excludes the shared GPU queue. Runtime is machine-specific.

The additional checkpoints come from [MouseLand’s official Hugging Face repository](https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837): `cpdino` (DINOv3 ViT-L) and `cpsam_v2` (SAM ViT-L). We verified complete learned-weight loading and the downloaded SHA256 hashes. Source and recipes are pinned in [pilot-plan.json](../results/cellpose-embedding-probe-20260914/pilot-plan.json); the [adjacent extension](../results/cellpose-embedding-probe-20260914/adjacent/pilot-plan.json) adds the immediate next frames without altering that recipe.

The original embedding-plan draft incorrectly assumed that the existing pilot frames were adjacent. It was archived without execution; the corrected plan adds t26/t76 before any scores were inspected. Detector inference was already running under its unchanged plan.

The native encoders retain the incumbent’s training exposure. The Cellpose release does not provide an image-level training manifest sufficient to certify independence from this competition data. Freezing predictions before this evaluation prevents new label-dependent edits; it does not erase earlier exposure or make this pilot an independent embryo test.

Increasing proposals can recover missing endpoints and improve small-radius localization, but it also adds competing associations and can affect the final scorer’s count penalty. A full-clip run must measure the combined result. These pilot percentages cannot be substituted for that final metric.

For generalization, train the detector and any association encoder with the target embryo excluded from the start, then compare complete graphs with fixed selection/tracking controls and report both embryos separately. Fine-tuning only a new head on top of the all-train incumbent is not an embryo-held-out experiment.

## Reproduction and evidence

Use `/kaggle/envs/detector-screen-cellpose/bin/python` for detector runs and `/kaggle/envs/cell-tracking-notebooks/bin/python` for native embeddings/evaluation, with `PYTHONNOUSERSITE=1` and two BLAS threads. The shared GPU lock is honored per volume/pair.

1. `python -m tools.cellpose_models_probe prepare`; `detect --model cpdino` and `detect --model cpsam_v2`.
2. `python -m tools.cellpose_adjacent_probe prepare`; `detect --model cpdino` and `detect --model cpsam_v2`.
3. `python -m tools.cellpose_embedding_probe prepare`, then `banks`, `native`, `evaluate`.
4. `python -m tools.cellpose_probe_report`.

The [validation receipt](../results/cellpose-embedding-probe-20260914/validation.json) verifies all 216 banks and 216 native matrices, independently recounts 180 nodes and 88 edges directly from the canonical GEFF files, and records browser/offline/mobile checks. Re-run the numeric checks with `python -m tools.validate_cellpose_probe`.

- Metrics SHA256: `63a056aedac8f4711d04ab2400f9538f2703eb9471f7ca294aacaf877a5f5faa`.
- [Machine-readable metrics](../results/cellpose-embedding-probe-20260914/pilot-metrics.json), [bank hashes](../results/cellpose-embedding-probe-20260914/banks-lock.json), [native prediction hashes](../results/cellpose-embedding-probe-20260914/native-lock.json), [embedding plan](../results/cellpose-embedding-probe-20260914/embedding-plan.json).
- The earlier 400-frame baseline (Cellpose R3 71.76%, R7 95.89%; incumbent R3 85.82%, R7 99.50%) is a different cohort. The numbers in this report must be compared within the 24-frame cohort.
