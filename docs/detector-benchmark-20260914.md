**The current Temporal U-Net ensemble has the highest measured recall on this
panel, with verified overlap in its published training records.** None of the tested released checkpoints
improved annotated-node recovery at the competition's 7 µm cutoff. The incumbent
matched **2,371 / 2,383 nodes (99.50%)**, with 116,533
candidates across 400 assessment frames. Cellpose `cpdino-vitb` was the strongest
new checkpoint at **2,285 / 2,383 (95.89%)**, with 156,226 candidates.

These are results on available labeled competition clips. The
[checkpoint-linked provenance audit](incumbent-provenance-20260914.md) verifies
all six files in the secondary snapshot manifest, including the exact incumbent
weight. Its recorded training list contains all 199 public clips and all 40
assessment clips; all 40 monitoring clips also occur in training. Training source
is shipped, although the exact historical run is not fully reproducible from it.
Treat the ensemble as a training-exposed reference. No hidden competition-test
label use was established. This comparison does **not** establish generalization to
unseen embryos or the relative potential of these architectures after fine-tuning.
The [subsequent provenance audit, tighter-error measurements and development plan](cellpose-detector-plan-20260914.md)
supersede the original recommendation to retain the incumbent on this evidence.

Eight pretrained model families were tested locally on the RTX 4090. Seven ran
on the complete assessment panel; AnyStar was screened on the separate pilot.
Repositories were downloaded under `/home/mpf/code/kaggle`. The
[metric evidence](../results/detector-screen-20260914/summary.json),
[exact commits, checkpoint hashes and verification receipts](../results/detector-screen-20260914/provenance.json),
and [reproduction instructions](../tools/detector_screen/README.md) preserve the
experiment.

The complete 400-frame results, ordered by each family's best measured 7 µm
recall, are:

- [Cellpose](https://github.com/MouseLand/cellpose), `cpdino-vitb`: **95.89%**;
  156,226 candidates, **1.34×** incumbent. At a per-frame cap equal to the
  incumbent's count, recall is **93.16%**. It recovers 2 incumbent misses but loses
  88 incumbent matches.
- [OrganoidTracker 2](https://github.com/jvzonlab/OrganoidTracker): **95.80%**;
  320,015 candidates, **2.75×** incumbent. Same-cap recall is **76.71%**. It
  recovers 9 incumbent misses but loses 97 incumbent matches.
- [PAC-MAP](https://github.com/DeVosLab/PAC-MAP), spheroid checkpoint:
  **91.19%**; 142,786 candidates, **1.23×** incumbent. Same-cap recall is
  **89.05%**.
- [CELLECT](https://github.com/zzz333za/CELLECT): **90.94%**; 155,152
  candidates, **1.33×** incumbent. Same-cap recall is **88.04%**.
- [Xenopus StarDist3D](https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari):
  **85.10%** using mask centroids, versus **84.77%** using predicted polyhedron
  origins. Both produce 145,475 candidates, **1.25×** incumbent. Centroid
  same-cap recall is **72.26%**.
- [Spotiflow](https://github.com/weigertlab/spotiflow), `synth_3d`:
  **70.92%** at threshold 0.05, versus **68.49%** at the default 0.3. The loose
  setting produces 58,096 candidates and has **69.74%** same-cap recall. This
  fluorescent-spot checkpoint was transferred to nuclear-scale inputs.
- [NucVerse3D](https://github.com/Segovia-lab/NucVerse3D), generalized unscaled
  checkpoint: **69.07%**; 52,668 candidates, **0.45×** incumbent. The 1× cap
  leaves its outputs unchanged.

The separate 12-frame pilot has 91 annotated nodes. AnyStar-mix recovered
**44 / 91 (48.35%)**, using 4,949 candidates; neither origins nor mask centroids
justified extending that preset. A loose Xenopus threshold recovered 91 / 91
but required **10.39×** incumbent candidates and fell to **62.64%** at the same
candidate cap. FOCUS-3D's earlier **89 / 91** result is retained as a pilot
reference, not a new 400-frame result.

The error range that matters here is narrow. Of the incumbent's 12 misses,
**11 have a nearest candidate 7.013–8.723 µm away**; two are less than 0.1 µm
outside the cutoff. The remaining miss has a candidate 5.873 µm away that is
used by another GT node under one-to-one matching. A nearby candidate need not
represent the same biological cell, so these cases warrant inspection rather
than an automatic shift.

OrganoidTracker is the most complementary tested checkpoint on these cases:
it recovers nine, and **six of those nine survive the incumbent candidate cap**,
at errors of 3.495–5.420 µm. Cellpose's two recoveries also survive its cap.
Three incumbent misses remain unrecovered by every complete assessment method.
This supports testing selective local refinement of cases near the cutoff,
in crowded regions or at image boundaries. No fusion rule was fitted or
evaluated in this experiment.

The incumbent makes both annotated endpoints available for **1,152 / 1,159
sampled GT edges (99.40%)**. Cellpose reaches **1,099 / 1,159 (94.82%)**.
These are detector diagnostics under the official per-frame matches, not
predicted-edge accuracy, division accuracy or competition scores. Sparse GT
does not identify every real cell, so unmatched candidates are not known false
positives. Counts also precede temporal pruning and cannot directly predict
the final node-count penalty.

The frozen assessment contains 40 clips, 20 per embryo, selected at evenly
spaced ranks of annotation count; five adjacent frame pairs were sampled per
clip. Its clips exclude the six pilot clips. Evaluation uses the installed
`tracksdata` optimal one-to-one distance matcher, native ZYX spacing
`(1.625, 0.40625, 0.40625)` µm, and integer rounding/clipping. Secondary checks
cover 5 and 6 µm and confidence-ranked per-frame candidate caps. Only complete,
identical cohorts are compared.

All **4,168 prediction artifacts** passed coverage and finite-array checks,
including the threshold/centroid variants and prior reference. New inference
outputs also passed their receipt checks. Physical
coordinate fixtures, source-parity probes and matching-gate assertions accompany
the adapters. Every realized output is hashed. NucVerse retains upstream
unseeded Wiener preprocessing, so a fresh run need not be bitwise identical.
Cellpose kept eager batch-8 inference after a compiled probe changed its masks;
its recorded CPU-affinity change makes timings descriptive rather than a
controlled speed comparison.

This is a local pretrained-inference screen. No fine-tuning, temporal linking,
submission or active detector configuration changed. Kaggle notebook integration
was not tested. Large images, weights, masks and detailed outputs remain in
ignored `work/detector-screen-20260914/`.
