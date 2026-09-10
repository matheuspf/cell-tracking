# Evidence review: why v5 is not v4 with more training

Reviewed 2026-09-10, repository `cec119434f7001ea79beeb137a8f1063e5b0deeb`.
Primary evidence starts at [v4 CONTINUATION](../multidata-training-v4/CONTINUATION.md).
All numerical observations below are committed local results, not authoring-session
microscopy experiments. See [SOURCES.md](SOURCES.md) for immutable source links.

## Established outcomes

The incumbent remains C0/v3 A_residual_m3.0, **0.934802374260586**, with
**123,135 edge TP / 4,965 FP / 5,748 FN**, **29 division TP / 92 FP / 122 FN**,
and 4,108,943 predicted observations across 199 clips. Per embryo: 44b6
0.931664468721842; 6bba 0.935221784097327. V2's 0.934206 is historical, not the
current hurdle. The quoted public 0.946 notebook result concerns another population.

V4 completed 804,000 retained optimizer updates across 69 production fits. C4's
+0.000146776125 pooled gain failed on 6bba and its second seed lost 0.000260245840
pooled. C4_m6 reached 0.9351783702566538 but also failed 6bba. C7's Zoo rendering
learned the renderer without a qualifying Biohub gain. More data and compute alone
were not enough *for that representation and deployment pipeline*.

Sources: [report](../../results/multidata-training-v4/final_report.md),
[NEXT_AGENT](../../results/multidata-training-v4/NEXT_AGENT.md),
[v3 report](../../results/strong-tracker-v3/final_report.md).

## Distinguish an exhausted test from an untested capability

### A. No replacement detector population was tested

V4 `models.Detector` is a triplanar query encoder with scalar center and offset
heads. It refines supplied centers. It is not a full-volume candidate-producing
network. Its loss does not test whether a new nucleus, missing daughter, or
alternative split can be added and supported across time. The detector arms also
rebuilt continuations, making their total regression partly an association effect.

Therefore **"v4 detector arms failed" does not establish that new detections have
no value**. V5 separates fixed-node scoring, coordinate-only changes, new proposals,
and subsequent association changes. It keeps the native resolution/units explicit.

Sources: [models.py](../../tools/multidata_training_v4/models.py),
[detector attribution](../../results/multidata-training-v4/detector_attribution.csv).

### B. A small fork classifier is not the strong tracker backbone

V4 uses a three-convolution 2D frame encoder and masked pooled temporal statistics,
plus a separate relative-geometry MLP. It did not fine-tune the installed Harmonic
Fusion temporal 3D U-Net and node transformer against supported real transitions.
The distinction is module identity and input evidence, not the number of optimizer
steps or how much VRAM happened to be allocated.

V5 N adapts that native network. V5 H uses a separately pretrained, edge-centric
tracking model. Both use the rich real annotated transitions instead of making the
few available division bags carry the entire adaptation problem.

### C. Proposal, gate and decoder attrition are different limits

For C4, the exact-ID diagnostic has 104/151 fork observations covered by the
six-candidate next-frame pool, 36 after its geometry gate, one after the margin,
and zero newly accepted exact-ID fork edits. These are not the official tolerant
local division matches; they cannot be substituted for official TP/FP/FN.

V3's much larger pool also failed: approximately 190 million hypotheses, 98.4%
unknown labels, only 20 and 92 source positive groups, and a flood of false forks.
V5 will neither drop all guards nor repeat an enormous pair enumeration. It scores
whole temporal explanations with shared feature extraction and deduplicated timing
alternatives, using a mature representation and explicit birth competition.

Sources: [candidate coverage](../../results/multidata-training-v4/candidate_gt_coverage.csv),
[v3 report](../../results/strong-tracker-v3/final_report.md).

### D. The native association signal is worth preserving

V4 C4_relinked beat its matched C1_relinked control by +0.014698468708, but remained
0.012543955843 below C0. This says relative external improvement inside a worse
linker is not enough. V5 requires frozen-backbone/current-decoder controls and a
C0 comparison for every learner. It does not overwrite the strong learned links
with a geometry-only assignment and count recovery toward 0.95.

### E. Calibration cannot manufacture missing visual evidence

Generator temperatures and fork thresholds were not biological posteriors. Reused
source embryos also do not provide an untouched validation set. V5 reports both
limitations, but does not let lack of perfectly independent inner biology prohibit
all development: source-only dependent development blocks are allowed and labeled
as such. Opposite-embryo outcomes never tune that direction's weights or calibration.

## Why the new alternatives are plausible, not guaranteed

**HOCT** represents candidate links and interactions between them in a pretrained
tracking transformer. Its official implementation distributes checked pretrained
weights and supports a frozen-backbone edge probe from sparse corrections. This
is a different hypothesis from our from-scratch fork MLP. The abstract's benchmark
claims are not Biohub measurements. Its coordinate-only convenience function is
currently a stub: the plan explicitly uses the full graph/region-feature interface.

**Native 3D adaptation** retains strong image features and supplies real, matched
candidate neighborhoods during supervised learning. It tests whether the network
can learn identity/displacement/appearance mistakes that fixed heuristic features
could not, without requiring a dense real cell mask dataset.

**Multi-hypothesis image observations** allow time to choose a single cell versus
two distinct nuclei and recover omitted objects. This is inspired by segmentation-
hierarchy tracking, but v5 is not claiming a complete Ultrack implementation. First
use existing neural heatmaps plus image-supported marker watershed and exclusivity.
FOCUS is optional; it must not become another unexecuted external dependency.

## The 0.95 budget

The exact additive incumbent division term is 0.1*(29/243). Thus its adjusted-edge
contribution is **0.9228682178819851**. With that contribution unchanged, 0.95
requires division Jaccard >=0.2713178211801481. Illustrative count configurations:

| Division TP | Division FP | Division FN | Calculated combined score |
|---:|---:|---:|---:|
| 29 | 92 | 122 | 0.934802374261 |
| 66 | 92 | 85 | 0.950028711709 |
| 50 | 30 | 101 | 0.950492527274 |
| 60 | 60 | 91 | 0.951304236839 |

Alternatively, unchanged divisions require adjusted-edge contribution
**0.9380658436213992**. Neither route is a forecast. Actual graph changes alter
matching, count multipliers, division evidence, and per-sample weights. The local
`contracts.py` budget helper only calculates these conditional scenarios.

This makes the intent concrete: aim to recover tens of reliable additional
bifurcations and/or a material share of the remaining association errors, not
extract another 0.0001 from annotation filtering.
