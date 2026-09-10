# Evidence review and changed hypothesis

Reviewed 2026-09-10 at cell-tracking commit
`0a3105a8f9d1b0954707675b42b9868832df4ac1`.

## What is measured, and what is not

The parent continuation records eight completed v5 comparisons and ongoing
native fits/replicas/fresh validation. It is not a final study. Use the latest
local receipts at S600, keeping this authoring snapshot identifiable.

| Complete result in reviewed snapshot | Full local score |
|---|---:|
| C0, independently reproduced | 0.934802374260586 |
| J, primary-only native evidence plus new decoder | 0.8773353943845731 |
| P_union_native_J | 0.6751326227250869 |
| P_DC_native_J | 0.8686524362605309 |
| H_general_J | 0.906022657469521 |
| H_probe_J | 0.9135868579084195 |

The table lists relevant complete comparisons, not all registered configurations.
No pending native model result is inferred. C0's recorded embryo scores are
0.931664468721842 (44b6) and 0.935221784097327 (6bba).

The new-peak bank matched 1,614 annotated observations missed by C0, without
losing a C0 match in that candidate diagnostic. But the union tracker exported
4,521,204 nodes, 30,185 edge FP and 4,774 division FP. New candidate recall alone
was not a tracking gain. The truth-assisted fixed-node and augmented-node
heuristics reached 0.9576193465416518 and 0.9721455963275534; they are not feasible
inference systems or rigorous upper bounds.

## Why this is not simply v5 again

`tools/image_native_tracking_v5/observations.py::regions` computes a marker
watershed from a pooled image/probability map, truncates it by distance to seeds,
and returns properties, validity, collision flags and centroids. It does not
return the label image. The continuation says the watershed combines C0 markers
and new peaks even when P0 graph nodes are fixed. Adding markers can therefore
change P0 morphology without changing its point universe.

This implementation already used SOME image-derived morphology. Claiming that
all prior work was literally points only would be inaccurate. What remains untested
is independent pretrained segmentation with masks retained as the objects used
for overlap, motion, split alternatives and full tracking optimization. V6 requires
that representation and an ablation proving the extra signal actually matters.

## Model-interface problem that must not be repeated

The v5 audit found physical coordinates/diameters and squared-micrometer inertia
were fed to a model whose pinned reference extractor supplies voxel-valued features.
All 655 audited source regions matched the unscaled upstream extractor. Changing
units altered mean absolute logits by 0.812178 and 0.300645 on two source tiles.
No full upstream-default-voxel comparison was registered/run in the reviewed
snapshot. Do not conclude from those HOCT scores that valid mask-based HOCT fails.

V5 also recorded 3,456,758 valid regions for 4,108,943 C0 observations. Missing
regions affected millions of edges. The negative H comparisons combine region
coverage, unit/domain mismatch, new score models and a new decoder; they do not
isolate the value of true instance segmentation.

## Falsifiable v6 hypotheses

H1: independent image segmentation yields more stable, meaningful object extents
than the shared marker watershed, without catastrophic merge/duplicate behavior.
H2: on identical nodes, centers and candidate links, real boxes/masks and shared
image-derived warps improve association/division scoring relative to centroid-only
and shape-erased controls.
H3: keeping competing segmentations until temporal optimization avoids the new-peak
union explosion and recovers useful missing/split objects.
H4: a hybrid using the full incumbent native evidence preserves its strong
continuations while exploiting region evidence where it is informative.

Reject or qualify these hypotheses separately. A pretty mask overlay is not H2;
more matched points is not H3; a solver-only gain is not evidence for H1.

## Biological/output distinction

The challenge's public introduction describes fluorescent nuclei. Segment the
observed compartment and verify the exact marker locally. Nuclear volume, shape
and intensity can provide identity evidence, but cannot establish unseen cell
membranes. Whole-cell contact graphs are not justified by nucleus masks.
Kaggle still accepts center/edge rows, not bbox/mask submissions. The conversion
happens after region-level tracking; segmentation remains internal evidence.

Sources and exact paths are in SOURCES.md. Existing v1-v4 negative findings remain
valid for their measured treatments, not for every future segmentation method.
