# Evidence review and scientific change

## Pushed state inspected

User repository: `matheuspf/cell-tracking`, branch `handover/image-native-tracking-v5`,
commit `0a3105a8f9d1b0954707675b42b9868832df4ac1`.
`handover/image-native-tracking-v5/CONTINUATION.md` explicitly describes ongoing
execution, recoveries and queued native-model/fresh-image work. The committed
`results/image-native-tracking-v5/ablation_scores.csv` contains eight full-population
configurations, including two truth-assisted oracles. No complete final v5 adoption
receipt was established in this review. Local state can be newer: M600 must record it.

| Completed configuration | Full local score | Meaning |
|---|---:|---|
| C0 | 0.934802374260586 | Retained v3/v4 incumbent, 199 clips |
| H_general_J | 0.906022657469521 | HOCT with registered region adapter and J |
| H_probe_J | 0.9135868579084195 | Adapted HOCT with that adapter and J |
| J_native_frozen | 0.8773353943845731 | Changed evidence and decoder together |
| P_DC_native_J | 0.8686524362605309 | New-point confirmation with same J path |
| P_union_native_J | 0.6751326227250869 | Raw new-point union; severe link/fork errors |
| Oracle_fixed | 0.9576193465416518 | Annotation-assisted feasibility, not deployable |
| Oracle_augmented | 0.9721455963275534 | Annotation-assisted feasibility, not a bound |

The retained graph has 4,108,943 nodes; edge TP/FP/FN 123,135/4,965/5,748;
division TP/FP/FN 29/92/122; adjusted-edge contribution 0.9228682178819851.
These are committed measurements, not measurements rerun in this authoring session.

## What was already tried (do not erase this distinction)

`tools/image_native_tracking_v5/observations.py::regions` runs thresholded seeded
watershed on pooled images, caps the expansion at six working-grid voxels, and
returns 15 morphology/intensity descriptors, validity/collision flags and centroids.
It does not return instance masks for subsequent voxel-overlap scoring. Regions
are seeded from the point population and foreground thresholds, not independently
predicted by FOCUS or Cellpose. Their geometry is image-derived, not fake spheres,
but that experiment is not a learned segmentation-first tracker.

The v5 continuation records 3,456,758 valid regions for 4,108,943 C0 nodes and
4,848,861 candidate edges missing an endpoint region. It also reports a discovered
input-unit discrepancy: the registered adapter supplies physical diameters and
inertias while the pinned upstream HOCT extractor supplies voxel-unit features.
The native-voxel full-data comparison was NOT measured in that snapshot. The H
results therefore do not isolate the quality of the pretrained model or the
value of better instance segmentation. A fresh unit-parity test is mandatory.

Raw P_union raised sparse recall only slightly while producing 30,185 edge FP and
4,774 division FP. This is evidence against indiscriminately adding centers and
rebuilding links. It motivates object-level duplicate exclusion, mask confidence,
shape continuity and competing single-versus-split explanations, not merely more
peak candidates. Oracle scores establish possible graph headroom only; their
many false forks and GT access preclude deployment and do not predict v6 gains.

V4 completed actual external pretraining (804,000 retained optimizer updates),
yet no complete qualifying external advantage was adopted. Do not explain that
negative result as 'training never ran'. Its images/trajectories did not supply
real dense mask labels, and its detector was a query/offset experiment. Reusing
those sources as dense real-cell boundary truth would repeat a target mismatch.

## Falsifiable new hypotheses

H1: On exactly the same point detections and candidate links, real learned mask
extent/overlap/appearance separates correct and contradictory associations better
than centers, boxes, or the old watershed descriptors. This is tested without
letting detection count changes explain the result.

H2: Independent learned instance detections recover useful observations and avoid
false splits/duplicates better than the old point-union arm. This needs a complete
tracking score; a high annotation-center coverage alone cannot establish it.

H3: Retaining segmentation ambiguity until temporal optimization with Ultrack
outperforms forcing one per-frame mask partition. Test the same mask evidence,
not two unrelated models with different linking costs.

H4 (conditional): A small segmentation student trained on carefully selected
source-embryo teacher masks can preserve tracking utility at practical inference
cost. Teacher agreement is pseudo-supervision, not biological truth; a student
failing tracking parity does not replace the teacher.

## Nonclaims and guardrails

Bounding boxes are useful gates/crops and a controlled baseline, not substitutes
for irregular 3D support in crowded regions. A model may predict nuclei rather
than whole-cell boundaries; identify the fluorescent compartment before selecting
FOCUS nuclei versus membrane weights. Neither sparse point labels nor a smooth
looking track validates a mask's exact boundary. Final CSV still uses a single
representative coordinate and temporal links; full masks remain internal state.
The research target is >=0.95 local, preserving the 0.934802374260586 incumbent.
