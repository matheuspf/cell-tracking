# Source and results review

## What the branch actually establishes

The latest inspected cell branch is image-native-tracking-v5 at `03ab557`.
Its CONTINUATION.md explicitly calls itself an in-progress checkpoint. At its
20:36 UTC update on 10 September 2026, seven of eighteen complete configurations
had been measured, native replication and other evaluations were still active.
A new plan must not label the pending arms failures or silently interrupt them.
Local Codex may append newer completed evidence to a v6 intake receipt, with
hashes/dates, but must not select late v5 weights on the target and call that a
preregistered independent comparison. Keep C0 as the fixed historical control.

| Complete 199-clip result in the inspected snapshot | Pooled score |
|---|---:|
| C0 / v3 A_residual_m3.0 | 0.934802374260586 |
| Frozen primary native evidence with J decoder | 0.8773353943845731 |
| Raw full-field peak union with native evidence/J | 0.6751326227250869 |
| DeepCenter-confirmed proposals with native evidence/J | 0.8686524362605309 |
| H_general_J | 0.906022657469521 |

C0 per embryo: 44b6=0.931664468721842, 6bba=0.935221784097327.
The P_union arm produced 4,521,204 nodes and division TP/FP/FN=41/4,774/110.
More candidate recall clearly did not yield a better final graph in that arm.
This does NOT mean all unlabeled peaks are false cells: sparse annotation cannot
establish dense detection precision. The J arm changes evidence and decoder
relative to full C0, so it is not a clean decoder-only ablation either.

The fixed/augmented-node truth-assisted diagnostics (0.9576193465416518 and
0.9721455963275534) show feasible corrections in those tested constructions.
They are not global upper bounds, predictions, valid inference, or target success.
The 0.95 gap from C0 is 0.015197625739414.

## Why segmentation-first is new, despite v5 morphology

`tools/image_native_tracking_v5/observations.py` currently uses downsampled native
images and neural peaks. It seeds watershed at candidate points and bounds region
expansion to six native voxels. It returns 15 scalar properties, valid flags,
collisions and optical centroids. `cache.py` saves those summaries and sampled
native features, not persistent instance masks/boxes used for inter-frame overlap.

Thus v5 did extract image-supported shapes, but it did not test a full external
instance segmenter whose mask identity, deformed overlap and split/merge alternatives
remain first-class variables throughout tracking. V6 must do that. Replacing one
point detector then using only nearest centroids does not meet this direction.

C0 points are still useful as image-derived prompts, anchors and fallback nodes.
They are not mask truth. In the first controlled arm preserve C0 nodes/coordinates
and full original evidence; attach masks and test the incremental association
information. In independent arms allow segmenters to discover/reconstruct objects
without requiring a C0 seed. A bad enrichment result must not block these arms.

## Two concrete integration defects to avoid

1. **Units.** V5's audit found its HOCT adapter passed micrometer positions/diameters
   and squared-micrometer inertia, while the pinned upstream extractor returns
   voxel-unit features. The audit came after outer scoring. Changing to the native
   voxel convention changed logits and some parent choices on source-only tiles.
   H_general_J is therefore not a clean verdict on the pretrained model. V6 must
   verify each learned model's feature units/normalization BEFORE target scores.
   Canonical raw spacing is (1.625,0.40625,0.40625) micrometers; v5's [1,4,4]
   downsampling produces 1.625-micrometer isotropic native voxels. Do not confuse
   either with isotropic raw pixels. Preserve actual resampling transforms.
2. **Moved-node serialization.** `common.save_delta` writes added/removed node IDs,
   but not changed coordinates of existing IDs. A segmentation-derived recentering
   can disappear during round trip if this writer is reused unchanged. V6 needs
   immutable full graph outputs or explicit updated-node records, and coordinate
   round-trip tests. C0 fixed-coordinate controls must truly remain fixed.

## Data, labels and evaluation

Tracked `docs/competition.md` records 199 (100,64,256,256) uint16 TZYX clips, two
embryos, sparse GEFF graphs, physical-distance matching and a node-count-adjusted
edge/division metric. Masks are internal inference evidence; the submission still
requires integer voxel nodes and temporal edges, not mask files or bboxes.
Read current local official references/scorer before freezing an execution.
The referenced overview file was not available through GitHub; its local copy
is a Codex intake task, not something this author claims to have reread.

The downloaded synthetic/Zoo/RIKEN inventory supplies **no dense cell masks**.
Synthetic centers are not segmentation ground truth; Zoo lacks paired microscopy;
RIKEN label semantics are incomplete. Do not train a standard exhaustive mask
loss by marking all unannotated cells/voxels background. Optional dense-mask
training requires genuinely documented coverage, external mask data, or explicitly
weak/pseudo-label losses with unknown regions ignored.

The repeatedly used two embryos and inherited teacher/checkpoint exposure limit
validation independence. Preserve the operational 199-clip comparison, use
opposite-source fitting and frozen recipes, and report it as exploratory rather
than inventing new unseen embryos or a leaderboard score.

## Author review scope

Read via GitHub: latest branches/PR context, AGENTS.md, data skill, competition and
external-data guides, v5 continuation, observation/cache/common/evaluation source;
FOCUS README/inference notebook/loader; Ultrack README/core/API; StarDist README;
current Cellpose documentation and official FOCUS model card. No microscopy,
private weights, training or official graph scoring was executed here.
