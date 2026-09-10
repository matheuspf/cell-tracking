# Segmentation-first tracking v6

**New execution handover, not measured v6 results.**

The user requests a new direction: track image-supported 3D cell regions, using
bounding boxes and instance masks from FOCUS-3D and other segmenters, rather than
reducing each object to a center before association. Ultrack is the principal
segmentation-hypothesis/lineage backend. It is not itself a pretrained cell detector.

The delivery parent is `handover/image-native-tracking-v5` at
`0b6ce2d7a77baed54385d4c9b08acbefb6aa7c03`, including the update received during
review. The initial evidence review was at `0a3105a8f9d1b0954707675b42b9868832df4ac1`.
Read [PUBLISH_NOTE.md](PUBLISH_NOTE.md) for the later ninth completed configuration
and its scope. V5 is still IN PROGRESS, not a final negative study. The last
validated incumbent remains C0 / v3 A_residual_m3.0, **0.934802374260586**.
Target: **at least 0.95 on the full official local population**. No leaderboard
gain is implied. Keep all prior outputs and active jobs intact.

## Run locally

```bash
git fetch origin
git switch --track origin/handover/segmentation-tracking-v6
```

Open local Codex with the user's selected GPT 6 Pro model and give it
[CODEX_PROMPT.md](CODEX_PROMPT.md). No ZIP, activation patch, or PR merge is needed.
This handover is a plan with tested mask contracts, not a preimplemented detector
integration. Codex must implement and run S600-S680, not return another plan.

Read [REVIEW.md](REVIEW.md), [DETECTORS.md](DETECTORS.md),
[EXPERIMENTS.md](EXPERIMENTS.md), [IMPLEMENTATION.md](IMPLEMENTATION.md), and
[VALIDATION.md](VALIDATION.md). [SOURCES.md](SOURCES.md) pins the evidence.

## What must materially change

1. Run independent learned 3D instance segmentation on raw volumes: FOCUS-3D with
   authorized weights, Cellpose-SAM volumetric inference, and a bounded StarDist3D
   comparison if useful. Keep labels, masks, boxes, confidences and transforms.
2. Measure the incremental information from boxes, masks, morphology and
   image-derived motion-aligned overlap on IDENTICAL candidate nodes and centers.
3. Run actual Ultrack segmentation-hypothesis selection and mask linking, both
   standalone and with verified native association evidence. Preserve alternatives
   rather than immediately unioning all model detections into extra cells.
4. Validate division evidence from warped parent/daughter regions, plausible
   region evolution and persistent daughter separation. Do not assume perfect
   volume or fluorescence conservation during mitosis.
5. Export Kaggle's unchanged point/edge schema only AFTER region tracking.

For nuclear fluorescence the segmented object is the visible nucleus, not an
inferred membrane or whole-cell boundary. Confirm the marker from local official
references and images; retain `observed_compartment` in every dataset adapter.

## Available helper checks

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/segmentation-tracking-v6 -p 'test_*.py' -v
python handover/segmentation-tracking-v6/preflight.py --help
```

The mask helpers use NumPy. They are not a segmenter, production warp, optimizer,
or substitute for the official evaluator. They explicitly test equal-centroid /
equal-bbox objects with different masks, physical units, half-open boxes, warp
sign, division unions and legal graph export.
