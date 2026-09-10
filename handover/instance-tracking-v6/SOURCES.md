# Sources and inspected identities

Prepared 2026-09-10. Repository evidence was read through the GitHub app; public
model documentation was also checked on the web. Only code/docs were inspected
here, not microscopy or local weights. Live sources can differ later: local
Codex records resolved commits, source/weight hashes and source terms.

## User repository, pinned v5 snapshot

`matheuspf/cell-tracking@0a3105a8f9d1b0954707675b42b9868832df4ac1`:
- `AGENTS.md`, `docs/competition.md`.
- `handover/image-native-tracking-v5/CONTINUATION.md`: explicit in-progress status,
  partial completed scores, valid-region coverage and HOCT unit audit.
- `results/image-native-tracking-v5/ablation_scores.csv`: eight complete configurations,
  all 199 clips, including two annotation-assisted oracle diagnostics.
- `tools/image_native_tracking_v5/observations.py`: pooled seed watershed, physical
  regionprops, descriptor return rather than retained overlap masks.
- Earlier v4 continuation/results and external-data guide are inherited context;
  they do not establish v6 results or real dense segmentation annotations.

Browse the pinned user snapshot:
https://github.com/matheuspf/cell-tracking/tree/0a3105a8f9d1b0954707675b42b9868832df4ac1

## FOCUS-3D

Code pin: `5c4b53f743a0fbbae056e2c1a139895ae819f069`.
https://github.com/yu-lab-vt/FOCUS-3D
https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/notebooks/01_inference.ipynb
Notebook blob: `b23f465ccc2b8f17abf73bd87e5478d86b58a9d8`.
https://huggingface.co/Qinghua-thu/FOCUS-3D
The card lists general, nuclear and membrane checkpoints, declares Apache-2.0
weights and requires contact-sharing/access acceptance. GitHub code declares
BSD-3-Clause. These are source statements, not permission to accept a gate.
No checkpoint bytes or real Biohub performance were verified in authoring.

## Ultrack

Code pin: `5c94d845eb0a7b78c8dc24492ef00f218a467995`.
https://github.com/royerlab/ultrack
https://royerlab.github.io/ultrack/api.html
https://royerlab.github.io/ultrack/install.html
https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/utils/edge.py
`edge.py` blob: `61b449ff6d3157b2174571315f77ec696dabcba6`.
https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/linking/processing.py
`processing.py` blob: `2e95b0bf12ed021719c15207b667390e1e00d9c6`.
Inspected contracts: OR foreground, mean instance boundaries, optional smoothing;
Node.IoU default linking, target-bbox shift, scale for distance; public selected-
mask/track exports and annotation flags. Gurobi is optional per installation docs;
verify supported installed fallback, do not infer it from an unrelated solver.

## Cellpose-SAM

https://cellpose.readthedocs.io/en/latest/do3d.html
https://cellpose.readthedocs.io/en/latest/
Documentation fetched identifies the Cellpose 4.x API. Verified: explicit axis and
anisotropy controls, orthogonal flows plus 3D dynamics, flow_threshold unused for
3D, and separate slice-stitching route. Resolve/install/package and weight hashes
locally before training/inference. No exact release is pretended pinned here.

## StarDist

https://github.com/stardist/stardist
https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py
Inspected registry blob `d1699e8287da4236aed0509974fe2fa0b3a5a723`.
3D_demo release archive/checksum is recorded in model_registry.json. It is a demo
3D prior, not a universal performance promise. The README documents star-convex
object supports and real dense-mask requirements for supervised training.

## Scorer

Historical active pin: `royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`.
Use current local official scorer contract after a read-only comparison, not a
silently switched revision. Do not treat our reference helper as that evaluator.
