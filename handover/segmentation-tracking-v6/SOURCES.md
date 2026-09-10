# Source ledger — reviewed 2026-09-10

## User repository evidence

All user-repository reads are pinned to
`matheuspf/cell-tracking@0a3105a8f9d1b0954707675b42b9868832df4ac1`.

- [Current continuation](https://github.com/matheuspf/cell-tracking/blob/0a3105a8f9d1b0954707675b42b9868832df4ac1/handover/image-native-tracking-v5/CONTINUATION.md): in-progress scope, eight complete comparisons, unit audit, region/candidate coverage, raw-model/ensemble distinction and pending work.
- [Observation implementation](https://github.com/matheuspf/cell-tracking/blob/0a3105a8f9d1b0954707675b42b9868832df4ac1/tools/image_native_tracking_v5/observations.py): pooled marker watershed and returned properties rather than retained masks.
- [Competition snapshot](https://github.com/matheuspf/cell-tracking/blob/0a3105a8f9d1b0954707675b42b9868832df4ac1/docs/competition.md): data/CSV layout, scale, 199 clips and sparse labels. Dates/rules are historical snapshot facts; local execution must revalidate.
- [External data guide](https://github.com/matheuspf/cell-tracking/blob/0a3105a8f9d1b0954707675b42b9868832df4ac1/docs/external-data-guide/README.md): centers/graphs versus unavailable true masks. Existing v4 receipts, not this earlier inventory, govern what was actually trained.

No local microscopy, mask, weight or GPU experiment was accessed/run during
handover authoring. Repository reports are evidence of recorded local runs, not
an independent re-execution here. The current branch is not assumed final.

## Primary software/model sources inspected

- [FOCUS-3D README](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/README.md) and [inference notebook](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/notebooks/01_inference.ipynb): volumetric instance segmentation, output masks, z_ratio, headless infer_volume, tiling and refinement. Notebook demonstration settings are not measured Biohub settings.
- [FOCUS model card](https://huggingface.co/Qinghua-thu/FOCUS-3D): current general/membrane/nuclei checkpoints; Apache-2.0 weight declaration and contact-sharing access gate. No weights downloaded or terms accepted here. Resolve exact authorized weight hashes locally.
- [Ultrack README](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/README.md): segmentation uncertainty and hypothesis tracking.
- [Ultrack labels_to_contours](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/utils/edge.py): union foreground, averaged outer boundaries, smoothing and empty normalization caveat.
- [Ultrack linking](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/linking/processing.py): region IoU, distance-weighted links and target-to-source rounded bbox shifts.
- [Ultrack tracking config](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/config/trackingconfig.py): solver choice CBC/Gurobi, negative event penalties, utility transforms and window controls.
- [Ultrack API](https://royerlab.github.io/ultrack/api.html): Tracker.segment/add_flow/link/solve, labels_to_contours, custom links and exports. Public docs may drift from the pinned implementation; test actual signatures.
- [Ultrack algorithm abstract](https://arxiv.org/abs/2308.04526): selection of disjoint segmentation hypotheses with temporal overlap. No benchmark claim is assumed to transfer to this competition; no PDF figures were analyzed here.
- [Cellpose 3D documentation](https://cellpose.readthedocs.io/en/latest/do3d.html): orthogonal flows plus 3D dynamics, anisotropy, 3D flow_threshold behavior and slice-stitching distinction. Source ref resolved to [a54cb48849b7e225a81e8e43dcb042d42427f543](https://github.com/MouseLand/cellpose/tree/a54cb48849b7e225a81e8e43dcb042d42427f543).
- [StarDist source](https://github.com/stardist/stardist): 3D star-convex object representation, supervised mask requirements; [model registry](https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py), inspected blob `d1699e8287da4236aed0509974fe2fa0b3a5a723`, registers 3D_demo. Pin exact package/config/checkpoint locally; no broad 3D foundation-model claim.
- [Challenge introduction by its organizer](https://www.linkedin.com/posts/loicaroyer_celltracking-computervision-bioimaging-activity-7477489197923336192-ZD8J): fluorescent nuclei as visible objects. Confirm precise marker and acquisition metadata locally before naming model compartment.

These are implementation and model-card facts, not claims that any segmenter
already improves this repository. License declarations do not by themselves
establish eligibility for every intended use. Use existing authorized access,
record current terms and source provenance, and isolate uncertainty per provider.
