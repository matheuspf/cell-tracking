# Sources and inspection scope — 10 September 2026

## Pinned user repository

[Cell-tracking v5 snapshot](https://github.com/matheuspf/cell-tracking/tree/03ab55727c5248b292b248113eaa54123791ef03),
root tree `79170d7cf013540c5e7b285f34a43cb12a7bbfe6`, inspected through GitHub app.
This is the correct cell project, not the unrelated Kaggriculture PPO conversation.

- [Continuation](https://github.com/matheuspf/cell-tracking/blob/03ab55727c5248b292b248113eaa54123791ef03/handover/image-native-tracking-v5/CONTINUATION.md):
  in-progress status, measured C0/J/P/H comparisons, pending training, units audit,
  resources/recovery and local artifact paths. No inference about unfinished scores.
- [AGENTS.md](../../AGENTS.md), [data skill](../../.agents/skills/competition-data/SKILL.md),
  [competition guide](../../docs/competition.md): raw TZYX/physical/output contracts.
  The linked ignored official overview body was unavailable through the app; local
  refresh/scorer verification is X600, not an author-side completion claim.
- [External-data inventory guide](../../docs/external-data-guide/README.md): downloaded
  synthetic/Zoo/RIKEN labels and explicit absence of dense cell masks.
- [observations.py](../../tools/image_native_tracking_v5/observations.py),
  [cache.py](../../tools/image_native_tracking_v5/cache.py): seeded, bounded watershed
  properties, native grid and saved summaries rather than persistent overlap masks.
- [common.py](../../tools/image_native_tracking_v5/common.py): C0 paths, graph deltas,
  atomic writes, resource locks; coordinate-only updates absent from save_delta.
- [evaluate.py](../../tools/image_native_tracking_v5/evaluate.py): fresh official
  matching, graph hashes, all199 completion and exact aggregation.

`experiments.json` records Git blob anchors for critical reviewed files. Author
review did not open local images, detailed GT, checkpoints, running processes or
ignored score arrays; the in-progress continuation is the latest fetched evidence.

## FOCUS-3D

[Source](https://github.com/yu-lab-vt/FOCUS-3D/tree/5c4b53f743a0fbbae056e2c1a139895ae819f069),
[README](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/README.md),
[headless notebook](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/notebooks/01_inference.ipynb),
[loader](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/src/focus3d/segmentation/FOCUS3D/inference_win.py).
These were source-read, not installed/executed. The documented output is a 3D
instance map; the inference loader's permissive checkpoint behavior was inspected.

[Official model card](https://huggingface.co/Qinghua-thu/FOCUS-3D) lists general,
membrane and nuclei checkpoints, Apache-2.0 model weights and an access condition
requiring contact sharing. Availability/terms must be recorded when locally used;
no gated terms were accepted and no weights or microscopy were uploaded/downloaded
by this author. Do not infer Biohub performance from the method description.

## Ultrack

[Source](https://github.com/royerlab/ultrack/tree/5c94d845eb0a7b78c8dc24492ef00f218a467995),
[core track interface](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/main.py),
[database/hypotheses](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/README.md),
[API documentation](https://royerlab.github.io/ultrack/api.html),
[installation](https://royerlab.github.io/ultrack/install.html).
The inspected head fixes window-boundary phantom selections/dangling parents.
`labels_to_contours`, image flow, explicit links and solver inputs are documented;
actual output/units/solver compatibility are local integration tests. Read versions
matching the pinned package. No Ultrack runtime or licensed solver ran here.

## Alternatives and cautions

[Cellpose 3D documentation](https://cellpose.readthedocs.io/en/latest/do3d.html)
explicitly describes orthogonal 2.5D flow with 3D dynamics, anisotropy/axis settings
and ignored 3D flow_threshold. Audit the selected checkpoint and terms locally;
this document does not select an unverified weight file or infer segmentation AP.

[StarDist source README](https://github.com/stardist/stardist) describes 2D/3D
star-convex segmentation and dense mask requirements. Standard 2D pretrained models
are not 3D weights. Pin actual 3D assets/dependencies during X620. A nuclear-shape
prior is a rationale to test it, not a claim that it fits all fluorescent cells.

[Trackastra](https://github.com/weigertlab/trackastra) is an optional mask-conditioned
tracking comparison, not the mandatory path. Its exact pretrained 3D domain/units
were not audited to the same depth as the two user-requested repositories.

All comparisons, loss/gating proposals and budgets in this handover are new design
choices grounded in the above interfaces. They are not published performance
claims. Local code/weight/data license and competition-rule compatibility must be
checked separately; a source license alone does not authorize every data asset.
