# Baseline, evidence and novelty boundary — revision 2

## Immutable starting point

Use `flexonafft/biohub-harmonic-fusion`, v29, script version 347965685. The user's
reported public score is 0.946; the repository verification is dated 2026-09-08,
not a live leaderboard verification by this planner. Another public notebook is
recorded tied at 0.946. Do not silently upgrade the source to a newer version.

Original local archive:
`/kaggle/notebooks/biohub-cell-tracking-during-development/flexonafft/biohub-harmonic-fusion/`.
Readable export: `/kaggle/working/biohub-harmonic-fusion.py`.
Inputs include the existing pilkwang support pack, DeepCenter center-prior and
seed314159 temporal U-Net datasets. The downloaded originals and weights are
ignored by Git and must be audited locally. The planner inspected committed
adapters and reports, not these unavailable original artifacts.

## Pipeline to preserve

The source adapters describe temporal 3D U-Net detections/features and a node
transformer, detection TTA and secondary harmonic fusion, candidate associations,
ILP graph selection, motion repair, one-frame/gap2 completion, safe divisions,
pruning, line-fit smoothing and integer CSV export. Eight-view detection TTA is
not proof that the learned association head is also augmentation-averaged.

The native adapter records `(T,Z,Y,X)`, downsampling `(1,4,4)`, metadata intensity
quantiles 0.001/0.999, lower clipping without upper clipping, two-frame image
context and five-frame decoding context. It explicitly documents truncation in
the installed feature extractor. These are source-review leads, not permission
to replace the full public implementation: the v5 native comparison omits public
TTA and secondary fusion. Verify raw-voxel, downsampled-grid and physical units,
center offsets, original coordinate precision and every tensor interface locally.

The traced parent probability normalization is `softmax(raw, dim=0)`: competing
parents normalize within a target column, not the reverse. Candidate threshold
is >0.48; traced ILP costs are edge=-p, appearance=0, disappearance=2, division=1.2.
A fork in this raw objective is dominated by making one daughter a free birth:
removing an edge of probability p and a division costs p-1.2 <= -0.2. Thus generic
ILP division-cost tuning is neither a baby-step inference fix nor part of this
study. Safe-division postprocessing remains essential to the public pipeline.

## Historical measurements (not fresh results)

| Configuration | All-199 local score |
|---|---:|
| Original public B0 | 0.911774 |
| Raw neural graph before repairs | 0.914903 |
| Full public pipeline with motion bypassed, B1 | 0.934206 |
| v3 selected residual pipeline, context only | 0.934802374260586 |
| v6 P0 residual pipeline, context only | 0.9348649864131336 |

B0 -> B1 per embryo: 44b6 0.912521 -> 0.931612; 6bba 0.911597 -> 0.934525.
Edge counts: 122201/6885/6682 -> 123023/4930/5860 TP/FP/FN.
Division counts: 23/97/128 -> 29/92/122. Removing all smoothing or safe divisions
was worse in the v2 complete ablations. E07 therefore protects branch points
while retaining ordinary smoothing; it does not simply disable smoothing.

V2 evaluated 104 complete variants. It already investigated learned native
association/fork selectors, deletion risk, temporal image classifiers and repair
combinations. V4's external-data training produced no adopted candidate. Do not
repeat these studies under new names or present the known B1 gain as new research.
The eight new arms are registered hypotheses, **not certified historically novel**:
check older local logs for an identical implementation, and cite/reuse exact
matching evidence rather than secretly re-searching it. Source-only differences
are insufficient if the actual tensors/graphs are unchanged.

## Validation limitations

Public checkpoints were trained/selected using supplied embryos, and the two
embryos and overlapping clips have been repeatedly examined. New splits, renamed
files or replaying the metric do not make them clean OOF. The 0.946 public score
uses a different population. Never add a local delta to it. Increasing the number
of candidates increases selection risk even without fitted parameters. Revision 2
therefore fixes each recipe, caps combinations, requires paired full-cohort
measurements and preserves failures. It reduces unbounded tuning, not all bias.

## Source map (pinned main unless stated otherwise)

- `AGENTS.md`; `.agents/skills/competition-data/SKILL.md`: runtime/data discipline.
- `docs/notebooks.md`; `configs/notebooks.json`: notebook versions and inputs.
- `docs/competition.md`: sparse labels, units, aggregation and submission overview.
- `tools/annotation_selection/public_lane.py`: original-notebook isolation.
- `tools/image_native_tracking_v5/native_adapter.py`: feature truncation and native interfaces.
- `tools/strong_tracker_v3/fresh.py`: normalization, ILP trace, fresh inference plumbing.
- `tools/strong_tracker_v2/replay.py`: actual repair order, flags and stage hooks.
- `tools/strong_tracker_v3/replay.py`: beware its hard-coded no-motion default.
- `results/strong-tracker-v2/v2_report.md`: complete ablations, counts and contamination caveats.
- `README.md`; `handover/segmentation-tracking-v6/CONTINUATION.md`: later retained methods.

## External implementation references, checked 2026-09-13

- PyTorch `grid_sample` documentation: coordinate conventions and volumetric
  interpolation. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
- MONAI inference documentation: overlapping-window aggregation and coverage
  normalization. https://monai-dev.readthedocs.io/en/stable/inferers.html
- Wang et al., *Aleatoric uncertainty estimation with test-time augmentation for
  medical image segmentation*, arXiv:1807.07356. https://arxiv.org/abs/1807.07356

These support implementation concepts and a general test-time-consistency
rationale, not evidence of a gain on Biohub. E04/E05/E06/E08 are proposed transfers
of those ideas. No MONAI dependency or new paper model is required. Check the
installed PyTorch API rather than upgrading to the documentation's current version.
