# Segmentation-first cell tracking v6 — active Codex handover

**New direction: track image-supported 3D objects, not just their point estimates.**
Repository: `matheuspf/cell-tracking` (not the Kaggriculture/PPO repository).
Branch: `handover/segmentation-first-tracking-v6`.
Parent: `handover/image-native-tracking-v5` at
`03ab55727c5248b292b248113eaa54123791ef03`, inspected 10 September 2026.

The parent is an **in-progress** v5 snapshot. C0 remains the measured incumbent:
**0.934802374260586**, target **>=0.95 complete local score**. Neither this handover
nor the parent's truth-assisted diagnostics establish a deployable improvement.

## Give Codex this prompt

> Read AGENTS.md, this document, and handover/segmentation-first-tracking-v6/.
> Execute X600–X690. Build real 3D instance detectors/segmenters and retain their
> masks, boxes, shape and appearance through association and division decisions.
> Prioritize FOCUS-3D when authorized weights exist, test an independent available
> segmenter, and integrate Ultrack. Compare center-only, box-only and mask-aware
> tracking fairly, then test full detector-driven reconstruction and bounded
> multi-segmentation hypotheses. Do not reduce masks to centroids before linking.
> Preserve C0 and v5. First inventory unfinished v5 jobs and results; do not kill,
> restart or overlap GPU jobs automatically, or switch a checkout used by running
> processes. Use a separate worktree when necessary. No need to finish or rerun v5
> as a prerequisite. Test the new direction rather than more point/fork-head grids.
> Respect sparse annotations, opposite-embryo calibration, units, source licenses,
> offline inference, and full official scoring. Missing FOCUS access closes that
> backend only, not the study. Do not accept gated terms or upload images for me.
> Implement, run, compare, and return maintained code, an inference package,
> measured results, visual mask/track review and CONTINUATION.md. Commit sanitized
> changes to this branch. Do not merge, submit, rent hardware, alter raw inputs,
> overwrite prior artifacts, or claim target success from oracle diagnostics.

No patch installer or activation overlay is required. This branch initially adds
only handover files; the existing tracker is unchanged. Detector environments,
weights and production modules are created by local Codex during execution.

## Start

```sh
git fetch origin
git switch --track origin/handover/segmentation-first-tracking-v6
# Do not run the switch in a checkout serving live v5 processes.
# Instead, if the local v6 branch does not yet exist:
# git worktree add -b handover/segmentation-first-tracking-v6 ../cell-tracking-seg-v6 origin/handover/segmentation-first-tracking-v6
```

Use the existing project Python (the v5 snapshot names
`/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`); inspect before
assuming that path exists. Read-only checks already supplied:

```sh
PYTHONNOUSERSITE=1 python handover/segmentation-first-tracking-v6/scripts/preflight.py
PYTHONNOUSERSITE=1 python -m pytest -q handover/segmentation-first-tracking-v6/tests
```

Read [REVIEW.md](handover/segmentation-first-tracking-v6/REVIEW.md),
[DETECTORS.md](handover/segmentation-first-tracking-v6/DETECTORS.md),
[PLAN.md](handover/segmentation-first-tracking-v6/PLAN.md),
[CONTRACTS.md](handover/segmentation-first-tracking-v6/CONTRACTS.md), and
[EVALUATION.md](handover/segmentation-first-tracking-v6/EVALUATION.md).
[SOURCES.md](handover/segmentation-first-tracking-v6/SOURCES.md) pins the review.
The registry contains planned experiments, not executed outcomes.

Keep new maintained modules under `tools/segmentation_tracking_v6/`, runtime wrapper
under `scripts/run_segmentation_tracking_v6.sh`, large outputs under
`/kaggle/working/cell-tracking/segmentation-first-tracking-v6` or ignored `work/`,
and compact results under `results/segmentation-first-tracking-v6/`.
