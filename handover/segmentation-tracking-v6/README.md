# Segmentation tracking v6

This is the single active handover for this iteration on
`handover/segmentation-tracking-v6` (PR #8).

**Execution record:** [CONTINUATION.md](CONTINUATION.md) and
[measured results](../../results/segmentation-tracking-v6/final_report.md).
**Plan:** [EXPERIMENTS.md](EXPERIMENTS.md) is the only execution specification.

The goal is to integrate an instance segmenter and Ultrack, retain real masks
and bounding boxes during tracking, and evaluate the result against the existing
tracker. The historical baseline is C0 = 0.934802374260586; the local target is
0.95 or better. C0 and P0 were scored on all 199 clips. P0 retained a small
point-control gain at 0.9348649864131336; learned-mask and Ultrack comparisons
remain blocked by unavailable local runtimes. No segmentation gain is claimed.

Use the existing repository, data, local runtimes and authorized model assets.
There is no ZIP, patch, activation script, separate handover, or manual setup step.
Do not use the earlier downloadable `segmentation-first-v6` plan. The broader
initial v6 proposal is superseded by EXPERIMENTS.md in this directory.

The existing `region_contracts.py`, `test_region_contracts.py` and `preflight.py`
remain supporting utilities, not additional plans or production integrations.
Old v1-v5 code, results and artifacts are unchanged.
