# Author-side validation, not detector execution

Date: 10 September 2026. Planning parent is
`03ab55727c5248b292b248113eaa54123791ef03` in matheuspf/cell-tracking.
The latest fetched v5 continuation is in progress; its reported results were read,
not independently reproduced. No final status is inferred for pending experiments.

Executed locally on Python 3.13.5:

```sh
PYTHONNOUSERSITE=1 python -m pytest -q handover/segmentation-first-tracking-v6/tests
python -m py_compile handover/segmentation-first-tracking-v6/scripts/*.py
```

**31 synthetic/reference tests passed.** Coverage includes physical spacing and
covariance, half-open boxes, owned mask crops, invalid shapes/dtypes/units, same-box
but disjoint masks, integer motion-compensated overlap, split-child compatibility,
lineage degrees/timing/IDs, moved-center full graph round trips, dense storage
accounting, planned registry validation and read-only foreign-CWD preflight.
Python syntax compilation, JSON parsing and authored-document link checks passed.

These helper tests do not implement or validate a detector, optical flow, learned
association, Ultrack integration, official metric, microscopy mask quality, GPU
runtime or the future inference package. The object helper is deliberately a small
NumPy reference, not a replacement production tracker or optimized mask store.
No weights, detailed ground truth or microscopy were loaded, and no competition
submission, gated access acceptance or external image upload occurred.

The authoring directory contains only these new files, not the complete source
checkout. Its preflight therefore reports missing source anchors rather than
pretending to verify local v5 artifacts. The Git blob pins come from the GitHub
source reads. The local execution must verify them against its checkout and record
any later source drift; do not rewrite the pins to hide drift. It must also inspect
live v5 jobs, remaining resources and authorized detector weights itself.

This branch initially adds only the handover, registry, reference helpers and tests.
Existing v5 runtime, prior results, raw inputs and checkpoints remain unchanged.
The next agent implements and executes X600–X690 and records measured outcomes.
