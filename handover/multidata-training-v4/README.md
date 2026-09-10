# Multi-dataset training v4

**Status: executed and measured.** Start with [CONTINUATION.md](CONTINUATION.md)
to review current results and propose the next study. See the
[measured report](../../results/multidata-training-v4/final_report.md),
[interactive dashboard](../../results/multidata-training-v4/dashboard.html), and
[execution handover](../../results/multidata-training-v4/NEXT_AGENT.md).
MANIFEST.json remains the original authoring snapshot; measured artifacts have
their own manifest.

## Start from current results

Give the next agent [CONTINUATION.md](CONTINUATION.md). It identifies the completed
comparisons, unresolved questions, evidence to read, and local dependencies.
The selected result remains v3 at **0.934802374260586**. W400–W490 and the scheduled
follow-up tests have completed; do not restart the old execution prompt.

The remaining sections preserve the original pre-execution handover.

Parent: `handover/strong-tracker-v3@292ecc2569f3a16676de92d80e382be02024d904`.
Incumbent: v3 `A_residual_m3.0`, full local score **0.934802374260586**.
The v2 score is historical, not the new promotion baseline.

This iteration changes learned representations using the downloaded datasets.
It is not another annotation-filter, graph-repair sweep or dataset-download study.
Train on paired synthetic microscopy and on compatible real trajectory exports,
then adapt to Biohub's sparse supervision. Separate their contributions with
same-architecture, matched-compute controls. Preserve the incumbent on failure.

## Original execution instructions (completed)

Check out `handover/multidata-training-v4` and give local Codex / GPT 6 Pro
[CODEX_PROMPT.md](CODEX_PROMPT.md). No activation script or overlay is required.
The agent implements and executes W400-W490; the training CLI described in
IMPLEMENTATION.md is work to implement, not a command already supplied here.

```bash
PYTHONNOUSERSITE=1 python -m unittest discover \
  -s handover/multidata-training-v4 -p 'test_*.py' -v
python handover/multidata-training-v4/preflight.py
```

The preflight only checks paths/counts; it does not mutate or download inputs.
Pass `--archive-root`, `--prepared-root` and `--v3-root` when storage has moved.
A missing optional dataset must not stop training on available eligible sources.

## Read

- [REVIEW.md](REVIEW.md): what v3 established and why more supervision is the pivot.
- [DATASET_PLAN.md](DATASET_PLAN.md): exact roles, exclusions, grids and data leakage.
- [EXPERIMENTS.md](EXPERIMENTS.md): controlled curriculum, validation and decisions.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): model interfaces, losses and execution contracts.
- [experiments.json](experiments.json), [dataset_registry.json](dataset_registry.json).
- [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md), [SOURCES.md](SOURCES.md).

The NumPy reference helpers and 40 tests check coordinate, censoring, supervision,
split and decoder contracts. They are not a trained model, data audit, or official
metric integration. All real-data/GPU work is assigned to local execution.

Use the committed `docs/external-data-guide/` and existing preparation code.
Never count duplicate archives as extra specimens; never pretend missing images,
masks or temporal links exist. Keep v1-v3 results, locks and input data unchanged.
