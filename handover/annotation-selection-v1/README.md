<!-- local-execution-result -->
Local execution is complete. See [measured results](../../results/annotation-selection-v1/README.md) and [reproduction instructions](../../docs/annotation-selection-v1.md). The original handover below is preserved.

# Annotation-selection study v1

**Status: handover only; no competition-data experiment has run in this PR.**

Goal: measure whether annotation membership is predictable on unseen embryos,
and whether label-blind filtering of a frozen tracking graph improves the actual
competition score. Do not assume the earlier hypothetical 10% prevalence,
0.886 edge Jaccard, 0.60 division Jaccard, count ratio 1, or 99% retention.
Do not infer what other competitors are doing from their leaderboard scores.

## Execute with local Codex / GPT 6 Pro

From this repository, open the local Codex session with the user's selected
GPT 6 Pro model and give it [CODEX_PROMPT.md](CODEX_PROMPT.md). The agent is to
**implement and execute** [PROTOCOL.md](PROTOCOL.md), not return another plan.
There is no activation patch, installer, or generated-repo overlay to apply.
Do not rerun the full bootstrap or redownload the already available data.

Available now, without additional Python dependencies:

```sh
export PYTHONNOUSERSITE=1
python -m unittest discover -s handover/annotation-selection-v1 -p 'test_*.py' -v
python handover/annotation-selection-v1/audit_metadata.py
```

The first command runs synthetic checks. The second reads local training
metadata and writes `work/annotation-selection-v1/metadata.json`; it does not
load image chunks, run the official matcher, or establish exact true-cell counts.

## What this must answer

| Question | Required answer |
|---|---|
| How much is annotated? | Exact annotated observation counts; annotation/estimated-total ratios; detector-conditional matched fractions; separate calibrated true-cell estimates where justified. |
| Can annotations be predicted? | Held-out embryo recall at fixed keep budgets, deleted-group annotation rate, lift, phi/MCC, ROC/AP and calibration; not accuracy alone. |
| Does that help the actual score? | Fresh official graph matching and division scoring before/after filtering, full sample coverage, and exact run-level aggregation. |
| Is it selection bias or ordinary tracking improvement? | Random and confidence-only controls, provenance-clean versus public-checkpoint lanes, count-adjustment ablation, and independent candidate-quality audit. |
| Would the result plausibly transfer? | Both embryo directions, source-only tuning, overlap-aware grouping, uncertainty and an explicit two-embryo limitation. |

## Existing local resources

Use the repository's `AGENTS.md`, competition skills, `docs/competition.md`, and
`docs/notebooks.md`. Canonical data is
`/kaggle/input/competitions/biohub-cell-tracking-during-development`.
Inspection environment: `cell-tracking` (Python 3.12).
Notebook/GPU environment: `/kaggle/envs/cell-tracking-notebooks`.
The 0.946 public notebook mirrors and their input artifacts already exist locally;
their training provenance must be established before claiming clean validation.

Raw data, notebook mirrors, weights, predictions, and cached references stay out
of Git. This repository is public. Use ignored `work/annotation-selection-v1/`
and `/kaggle/working/cell-tracking/annotation-selection-v1/` for generated files.
Only sanitized aggregate results and executable code should later be committed.

## Contents

- [PROTOCOL.md](PROTOCOL.md): hypotheses, estimands, leakage controls, experiments and decisions.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): modules to implement, tables, runtime and parity tests.
- [experiments.json](experiments.json): bounded, preregistered initial experiment grid.
- [SOURCES.md](SOURCES.md): inspected repository/source pins and verification requirements.
- `analysis.py`: strict aggregation and membership/sensitivity helpers; **not a graph evaluator**.
- `audit_metadata.py`: read-only metadata inventory, Zarr v2/v3 support.
- `test_helpers.py`: synthetic checks; local official-metric parity remains mandatory.
- [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md), `STATUS.json`: reporting contract and initial status.
