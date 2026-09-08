# Source ledger and inspected state

Prepared 2026-09-08 through the GitHub app. User repo base:
`matheuspf/cell-tracking@08be34e4e67a20a48ae263d053c879fa7390d7c0`.
Official source ref resolved to
`royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`.
Revalidate actual local/public scorer agreement before experiments; a pinned snapshot
is reproducibility evidence, not proof that Kaggle will never change its evaluator.

## Primary implementation sources inspected

- [Metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md)
  — sparse matching, signed count adjustment, local division matching, run aggregation.
  Observed blob: `df84639194776282af9b788828ad6a9b870e19bb`.
- [Metric implementation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/metrics.py)
  — `evaluate` mutates matches; `per_sample_metrics` applies the count adjustment;
  `summarise` uses current edge denominators as weights and pooled division counts.
  `evaluate_datasets` does not apply count adjustment. Duplicate/nonconsecutive-edge
  handling and out-degree guards also exist; use valid graphs rather than relying
  on those sanitizers. Observed blob: `e536cdc9f0877542ab227ec701ef0fdbb667189a`.
- [Evaluation entrypoint](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/evaluate.py)
  — only scores the directory intersection and can skip exceptions; reads estimated
  node total through GEFF metadata. Add strict completeness checks around it.
  Observed blob: `b17afbbfec40a7e477d30b947ed0987625bf59ae`.

Before local integration, also inspect `src/tracking_cellmot/division_metrics.py`,
`io.py`, upstream tests, installed tracksdata matching code, and the current local
Kaggle evaluation/reference snapshot. This PR does not vendor or alter upstream code.

## User repository resources inspected

`AGENTS.md`, `README.md`, `docs/competition.md`, `docs/notebooks.md`,
`tools/inspect_data.py`, and `.agents/skills/competition-{data,forum,rules}/SKILL.md`.
The repo documents canonical data paths, two runtime environments and mirrored
Harmonic Fusion / 942 TTA notebooks. It reports 199 training image/GEFF pairs and
203 total image arrays; these are repository-reported counts, not data audited in
this ChatGPT session. No raw microscopy, checkpoints or local reference bodies
were available here. Current user dataset and training provenance remain to measure.

Local forum bodies to read, using the existing extractor/skills (metadata titles
alone are not evidence): 716062, 716793, 727154, 728324, 739018, 739686.
Check organizer authorship and distinguish host statements from participant ideas.
The local rules and annotation-selection clarifications take precedence over prior
chat summaries. Preserve request throttles; no posting is authorized.

## Provenance discipline

Record the exact metric commit, graph-library build, input metadata hashes,
notebook script-version IDs, support-pack checkpoint SHA-256 values, and training
sample manifests in each result. Stop official-score claims if the Kaggle scorer
and public source disagree until reconciled. Keep old/new comparisons separately
named; never silently switch implementations halfway through an experiment.

Prior conversation numbers are hypotheses/examples only. Public scores alone do
not establish whether another competitor is filtering annotations. No such claim
is an assumption of this study.
