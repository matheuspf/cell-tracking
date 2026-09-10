# Local execution: multi-dataset training v4

Implement and execute this handover in `matheuspf/cell-tracking` on the
`handover/multidata-training-v4` branch. Use the user's selected GPT 6 Pro model,
existing competition data, downloaded external datasets, and one RTX 4090.
Produce trained models, measured transfer results and a runnable inference
candidate, not another plan or merely a data reader.

Read root AGENTS.md and relevant competition skills; then read the complete
`docs/external-data-guide/`, v3's measured report/NEXT_AGENT, and this handover.
The new incumbent is v3 A_residual_m3.0 = 0.934802374260586 over 199 clips.
Preserve it and all sealed v1-v3 artifacts. Do not rerun the old searches.

Prioritize actual external-data training:
1. Verify/locate the existing static and six-frame synthetic data, prepared Zoo
   graphs, source hashes and source-use evidence. Reuse downloads and converters.
2. Build modality- and annotation-aware adapters. Run tiny overfit/gradient and
   coordinate tests before long training. Read actual local metadata.
3. Train image/center/event representations on paired synthetic data and a
   separate geometry-only representation on eligible experimental Zoo tracks.
4. Fine-tune using source-embryo Biohub supervision with unknown-label masks.
   Train real-only matched-compute controls and the prescribed dataset ablations.
5. Evaluate all selected frozen models through bounded event proposals and a
   decoder that can actually represent a fork. Then test detector changes
   separately, followed by a limited combination and fresh-image inference.

Do not feed Zoo graphs into image models with zero-filled native logits; do not
invent RIKEN links; do not train annotation-membership labels from dense synthetic
cell truth. Do not use every Zoo species as independent zebrafish evidence.
Unknown source terms/calibration block the affected use, not the entire run.
RIKEN remains audit-only unless its permitted use and missing labels are resolved.

Fix the declared raw fork-objective contract in an isolated branch of inference:
v3 proved forks impossible under its original weights. Demonstrate no-op parity,
a true-fork positive fixture and a nondivision control using the actual solver.
Keep that decoder identical across the data-ablation comparisons. A blanket
lower division penalty or a 190-million-alternative search is not the experiment.

Use the source registry and exact supervision masks. The released simulator
used 44b6-derived calibration; carry this exposure into lineage manifests and
never claim target-independent training when 44b6 is the target. Public teacher
exposure also persists. Label the full study operational exploratory.

Choose missing low-level adapters yourself from inspected local source. Fix
implementation failures and continue independent eligible lanes. Do not stop
at scaffolding, a synthetic-only accuracy result, or a one-batch GPU smoke test.
Use resumable actual-step and unique-example coverage logs. Respect the total
resource cap, emitting measured partial results when a lane cannot complete.

Before target outcomes, freeze both source directions' models, candidate budgets,
calibration procedure and prediction files. Preserve full sample coverage, current
metric pins and GT-free inference. No target-label threshold tuning masquerading
as validation. Report absolute score and delta against v3, per embryo and pooled.
No prior local delta may be added to a leaderboard score.

Commit and push sanitized code/configs/aggregate reports/status/NEXT_AGENT to this
v4 branch after reviewing staged files. Keep models, image data, detailed labels
and predictions in ignored local stores. Do not merge the PR, upload to Kaggle,
publish notebooks/forum posts, accept new gated terms, upload microscopy to remote
services, rent hardware, or initiate paid API calls. Do not change unrelated work.
