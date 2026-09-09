# Inspected evidence and version pins

Authoring date: 2026-09-09. User repository read and written through the GitHub app.
Base: matheuspf/cell-tracking@a58c80b041db0a05262a58b3d1b5f34045b8d2ff.
No competition arrays, pretrained weights or event-level local artifact store were
accessed in this authoring session. Numerical facts are committed v2 evidence;
new mechanisms/targets are hypotheses, not measured gains.

## Measured evidence

- results/strong-tracker-v2/v2_report.md (blob d6469ca6dce8449ab510b0fc5963c216f8fbff79).
- results/strong-tracker-v2/summary.json (blob 551449c9bdebdf1dfa37a862f5aee62bda6f93d7).
- results/strong-tracker-v2/candidate_coverage.json (blob dabab0f368759369b922e6c36acfcf6468de3ca2).
- results/strong-tracker-v2/winning_config.json (blob 96da8f011a2f46e9bfb7bec1c97aa16b07219089).
- docs/strong-tracker-v2.md; root AGENTS.md; historical v1/v2 protocols.

## Implementation inspected

All paths under tools/strong_tracker_v2/ at the base revision:
training_data.py (9219b93d51cc3087f0cafa22bdacb1cbbd900d28),
hypotheses.py (04bd0a41eedeac6f3a923e70f7c0696598060fdd),
infer.py (d4d2c561b4f26e42d7fb910857ff862068a049e3),
decode.py (7b689b5abae151695e9043af8383673a39b5b986),
fit.py (655a40e68bf0172401e17ffaa42f9a50a9638be9),
native.py (20519c87545e397cc8ff90cf03e7538ab479456e),
replay.py (4a5ec201fba1e11f8cc13f5d610d1f74df840850),
selected.py (4dec231425c70f475f477eefee903a0089b85e26).

## Primary metric reference

https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/division_metrics.py

The current metrics.md web search result on 2026-09-09 describes local parent and
daughter windows with +/-1 timepoint tolerance. V2 records the above scorer pin.
Local Codex must verify current official/Kaggle snapshot agreement and retain the
old pin for historical comparisons; no silent metric switch between variants.
The local official division evaluator supplies the actual implementation contract.
No deprecated weak-component/reachability division shortcut is permitted.

External tracking implementations were considered but are not required dependencies:
https://github.com/weigertlab/trackastra documents mask+image association and
separate greedy/ILP linking. That does not establish benefit here or a ready 3D
point-only adapter. This plan intentionally prioritizes the user's cached neural
scores and native graph alternatives before replacing its proven tracker.
