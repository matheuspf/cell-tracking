# Local implementation contracts

The existing repository is a data/reference/notebook scaffold, not a verified OOF
tracking pipeline. Do not invent an existing trainer or assume the mirrored public
checkpoints have held-out provenance. The supplied helpers are implemented; the
modules below are work for local Codex. All commands introduced by Codex must be
runnable from a clean checkout and documented in the final report.

## Suggested additive package

Implement under `tools/annotation_selection/` with tests under
`tests/annotation_selection/`. A single CLI is preferable:
`PYTHONPATH=tools python -m annotation_selection <stage>`.
This CLI does not exist in the initial handover; build and test it first.

| Module | Contract |
|---|---|
| `inventory.py` | Extend metadata-only audit with streamed GEFF arrays, ID/coordinate/edge validation, exact annotation statistics and source fingerprints. |
| `splits.py` | Embryo assignment, crop/time overlap graph, supergroups, inner folds, manifest lock and forbidden-overlap assertions. |
| `metric_adapter.py` | Pinned official matcher/graph reader and division evaluator; exact sample-set validation, fresh graph copies, full expected sample coverage and count-estimate checks. |
| `candidates.py` | Image-only classical candidates, physical NMS, deterministic links/forks, and a separately named public-notebook export adapter. No GT-centroid injection. |
| `features.py` | Image/geometry/temporal feature allowlist, fold-specific fitted transforms, predicted-tracklet features, memory-bounded chunk reads. |
| `labels.py` | Evaluation-only one-to-one annotation membership, candidate ambiguity flags and baseline TP/link tables. Must not be imported by inference. |
| `train.py` | Source-only inner training/selection, constant/logistic/tree probes, small patch probe, provenance manifest and checkpoint hashes. |
| `filter_graph.py` | Identity, random, quality, membership-node, tracklet and predicted-fork-protected policies; stable IDs and incident-edge deletion. |
| `evaluate.py` | Frozen outer evaluation, exact fresh official counts and score, old/new TP accounting, division outcomes and aggregation parity. |
| `report.py` | Coverage/retention/score tables, uncertainty, attribution, HTML/Markdown and artifact ledger. |

Reuse the existing `tools/competition_paths.py`, `tools/inspect_data.py`, path
aliases and notebook export/runtime tooling. Do not modify extractor throttles or
hand-edit reference manifests. Do not import untrusted notebook shell/install cells
blindly; extract inference into a controlled adapter with equivalent settings and
record every deviation. Resolve actual notebook input paths from docs/configs and
local files. Public notebook runs share `/kaggle/working/submission.csv` and
`tracking_repo`: isolate these locations, preserving previous outputs.

## Tables and manifests

Store Parquet/CSV/JSON in ignored work/output storage; include schema version,
run ID, code revision and producer command. Use `(dataset,candidate_id)` composite
keys and stable predicted IDs. Keep train/source/evaluation artifacts separate.

**`sample_inventory.csv`:** dataset, embryo, overlap_group, image_shape, axes,
physical_scale, annotated_nodes, annotated_edges, gt_divisions, estimated_total,
estimate_source, estimate_valid, metadata_hash. Detailed annotation summaries must
respect the outer embargo. Invalid estimates stay invalid, never substituted.

**`candidate_features.parquet`:** dataset, candidate_id, t/z/y/x in documented
units, generator_id, prediction_graph_hash, feature_version, allowed features,
predicted_tracklet_id. No GT-derived values. Record the feature-column allowlist.

**`membership_labels.parquet` (evaluation area):** dataset, candidate_id,
matched_gt_id, annotation_label, ambiguity_stratum, matching_revision. Store missed
GT nodes separately; never turn them into synthetic positive candidates. Node-match
labels after filtering are not the same task as baseline membership labels.

**`fold_manifest.json`:** source/outer embryo IDs, source inner groups, overlap
proof, training sample hashes, predictor training provenance, embargo state,
feature/model/policy hashes, prediction freeze time, evaluation start time.
Train/test disjointness must cover every supervised upstream component.

**`candidate_scores.csv`:** dataset, candidate_id, annotation_label, score, fold,
model_id, provenance_lane. Export labels only for evaluation. `analysis.py membership`
accepts the first four columns and computes per-video top-fraction operating points;
it does not verify OOF provenance or implement coherent filtering.

**`score_rows.csv`:** dataset, fold, lane, model_id, policy, seed, requested_keep,
realized_keep, estimated_total, num_pred_nodes, edge_tp, edge_fp, edge_fn,
division_tp, division_fp, division_fn, gt_node_recall, graph_hash, metric_revision.
Feed one variant at a time to `analysis.py summary`; `--expected` is a JSON array
of all locked sample names for that fold/variant, generated from the split manifest.
Reject mixed variants, duplicate datasets and GT/estimate drift before aggregation.

**`retention.csv`:** requested/realized r, natural membership prevalence, retained
matched nodes, total matched nodes, t, precision, specificity, phi, deletion
annotation rate/lift, baseline TP surviving, newly recovered TP, FP retained,
J0/J1, D0/D1, S0/S1, c0 per sample. Report fold-specific and pooled values with
correct denominators. Raw TP survival and rematched TP count can differ.

## Runtime and resource controls

Use the existing Python 3.12 inspection environment for metadata/CPU work, and
`/kaggle/envs/cell-tracking-notebooks` for the already verified CUDA stack.
Set `PYTHONNOUSERSITE=1`; capture actual packages before changing dependencies.
Do not overwrite the user's environment lock from an unrelated environment.
Add only necessary packages in an isolated environment if required and record pins.

Stream Zarr chunks/time windows. Cache label-free features once per frozen baseline;
never load the full 80+ GiB image collection into RAM. Pilot on two source-only
clips with different density before scaling. Start with <=4 loader workers and
bounded queues; profile before increasing. Run notebook models serially. Reserve
memory headroom: initial soft limits 20 GiB GPU, 32 GiB host RAM, 50 GiB patch cache
and 150 GiB new outputs, each lowered to available capacity. Do not delete user
artifacts to satisfy limits. Mixed precision is allowed after checking output
stability; batch-size backoff must preserve evaluation examples.

Initial budget caps are controls, not completion-time promises: metadata/classical
and tabular work first; at most 12 GPU-hours total for the small image probes.
Pilot projects full notebook inference cost before launching all clips. If the cap
would be exceeded, finish cheap stages, run an explicitly labeled pilot, and report
which full-data lane remains incomplete. Do not call a subset full OOF. Preserve
resumable per-sample outputs, model hashes, RNG states and a machine-readable stage
ledger. Do not stop all work because one optional lane is blocked.

## Required local tests beyond the supplied 29 synthetic tests

1. Run upstream tests for the pinned metric and independently compare upstream
   `per_sample_metrics`/`summarise` against `analysis.strict_summary` on random valid
   counts, unequal sample weights, no-division cases and changing FP denominators.
2. Golden graph fixtures: correct/wrong/missing links; unmatched-to-unmatched
   edges; extension beyond annotation; duplicate centers; one-to-one competition;
   anisotropic distance and time gating; empty prediction; genuine/false/shifted
   forks; merged branches; duplicate/nonconsecutive edges. Assert exact TP/FP/FN.
3. Identity filtering is bit-for-bit graph-equivalent; keep=0 is handled explicitly;
   no dangling/duplicate edges; no time bridging; deterministic tie handling; fork
   closure uses no GT; reported retention includes closure and rounding.
4. Missing sample, unreadable graph, missing/NaN/zero total estimate or changed GT
   checksum fails full-run scoring. Do not silently inherit upstream skipped rows.
   Zero annotated-event fixtures follow upstream conventions without fabricated ones.
5. Split and feature leakage tests: disjoint overlap groups, no target labels in
   source fitting, no GT-derived feature columns, no fitted transform crossing folds,
   no evaluation-match attributes serialized in baseline graphs. Fail unknown
   checkpoint provenance from the *clean* lane while permitting diagnostic labeling.
6. A GT-unavailable inference smoke test reproduces frozen outputs. Instrument file
   reads or pass a label-free input directory; matching/label files are absent. Masking
   output columns alone is not proof that inference did not load annotations.
7. CLI tests from repository root and a different current directory, clean checkout
   import resolution, interrupted-run resume, immutable input checksum checks, JSON
   without NaN, and a staged-file audit excluding raw data/weights/credentials.

Do not present these integration tests as already passing in this PR. The 29
included tests cover arithmetic and synthetic metadata only.
