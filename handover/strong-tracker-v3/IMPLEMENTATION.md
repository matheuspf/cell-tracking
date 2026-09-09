# Implementation and integration contracts

## Additive structure

Implement `tools/strong_tracker_v3/` with an explicit immutable RunContext carrying
repo/data/V1/V2/V3 paths, config revision and metric revision. Imports must not
create outputs or read labels. Do not monkeypatch globals in v1/v2 modules. Reuse
pure graph I/O, scorer adapters and hashing through explicit parameters, with
parity tests. Proposed modules: context, incumbent, features, disagreements,
event_proposals, source_labels (evaluation/training only), event_model, decode,
rescue, evaluate, inference, report. Provide one CLI and
`scripts/run_strong_tracker_v3.sh` resolving its repo root from its own path.
The CLI does not exist in this handover; Codex must implement it.

Read existing modules, not just this description:
- v2 selected.py: exact `selected_predictions/*.npz` and `selected_prediction_lock.json`.
- replay.py: `replay/no_motion/<dataset>/final_export.npz`, raw/native/cache reuse,
  six phases, exact notebook hash and documented half-integer compatibility.
- native.py: current confidence provenance is stable-ID based, not nearest center.
- hypotheses.py, training_data.py, decode.py, infer.py, fit.py: the old-graph
  training assumptions and per-owner guard described in REVIEW.md.
- v2 metric/common adapters and v1 official source: counts, time-aware matching,
  per-sample estimates and weighted aggregation. Beware side effects from matching.

Canonical cache paths are given in experiments.json. Check paths before execution;
output paths must not resolve into data, V1, V2, or the repository's public result
folders. Restoring absent caches is a local execution step with recorded source
hashes, never silently fabricating evidence. Store raw predictions outside Git.

## Canonical graph and edits

Baseline nodes are `[node_id,t,z,y,x]` with explicit per-dataset physical scale.
Edges are `[source_id,target_id]`. Preserve stable IDs; derived nodes require a
separate collision-free namespace and image-derived provenance. Donor graphs
must be projected to this universe explicitly before comparing scores. An ID match
alone does not validate a changed coordinate/center or inserted point.

A local edit contains read/write boundary, affected source IDs, removed edges,
added edges and complete owner alternatives. `graph_edits.py` here is a strict,
stdlib reference atomic editor for fixed-node actions; it rejects duplicate,
missing, dangling, merge, over-degree and nonconsecutive edits. It is NOT a learned
selector, conflict solver, node-insertion API, or scalable production graph engine.
Local Codex can implement a faster equivalent and must test parity. Checking
individual valid edits is insufficient: conflicts must be tested after combination.

Preserve all outside-boundary edges. No-op must return exactly the same arrays,
not reindex, prune or re-round. If coordinate refinement is enabled, use a distinct
ledger and re-evaluate matching. Division decisions must satisfy parent ownership,
two distinct daughter paths and global event uniqueness; early/late validation
is per the actual official evaluator, not an invented reachability surrogate.

## Tables and outputs

`incumbent_manifest.json`: expected clips, image/graph/estimate hashes, native ID
mapping, actual scale, code/runtime versions, exact 199-clip score parity.
`candidate_manifest.json`: incumbent hash, proposals before/after each gate,
source-only coverage, graph topology version, native features/heatmap provenance.
`disagreements.parquet`: canonical IDs, teacher votes, edge probabilities and
coordinate differences; NO target GT fields.
`event_features.parquet`: predicted trigger, candidate pair/time/path and image
features, ownership, masks, proposal inclusion probabilities, event grouping key.
`training_labels.parquet`: source-only evaluability, label or valid-alternative bag,
source annotation group, source matcher version; kept outside inference inputs.
`round_lock.json`: both direction source model hashes, feature schemas, primary
choices, grid, prediction hashes and outer-score exposure status.
`score_rows.csv`: dataset, embryo, variant, all edge/division counts, predicted
node count, estimated_total, node recall, graph/input/scorer hashes, runtime.
`edit_ledger.parquet`: complete edits with predictions; separate evaluation ledger
records GT effects and never enters inference.
`summary.json`: exact run aggregation, per-embryo and pooled change versus v2,
secondary change versus v1, decision and all caveats.

Stable dataset/candidate identities are keys, not predictive features. Do not
persist matching attributes in an ostensibly label-free baseline. V2's generic
matched-pair intersections and GT-edge recovery are different reports.

## Tests required locally

1. Supplied helper tests pass. All existing applicable v1/v2 and upstream metric
   fixtures continue to pass without altering their expected outputs.
2. Count helper parity to official per_sample_metrics/summarise, unequal weights,
   positive count estimates, no division events, clipping, empty/unreadable/missing
   graph handling. No skipped samples; unchanged GT and estimates per comparison.
3. Recomputed incumbent features include changed adjacency and centers. A test
   altering only topology changes temporal features; stale graph hash rejects cache.
4. Golden canonical-ID mapping, differently smoothed donor graph, inserted node
   ambiguity, and coordinate-refinement cases. Count recovered GT edges separately
   from changed predicted edge IDs.
5. Owner guard fixture: one free daughter and one externally owned daughter at
   1um; also two owned daughters, genuine donor reassignment, and absent confidence.
   Owner score cannot be replaced with the minimum across unrelated daughters.
6. Event fixtures: wrong current continuation but valid alternative pair; daughter
   persistence in candidate paths absent from selected graph; early/late split;
   clip boundaries; swapped daughter order; shared target; shifted duplicate event;
   false neighboring division; incomplete sparse annotation => ignore, not negative.
7. Conflicting edits legal alone but invalid together; full donor closure; no-op
   tie; timeout/infeasibility abstention; stable outside-boundary hash. Small MILP
   solutions match exhaustive enumeration. Score and all constraints checked anew.
8. Positive/negative crop centers and normalization follow the same inference
   pipeline. No GT-centered positives, target-calibrated priors, embryo identifiers
   or evaluation files in prediction code. Full source-only manifest checks.
9. Annotation-unavailable fresh-process replay with raw image inputs, dependency
   loads and feature creation occurring AFTER read blocking. Strict graph export.
10. Wrapper --help from foreign working directory, explicit interpreter, resume
    checksums, missing cache/resource failure paths and staged-file privacy audit.

## Resources and portability

Prefer existing `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`.
Resolve alternative user interpreter via a flag. Source scripts/root_remote_env.sh
only when it is appropriate to this host, never source unknown credentials into
logs. CUDA/driver changes and broad installs are not needed for the primary lane.

One GPU, soft GPU peak 20 GiB, host task RAM min(32 GiB, available budget), initial
CPU process pool 4 with <=2 BLAS threads each, total task thread cap 32 unless
measured resources justify more. Stream image ROIs and avoid a full multi-million-
node image tensor. Cache label-free crops/features once, use pinned LRU heatmaps,
and record physical transforms. Initial new-disk cap 60 GiB, GPU training cap
8 hours for event fits and 12 hours across all optional fits. These are experiment
budgets, not predictions of completion time. Lower caps to actual free resources.

Use isolated configuration objects/processes for repair ablations: v2's original
namespace cache and file-exists shortcuts are not safe for untracked v3 settings.
Every reusable output needs input+config+code hashes. Long local runs may use the
user's local tmux; the authoring assistant is not running background experiments.
No automatic cloud operations, gated access or public benchmark submission.
