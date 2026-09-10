# Evaluation, promotion and preservation

## What 0.95 means

Target is the current published official local aggregation on all 199 training
clips, not hidden test or the historical 0.946 public notebook score. The baseline
is 0.934802374260586 (44b6=0.931664468721842; 6bba=0.935221784097327).
Do not compare a new model only against v1, v2, a weak direct alternative or a
matched-control score that is lower than the incumbent.

Recompute matching and local division assignments for every different graph.
Verify GT checksums, per-sample supplied total estimates and the exact sample
manifest. Current edge denominators TP+FP+FN determine adjusted-edge weights;
division counts are pooled. Never average sample Jaccards or substitute a global
node count ratio. A zero-event sample follows the pinned evaluator, not invented
perfect precision. Recheck official drift; if it changes, score both incumbent
and candidate under the same reconciled version before comparing.

## Data and selection boundaries

Train each direction only on its source embryo's direct annotations. New source
features must not call opposite-embryo historical E teachers as clean supervision.
Use public native predictions as inherited exposed evidence only when declared.
No biological-independence claim follows from a different architecture, new random
seed, synthetic validation, time blocks, or renamed filenames.

Prior studies exposed both embryos. New variants are operational exploratory.
Source-only inner folds are allowed only with demonstrated disjoint physical/time
supports. Otherwise preregister fixed recipes and finish both directional models
before comparative evaluation. Do not invent independent random clip/frame folds
or node-bootstrap confidence intervals. Report both embryos and seed variation.

Bound selection to the recorded slot table; record after-outcome decisions as
exploratory amendments before new runs. Neither a failed development pilot nor an
oracle may supply a deployable GT-assisted decision. A source-fit metric diagnoses
optimization only. A new untouched biological dataset, if lawfully acquired later,
is separate validation, not assumed to exist here.

## Decision classes

- `target_reached_local`: pooled >=0.95; neither embryo below its incumbent by
  more than 1e-10; full sample/graph/metadata checks; cold image inference and
  package parity; same selected learned arm replicated in a second seed with
  the target and embryo conditions also met. This is still local exploratory.
- `improved_below_target`: a positive, repeatable pooled improvement and no embryo
  regression, all integrity/delivery gates pass, but 0.95 has not been confirmed.
  Retain it as a new local candidate; report the remaining gap explicitly.
- `promising_unreplicated`: target or gain reached only once, inconsistent embryos,
  missing confirmation, or unresolved model-data provenance. Do not silently
  replace the incumbent. Name the successful and failed evidence.
- `no_improvement`: valid completed alternatives fail the above. Preserve C0,
  quantify bottlenecks and make the next-agent handover useful.
- `incomplete_or_invalid`: name missing dependencies/experiments and salvage valid
  diagnostics without presenting partial sample coverage as a full score.

Deterministic zero-shot systems need reproducible inference, not an artificial
second training seed. Any stochastic decoding/segmentation must be seeded and
repeated. Learned ensembles require actual evaluation of the exact ensemble;
no averaging or adding score deltas from separately evaluated models.

## Required artifacts

Local output root defaults to `/kaggle/working/cell-tracking/lineage-reconstruction-v5`.
Produce final_report.md, dashboard.html, NEXT_AGENT.md, status.json, score_rows.csv,
family_summary.csv, candidate_coverage.csv, region_mapping_audit.json,
training_exposure.csv, checkpoint_manifest.json, input_manifest.json,
variant_lock.json, runtime_summary.json, selected_config.json,
inference_receipt.json and artifact_manifest.json. Include valid fields only;
unrun experiments have null results and explicit reasons, not placeholder zeros.

The report must distinguish center errors, candidate edge coverage, score ranking,
solver/candidate conflicts, timing tolerance and pathological fallback. Give
zero-shot versus adapted scores; A0 versus new-node A1 effects; native/new/combined
evidence; actual runtime and memory; no-op controls and all rejected variants.

Keep masks, crops, detailed GT identities, raw data, downloaded model weights,
prediction graphs and credentials out of public Git. Commit a sanitized result
subset, code and next-agent prompt to the v5 branch. Preserve v1–v4 roots and user
work. Do not reset expired v4 queues or treat this plan as authorization to rent
hardware, accept gated terms, submit to Kaggle or merge pull requests.
