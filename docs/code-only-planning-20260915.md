# Start here: planning from the Git checkout only

**Snapshot: September 15, 2026.** This handover is for an agent devising the next
plan without microscopy, local experiment directories, model weights or private
conversation history. The linked documents, code and numerical evidence are in
Git. Absolute `/kaggle/`, `/home/mpf/`, `/root/` and `work/` paths in receipts are
artifact identities from the original machines, not promises that those files
exist in a clone.

## Read these first

1. [Current pipeline error report](pipeline-errors-20260915.md): the strongest
   complete local result, error attribution, score-impact calculations and
   recommended next priorities.
2. [Machine-readable error summary](../results/pipeline-errors-20260915/summary.json):
   both pipelines, every clip, both embryos, category/stage counts, accounting
   scenarios and source hashes. Its `models.pooled` key means C4_m6;
   `models.best` means the retained P0 pipeline. These internal keys do not mean
   P0 has the larger pooled score.
3. [Incumbent provenance audit](incumbent-provenance-20260914.md) and
   [audit receipt](../results/incumbent-provenance-20260914/audit.json): why the
   strong local detector results are training-exposed.
4. [v4 final report](../results/multidata-training-v4/final_report.md) and
   [v6/P0 final report](../results/segmentation-tracking-v6/final_report.md): what
   was measured, adopted, rejected or left unimplemented.
5. [External-data inventory](external-data-guide/README.md): committed source
   routes, schemas, preparation receipts and limitations. Its original branch
   context is historical; use the current error report for model selection.

These files are enough to draft a grounded plan without starting a server or
opening local artifact links. Historical `handover/**/CODEX_PROMPT.md`, experiment
locks and continuation files document individual studies. They are not a new
instruction to launch or resume every old experiment. Distinguish a proposed
handover from an executed result.

## Current decision and main evidence

- **Highest complete non-oracle local score: C4_m6, 0.935178370257** on 199 clips.
  It was not adopted because of the 6bba regression versus C0 and the missing
  qualifying replication.
- **Retained pipeline: P0, 0.934864986413** on those same 199 clips, after its
  deterministic-repeat, both-embryo and fresh-image checks.
- **Underlying selected v3/C0: 0.934802374261.** C4_m6 and P0 preserve the same
  4,108,943 final observations. Their downstream graph decisions differ.

The current audit re-evaluated all 398 C4_m6/P0 graphs with the pinned official
metric. It reproduced 133,318 sparse annotations and 128,883 annotated edges
across 19,900 frames. These are **two repeatedly used training embryos**, not
199 independent embryos. The dominant released detector checkpoint's training
manifest includes all 199 clips. No observed local delta establishes a hidden
leaderboard improvement.

For C4_m6:

- Divisions: **30 TP / 92 FP / 121 FN**. **97** misses have sufficient official
  local-window parent/daughter matches but no parent-side predicted fork;
  **24** lack sufficient local matches.
- Links: **123,133 TP / 4,968 FP / 5,750 FN**. **3,040** FN have an unmatched
  endpoint; **2,710** FN have both endpoint centers matched.
- **4,907** link FP touch an unmatched prediction. **2,467** use one within 7 µm
  of the expected annotation, while a different prediction owns that assignment.
- Final center recall is **130,836 / 133,318 = 98.14%**. Of 2,482 unmatched
  annotations, **1,765** have a raw candidate nearby but no nearby final center.

The strongest next lead is division-versus-continuation/birth modeling on fixed
incumbent observations, followed by observation selection and temporal identity.
This is a diagnosis to inform the next plan, not a prescribed new experiment
lock. Final-graph evidence cannot distinguish absent candidates from rejected
scores, acceptance gates or solver decisions; instrument those stages when
designing the comparison.

## Recent studies that should inform the plan

- **Cellpose detector screening/refinement:**
  [detector benchmark](detector-benchmark-20260914.md),
  [refinement results](cellpose-refinement-results-20260914.md), and
  [embedding controls](cellpose-embedding-results-20260914.md).
  Small-radius localization improvements and extra proposals have not reliably
  improved full tracking. Compare matching radius, encoding, population and
  exposure before interpreting any recall number.
- **Cellpose + ultrack:**
  [current errors](cellpose-ultrack-current-errors-20260914.md),
  [division adaptations](cellpose-ultrack-error-analysis-20260914.md), and
  [implementation/reproduction](../tools/cellpose_ultrack/README.md).
  These newer event-cost, learned-pair and daughter-persistence runs use six
  reused clips. They do not replace the complete 199-clip incumbent evidence.
- **Alternative trackers:** [HOCT reassessment](hoct-reassessment-20260914.md)
  and [other trackers](other-trackers-20260914.md). Preserve their controls and
  observations when proposing another replacement architecture.
- **Native model and exposure comparison:**
  [incumbent comparison](incumbent-comparison-20260914.md) and its committed
  `results/incumbent-comparison-20260914/` receipts. The source-only whole-model
  training report is an archived progress snapshot, not proof a worker is still
  running or has completed. No completed source-only final target score is
  established by that progress snapshot. Model weights/resume states are local.
- **Expanded public946 study:**
  [measured report](../results/public946-minimal-generalization-v1/REPORT.md),
  [experiment matrix](../results/public946-minimal-generalization-v1/experiment_matrix.csv),
  and [per-embryo scores](../results/public946-minimal-generalization-v1/per_embryo_scores.csv).
  It is incomplete: B0 has a fresh 199-clip score, while B1 has only partial
  execution and the larger matrix lacks results. The historical public 0.946
  leaderboard score is not a fresh local score or a new submission.
- **Native-resolution detector v8/v9 and integration handovers:** read their
  `handover/` plans as proposals with explicit contracts. Their presence in Git
  is not evidence that their planned experiments were run successfully.

## Code map

The current best pipelines build on the public TemporalUNet detector and
association model, the selected v3 tracking graph, and conservative downstream
association/division changes. The code needed to reason about those changes is
committed:

- `tools/strong_tracker_v3/incumbent.py`, `inference.py`, `association.py`,
  `decode.py`, `features.py`, `event_proposals.py`: incumbent construction,
  observations, continuation evidence, candidates and decoding.
- `tools/multidata_training_v4/models.py`, `proposals.py`, `calibrate.py`,
  `decode.py`, `infer.py`: learned event models, candidate gates and margins,
  calibration, graph conflict resolution, and variants including `C4_m6`.
- `tools/segmentation_tracking_v6/controls.py`, `point_child.py`, `infer.py`:
  the P0 point-control fit, application and inference interface. The folder name
  does not establish that learned segmentation produced the retained result.
- `tools/annotation_selection/metric_adapter.py` and
  `handover/annotation-selection-v1/analysis.py`: graph-ID mapping, official
  scorer integration and an independent strict aggregation implementation.
- `tools/center_comparison/tracking_index.py`: disjoint edge-error partition,
  official division-window diagnosis, source checks and accounting scenarios.
- `tools/center_comparison/tracking_server.py`, `tracking_view.js`,
  `tracking_style.css`, `index.html`: read-only error API and the report/case UI.
  `detection_index.py`, `detection_server.py`, `detection_view.js` implement the
  existing observation review and native image endpoint.
- `tools/detector_screen/`, `tools/cellpose_refine/`,
  `tools/cellpose_ultrack/`, `tools/incumbent_comparison/`,
  `tools/hoct_reassessment/`, `tools/other_trackers/`: recent experiment code.
  Their configs, numerical summaries, environment and validation receipts are
  under matching `configs/` and `results/` paths.
- `tools/biohub_external_data/` and `docs/external-data-guide/`: maintained
  preparation code and portable inventory. `archive/` contains historical
  public-download helpers, not private conversations or bundled datasets.

## Contracts the plan must preserve

Images are native **Zarr v3** time/z/y/x arrays; annotations are sparse **GEFF**
graphs. Read actual metadata before execution. The documented physical scales
are z=1.625 µm and y=x=0.40625 µm per voxel. Node matching is optimal one-to-one
assignment within 7 µm, separately at each timepoint. Node IDs must be mapped
explicitly between persisted graphs and graph-library internal indices.

The score is weighted adjusted edge Jaccard plus 0.1 × micro division Jaccard.
Division matching uses independent local windows and a timing tolerance;
immediate-edge topology alone is not the division verdict. Node-count
adjustment uses a provided coarse total-node estimate, not sparse annotation
count. Unmatched predictions and unevaluable edges are not automatically false
positives. Edge and division flags can refer to the same biological event.

The pinned scorer revision is
`075fc5f5a52d11077f9dc2b074644618f26939e2`, from
[the official metric repository](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2).
The official source checkout is ignored; its public identity, integration code,
count summaries and reproduction checks are committed.

Notebook submissions produce native integer-coordinate node/edge rows in
`submission.csv`. See [competition data/contract notes](competition.md) and
[notebook setup](notebooks.md). The September reference snapshot specifies a
12-hour notebook runtime with internet disabled; refresh current official
requirements before execution or submission.

## What Git contains and what it deliberately excludes

**Included:** implementation code, configs and runtime pins; dated plans and
reports; aggregate and per-frame/per-clip diagnostic statistics; source/model
hash receipts and candidate-bank manifests; validation receipts; and standalone
HTML/SVG reports containing derived statistical evidence. Some detailed JSON
reports are retained so a planning agent can inspect strata rather than depend
on a prose conclusion.

**Excluded:** canonical microscopy/GEFF, external downloads and raw references;
proposal/feature arrays; full predicted graph files and SQLite image-review
indexes; model weights, training checkpoints and inference packages; generated
submissions and logs; credentials and private conversation/session stores.
Three personal session utilities also stay local under explicit ignore rules.
Excluded files were preserved on the working machine, not deleted.

The complete source paths and hashes for local C4_m6/P0 graph recovery are
recorded by `tools/center_comparison/best_predictions.py` and the audit receipts.
The default working roots are `/kaggle/working/cell-tracking` and the canonical
competition input directory. The UI's image/case pages require those local
artifacts; a code-only agent should read the committed report and JSON instead.
Do not treat an inaccessible localhost link as missing analytical evidence.

## Validation and useful next-plan deliverables

The [report summary](../results/pipeline-errors-20260915/summary.json) records
provenance for all 398 fresh graph evaluations. The
[tracking validation receipt](../results/pipeline-errors-20260915/validation.json)
records 469 checked scenes, source hashes, real native-frame checks and
desktop/mobile browser coverage. The written report records the scoped test
results. These receipts are historical evidence; a new environment must validate
its own dependencies.

The September 15 Git snapshot also passed `ruff check tools tests` and the full
`pytest -q tests` suite: **220 passed, 19 skipped**, with `PYTHONNOUSERSITE=1`
and `PYTHONPATH=tools:.` in the existing annotation-selection runtime. All staged
JavaScript and shell files passed syntax checks. The four entry documents
(README, this handover, the pipeline report and viewer guide) have 49 relative
links, all verified against the staged Git tree.

Useful outputs from the next planning agent are: a chosen primary bottleneck;
specific code entry points to instrument or change; a fixed-observation/control
comparison; an embryo and checkpoint-exposure protocol; candidate/score/decoder
stage diagnostics; resource and Kaggle-runtime estimates; and adoption criteria
based on the full combined metric. Separate required artifact retrieval from
analysis that can already be performed from Git.
