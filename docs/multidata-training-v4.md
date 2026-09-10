# Multi-dataset training v4

The maintained implementation lives in `tools/multidata_training_v4/`. It uses the
existing CUDA study interpreter and additive outputs under
`/kaggle/working/cell-tracking/multidata-training-v4/`. Raw inputs and v1–v3 remain
read-only. W400–W490 are complete; the selected result remains v3
`A_residual_m3.0` at **0.934802374260586**. Read the
[measured report](../results/multidata-training-v4/final_report.md) and
[continuation handover](../handover/multidata-training-v4/CONTINUATION.md) before
proposing another study. The commands below document the completed implementation;
scientific reruns must use a new output namespace.

Run from any directory with the absolute wrapper path, or from the repo root:

```bash
scripts/run_multidata_training_v4.sh sources
scripts/run_multidata_training_v4.sh evaluate --variants C0
scripts/run_multidata_training_v4.sh index
scripts/run_multidata_training_v4.sh sanity
scripts/run_multidata_training_v4.sh provenance
scripts/run_multidata_training_v4.sh read_audit
scripts/run_multidata_training_v4.sh train
scripts/run_multidata_training_v4.sh calibrate
scripts/run_multidata_training_v4.sh infer
scripts/run_multidata_training_v4.sh evaluate --variants C0,decoder_only,original_objective_C4,C1short,C1,C2,C3,C4,C5,C6,C4_Gonly,C4_m2,C4_m6,C1_relinked,C4_relinked,D_real,D_synthetic,D_synthetic_C4
scripts/run_multidata_training_v4.sh report
scripts/run_multidata_training_v4.sh delivery
```

Pass `--python /path/to/CUDA/python` before the stage to change the interpreter.
Use `V4_OUTPUT` to choose a separate new experiment root under the same study parent; preserve prior roots.
Source and preparation paths follow the documented restored workspace layout.
Training resumes optimizer state and exact deterministic exposure cycles every
1,000 actual updates. Completed files are verified before reuse. Scientific changes
require a new output namespace; do not silently reuse incompatible checkpoints.

`queue` is a local scheduled continuation after the training checkpoint manifest:
calibration → all planned GT-free predictions → full frozen official evaluation.
It observes the user-authorized deadline and reserves ten minutes for delivery.
It does not contact a paid service or create a remote automation.

`secondary` waits for the primary comparison receipt, repeats C1 and the strongest
qualifying external arm (or the prespecified C4 replication when none qualifies),
then freezes both directions before second-round scoring. `queued_stress` runs
the already locked generator corruptions when their pretrained weights exist.
`finish_queue` waits for replication and conditionally runs the predeclared W470
rendering treatment, then coverage diagnostics, reporting,
fresh-image delivery, the predeclared additional density pilots when time permits,
head-path and unfamiliar-name checks, rendered holdout diagnostics, and final
validation/export. Git review, commit and push remain explicit actions.
These are local subprocess queues bounded by `authorized_run_window.json`.
Browser validation uses the existing project Conda interpreter, which supplies
Playwright and headless Chromium. Override it with `V4_BROWSER_PYTHON` if needed;
the CUDA training environment stays separate.

W470 (`rendering_trial`) requires a measured C4/C6 source-to-target gap, external
utility, and at least 95 minutes left. C7 mixes explicitly rendered Zoo-fish
optical patches with synthetic images using source-only appearance statistics.
Its decoder, step budgets and temperature procedure remain fixed. Any second
seed is separately frozen; an unreplicated C7 cannot replace the incumbent.
The density follow-up uses measured pilot runtime to choose all four predeclared
clips, the two 90th-percentile clips, or no extra clips, reserving final validation.

Individual follow-ups are `stress`, `coverage`, `additional_pilots`,
`dashboard_check` and `finalize`. The finalizer verifies preserved studies,
same-seed matched compute, all score rows, package receipts and the offline
dashboard before exporting an explicit allowlist of sanitized files.
`head_parity` separately verifies the individual-graph C4 deployment path against
the scored shared-patch primary path on all 199 clips, with labels unavailable.
`unknown_name_pilot` runs a fixed image under an unfamiliar filename with the
source model supplied explicitly and checks complete graph and CSV-name parity.
`rendered_holdout` compares frozen C4/C7 optical towers on the rendered Zoo
holdouts after the transfer outcomes; it does not update models or calibration.

The detector is a native-pixel center-query/offset model; its bounded integration
refines incumbent center proposals. Geometry uses label-blind point observations,
per-axis robust normalization and six candidates. The optical event tower uses
actual parent and daughter triplanes with explicit temporal validity. Synthetic
dense negatives, Zoo weak graph labels and Biohub sparse supported labels have
different masks and weights. RIKEN and unlicensed/uncleared species are excluded.

Detector offsets invert the native `/3` target normalization, then apply a
0.5 µm displacement cap before integer rounding. The realized rounded distance
may exceed that cap; the per-clip receipt reports it. A failed raw-logit
consistency attempt is retained locally under `failed_fits`; production real-only
adaptations use bounded sigmoid/softmax consistency. They match supervised rows
and optimizer updates, with one additional detached teacher forward.

A pre-outcome audit also corrected unsupported edge negatives for possible
unannotated second daughters. `sparse_mask_repair` preserves the former G/I
checkpoints and caches and restarts every affected fit at its full budget.
`repair_checks` refreshes the source-read, stress and runtime checks against the
corrected weights. The original failed fits never enter the reported comparisons.

Use the generated local package's `run.sh` with explicit image, output, v1, v2 and
source-model paths. `--disable-new-heads` preserves v3 identity. The package bundles
new heads and reuses the verified baseline dependencies; it does not require external
training datasets or GT and does not submit to Kaggle.

The study is operational exploratory. Released synthetic data depends on 44b6;
the public incumbent and historical teachers retain inherited exposure. External
validation temperatures are shared calibration access even for real-only neural
weight controls. Neither generator holdouts nor Zoo time blocks establish an
independent biological validation set.
