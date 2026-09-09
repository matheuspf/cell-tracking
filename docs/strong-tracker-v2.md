# Strong tracker v2: executed study

This is the additive execution of [the v2 handover](../handover/strong-tracker-v2/CODEX_PROMPT.md). The sealed v1 study and raw inputs remain unchanged. New code lives in `tools/strong_tracker_v2/`; local outputs live at `/kaggle/working/cell-tracking/strong-tracker-v2/`.

The selected local pipeline disables the original Harmonic Fusion motion relinker, preserves the cached neural associations, and keeps the remaining repairs. The strict-bounds export scores **0.934206**, against **0.911774** identity: **+0.022432**, with positive change in both embryos. See [the measured report](../results/strong-tracker-v2/v2_report.md) and [offline dashboard](../results/strong-tracker-v2/dashboard.html). Public upstream contamination and reused outcomes make this exploratory; no leaderboard uplift is inferred.

## Environment

Run from `/root/code/kaggle/cell-tracking`. The wrapper sources `scripts/root_remote_env.sh`, removes the conflicting CUDA toolkit library override, and explicitly executes `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`. It sets `PYTHONNOUSERSITE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, GPU 0, and two-thread BLAS/OpenMP/Polars limits. The inspection/Playwright interpreter is `/root/.conda/envs/cell-tracking/bin/python`.

The pinned official evaluator is `work/annotation-selection-v1/official`, revision `075fc5f5a52d11077f9dc2b074644618f26939e2`. The documented upstream metadata exception remains: this validated metric runs on the existing PyTorch 2.8.0+cu128 stack even though upstream package metadata requests a later version. No shared stack was upgraded.

## Reproduction and resume

These commands reuse the completed v1 caches in place. They are v2 commands, not the v1 stage commands with hard-coded output paths. Long stages were run in tmux with logs in the v2 namespace. Run the sequence below serially for a conservative reproduction; the execution also overlapped independent CPU stages within about 40 active threads.

```bash
cd /root/code/kaggle/cell-tracking
study_out=/kaggle/working/cell-tracking/strong-tracker-v2
mkdir -p "$study_out/logs"
if [ ! -f "$study_out/resource_monitor.stop" ]; then
  bash scripts/run_strong_tracker_v2.sh resources > "$study_out/logs/resources.log" 2>&1 &
  v2_monitor_pid=$!
  trap 'touch "$study_out/resource_monitor.stop"' EXIT
fi
bash scripts/run_strong_tracker_v2.sh census --workers 16
bash scripts/run_strong_tracker_v2.sh replay --limit 2 --workers 1
bash scripts/run_strong_tracker_v2.sh replay --workers 4
bash scripts/run_strong_tracker_v2.sh registration
bash scripts/run_strong_tracker_v2.sh oracles --workers 16
bash scripts/run_strong_tracker_v2.sh training_data --workers 16
bash scripts/run_strong_tracker_v2.sh fit
bash scripts/run_strong_tracker_v2.sh infer --workers 16
bash scripts/run_strong_tracker_v2.sh evaluate --variant repair --workers 16
bash scripts/run_strong_tracker_v2.sh stages --workers 4
bash scripts/run_strong_tracker_v2.sh census_stages --workers 4
bash scripts/run_strong_tracker_v2.sh replay --variant no_motion --workers 4
bash scripts/run_strong_tracker_v2.sh bypass_score --workers 4
bash scripts/run_strong_tracker_v2.sh evaluate --variant filter --workers 12
bash scripts/run_strong_tracker_v2.sh risk --workers 2
bash scripts/run_strong_tracker_v2.sh supplement --variant risk-infer --workers 2
bash scripts/run_strong_tracker_v2.sh supplement --variant risk --workers 2
bash scripts/run_strong_tracker_v2.sh temporal --steps 5000
bash scripts/run_strong_tracker_v2.sh supplement --variant image-infer
bash scripts/run_strong_tracker_v2.sh supplement --variant image --workers 12
bash scripts/run_strong_tracker_v2.sh combinations --workers 2
bash scripts/run_strong_tracker_v2.sh supplement --variant combo --workers 2
bash scripts/run_strong_tracker_v2.sh selected
bash scripts/run_strong_tracker_v2.sh classifier_metrics
bash scripts/run_strong_tracker_v2.sh attribution
touch "$study_out/resource_monitor.stop"
if [ -n "${v2_monitor_pid:-}" ]; then wait "$v2_monitor_pid"; fi
bash scripts/run_strong_tracker_v2.sh report
source scripts/root_remote_env.sh
export PYTHONPATH="$PWD/tools:$PWD/work/annotation-selection-v1/official/src"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m pytest -q tests \
  work/annotation-selection-v1/official/tests/test_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_sandbox_examples.py \
  > "$study_out/logs/final-tests.log" 2>&1
/root/.conda/envs/cell-tracking/bin/python tools/strong_tracker_v2/dashboard_check.py
bash scripts/run_strong_tracker_v2.sh finalize
```

Completed sample outputs are resumable. Config/model locks reject changed locked values. A changed scientific setting belongs in a new variant or study namespace. Temporal checkpoints record optimizer, NumPy and torch RNG states, positive sampling position, unique-example coverage and learning curves for exact step resume. Full fits use all source positives; the 10/30/100% learning curves each run 5,000 steps.

The replay adapter executes only the notebook configuration and repair declarations, avoiding notebook installation, materialization, prediction and submission side effects. It records six true coarse graph phases. Two ULP-level half-integer serialization differences are explicitly reconciled with the sealed baseline; native floats, failure receipts and separately scored native-rounding results are retained. The six pre-existing out-of-bounds baseline points are never silently clipped in identity scoring.

The winning setting is `OUTPUT_MOTION_RELINK=False`; other notebook repair parameters are unchanged. `selected_prediction_lock.json` records the original notebook hash, exact changes, all sample graph hashes, strict validators and byte parity between independently annotation-blocked export and scored graphs. This is a cached-prediction experiment, not a fresh hidden-set notebook runtime measurement.

## Validation and preservation

```bash
source scripts/root_remote_env.sh
export PYTHONPATH="$PWD/tools:$PWD/work/annotation-selection-v1/official/src"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m pytest -q \
  tests \
  work/annotation-selection-v1/official/tests/test_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_sandbox_examples.py
```

Inference audit hooks block GT GEFFs, evaluation/oracle/census files and the GT-bearing inventory before input/model loading in inference workers. Every scored graph uses fresh official matching, full local division evaluation and exact run-level aggregation. Group-deletion experiments are source-only evaluation targets; oracle intervention fields never enter deployed features or repairs.

All detailed GT windows, coordinates, weights, predictions and image stores remain local. Only code, reproducible configs and sanitized aggregate evidence are tracked. No Kaggle submission, network model inference, forum posting, broad install or remote publication occurs.

Before replacing the Vast instance, copy out the entire v2 output root, ignored `work/strong-tracker-v2/`, and the new Git commit. Reproduction also needs the preserved v1 store, original competition inputs and pinned evaluator. `/root` and `/kaggle` are on the nonpersistent instance disk.
