# Continuation — pipeline error training

Read REPORT.md and STATUS.json first. This file is generated from the current execution receipts.

## Current state

State: executing. Production default: P0. No new target fitting or threshold selection is authorized.

Active stage: point_predictions. Completed division matrices: 199/199.
Completed point predictions per arm: A10=199/199, A10_replication=199/199, O10_swap=60/199.

Inspect actual processes before launching any resumable queue; active.json can refer to a completed child.
Do not kill or resume a different study. Historical native 400-epoch fits remain untouched.

```sh
ps -eo pid,ppid,etime,rss,args | rg 'pipeline_error_training|train_native'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
```

## Runtime and exact commands

Run from the repository root. Do not start a duplicate queue while its process is alive.

```sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
$STUDY_PY -m pipeline_error_training queue
$STUDY_PY -m pipeline_error_training.observation_queue
$STUDY_PY -m pipeline_error_training.finish_queue
$STUDY_PY -m pipeline_error_training.report
```

For early CPU audits while the finish queue is running in another terminal, use
`$STUDY_PY -m pipeline_error_training.early_scoring`. It requires every completed clip and
serialization file. A global scorer lock serializes audits and shared table exports;
the final queue reuses completed scoring receipts. This does not change model selection.

The post-primary queue requires both primary progress files to say complete. It performs source-only
qualification and replication before the target freeze. Every failed independent prediction lane is retained.
If a frozen stage fails, preserve its files under the new invalid/ root, record a named implementation repair,
and rerun the same stage with unchanged scientific weights, calibration, bank and decision policy.

## Pending or unsuccessful registered experiments

- D10_random: not run — Organoid family did not qualify in both source directions; conditional control was not authorized
- O10_swap: not run — 2/2 directional fits complete; all-199 target score unavailable
- O10_restore: not run — 2/2 directional fits complete; all-199 target score unavailable
- C10: not run — Required execution or validation is incomplete
- V10: not run — Required execution or validation is incomplete

## Host restart recovery

The host restarted at 2026-09-16T11:10:47+00:00; the exact study interruption time is unknown. All 129 completed clip matrices and the frozen models were verified before resuming. Incomplete outputs and the previous ledger remain under invalid/host_restart_20260916/. GPU accounting conservatively includes 2.716 hours from the unclosed lease through the new boot, including possible downtime. This is an upper bound, not observed active execution. Training and other studies were not resumed. See host_restart_recovery.json.

## Completed-fit crop caches

The frozen fits' regenerable training crops (23.45 GiB) were retired to reserve space for inference. All 32 model/optimizer/RNG state files were hash-verified before and after. The data loaders rebuild missing crops on demand from pinned images. See training_crop_retirement.json and work/pipeline-error-training-20260915/maintenance/training_crop_retirement/manifest.json.

## Artifact locations and restrictions

- Resumable weights, optimizer/RNG state, full graphs, source labels, image embeddings and logs: work/pipeline-error-training-20260915/.
- Queue state: queue/progress.json, observation_queue/progress.json, finish_queue/progress.json under that root.
- GPU accounting: gpu_budget/ledger.json; limits are 20 GiB total device memory, 50 GiB study RSS and 48 total lease-hours. The original reservations are 4/24/12/8 hours. Before target evaluation, gpu_reservation_settlement.json records unused first-seed hours shared with final inference; all 12 replication hours remain reserved.
- Concrete validation failures: fresh_image_validation.json and native_refresh_validation.json, with log hashes; original failed attempts stay under invalid/.
- Missing clean upstream fits block a clean end-to-end transfer claim; they do not block the independent operational lanes.
- Do not change the production default, merge, upload to Kaggle or publish weights.
