# Continuation — pipeline error training

Read REPORT.md and STATUS.json first. This file is generated from the current execution receipts.

## Current state

State: executing. Production default: P0. No new target fitting or threshold selection is authorized.

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

The post-primary queue requires both primary progress files to say complete. It performs source-only
qualification and replication before the target freeze. Every failed independent prediction lane is retained.
If a frozen stage fails, preserve its files under the new invalid/ root, record a named implementation repair,
and rerun the same stage with unchanged scientific weights, calibration, bank and decision policy.

## Pending or unsuccessful registered experiments

- D10_frozen: not run — 2/2 directional fits complete; all-199 target score unavailable
- D10_adapted: not run — 2/2 directional fits complete; all-199 target score unavailable
- D10_random: not run — Organoid family did not qualify in both source directions; conditional control was not authorized
- D20_compact: not run — 2/2 directional fits complete; all-199 target score unavailable
- D20_temporal: not run — 2/2 directional fits complete; all-199 target score unavailable
- D20_no_pretrain: not run — 2/2 directional fits complete; all-199 target score unavailable
- A10: not run — 2/2 directional fits complete; all-199 target score unavailable
- O10_swap: not run — 2/2 directional fits complete; all-199 target score unavailable
- O10_restore: not run — 2/2 directional fits complete; all-199 target score unavailable
- C10: not run — Required execution or validation is incomplete
- V10: not run — Required execution or validation is incomplete
- A10_replication: not run — 2/2 directional fits complete; all-199 target score unavailable
- D10_adapted_replacement: not run — 2/2 directional fits complete; all-199 target score unavailable
- D20_temporal_replacement: not run — 2/2 directional fits complete; all-199 target score unavailable

## Completed-fit crop caches

The frozen fits' regenerable training crops (23.45 GiB) were retired to reserve space for inference. All 32 model/optimizer/RNG state files were hash-verified before and after. The data loaders rebuild missing crops on demand from pinned images. See training_crop_retirement.json and work/pipeline-error-training-20260915/maintenance/training_crop_retirement/manifest.json.

## Artifact locations and restrictions

- Resumable weights, optimizer/RNG state, full graphs, source labels, image embeddings and logs: work/pipeline-error-training-20260915/.
- Queue state: queue/progress.json, observation_queue/progress.json, finish_queue/progress.json under that root.
- GPU accounting: gpu_budget/ledger.json; limits are 20 GiB total device memory, 50 GiB study RSS and 48 total lease-hours. The original reservations are 4/24/12/8 hours. Before target evaluation, gpu_reservation_settlement.json records unused first-seed hours shared with final inference; all 12 replication hours remain reserved.
- Concrete validation failures: fresh_image_validation.json and native_refresh_validation.json, with log hashes; original failed attempts stay under invalid/.
- Missing clean upstream fits block a clean end-to-end transfer claim; they do not block the independent operational lanes.
- Do not change the production default, merge, upload to Kaggle or publish weights.
