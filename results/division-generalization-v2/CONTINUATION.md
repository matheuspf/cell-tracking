Read REPORT.md and STATUS.json. Status: incomplete_resumable.

Run from `/home/mpf/code/kaggle/cell-tracking` on the current branch. Inspect actual processes first; never resume an older study or launch duplicate workers.

```sh
ps -eo pid,ppid,etime,rss,args | rg division_generalization_v2
nvidia-smi
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
$STUDY_PY -m division_generalization_v2 queue
$STUDY_PY -m division_generalization_v2 report
```

The queue checkpoints every 128 joint updates and at milestones. Each matched arm loads its seed/source's identical 2,048-update model, optimizer, RNG and data-order state. Do not shrink 4,096 updates or treat interrupted fits as failed architectures.

Heavy artifacts are under `work/division-generalization-v2`, a symlink to `/kaggle/working/cell-tracking/division-generalization-v2`. Recreate that empty isolated root on another machine; inherited inputs resolve through input_manifest.json. Existing raw inputs and work/pipeline-error-training-20260915 are read-only references. No shared environment changes are needed.

Any stage failure is preserved with its exact command and log in queue/failure.json and logs/. Fix implementation defects, archive invalid attempts with reasons, and retain scientific settings. An unclosed GPU lease requires conservative accounting reconciliation before restart.

Outstanding gates are explicit in STATUS.json, validation.json, training_receipts.json and source_scores.csv. Target export requires source qualification; failed cheap control does not block main training. Both directional packages and both seeds must be frozen before target scoring. Full target scoring and fresh image proof, not partial clips, govern recommendation and the ≥0.95 claim.
