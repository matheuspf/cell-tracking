Read [REPORT.md](REPORT.md) and [STATUS.json](STATUS.json). Status: complete.

The execution queue finished. The report records whether the measured nominee passed all recommendation and replication gates; completing the study does not itself establish a score of ≥0.95. P0 remains the production default.

Review [target_freeze.json](target_freeze.json) for the source-only checkpoint, application and family choices; [replication.json](replication.json) for completion and recommendation checks; and [fresh_image_validation.json](fresh_image_validation.json) for the renamed-image replays. Training, calibration, source screens, target scores, error transitions and resource receipts are linked from REPORT.md.

Heavy artifacts remain under `work/division-generalization-v2`, a symlink to `/kaggle/working/cell-tracking/division-generalization-v2`. Input paths and hashes are recorded in input_manifest.json. Preserve those artifacts, the recovery archives and the original baselines. No completed training or inference needs to be rerun to inspect these results.

To regenerate the concise report from existing receipts, run from `/home/mpf/code/kaggle/cell-tracking`:

```sh
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m division_generalization_v2 report
```

To reverify every saved target graph and refresh its portable hashes and error identities, run `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m division_generalization_v2.delivery` with the same environment settings. Raw-scene panels and their hashes are listed in diagnostic_gallery.json; the local gallery is `work/division-generalization-v2/diagnostics/gallery/index.html`.

For a deliberate independent reconstruction, use the committed study configuration and verified inputs in a separate isolated work root. Preserve the 4,096 joint-update floor, paired prefixes, source-only decisions and startup access guards. The original execution, invalid attempts and crash costs remain part of this study's provenance.

No production promotion, merge, Kaggle submission, weight publication or leaderboard claim was performed.
