# Image-native tracking v5

Execution implements the [v5 handover](../handover/image-native-tracking-v5/CODEX_PROMPT.md).
Read [CONTINUATION](../handover/image-native-tracking-v5/CONTINUATION.md) for the
current measured state and unfinished work. The incumbent is C0/v3 A_residual_m3.0
at 0.934802374260586; the study target is at least 0.95.

All new artifacts are isolated under
`/kaggle/working/cell-tracking/image-native-tracking-v5`. Raw images, sparse GEFF
labels and v1–v4 artifacts remain unchanged. New source code is in
`tools/image_native_tracking_v5`; `scripts/run_image_native_tracking_v5.sh` selects
the existing annotation runtime and explicit roots.

The independent model families are official pretrained HOCT with supported-source
edge probing, the installed native association transformer with a frozen encoder,
and actual temporal 3D U-Net adaptation. A fixed/expanded-node factorial combines
full-frame new optical peaks, optional DeepCenter confirmation and five-frame
joint graph selection. Full comparisons preserve source-only fitting/calibration
and repeat learned finalists with the registered second seed.

The complete C0 score and two fresh full-image pilots have been reproduced. The
initial source fits, image extraction and full comparisons are still executing at
this commit. No new pipeline has yet established target success. Local progress
is available in `OUTPUT/dashboard.html`, `OUTPUT/status.json` and
`OUTPUT/supervisor_state.json`.

The supervisor runs independently of the chat in tmux and reuses completed shards
and optimizer/RNG checkpoints after validation. A reboot on 2026-09-10 interrupted
the first production native fit; recovered artifacts passed their recorded hashes,
and the repeated loss sequence matched at logged precision. Resource limits remain
20 GiB soft GPU memory, 28 GiB process RSS, at least 8 GiB free disk, 18 complete
configurations and 48 summed GPU training hours.

Direct source fitting does not erase upstream checkpoint/teacher exposure. Both
embryos have been repeatedly reused. Reported comparisons are operational
exploratory results, with no independent-biological-validation or leaderboard claim.

Useful commands from any working directory:

```bash
/root/code/kaggle/cell-tracking/scripts/run_image_native_tracking_v5.sh report
/root/code/kaggle/cell-tracking/scripts/run_image_native_tracking_v5.sh \
  --python /root/.conda/envs/cell-tracking/bin/python dashboard_check
```

`report --final` is reserved for completed registered comparisons and passed fresh
validation. The early inference bundle is an integration artifact; use the final
validated package once the continuation records completion. No Kaggle submission
is performed by this study.
