# V5 execution continuation

This is an **in-progress execution checkpoint**, not the final study result.
The user requested implementation and execution of X500–X580, complete measured
results, fresh inference, a dashboard, and sanitized commits on this branch.
Continue the current study; do not restart v4.

## Current measured evidence

- Fresh official C0 scoring on all 199 clips exactly reproduces
  **0.934802374260586**. Per embryo: 44b6 **0.931664468721842**;
  6bba **0.935221784097327**. Keep C0 as fallback.
- Two full image-to-C0 pilots, chosen by image-derived median detection density,
  reproduce both complete graphs and the original pre-ILP arrays exactly.
- The official HOCT general_v1 JIT runs with genuine 19-dimensional region/position
  inputs and 288-dimensional edge embeddings. The first source probe completed
  2,000 optimizer updates. Other source/seed fits are queued.
- Real-source tiny N1/N2 overfits passed. N1 kept the actual U-Net byte-identical;
  N2 changed its first encoder tensor with a nonzero gradient. These are tiny
  integrity checks, not complete learned-model comparisons.
- Full-field native peak discovery, watershed morphology, DeepCenter confirmation,
  common candidate banks and a rolling five-frame MILP are implemented and running.
  Source pilots expose substantial false-edge/node-adjustment risk from raw P1.
- Actual solver/sparse-supervision fixtures pass, including duplicate hypotheses,
  timing identities, two positive daughters, empty/censored losses and no-op parity.
- The offline dashboard passes desktop/mobile browser and CSV-download checks.
  It displays progress and only complete official score rows.

No new operational 199-clip comparison had completed at this checkpoint. No target
success or promotion has been established. Production N2 fits, replication, final
fresh inference, optical review, and final reporting remain required.

## Server recovery on 2026-09-10

The server rebooted after logs stopped near 12:56 UTC. At 13:32 UTC no v5 processes
were running. Kernel logs were inaccessible, so the cause is unknown. The last
resource sample was within the registered limits; that does not establish the
cause of the reboot.

Recovery verified every saved NPZ hash in 122 observation shards, 120 HOCT feature
shards, 122 DeepCenter shards and existing candidate banks, plus all protected
incumbent dependencies. The native log reached update 2,941; the last complete
optimizer/RNG checkpoint was update 2,000. It was copied separately into
`recovery_checkpoints/` before resuming. All 48 repeated logged losses matched at
their recorded five-decimal precision. Resume snapshots now occur every 250
updates, with the model recipe unchanged.

The detached tmux session is **`cell-tracking-v5`**. The supervisor runs one
sequential native training lane and one serialized auxiliary GPU lane. CPU process
pools share a lock and never exceed six pool workers. Check actual state rather
than starting duplicate jobs:

```bash
tmux list-sessions
pgrep -af image_native_tracking_v5
cat /kaggle/working/cell-tracking/image-native-tracking-v5/supervisor_state.json
tail /kaggle/working/cell-tracking/image-native-tracking-v5/logs/supervisor.log
```

After another reboot, verify cache/checkpoint hashes against their sidecars and
`recovery_20260910.json`, inspect any partial outputs, then restart the supervisor
only if no existing supervisor is running:

```bash
cd /root/code/kaggle/cell-tracking
tmux new-session -d -s cell-tracking-v5 -c "$PWD" \
  'exec scripts/run_image_native_tracking_v5.sh resume supervise >> /kaggle/working/cell-tracking/image-native-tracking-v5/logs/supervisor.log 2>&1'
```

The supervisor does not hide failures or retry failed interfaces indefinitely.
Inspect `supervisor_state.json` and the named job log. Completed source checkpoints,
calibrations, cache shards and officially scored graphs are reused after checks.
The native queue also completes calibration if a checkpoint survived but its
calibration did not.

## Paths and environment

- Repository: `/root/code/kaggle/cell-tracking` (also `/root/cell-tracking`).
- New outputs: `/kaggle/working/cell-tracking/image-native-tracking-v5`.
- New maintained code: `tools/image_native_tracking_v5/`.
- Runtime: `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`.
- Browser checks: `/root/.conda/envs/cell-tracking/bin/python`.
- Canonical data: `/kaggle/input/competitions/biohub-cell-tracking-during-development`.
- Source-only HOCT install: `work/image-native-tracking-v5/python`.
- Prior studies remain read-only siblings of the v5 output directory.

Use `scripts/run_image_native_tracking_v5.sh MODULE ...`; it sets the required
Python isolation, thread counts, GPU 0 and source paths. It supports `--python` and
works from a foreign working directory. Do not install into system Python or
upgrade the working baseline environment.

The live dashboard is `OUTPUT/dashboard.html`; `OUTPUT/status.json` and
`OUTPUT/training/*_progress.json` contain current progress. Refresh with the `report`
module. The tmux `dashboard` window refreshes it every minute. This early package
at `OUTPUT/inference_package_early` is an integration artifact and is **not yet
validated for export**. Preserve the crash-interrupted fresh test directory.

## Frozen experimental scope

`OUTPUT/execution_protocol.json` registers **18 total complete configurations**:
16 operational configurations including C0, and the fixed/augmented legal
truth-assisted oracles. The ctc_v0 alternative is a source-only diagnostic, freeing
the two oracle slots. The earlier registration snapshot is retained separately.
Do not add an unregistered target-label threshold sweep.

Both source directions use the same recipe: 8,000 N1 updates, followed by 12,000
N2 updates, effective batch four full-frame windows, AdamW head LR 1e-4 and encoder
LR 1e-5. Repeat with seeds 20260910 and 314159. The native queue completes all eight
fits. H1 uses true frozen HOCT embeddings, 2,000 linear-probe updates and no hard
ILP consistency. H2 adds one source-fitted residual calibration with frozen native
evidence. N2 was selected for the P1 factorial from source interface/overfit evidence
before any outer operational scores.

Source calibration uses the first ten lexicographic source clips. Sparse unknowns
stay unknown. Reused embryos, C0 teachers and public checkpoint exposure remain
explicit. Node universes and candidate banks are separate for fixed and expanded
populations. Complete native comparisons intentionally use the primary model
without the incumbent's secondary/eight-view harmonic ensemble; N0 isolates this
change. No labels or cached selected graphs may enter fresh inference.

## Remaining work

1. Finish the supervisor jobs, inspect every failure, and require exactly 199
   officially scored clips for every registered configuration. Preserve all failed
   attempts and input/model/config fingerprints.
2. Complete the original HOCT decoder and ctc_v0 source diagnostics. Their tmux
   window is `HOCT-diagnostics`; inspect its log. They are not complete scores.
3. Complete candidate coverage and both graph-legal oracles. Keep official local
   division evidence distinct from exact-ID daughter pairs. Record candidate,
   missing-region, score, protection and conflict attrition.
4. Compute GT-edge and predicted-edge regret, node/matching effects, and exact
   additive score decomposition. Decide whether conditional P2 is justified from
   source evidence; do not substitute center refinement for dense proposals.
5. Build the final inference package after all required weights/calibrations exist.
   Execute at least six full density-spanning clips from both embryos, including
   actual H/N/P/DeepCenter/J paths, C0 disablement, unfamiliar filenames, early
   offline guards, cache denial, CSV roundtrip and exact graph/numeric parity.
   An early C0 retry is queued in tmux `package-check` behind the auxiliary GPU lock.
6. Generate the local optical-review pack after prediction freeze. Do not invent
   human judgments or turn review images into new labels.
7. Finish `final_report.md`, all required CSVs/manifests, dashboard validation,
   dependency manifest, final package, truthful STATUS and this continuation.
   The current report generator still needs detailed final outcome/failure prose.
8. Run final preservation/sanitization checks; commit and push additive v5 code and
   sanitized results to `handover/image-native-tracking-v5`. No merge, Kaggle upload,
   notebook/forum publication, new hardware or paid API calls are authorized.

Promotion requires positive pooled gain and no embryo regression beyond 1e-8 in
both same-recipe seeds. Export the primary seed, not the better replica. A selected
score below 0.95 is a subtarget gain. If no candidate passes, retain C0 and report
the negative result with complete measurements.

Unrelated user work was already dirty at the start, including
`handover/strong-tracker-v2/CODEX_PROMPT.md`, external-data-guide/remote files and
`Untitled`. Stage only an explicit v5 allowlist; do not reset or clean the workspace.
