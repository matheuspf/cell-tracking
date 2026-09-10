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
  inputs and 288-dimensional edge embeddings. All four source/seed probes completed
  2,000 optimizer updates. Original decoder source tiles completed for both embryos
  using the upstream SCIP fallback; the empty source6 central-tile attempt is retained.
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

The first new complete comparison, J with frozen primary native evidence, scores
**0.8773353943845731**, delta C0 **-0.0574669798760129**. Its 44b6 score is
0.7473208427350831 and 6bba score is 0.9021821466767624. All 199 clips were freshly
scored with one CPU scorer alongside the six-worker control decoder. This changes
both the primary-only evidence and decoder relative to full C0, so it does not
isolate the decoder alone. No target success or promotion has been established.
The fixed-node truth-assisted legal graph scores **0.9576193465416518** and the
augmented-node counterpart scores **0.9721455963275534**, both on all 199 clips.
These are heuristic feasibility diagnostics, not deployable models, global upper
bounds, or target success. Production N2 fits, replication, final fresh inference,
optical review, and final reporting remain required.

The first full new-observation control, **P_union_native_J**, scores
**0.6751326227250869** (delta C0 **-0.2596697515354991**) on all 199 clips. It exports
4,521,204 nodes, with edge TP/FP/FN 109,131/30,185/19,752 and division TP/FP/FN
41/4,774/110. Improved sparse node recall does not compensate for these association
errors. This is a negative result for this frozen-representation proposal arm;
HOCT, adapted native representations and confirmation controls remain pending.

At the 17:07 UTC recovery update, all 199 observation, HOCT-feature and DeepCenter
shards and target candidate banks were complete. Both primary N1 fits completed
8,000 updates and source calibration; primary N2 fits were running beyond 7,600
updates for 44b6 and 2,800 for 6bba. All opposite-source HOCT and N1 score arrays
were complete and waiting for the serialized CPU decoder. All 199 P_union_native_J
graphs were complete; their official scoring finished at 17:11 UTC alongside the
six control workers. The additional actual
3D-network pixel-response test passed. No training recipe was reduced.

Full candidate coverage is complete: 130,836 of 133,318 annotated nodes matched
C0, 125,426 of 128,883 annotated edges occurred in its bank, and the expanded bank
recovered 1,614 additional annotated nodes without losing a C0 match. The 2,043,878
new peaks are predominantly unlabeled; this is not an estimate of dense detection
precision. Official locally feasible division evidence covered 118 of 151 events;
the separate exact-ID candidate-pair count was 99. Detailed identities stay local.

The guarded `inference_package_early4` C0 run completed from an unfamiliar filename
with exact full-graph parity and CSV roundtrip. Actual annotation/cache/DNS denial
tests passed before numerical imports. Prior guard failures exposed the secondary
checkpoint config, v2 hash-only teacher lock, and newly generated output GEFF;
precise pinned exceptions were added, and every failed attempt was preserved.
Eight early full-image C0/H/N1/P/DeepCenter paths also completed on the other
embryo in 235.27 seconds, with CSV and read-audit checks passing. Only C0 had graph
parity checked in that early trial. All six-clip comparisons, including actual N2,
remain required. The early bundle is an integration artifact, not the final export.

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

The detached tmux session is **`cell-tracking-v5`**. Its original supervisor is
preserved. Scheduling now allows one native fit per source, with at most two
simultaneous native optimizers and one serialized auxiliary inference/calibration
lane. The initial source6 N2 helper occupies that auxiliary lane until it finishes.
Each source's replica helper waits for its own primary N2 model, then runs the
registered N1 and N2 replica fits. Per-fit locks coordinate these helpers with the
original queue, which completes every source calibration. The complete optimizer
function body is structurally identical to the pre-scheduling version. A 600-update
benchmark reached 2.50 combined updates/second, versus the earlier single-source
1.40, with 9.34 GiB GPU and 13.54 GiB sampled summed RSS peaks. See
`source_lane_schedule.json` and `GPU_concurrency_benchmark.json`; the running
supervisor's original `gpu_policy` string describes its startup queue only.
CPU process
pools share a lock. The existing control pool retains six workers; future decoder
pools use eight after measured CPU/RSS profiling, with other pools at six or fewer.
An event index avoids repeated scans; every MILP array remained exact on 16 source
fixtures. HiGHS's configured two-second limit can be exceeded by presolve/runtime;
the report records actual elapsed solver time. Check actual state rather
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

The concurrency helpers are separate from the original supervisor. After checking
that none is already running, restore `source_lane 44b6` and `source_lane 6bba` in
their own detached windows. If source6 primary N2 is still incomplete, restore its
`aux_native 6bba N2` helper as well. These wrappers reuse complete, hash-verified
fits and lock unfinished fits; do not invoke bare optimizer jobs in parallel.

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

Additional detached waiters in the same tmux session run `regret --wait`,
`division_review --wait`, `fresh_validate --wait`, `final_analysis --wait`, and
`finalize --wait`. The fresh validator starts once primary models/calibrations exist,
running the frozen six image clips during replica training. It waits for complete
scores before comparing results. The final package must preserve every tested
runtime/primary-weight byte and then execute its selected default and disable switch.
Other waiters require complete frozen results, then run diagnostics,
manifests, preservation checks, final reports and
browser validation. The final-stage waiters exec the current maintained module when
their prerequisites become ready. After a reboot, restart each waiter once only
after checking process state. The supervisor covers the compute queue; these
artifact waiters are separate. They never stage, commit, push or submit anything.

J had 882 windows stop with a feasible time-limited solution; expanded P1 also has
many feasible timeouts. Fresh tests retain the original exact graph/integer-count
parity requirement. They record predecoder observation, candidate, model-score and
configuration fingerprints to distinguish input mismatches from solver decisions.
Any failed parity variant stays in the measured report and is excluded from
promotion. A completed failed test must not be described as a verified pipeline;
the final export must pass its own fresh tests, with C0 retained as fallback.

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

J/native/ctc calibration uses the first ten lexicographic source clips; HOCT H0/H1/H2
calibration uses the supported source feature rows from the full training source.
Sparse unknowns stay unknown. Reused embryos, C0 teachers and public checkpoint exposure remain
explicit. Node universes and candidate banks are separate for fixed and expanded
populations. Complete native comparisons intentionally use the primary model
without the incumbent's secondary/eight-view harmonic ensemble; N0 isolates this
change. No labels or cached selected graphs may enter fresh inference.

## Remaining work

1. Finish the supervisor jobs, inspect every failure, and require exactly 199
   officially scored clips for every registered configuration. Preserve all failed
   attempts and input/model/config fingerprints.
2. Preserve completed original-HOCT/ctc_v0 source diagnostics, including the bounded
   source6 ROI retry. They are source diagnostics, not complete scores.
3. Preserve completed candidate coverage and both graph-legal oracles. Keep official local
   division evidence distinct from exact-ID daughter pairs. Record candidate,
   missing-region, score, protection and conflict attrition.
4. Compute GT-edge and predicted-edge regret, node/matching effects, and exact
   additive score decomposition. The final-objective source P1/PDC pilots are done;
   `P2_decision.json` records why the conditional extension is not scheduled. Do not
   substitute center refinement for dense proposals or label unmatched peaks false.
5. Execute the frozen six full density-spanning clips from both embryos while
   replica training continues, including
   actual H/N/P/DeepCenter/J paths, C0 disablement, unfamiliar filenames, early
   offline guards, cache denial, CSV roundtrip and exact graph/numeric parity.
   Build the final package after complete comparisons; verify the tested validation
   payload is byte-identical and execute the final default and disable switch.
   The early C0 and eight-path integration trials passed their documented checks;
   they do not replace six-clip N2 and graph/count validation.
6. Generate the local optical-review pack after prediction freeze. Do not invent
   human judgments or turn review images into new labels.
7. Finish `final_report.md`, all required CSVs/manifests, dashboard validation,
   dependency manifest, final package, truthful STATUS and this continuation.
   The report generator includes measured family, headroom, regret and gate prose;
   inspect the completed results and browser output before final publication.
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
