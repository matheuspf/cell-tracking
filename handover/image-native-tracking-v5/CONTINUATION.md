# V5 execution continuation

This is an **in-progress execution checkpoint**, not the final study result.
The user requested implementation and execution of X500–X580, complete measured
results, fresh inference, a dashboard, and sanitized commits on this branch.
Continue the current study; do not restart v4.

The **second reboot recovery** completed at 19:01 UTC on 2026-09-10. All study
processes had stopped. Saved model, cache and six complete evaluation sets passed
hash checks. The source44 replica N1 resumed at update 750 and source6 primary N2
at 7,750; the latter repeated 91 known logged updates. Four of five overlapping
printed losses match; the fifth differs by 0.00001. This is not bitwise training
reproducibility. Both snapshots, logs and two empty progress files were preserved.
See `recovery_second_20260910.json` and `resume_determinism_second.json`.
Training, decoding, scoring and all artifact waiters are running in the restored
`cell-tracking-v5` tmux session. Source44 primary N2 calibration also completed.
At 19:26 UTC, a measured scheduling overlap started the registered source6 N1
replica on the idle auxiliary lane before its primary N2 finished. This transient
third optimizer holds `gpu_aux`, excluding auxiliary inference/calibration. It
achieved 2.83 head updates/s while backbone throughput changed from 1.48 to 1.40
across different source windows, with 6.35 GiB GPU and 17.29 GiB RSS peaks. The
source helper and original queue will reuse/lock this same fit. See the separate
`GPU_head_overlap_schedule.json` and `GPU_head_overlap_benchmark.json` receipts;
they supersede the earlier two-optimizer scheduling ceiling for this one overlap.

At 19:54 UTC, primary source6 N2 completed 12,000 actual image-network updates.
All 12,374 eligible windows and 106,313 supported edges were seen. Its first encoder
tensor changed by 0.02013603 in L2 norm, with a nonzero first-step convolution
gradient. The checkpoint hash verified. Both primary N2 models and the source44
N1 replica completed first. Source6 replica N1 subsequently finished at 20:17 UTC,
giving six of eight completed native fits. Both N2 replications remain active.
Both primary N2 calibrations are complete and their opposite-source full-image
score extraction is running. All 199 confidence-ablation and N1 graphs are complete;
their official scorers are queued behind the active HOCT decoder batch.
The complete transient three-optimizer phase peaked at 10.22 GiB GPU and
17.29 GiB summed RSS, within both caps.

At 20:31 UTC, **H_general_J** completed official scoring on all 199 clips:
**0.906022657469521**, delta C0 **-0.028779716791064947**, with edge TP/FP/FN
121,667/7,559/7,216 and division TP/FP/FN 29/99/122. Seven of 18 registered complete
configurations are now measured. The adapted H graph variants remain in progress.

At 20:51 UTC, **H_probe_J** completed all 199 official scores at
**0.9135868579084195**, delta C0 **-0.02121551635216645**. Per embryo it scores
0.8440421993352256 on 44b6 and 0.92770934806879 on 6bba. Edge TP/FP/FN are
121,603/6,053/7,280; division TP/FP/FN are 29/149/122. This improves on H0 but
regresses both embryos against C0. Eight of 18 configurations are now measured.
The H/native combination and same-recipe replication remain in progress.

A source-only audit at 20:36 UTC confirmed a material HOCT interpretation limit.
The registered adapter supplies micrometer coordinates/diameters and squared-
micrometer inertia, while the pinned upstream extractor returns voxel-unit
features even when scale metadata is supplied. This was found after outer scores,
a protocol deviation from the requested pre-score scaling audit. All 655 actual
source-image regions matched unscaled upstream regionprops exactly. On the same
two source tiles, a native-voxel reference changed mean absolute JIT logits by
0.812178 and 0.300645, and candidate-parent argmax for 9/92 and 0/26 targets.
These CPU FP32 unit-sensitivity diagnostics used no GT scores and changed no
production features, fit, calibration, candidate bank or decoder. The full
199-clip upstream-default-voxel HOCT comparison was not registered or run. Retain
that explicit limitation rather than interpreting the H score as checkpoint
failure. See `HOCT_unit_contract_audit.json` and the dashboard's feature section.

The six-clip validation bundle is frozen at manifest SHA256
`c0b6dae1a34defcd8d896bfc83eae42c961fe3c21effba04b054f7dd6efa217c`.
All 155 listed files and the early annotation/cache/network guard self-test passed.
Primary N2 score extraction finished all 199 clips at about 20:53 UTC, and the
first fresh-image child began at 20:53:28 using unfamiliar name `volume_01` and
source model 6bba. It will execute all 12 primary/control paths. The primary N2
graph decoder is queued behind the active HOCT batch. Both N2 replicas continue
their full 12,000-update fits. No full six-clip parity result exists yet.
Do not edit this bundle's runtime, primary weights or calibrations without a new
complete six-clip validation attempt. Reporting-only changes are separate.

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
HOCT and the adapted native representation/proposal factorial remain pending.

The complete frozen-representation DeepCenter confirmation arm, **P_DC_native_J**,
scores **0.8686524362605309** (delta C0 **-0.06614993800005509**): 44b6
**0.6773841767024975**, 6bba **0.9092325763689433**. It adds 15,179 nodes and removes
12,005, with edge TP/FP/FN 118,674/9,532/10,209 and division TP/FP/FN 29/165/122.
Confirmation improves the raw P1 result but remains below the fixed-node N0/J
control and C0. Its 199-clip scoring finished at 17:56 UTC without altering recipes.

At the 17:07 UTC recovery update, all 199 observation, HOCT-feature and DeepCenter
shards and target candidate banks were complete. Both primary N1 fits completed
8,000 updates and source calibration; primary N2 fits were running beyond 7,600
updates for 44b6 and 2,800 for 6bba. All opposite-source HOCT and N1 score arrays
were complete and waiting for the serialized CPU decoder. All 199 P_union_native_J
graphs were complete; their official scoring finished at 17:11 UTC alongside the
six control workers. The additional actual
3D-network pixel-response test passed. No training recipe was reduced.

At 18:06 UTC, primary 44b6 N2 completed all 12,000 real image-network updates in
9,435.73 seconds. Its 6,294 eligible windows and 19,530 supported positive edges
were all seen, with zero missed windows. The first encoder tensor changed by
0.02243369 in L2 norm; its first-step gradient norm was 0.05363821. The checkpoint
hash verified. Source44 replica N1 started automatically; primary source6 N2 was
past 7,300 updates. Source44 N2 calibration waits on the shared auxiliary GPU lock.
There are now three complete native fits and six complete scored configurations.

Full candidate coverage is complete: 130,836 of 133,318 annotated nodes matched
C0, 125,426 of 128,883 annotated edges occurred in its bank, and new peaks matched
1,614 annotated nodes absent from C0 matching without losing a C0 match. The 2,043,878
new peaks are predominantly unlabeled; this is not an estimate of dense detection
precision. Official locally feasible division evidence covered 118 of 151 events;
the separate exact-ID candidate-pair count was 99. Detailed identities stay local.

The post-freeze HOCT feature audit found 3,456,758 valid image-supported regions at
4,108,943 C0 nodes and 10,298,109 finite scores among 15,179,751 candidate edges.
Missing endpoint regions explain 4,848,861 edges; another 32,781 valid-endpoint
edges lack scored model support. Inertia tensors are anisotropic at 99.90% of valid
regions, and every valid region has intensity variation; no sphere features were
substituted. Physical z also differs from the published normalization distribution:
75.42% of regions lie beyond three published standard deviations. This is a
descriptive model-domain diagnostic, not a causal explanation or new calibration.
All 199 observation/HOCT-score hashes are retained in `HOCT_feature_audit.json`.

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

The detached tmux session is **`cell-tracking-v5`**. The supervisor was restarted
after each reboot, preserving its earlier logs. Scheduling allows one native fit per source, with at most two
simultaneous native optimizers and one serialized auxiliary inference/calibration
lane. After the second reboot the original queue resumed source6 primary N2;
the auxiliary training helper is no longer needed.
Each source's replica helper waits for its own primary N2 model, then runs the
registered N1 and N2 replica fits. Per-fit locks coordinate these helpers with the
original queue, which completes every source calibration. The complete optimizer
function body is structurally identical to the pre-scheduling version. A 600-update
benchmark reached 2.50 combined updates/second, versus the earlier single-source
1.40, with 9.34 GiB GPU and 13.54 GiB sampled summed RSS peaks. See
`source_lane_schedule.json` and `GPU_concurrency_benchmark.json`; the running
supervisor's original `gpu_policy` string describes its startup queue only.
CPU process pools share a lock. Completed control graphs are reused. Remaining
control graphs and already loaded N1/HOCT pools use eight workers. A further
22-sample, 670-second benchmark measured 16.37 GiB peak summed RSS; two more peak
workers plus the prior inference subtree peak project 22.90 GiB. Newly launched
decoder batches use ten workers, with other pools at six or fewer. The projection
is not a measured ten-worker peak. Numerical functions passed exact AST checks.
See `CPU_worker_recovery_benchmark.json` and `CPU_worker_recovery_schedule.json`.
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
the recovery receipts, inspect any partial outputs, then restart the supervisor
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
their own detached windows. Let the main queue resume unfinished primary fits;
do not also start the old auxiliary training helper. These wrappers reuse complete, hash-verified
fits and lock unfinished fits; do not invoke bare optimizer jobs in parallel.

JSON and graph writes now fsync file bytes before rename and the parent directory
afterward. The resource monitor also flushes newly published native resume
snapshots every 30 seconds. It runs in its own `resources` window through fresh
inference and final analysis. An exited monitor child may remain as a zombie
until the original supervisor reaps it; it is not an active duplicate monitor.

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

The separate `auto_evaluate` helper in tmux `official-scorer` scores each registered
variant once all 199 prediction receipts exist, while the decoder pool continues.
One serial scorer is allowed beside at most ten pool workers, with fresh resource
samples and a 24 GiB RSS admission threshold. A measured six-worker/one-scorer overlap
peaked at 14.90 GiB RSS. Per-variant locks prevent overlapping original batches from
writing the same receipts; the official evaluator and aggregator are AST-identical.
See `evaluation_overlap_schedule.json`. Restore this helper once after a reboot,
after checking no existing copy is running. Its output is the canonical registered
evaluation, not an extra scored configuration.
If a regular evaluator already owns a ready variant while waiting for its CPU
slot, the background scorer now skips that busy variant and tries other ready
families. Per-variant exclusion is preserved. The added partial-lock release test
passes; there are now 20 implementation tests. See `scorer_skip_schedule.json`.

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
populations. Morphology comes from one marker watershed containing both C0 centers
and newly discovered peaks; P0 keeps C0 graph nodes and coordinates with their
features from this shared partition. Complete native comparisons intentionally use
the primary model without the incumbent's secondary/eight-view harmonic ensemble;
N0 provides the matched control for N1/N2. No labels or cached selected graphs may
enter fresh inference.

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
