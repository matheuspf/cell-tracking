# Local v11 execution

Run only on `handover/division-reliability-v11-ready`. The scientific contract is
`handover/division-reliability-v11/study.json`; `results/division-reliability-v11/execution_lock.json`
records the measured selection of U=8000 and E=4000. Historical v10 checkpoints
are preserved and excluded from model ancestry.

From the repository root, use the already prepared environment:

```sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=tools:.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
study_python=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
"$study_python" -m division_reliability_v11 report
```

`report` checks process identities through `/proc`, reads durable checkpoint
state, and refreshes sanitized results. Its timestamps alone do not prove that
training is running. Private stage logs, checkpoints, complete predictions and
telemetry are under `work/division-reliability-v11/`.

For a lightweight read-only view of live process identities, frame progress and
completed jobs without refreshing the report files:

```sh
"$study_python" -m division_reliability_v11.monitor
```

When no existing pipeline controller is running, resume the dependency queue:

```sh
"$study_python" -m division_reliability_v11 run --workers 3
```

The queue joins an existing upstream worker when one owns the study. Cell locks
and a controller lock reject duplicate execution. An optional upstream-only
queue is already in use for this execution; do not start another copy. If it
stops, the pipeline can resume remaining upstream cells itself.

The initial CPU concurrency is three. `controller/options.json` in the private
work directory can set `workers` or `workers_by_stage` (for example,
`prepare-actions` or `predict:C01`). Increases up to eight require measured
per-worker memory headroom; the 44 GiB study RSS limit, 10 GiB available-host
floor and durable-space floor continue to apply. Changes affect newly dispatched
jobs and never terminate existing workers. GPU work still shares one lease lock.

The controller preserves prior job/resource receipts before reusing a stage's
outputs. A temporary metadata observer also snapshots completed receipts from
the queue processes that were already running when this archival repair was
installed. It starts no model or data work and exits when those owners finish.
Resource reporting deduplicates archived copies and distinguishes original
execution timings from short verification/reuse processes.

A completed C00 cell later in the schedule can prepare its source observations
while an earlier cell trains C11:

```sh
"$study_python" -m division_reliability_v11.source_prefetch --source 44b6 --seed 314159 --workers 3
```

Start this only after that cell's C00 final receipt exists. A per-cell ownership
barrier makes the normal queue wait at its C00 package entry until preparation
finishes. The helper runs the same guarded source predictions, action banks and
diagnostics; it starts no training and opens no target data. Completed receipts
are reused after parent verification, and original job/resource receipts are
archived before the normal queue reuses them. It refuses a cell the live normal
queue has already reached. Inspect `controller/source-prefetch/` for its state.

After that later cell's source preparation is complete, its C01 fit and C11
prefix can also overlap the earlier cell:

```sh
"$study_python" -m division_reliability_v11.head_prefetch --source 44b6 --seed 314159
```

This uses the same ownership barrier and ordinary guarded fitting workers. It
stops at C11's durable midpoint, before mining. The main queue then reuses the
completed C01 fit and resumes the same C11 checkpoint. It refuses an active
worker or a cell the main queue has reached. Check measured aggregate memory
headroom before starting an additional fit; horizons and sampling stay locked.

Each upstream update is random full-source sampling. Durable state contains the
optimizer, learning-rate step, Python/NumPy/CPU/CUDA RNG, and sampler state.
Checkpoints occur every 250 updates or five minutes and at phase boundaries.
On restart, uncommitted history rows are archived before replay. Final weights
are retained only after the registered horizon. Fresh-process 10+10 versus 20
update proofs and compact mixed-objective replay are recorded in the results.
The actual host-restart recovery also compared 213 replayed production updates
with their preserved records. Samples, groups, learning rates, losses and
gradient norms matched exactly; `host_restart_resume.json` records the scope,
preserved hashes and conservative charge for the interrupted GPU lease.

After an upstream final is retained, validate its full sampling history, finite
losses and weights, final checkpoint equality, and optimizer charge accounting:

```sh
"$study_python" -m division_reliability_v11.upstream_final_audit --source 44b6 --seed 20260918
```

This distinguishes the retained trajectory from discarded updates replayed after
an interruption. All attempts remain in the resource journal. The audit's
immutable receipt is included in `validation.json` by `report`.

C11 pauses at its registered midpoint. Independent source-fit mining workers
read one immutable midpoint snapshot; their completed clip receipts can be
reused only with identical parents. The merged pool is consumed once on resume.
Source-calibration score caches likewise retain complete legal denominators and
verify their parent hashes. A fitted head cannot read calibration data.

After a real mining pass and resumed updates, inspect its full source pool:

```sh
"$study_python" -m division_reliability_v11.mining_audit --source 44b6 --seed 20260918
```

This verifies every selected negative group's sparse support, the unchanged
midpoint and bank hashes, all clip receipts, and recorded hard-slot draws and
their selection probabilities. It preserves the first observed resume snapshot;
reruns recheck the current artifacts while retaining that original timestamp.

After a compact final is retained, audit every recorded update and compare its
weights with the final resumable checkpoint:

```sh
"$study_python" -m division_reliability_v11.compact_final_audit --source 44b6 --seed 20260918
```

This checks the complete phase/group counts, finite losses and weights, final
learning rate, one mining pass, and optimizer/scheduler/sampler/RNG state. The
private immutable receipt is included in `validation.json` by `report`.

The CLI automatically sets `NVIDIA_TF32_OVERRIDE=0` before numerical imports for
C11 mining, calibration, prediction and cold inference. A trained midpoint
encoder exposed a TF32 singleton-reference discrepancy in the original parity
check. The source probes and complete C00 control in
`results/division-reliability-v11/compact_precision_repair.json` validate this
evaluation-only correction. Training and ordinary C00/C01 workers keep their
original settings. Do not export this override globally when resuming training.
C11 packages record the evaluation environment and entry-point hashes.

Target prediction stays locked until every registered cell is a retained package
or has an explicit scientific blocker. All retained target predictions freeze
together before target labels are opened. Missing arms remain null; a calibrated
disabled policy remains a separately labeled measured no-op. Cold validation
uses renamed complete clips, starts from images, and forbids the C00 cache.
The cold comparison requires all 100 raw-frame hashes, the explicit package and
dataset names, clean worker guards, output artifact hashes, graph equality and
actual CSV byte equality after replacing only the dataset column. A mismatch
leaves its comparison receipt and stops the queue before target scoring.
Up to three independent package workers run concurrently; each package's two
strata stay sequential. Selected images are copied before workers start. The
same GPU lease lock and aggregate resource guards remain active. Set
`cold_workers` to 1, 2 or 3 in `controller/options.json` before the cold stage
to match measured host headroom. This changes dispatch only; every selected
model/stratum still runs with the same input and equality requirements.

Useful checks and explicit stage entry points:

```sh
"$study_python" -m division_reliability_v11 validate
"$study_python" handover/division-reliability-v11/check_plan.py
"$study_python" -m division_reliability_v11 --help
```

The CLI's `--updates` option is for diagnostic overfit only. Production horizons
come from the immutable lock. Do not rerun diagnostic pilots over their existing
receipt directories, change the locked upstream core, or select a new recipe
from target outcomes. Preserve failed receipts when repairing an implementation
error with the same scientific recipe.

The runtime is `/kaggle/envs/cell-tracking-annotation-selection-v1`, because the
inspection-only Conda environment lacks the required training dependencies.
Dependency versions, the architecture source hashes, original references and
the pinned official scorer are recorded in the lock. This is local execution;
compatibility with a future Kaggle notebook runtime is not yet established.
