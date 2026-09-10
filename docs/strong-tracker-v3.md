# Strong tracker v3: execution and reproduction

V300–V370 builds on the completed v2 incumbent, `bypass_motion_bounds`, whose
independently verified score is **0.9342063149703403**. Read the
[measured report](../results/strong-tracker-v3/final_report.md),
[offline dashboard](../results/strong-tracker-v3/dashboard.html), and
[execution status](../handover/strong-tracker-v3/STATUS.json) for the completed
results. The original experiment specification remains in
[EXPERIMENTS.md](../handover/strong-tracker-v3/EXPERIMENTS.md).

All comparisons use the same 199 clips, fixed count estimates, fresh official node
matching and division assignment, and the same pinned scorer revision
`075fc5f5a52d11077f9dc2b074644618f26939e2`. This is operational exploratory evidence:
the two embryos have been reused and the public checkpoints have upstream
contamination. The opposite-embryo directions are preserved, but are not clean
out-of-fold validation. Unknown overlap does not support independent biological
replicates, inner folds, bootstrap intervals, or hidden-test claims.

## Runtime and immutable inputs

The executed runtime is Python 3.12.14, PyTorch 2.8.0+cu128 and one RTX 4090.
Use the existing study interpreter; do not install into system Python:

```bash
scripts/run_strong_tracker_v3.sh \
  --python /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python --help
```

The wrapper resolves its checkout from its own path, works from a foreign current
directory, sets `PYTHONNOUSERSITE=1`, restricts BLAS/OpenMP worker threads, and
removes the incompatible injected `/usr/local/cuda/lib64` library path. It does
not require the host-specific `root_remote_env.sh` script. An explicit interpreter
or `STRONG_TRACKER_PYTHON` overrides the default. CUDA-required stages fail when
CUDA is unavailable.

The wrapper also restricts execution to a shared set of at most 32 allowed CPUs.
CUDA and Zarr can create many idle support threads; this affinity bound limits
aggregate CPU execution as well as the configured compute-pool thread limits.

The CLI exposes explicit `--repo`, `--data`, `--v1`, `--v2`, `--out`, `--work` and
`--official` paths. The executed checkout must match `--repo`. Keep the scorer at
that checkout's `work/annotation-selection-v1/official`: the reused metric adapter
imports that location, and arbitrary external scorer relocation is not supported.
The wrapper resolves a moved checkout automatically. Defaults are recorded in the
artifact manifest. The canonical inputs are:

- Competition images: `/kaggle/input/competitions/biohub-cell-tracking-during-development`.
- Completed v1 artifacts: `/kaggle/working/cell-tracking/annotation-selection-v1`.
- Completed v2 artifacts: `/kaggle/working/cell-tracking/strong-tracker-v2`.
- Official metric checkout: `work/annotation-selection-v1/official`.
- New outputs: `/kaggle/working/cell-tracking/strong-tracker-v3`.
- New temporary work: `work/strong-tracker-v3`.

Do not run the old study drivers or write into their directories. V3 uses an
immutable `RunContext`, explicit graph inputs, and parity-tested pure adapters.
Label-free `inputs.json` is separate from the evaluation inventory. Graph, model,
feature, code and image-content fingerprints reject incompatible resumptions.
The complete image-content manifest hashes all 20,298 files in the 199 image
groups. Final preservation checks cover sealed v1/v2 file sizes and modification
times plus the exact selected-v2 hashes.

## Stage entry points

These are reproduction commands, not an instruction to restart completed work.
Inspect the corresponding receipts before resuming; preserve prior locks and
failed controls. A changed method needs a new namespace and a recorded decision.

```bash
scripts/run_strong_tracker_v3.sh incumbent
scripts/run_strong_tracker_v3.sh evaluate --variant incumbent,historical_v1 --round V300
scripts/run_strong_tracker_v3.sh census --workers 4
scripts/run_strong_tracker_v3.sh replay --workers 4
scripts/run_strong_tracker_v3.sh association --workers 4
scripts/run_strong_tracker_v3.sh events --variant prepare --workers 4
scripts/run_strong_tracker_v3.sh events --variant fit --steps 10000
scripts/run_strong_tracker_v3.sh events --variant probabilities --workers 2
scripts/run_strong_tracker_v3.sh events --variant profile-decode
scripts/run_strong_tracker_v3.sh event-decode --variant verify
scripts/run_strong_tracker_v3.sh event-decode --workers 4
scripts/run_strong_tracker_v3.sh rescue --workers 2
scripts/run_strong_tracker_v3.sh combinations --workers 2
scripts/run_strong_tracker_v3.sh preserve
scripts/run_strong_tracker_v3.sh report
```

The corrected image rescue is implemented separately in `rescue_fixed.py`; the
original `rescue.py` and its measured failed control remain unchanged. The heatmap
control has its own `heatmap_rescue.py` entry. Full image-byte hashing,
source-oracle scoring, aggregate publication and dashboard validation have
separate modules to keep evaluation evidence outside prediction.

The executed event decoder uses `event_decode_cached.py` to read only the event
arrays it consumes from immutable NPZ files. Its separate fingerprint and all-ten
dense-clip byte/ledger/solver-count parity proof preserve the original frozen
decoder and model files. The original eager-loading `events --variant decode`
entry remains available and produces the same graphs with more memory use.

Prediction generation and comparative evaluation are separate. After all expected
graphs and both source directions exist, score a frozen bounded round with:

```bash
scripts/run_strong_tracker_v3.sh evaluate \
  --variant VARIANT_A,VARIANT_B --round UNIQUE_FROZEN_ROUND --workers 2
```

The scorer requires exact 199-clip coverage for each variant and freshly evaluates
each complete graph. Source oracle graphs live under `evaluation/`, are explicitly
nondeployable diagnostics, and cannot become selected inference. No new setting
may be silently substituted for an already measured variant.

The final local gate requires a positive pooled delta versus v2 and nonnegative
delta in each embryo within `1e-10`. The invalid plateau-rescue control and
canonicalized teacher diagnostics are ineligible. A small gain is below `0.002`;
the original `+0.02` target is a stretch goal. Standalone gains are never summed to
infer a combination score.

## Selected inference package

The local `inference_package/` contains the selected policy, code, repair models
and dependency hashes. Prediction imports install an annotation-denial audit hook
before loading graph/image/model dependencies. The package does not bundle
annotations, evaluation count estimates, cached graph predictions, raw images or
upstream checkpoint weights.

`selected_inference_package.zip` archives that directory for transfer. Its size,
SHA-256 and dependency scope are recorded in the sanitized `inference_archive.json`;
the archive and repair weights remain local, outside Git.

```bash
/path/to/strong-tracker-v3/inference_package/run.sh \
  --python /path/to/CUDA-python \
  --images /path/to/image_directory \
  --output /path/to/new_prediction_directory \
  --v1 /path/to/annotation-selection-v1 \
  --v2 /path/to/strong-tracker-v2 \
  --source-model 44b6
```

`--source-model` is explicit (`44b6` or `6bba`); an unknown embryo does not select a
policy from its filename. The measured experiment uses the same policy in both
directions with the opposite source fit. Choosing a deployment source model for a
new embryo is a declared deployment input, not a measured generalization result.
Provide pinned original tracking code, primary/secondary checkpoints, DeepCenter
checkpoint and the frozen E teacher models using the dependency manifest.

The output is a validated `submission.csv` and per-clip graphs. Coordinates must
be integer and in bounds, node and edge IDs unique, links consecutive in time,
indegree at most one and outdegree at most two. The local CSV is not automatically
uploaded. Fresh-pipeline receipts distinguish actual neural inference, repairs,
teacher reconstruction, image-feature work, CPU decoding and comparison-only
reads. RTX 4090 timings do not establish Kaggle accelerator performance.

## Validation and transfer to another host

Tests under `tests/strong_tracker_v3/` cover explicit-input parity, graph ownership
and complete edit constraints, official event timing and source masking, daughter
symmetry, and the flat-background rescue counterexample. The original handover
fixtures and applicable v1/v2/upstream scorer tests remain available. Dashboard
validation checks actual aggregate/clip coverage, filtering, mobile layout and
zero network requests in headless Chromium.

Final publication is gated on the completed 32 × 199 score/match set, the exact
selected export and fresh reproduction, and the copied-package image pilots.
The evaluation-only export attestation contains graph hashes, with score/count
tables excluded from guarded inference. `candidate_manifest.json` and
`round_lock.json` index the frozen inputs. The local `edit_ledger.parquet` contains
prediction actions; `evaluation/edit_effects.parquet` and whole-stage regret
tables keep annotation evidence separate. Per-action division or combined-score
credit is not inferred from edge bookkeeping.

After the required receipts are complete, the publication entry points are:

```bash
scripts/run_strong_tracker_v3.sh ledger-delivery
scripts/run_strong_tracker_v3.sh preserve
scripts/run_strong_tracker_v3.sh report
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /root/.conda/envs/cell-tracking/bin/python -m strong_tracker_v3.dashboard_check
scripts/run_strong_tracker_v3.sh validation
scripts/run_strong_tracker_v3.sh delivery
scripts/run_strong_tracker_v3.sh artifacts
```

The browser check uses the existing project interpreter with Playwright. Supply
an equivalent available interpreter on another host. Delivery verifies the HTML
hash, package contents/policy, copied-package smoke and validation code hashes;
the final artifact pass indexes the resulting public files.

Read [NEXT_AGENT.md](../handover/strong-tracker-v3/NEXT_AGENT.md) and the published
artifact manifest before moving or resuming this study. Preserve the whole local
v3 artifact root, original images, v1/v2 dependencies, source/weight pins and the
working environment. Raw tensors, predictions, detailed annotation-derived
evidence and weights stay outside Git. Sanitized aggregates and code are committed.
The instance disk is not an external backup; no upload or persistence is implied.
