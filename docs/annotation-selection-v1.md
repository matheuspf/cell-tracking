# Local annotation-selection execution

This is an additive implementation of the [v1 protocol](../handover/annotation-selection-v1/PROTOCOL.md).
Outputs are local and ignored at `/kaggle/working/cell-tracking/annotation-selection-v1/`;
source snapshots/logs are under `work/annotation-selection-v1/`. The final output
report and dashboard contain measured results, not the original discussion's numbers.

The study uses fixed classical candidates and source-only selectors in both embryo
directions. Exact image patch matches demonstrate clip overlaps; missing global
crop/time origins prevent certifying independent inner groups. The recorded protocol
fallback uses fixed settings and makes the primary rule explicitly **not an inner-CV
selection**. Conservative whole-embryo supergroups make bootstrap confidence intervals
unavailable. Checkpoint provenance labels the separate public notebook lane contaminated.

## Environment and pinned source

The original inspection and notebook environments are preserved. An isolated venv
inherits the existing notebook packages and adds scikit-learn/pytest. Installed
package versions and source hashes are captured in `environment.json` and the local
artifact manifest. To reconstruct on a machine with the documented notebook runtime:

```sh
export PYTHONNOUSERSITE=1
/kaggle/envs/cell-tracking-notebooks/bin/python -m venv --system-site-packages \
  /kaggle/envs/cell-tracking-annotation-selection-v1
/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m pip install \
  scikit-learn==1.9.0 pytest==9.1.1
git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git \
  work/annotation-selection-v1/official
git -C work/annotation-selection-v1/official checkout --detach \
  075fc5f5a52d11077f9dc2b074644618f26939e2
```

Use those setup commands only if those study paths do not already exist. No reset,
clean, bootstrap, full competition download, or modification of notebook originals
is part of this workflow. The full data is verified using the existing
`python tools/download_data.py --verify-only --checksums` path.

The official scorer is imported directly from its pinned source checkout. Its
package metadata requests PyTorch >=2.9; this local run preserved the working
2.8.0+cu128 notebook stack and validated the actual scoring paths with 102 upstream
fixtures. This is a recorded dependency exception, not a fully resolved upstream
package installation.

## Commands

From any directory, the absolute path to `scripts/run_annotation_selection.sh`
resolves the repository and runtime. From the repository root:

```sh
bash scripts/run_annotation_selection.sh resources > work/annotation-selection-v1/resource-monitor.log 2>&1 &
bash scripts/run_annotation_selection.sh environment inventory candidates splits audit labels train image freeze infer
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection infer \
  --output /kaggle/working/cell-tracking/annotation-selection-v1/predictions_gt_unavailable
bash scripts/run_annotation_selection.sh lock
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection evaluate --workers 16
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection public --limit 2
bash scripts/run_annotation_selection.sh public public-prepare
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection public-evaluate --workers 16
bash scripts/run_annotation_selection.sh export matched-controls audit-viewer report validate finalize
```

These are the fresh-run stage order. For an existing run, resume its incomplete
stage from `status.json`; do not replay metadata/split creation or immutable lock
creation over a completed study. The CLI records commands, and per-sample scoring
resumes with input fingerprints. This execution began with eight clean scoring
workers and added eight workers using the same function and frozen inputs; a
single 16-worker pool provides the same experiment. Worker count affects runtime,
not the filters or score definition.

`candidates --limit 2` is an image-only runtime pilot; it is never full OOF.
`public --limit 2` isolates the original Harmonic Fusion pilot. The full public run
uses all 199 training images through an image-only symlink view. It relocates output
paths, excludes the unused GT-reading validator, and adds pre-ILP coordinate,
detection-probability and edge-probability exports. Original settings are retained.
The original no-export pilot and full export-hook graphs must agree exactly.
The `public-prepare` stage checks that comparison before freezing diagnostic
predictions. Completed public inference is reused only after source/adapter checks.

In the measured run, the notebook's neural inference finished first and its serial
graph-repair stage became the bottleneck. `public-repair-pilot` reused the frozen
neural GEFF graphs with the original repair functions and matched both original
pilot graphs exactly. The owned serial process was then stopped, with its partial
CSV/logs preserved, and `public-repair` scheduled four isolated clip shards. All
27 repair functions are checked byte-for-byte against the original adapter. Only
paths, the input clip list and reuse of existing neural graphs/retention diagnostics
change; CSV row IDs are renumbered when merging. The completion receipt verifies
all 199 datasets, unchanged raw neural graphs and final pilot parity. This is an
execution optimization after model lock, not a different tracker configuration.
The fresh-run commands above can also use the original serial path throughout.

The original public export has six z=64 centers in five clips. A strict spatial
check caught this before public scoring. The diagnostic lane preserves those
coordinates and scores the original graphs; the pinned metric has no image-shape
bound. Only image sampling uses half-sample reflection, consistent with the patch
sampler, while geometry uses original centers. Time, IDs, edges and fork checks
remain strict. This exception and the failed preparation log are recorded in
`public_coordinate_audit.json`; the clean lane retains strict spatial validation.

Stages persist per-sample graph/feature files, source models and lock receipts.
Repeated candidate generation verifies its configuration/artifact hash before
reusing an output. Matching and evaluation caches validate their source fingerprints.
Interrupted tabular fits repeat the incomplete source direction; complete directions
are retained. An early implementation performance bug in the synthetic control
loop was stopped and fixed before the models were locked; its log was preserved.

Do not change configuration in an existing locked run. A changed experiment or a
post-revelation follow-up requires a new output/study version. Split/model/prediction
locks are immutable; reports may be regenerated from completed measured artifacts.
Raw image/GT caches, predictions, checkpoints and matching tables stay out of Git.
The optional census and candidate-audit stages preserve existing label CSVs.
`audit-viewer` exports two offline z-stack viewers with blank manual assessments;
it does not infer biological truth from detector or annotation labels.

## Validation

```sh
PYTHONNOUSERSITE=1 python -m unittest discover \
  -s handover/annotation-selection-v1 -p 'test_*.py' -v
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m pytest -q tests
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/work/annotation-selection-v1/official/src" \
  POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m pytest -q \
  work/annotation-selection-v1/official/tests/test_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_metrics.py \
  work/annotation-selection-v1/official/tests/test_division_sandbox_examples.py
```

The integration tests exercise one-to-one/time/anisotropic matching, singleton and
empty graphs, fresh division counts, immutable identity filtering, graph validity,
strict sample completeness and estimate validation, official aggregation parity,
prepared-policy equivalence, source/outer disjointness, GT-file access rejection,
and CLI imports from a different working directory.

`validate` runs the helper, study and inspection checks, reuses verified upstream
pass logs when the pinned source is unchanged, and launches offline Chromium with
the existing inspection environment's Playwright installation. Activate the
`cell-tracking` inspection environment on `PATH` before running that stage. The
browser checks 36 lane/embryo/model/policy combinations, the high-retention view,
embedded-data export, mobile layout and both audit viewers. `finalize` requires
those receipts, closes resource monitoring, verifies complete sample sets, copies
execution logs/source, writes the artifact manifest and sanitized aggregate results,
and fills the handover status. It performs no Git commit, push or submission itself.

The actual local commit is made only after reviewing the staged file list. It
contains additive code/tests/docs and aggregate results, with all raw predictions,
patches, checkpoint weights and detailed matching tables left in ignored storage.

The report's source links are local. No submission, publication, push, forum post,
paid API call or hardware rental is part of execution.
