# Pipeline error training

Isolated implementation of `handover/pipeline-error-training-20260915`. The
historical P0, C4_m6 and C0 algorithms and output files are inputs. No production
policy is changed by this package.

## Runtime

Use the existing isolated runtime and suppress user site packages:

```sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
```

`results/pipeline-error-training-20260915/environment.json` records the measured
package versions. The Organoid adapter loads its existing Keras environment and
the pinned corrected v3 artifact explicitly. It does not download weights.

Large files belong exclusively to `work/pipeline-error-training-20260915/`.
Prediction entries install annotation/cache/network denial before numerical
imports. Operational predictions receive exact hashed P0 graphs and native
evidence; fresh inference receives only images, models and its current run.

## Execution order

The exact prepared manifests and common update lock are resumable. Do not rerun
preparation or revise the source split/model recipe after a target freeze.

```sh
$STUDY_PY -m pipeline_error_training queue
$STUDY_PY -m pipeline_error_training.observation_queue
$STUDY_PY -m pipeline_error_training.finish_queue
```

`finish_queue --wait-for-primary` can wait in a separate process while the two
primary queues execute. An incomplete primary lane must be repaired or have a
concrete disposition in `results/pipeline-error-training-20260915/lane_blockers.json`
before source nomination. Resuming after the target freeze verifies its original
model and selection hashes and preserves the frozen recipes.

The independent primary queues finish both directions, calibrate on source groups
and run the fixed complete source clips. `finish_queue` runs source qualification,
conditional controls and replication before freezing the entire target matrix.
It then predicts, scores and exports full metrics. It never replaces a failed
direction with P0 or picks another seed. Inspect and repair an implementation
failure before retrying that stage; preserve the original failed artifacts.

For a single resumable fit and its source calibration:

```sh
$STUDY_PY -m pipeline_error_training train --arm D20_temporal --source 44b6
$STUDY_PY -m pipeline_error_training calibrate --arm D20_temporal --source 44b6
```

The cooperative GPU lease is released between optimizer updates. Process/device
memory, disk and cumulative lease time are monitored. The budget reserves 4/24/12/8
GPU lease hours for pilots, first seed, replication and final work, respectively.
Other programs' device memory contributes to admission; their GPU time is not
charged. A failed resource admission is a blocker, not permission to disrupt them.

## Validation and results

```sh
$STUDY_PY -m pytest tests/pipeline_error_training -q
$STUDY_PY -m pipeline_error_training.native_validation
$STUDY_PY -m pipeline_error_training.fresh_validation
$STUDY_PY -m pipeline_error_training.validation --tests
$STUDY_PY -m pipeline_error_training.report
```

Fresh/native validation also needs the cooperative lease; run it at a queue
boundary using `maintenance/requests.json` while training is active. The training
CLI and explicit prediction-job entries expose `--help`; source-owning scripts
require explicit directions.

Full target metrics, both embryos, changed-error identities, source screens,
training hashes and resource evidence are exported under the dated results root.
Missing values are null/blank with reasons. `recommendation.json` is separate
from the adopted P0 default. No clean end-to-end OOF or leaderboard claim follows
from a source-only new head on these exposed upstream proposals.
