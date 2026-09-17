# Division generalization v2

This isolated study implements `handover/division-generalization-v2/`. It reads
the pinned P0/C4_m6 artifacts and never changes a production default or writes to
the preceding study. Direct supervision, inner calibration and mining are source
only; inherited P0 checkpoint/observation exposure remains explicit.

Use the existing isolated runtime (no installations):

```sh
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
$STUDY_PY -m division_generalization_v2 queue
$STUDY_PY -m division_generalization_v2 report
```

The work root is `work/division-generalization-v2`, backed on this machine by
`/kaggle/working/cell-tracking/division-generalization-v2`. Code and small receipts
are committed; graphs, raw scenes, weights, optimizer/RNG states and logs are not.

Stages can be resumed individually using `baselines`, `prepare --source 44b6`,
`validate`, `profile --source 44b6`, `train --source 44b6 --arm prefix --seed 20260916`,
`screen --source 44b6 --arm J_uniform --seed 20260916 --stop-at 4096`, and `report`.
Inspect actual processes before starting a queue; it holds its own worker lock
and cooperates with the inherited GPU lock. A `--stop-at` training boundary is an
interruption of the immutable fit budget, never a smaller scientific experiment.

The main model uses stationary t−3…t+3 native 16×64×64 scenes at two physical
scales, shared 16/32/64-channel convolutions, spatial query sampling and two
128-wide four-head temporal attention layers. Complete-action heads consume
native evidence as features. All output differences initialize to literal zero.
G30 uses only frozen native image/association/geometry features.

Source preparation uses a fixed seed Bernoulli 1/16 sample of prediction-only
anchors, independently of annotation component IDs. Separate positive exposure
does not estimate prevalence. Complete-action labels record exact official count
deltas with fixed node matches, local topology updates and global one-to-one fork
assignment; independent full-graph replays check this implementation. Unknown
alternatives are retained in archives but masked from ranking and risk losses.

Training uses 8 positive, 16 representative supported and 8 confuser groups,
32 effective groups per update, AdamW 3e−4/1e−4, clipping 1, 128 warmup updates,
plateau and cosine to 0.1 at update 4096. Every main branch starts from its own
source/seed's exact 2048-update model/optimizer/RNG checkpoint and performs 2048
further joint updates. Mining refreshes at 2048 and 3072 retain the random stream.
Raw scenes may be cached. Trainable encoder representations are never reused
across optimizer updates. See the execution lock for source-only selection and
the single conditional 8192-update continuation.

Correctness checks:

```sh
$STUDY_PY -m pytest -q tests/division_generalization_v2 \
  tests/pipeline_error_training/test_contracts.py \
  tests/pipeline_error_training/test_scientific_contracts.py \
  tests/pipeline_error_training/test_guards_and_observations.py
```

Full source screens precede the target freeze. Target inference workers install
annotation/cache/network denial before numerical imports and receive safe
prediction-only job descriptions. Both directions and seeds are predicted before
new target metrics are opened. Fresh proof reconstructs original P0 from renamed
complete image clips and compares exact persisted candidate graphs and CSV/GEFF
round trips. These local 4090 timings do not guarantee the Kaggle runtime.
