Cellpose refinement v1

This implements the first experiment in the
[reviewed detector plan](../../docs/cellpose-detector-plan-20260914.md): a small
residual head on frozen CPDINO features and native image patches. It preserves
the original Cellpose candidate count, confidence ranking and candidate order.
Every emitted integer center is within 3 µm of its rounded original center.

The encoder checkpoint is the exact benchmarked `cpdino-vitb`, SHA256
`3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f`.
Its inherited training population remains unverified. The new heads are
source-isolated adaptations, not certified leakage-free models.

The [configuration](../../configs/cellpose-refine-v1.json) fixes the following:

- Four retained fits: source 44b6 → target 6bba and the reverse, each with seeds
  20260914 and 314159. Both embryos have been examined in prior research.
- Source GEFF point supervision at times 9, 29, 49, 69 and 89 in every source
  clip: 1,002 GT observations from 71 clips for 44b6, and 5,770 from 128 clips
  for 6bba. Each produces a zero-offset query and three deterministic jitters
  within 2.5 µm, reduced near other annotated centers.
- Natural source proposals come from the existing 206-frame-per-embryo
  Cellpose panel. Mutual nearest correspondence within 5 µm and an observed
  competing-GT margin of at least 2 µm produces 595 supported natural queries
  for 44b6 and 1,673 for 6bba. Other real proposals are unknown and receive no
  supervision. These observations overlap some jitter GT; the counts are not
  independent biological samples or a disjoint unique-label total.
- The head has 398,755 trainable parameters. It consumes three 768-D frozen
  token features, a trilinearly sampled native `7×25×25` intensity patch, and
  the query's physical displacement from its nearest native voxel. It receives
  no embryo identity, global anatomical position, track ID or target labels.
- Frozen Cellpose features use the exact full-volume normalization and
  anisotropy-4 resize recipe. The half-pixel resize origin and stride-8 token
  receptive-field centers are explicit. Token values entering `CPDINO.out`
  are used; its random style output is ignored.
- AdamW, learning rate 3e-4, weight decay 1e-3, 100 warmup updates, cosine
  decay to 5% of the initial rate, batch 128 and exactly 3,000 updates.
  Half of each batch samples natural queries and half samples jitter/center
  queries. Their loss weights are 2 / 1 / 0.25 respectively. The objective is
  Smooth-L1 physical offset error with beta 1 µm; targets and model residuals
  are bounded at 3 µm. The final output layer starts at zero.

The raw graph files are read to prepare source-side labels; only the selected
timepoints and supported natural proposals contribute to training. Each fit
loads only its own embryo's immutable feature/target files. Python audit hooks
reject reads of the other embryo's training data and block annotation access
in inference. These hooks are not a kernel sandbox. Full-volume intensity
normalization is a fixed per-image operation, not preprocessing fitted across
source and target datasets.

Data and feature hashes are frozen before extraction; code/configuration hashes
are frozen before fitting. All four final checkpoints and all corresponding
predictions must exist and pass hashes before evaluation opens GT. No target
score selects a checkpoint. Later target-informed recipe changes are separate
exploratory iterations.

Runtime and commands

Use the existing isolated Cellpose runtime; do not change the system Python.
From the repository root:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.preflight
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.prepare
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.audit_inputs
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.extract
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.queue
```

Extraction and the queue can run concurrently: the queue waits for each source's
complete feature cohort. They share the existing detector-screen GPU lock with
other local work. Extraction releases it between frames. The feature cache
contains pooled query vectors and small patches rather than dense triplane
token volumes. Inputs remain unchanged; large caches/checkpoints live in ignored
`work/cellpose-refine-v1/`. Compact receipts/results live in
`results/cellpose-refine-v1/`.

Resume by rerunning the relevant command. Complete feature caches are hash
checked, training resumes from optimizer/RNG checkpoints, and completed fits
are verified before reuse. A configuration or locked-source mismatch fails
rather than silently mixing recipes. Failed-process logs remain in the work
directory. Evaluation uses the already verified notebook metric environment.

For annotation-free raw-volume inference, including fresh Cellpose proposals:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.cellpose_refine.infer --volume /path/to/native-volume.npy --head work/cellpose-refine-v1/models/source-44b6-seed-20260914/final.pt --output work/cellpose-refine-v1/replay/prediction.npz
```

Use the model trained on the opposite embryo for local held-out evaluation.
`centers_zyx` is the bounded integer export, `float_centers_zyx` is the continuous
diagnostic, and `scores` preserves Cellpose's original confidence. A raw volume
must have the verified native `64×256×256` shape and spacing
`(1.625, .40625, .40625)` µm. This local runtime is not yet a packaged Kaggle
notebook.

The retained-model replay check selects the first pilot volume per embryo from
the frozen manifest, copies each to a generic filename, runs the complete raw
detector with the opposite-embryo primary-seed head, and checks exact feature
and prediction equality against the locked caches. It can wait for the queue:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m tools.validate_cellpose_refine --wait
```

This checks two complete 3D volumes; it does not certify whole-clip execution or
the Kaggle notebook runtime. The selected frames and test script hash are
recorded before evaluation.

Verification and interpretation

The GPU preflight checks exact feature repetition, unchanged encoder weights,
exact reconstruction of upstream logits from the feature tap, an actual head
backward/optimizer step, zero initialization and bounded outputs. Coordinate
ramps check interpolation origins and token sampling. CPU contracts check
unknown-region gradients, ambiguity rejection, jitter replay, integer rounding
bounds, recursive exposure checks and read guards:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /kaggle/envs/detector-screen-cellpose/bin/python -m unittest discover -s tests -p test_cellpose_refine.py -v
```

Evaluation reuses the 400-frame assessment and official one-to-one matching at
1–7 µm. It reports integer and float results, absolute K=100/200/400 budgets,
censored localization error, shared-GT paired error, close-pair recovery and
sampled edge-endpoint availability. Counts and scores must remain identical to
the frozen baseline. Promotion requires tighter localization and continuous
error gains with no greater than 0.2 percentage point R7 loss on either embryo
at the declared budgets, for both seeds. This is an engineering criterion on
two reused embryos, not a statistical generalization guarantee.

Evaluation also reports signed Z/Y/X errors and recall by faintness, proximity
to the crop boundary, local Z position and annotated division neighborhoods.
These definitions are fixed in `diagnostics.py` before fitting. Local crop Z
is not acquisition depth, and a sparse annotated division family is not a
complete mitosis inventory. Strata overlap; matched-only errors have their own
conditional denominator. Target strata are computed only after predictions
are locked.

The head combines frozen semantic features with a new image branch. A gain by
this combined model does not by itself establish which branch caused it.
Unfreezing the encoder is gated on a useful first result. Discovering additional
cells needs a separate query source and justified negative supervision. No
temporal assignment, graph score, tracker promotion or submission is part of
this first experiment.
