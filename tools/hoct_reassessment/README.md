# HOCT reassessment

This study tests the pinned downloaded HOCT checkpoints before adding another
tracking family. Raw inputs and all earlier studies remain unchanged. Outputs
are in `work/hoct-reassessment-20260914`; concise receipts and scores are in
`results/hoct-reassessment-20260914`.

The locked recipe is `configs/hoct-reassessment-20260914.json`. It uses the same
six complete 100-frame clips as the Cellpose/ultrack experiments. These are
exploratory public clips, not independent biological validation or all 199 clips.

## Comparisons

- **Legacy physical/voxel controls:** identical v5 observations, candidates,
  source calibration, contextual windows and decoder, with paired CPU FP32
  inference. Convert only model position/diameter/inertia units. These arms
  retain the baseline's structural prior, protected edges and missing-feature
  fallback. Their score does not measure an independent tracker.
- **Standalone Cellpose + HOCT:** native voxel features from the real unchanged
  Cellpose masks, physical nearest-neighbor candidates, upstream tiled image
  normalization/features and parent-versus-orphan probabilities. Both
  `general_v1` and `ctc_v0` run without Biohub fitting, baseline edges or native
  model features. Separate no-division controls measure the branching tradeoff.

The standalone decoder solves the upstream adjacent-frame flow objective by
exact assignment. At default node cost -10 every observation is selected: even
an isolated node costs at most -10 + .5 + .25. With adjacent edges, birth,
termination and division costs separate by transition. Two source slots encode
the first child and optional second child. The reduction is verified against
exhaustive legal graphs and six actual upstream SCIP solves. It is not a
general replacement for HOCT's solver with missing observations or gap edges.

## Runtime

Use `/kaggle/envs/cell-tracking-notebooks/bin/python` and set
`PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, `POLARS_MAX_THREADS=2`.

HOCT source is reused from the verified v5 inference bundle; weights stay in
the old model directory. The missing optional `spatial-graph==0.1.1` dependency
was installed only into `work/hoct-reassessment-20260914/python` using the named
notebook runtime. Shared environments were not modified.

For standalone inference, `--execution cooperative_cuda --workers 1` uses the
existing detector/training GPU lock. Leases release after ten elapsed seconds,
checked between batches. Inputs/outputs remain CPU tensors; the same FP32 model
runs on the GPU with TF32 and autocast disabled. CPU/GPU real-batch validation
and the scheduling adjustment are recorded separately. Original CPU results
and the timing probe remain available. Other experiments are not stopped.

## Entry points

Run these modules from the repository root with the runtime above:

~~~text
python -m tools.hoct_reassessment.validate
python -m tools.hoct_reassessment.legacy score --workers 3
python -m tools.hoct_reassessment.legacy decode --workers 3
python -m tools.hoct_reassessment.cellpose prepare --workers 2
python -m tools.hoct_reassessment.runtime_check
python -m tools.hoct_reassessment.cellpose score --execution cooperative_cuda --workers 1
python -m tools.hoct_reassessment.decode --execution cooperative_cuda
python -m tools.hoct_reassessment.evaluate --arms v3 previous_H_general_J legacy_physical legacy_voxel
python -m tools.hoct_reassessment.evaluate --arms v3 ultrack_no_division cellpose_general_v1_cuda cellpose_general_v1_no_division_cuda cellpose_ctc_v0_cuda cellpose_ctc_v0_no_division_cuda
~~~

The subsequent single source-adaptation recipe is locked separately in
`configs/hoct-source-adaptation-20260914.json`. Run it after the frozen-checkpoint
evaluation has created the source-only node matches:

~~~text
python -m tools.hoct_reassessment.embeddings
python -m tools.hoct_reassessment.adapt fit --source 44b6
python -m tools.hoct_reassessment.adapt fit --source 6bba
python -m tools.hoct_reassessment.adapt predict
python -m tools.hoct_reassessment.decode --execution cooperative_cuda --models source_residual
python -m tools.hoct_reassessment.evaluate --arms cellpose_source_residual_cuda cellpose_source_residual_no_division_cuda
python -m tools.hoct_reassessment.diagnose
python -m tools.hoct_reassessment.report
~~~

The two source fits may run concurrently in separate processes. Each process
rejects files from the opposite embryo and refuses to overwrite a saved model.
Prediction freezes both model hashes before applying each model to the other
embryo. Training uses supported parent choices and their competing incoming
links; it does not label unannotated cells or unrecorded daughters as negatives.

Prediction commands do not import evaluation code. The standalone worker's
audit hook blocks annotation files, prior evaluation artifacts, native weights,
baseline graphs and legacy observation banks. It uses only Cellpose images,
masks and HOCT checkpoints. Complete graphs are hashed before official matching.
CSV exports are round-trip checked; summaries use exact official aggregation.

Caches intentionally reject changed input/code receipts. Preserve a completed
run and use a new namespace for a changed experiment. Reproduction at a later
code revision should regenerate stages in a fresh output directory rather than
rewriting historical receipts.
Set `HOCT_REASSESSMENT_OUTPUT` and `HOCT_REASSESSMENT_RESULTS` to fresh absolute
directories for the heavy outputs and concise receipts respectively. Install
the isolated optional dependency under the new output directory's `python/`
subdirectory as described above. The report's document and canvas destinations
are specific to this study; adjust them before publishing a separate study.

The incumbent has verified training/assessment overlap. HOCT's release registry
and release notes do not give checkpoint-linked training lists; its exact
pretraining exposure is unresolved. The source-only residual adaptation reports
both directions between embryos, but uses the same six public development
clips. It is not a held-out estimate for new embryos.
