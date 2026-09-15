# OrganoidTracker2 and CELLECT association experiments

The locked recipe is `configs/other-trackers-20260914.json`. Both experiments
start from the same frozen Cellpose observations used in the HOCT reassessment,
on six complete 100-frame clips. Outputs and weights stay in ignored
`work/other-trackers-20260914`; concise evidence is in the matching `results/`
directory. No Biohub labels are used for fitting or inference.

## What is tested

- OrganoidTracker2's released link and division classifiers receive their actual
  patch inputs, at the model's physical resolution. The adapter reuses upstream
  normalization, candidate creation, asymmetric coordinate channels, Platt
  calibration, boundary costs and graph pruning. The DPCT dependency is absent;
  SciPy/HiGHS solves its binary flow objective exactly. It may omit observations.
  Organoid-specific depth pruning, oversegmentation correction and track cleanup
  are excluded from this experiment on existing Cellpose instances.
- CELLECT's released U-Net supplies 64-channel embeddings, sizes and division
  features sampled at the rounded Cellpose centers. Its temporal matcher sees
  the five nearest candidates in physical distance, represented in XY-pixel
  units. Released ranking and division gates produce links. Incoming ties are
  resolved deterministically to prevent merges. All observations remain.
  This is an association transfer test, not the native CELLECT detector pipeline.
  Frame 99 is duplicated as its own image context to preserve the whole clip.

Each family has a separate no-division control. Frozen external checkpoints
are used without local parameter selection. These reused public clips do not
measure performance on new embryos; the v3 comparison has known training overlap.

## Reproduction

Run from the repository root with `PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `POLARS_MAX_THREADS=2`.

OrganoidTracker2 uses `/kaggle/envs/detector-screen-organoid/bin/python`:

~~~text
python -m tools.other_trackers.validate organoid
python -m tools.other_trackers.organoid
~~~

The remaining stages use `/kaggle/envs/cell-tracking-notebooks/bin/python`:

~~~text
python -m tools.other_trackers.validate flow
python -m tools.other_trackers.validate cellect
python -m tools.other_trackers.cellect
python -m tools.other_trackers.decode organoid
python -m tools.other_trackers.decode cellect
python -m tools.other_trackers.evaluate
python -m tools.other_trackers.report
~~~

The inference processes can run concurrently. The shared GPU lock grants
ten-second leases, checked between batches, and releases model allocations
between leases. CPU preprocessing and decoding use two threads per process.
Incomplete frames are not considered complete; matching code/input receipts are
required when resuming. Preserve completed output namespaces when changing code.

`OTHER_TRACKERS_OUTPUT` optionally sets a separate absolute output root. This
run overlapped the last two OrganoidTracker clips in a private output root,
sharing only the frozen models and candidate banks through read-only use of
symlinks. `python -m tools.other_trackers.finish` merges only complete private
clips before the main worker reaches them, decodes completed clips, then runs
official evaluation and reporting. It never overwrites a partial main clip.
The scheduling receipt and per-frame code/model hashes document that change;
the numerical recipe is identical for both workers.

The two OrganoidTracker model folders contain `model.keras` and `settings.json`
from `models_all/models_all/model_links` and `model_divisions` inside the existing
`work/detector-screen-20260914/organoid/models/models_all.zip`. The archive is the
corrected release from Zenodo record 18479952. Both CELLECT checkpoint files are
already present under `/home/mpf/code/kaggle/CELLECT/model`. Original source
checkouts and downloaded archives are left unchanged.

The official metric runs only after all requested graphs have been hashed.
CSV exports are round-trip checked, and endpoint/candidate error categories
must reconcile exactly with official sparse-label FP/FN counts.
