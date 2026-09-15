Detector screen, 2026-09-14

Completed findings: [benchmark report](../../docs/detector-benchmark-20260914.md).

This experiment tests pretrained cell-center proposals before temporal linking.
The primary metric is annotated-node recovery under the competition's 7 µm,
one-to-one distance matching. Secondary measurements cover 5/6 µm margins,
both GT edge endpoints being detected, candidate counts, and confidence-ranked
budgets relative to the incumbent. Sparse labels do not provide conventional
false-positive counts.

The frozen panel contains 12 pilot frames from six clips and 400 assessment
frames from 40 other clips: 20 clips per embryo, five adjacent frame pairs per
clip. Assessment has 2,383 GT nodes and 1,159 eligible GT edges. Clip selection
uses evenly spaced ranks of total annotation count, not model outcomes.
The [checkpoint-linked audit](../../docs/incumbent-provenance-20260914.md) verifies
that the incumbent secondary manifest includes every assessment clip in training.
The incumbent has training exposure to both embryos, with incomplete
checkpoint-level lineage. Treat it as potentially contaminated. This is a
pretrained-transfer comparison on available competition images, with no claim
of independent biological validation.

Raw inputs remain unchanged. Images, model assets, realized predictions, logs,
and detailed receipts are under `work/detector-screen-20260914/` (ignored).
Compact derived evidence is under `results/detector-screen-20260914/`.

The subsequent [Cellpose development review](../../docs/cellpose-detector-plan-20260914.md)
adds 1–7 µm curves, absolute candidate budgets, sparse annotated-pair diagnostics,
and optimistic bounded-refinement capacity from the same saved assessment outputs.
It does not train models or infer temporal links. Reproduce that CPU analysis with:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.analyze_headroom
```

Its output is `results/detector-development-20260914/localization-headroom.json`.
Detailed GT correspondences stay in ignored `work/detector-development-20260914/`.
The geometric capacity is a maximum-cardinality relaxation, distinct from the
official weighted matcher. Endpoint availability and displacement errors are
detector diagnostics, not competition scores or an evaluated assignment method.

Repositories are downloaded under `/home/mpf/code/kaggle`: `CELLECT`,
`OrganoidTracker`, `NucVerse3D`, `KapoorLabs-VollSeg`, `stardist`, `spotiflow`,
`PAC-MAP`, `AnyStar`, and `cellpose`, plus the `dinov3` inference dependency.
Exact commits and checkpoint hashes are exported in the results' provenance
file. Downloading a repository does not imply every latest source file was used:
StarDist execution uses the pinned installed 0.9.2 package.

The eight model families and frozen adapters are:

- `cellect_adapter.py`: released two-frame 3D U-Net and intraframe grouping MLP;
  native image geometry, source logarithmic normalization and boundary handling.
- `organoid_adapter.py`: published position model, target spacing `(2,.32,.32)`
  µm, current/next frames, threshold 0.1. Geometry fixes preserve fractional
  voxel centers and give each tile core exclusive ownership. XY core 352 was
  selected from input geometry before evaluation.
- `nucverse_adapter.py`: generalized unscaled checkpoint, native grid,
  upstream preprocessing, 200-step reconstruction, geometric mask centroids.
  Persistent Keras execution avoids per-patch session clearing. The original
  unseeded Wiener preprocessing is retained; realized outputs are preserved.
- `spotiflow_adapter.py`: `synth_3d` checkpoint, 4× XY block-mean sampling,
  correct block-center offset, subpixel centers. Thresholds 0.3 and 0.05 are
  evaluated from one forward pass. This is a fluorescent-spot checkpoint
  transferred to nuclear-scale inputs.
- `stardist_adapter.py`: Xenopus nuclei checkpoint, documented anisotropy,
  16 overlapping neural tiles, default probability threshold. Both source
  polyhedron origins and geometric mask centroids are evaluated. Threshold 0.1
  is an additional pilot-only candidate-count stress test. An eight-tile
  optimization was rejected because it moved some selected centers.
- `pacmap_adapter.py`: SH-SY5Y spheroid fine-tuned seed-0 checkpoint, source
  physical spacing and patch normalization, 5 µm distance-aware peaks/merging.
  Scores are regression amplitudes rather than probabilities. Full-frame
  adaptation preserves border candidates and can expose seam proposals.
- `cellpose_adapter.py`: official `cpdino-vitb`, native 3D reconstruction from
  orthogonal 2D inference, anisotropy 4, default BF16/batch 8, geometric centroids.
  Larger batches preserved pilot masks but did not improve speed. A compiled
  execution probe changed the instance count and was rejected.
- `anystar_adapter.py`: official synthetic-only AnyStar-mix, isotropic grid
  preserving XY spacing, source probability 0.5/NMS 0.3, both source origins and
  geometric mask centroids. This method is tested on the 12-frame pilot.

The incumbent is the existing two-checkpoint TemporalUNet3D candidate ensemble.
Its pre-ILP candidate files are reused from
`/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/inputs/`.
Incumbent inference is not retimed. The existing FOCUS-3D pilot is included as
a reference; its saved score values are placeholders, so its budget truncation
must not be interpreted as confidence ranking.

From the repository root, prepare the image-only panel and incumbent cache with:

```sh
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.prepare
```

Preparation requires the canonical competition data, existing annotation-study
inventory, incumbent candidate cache, and previous pilot manifest. The committed
results panel records the selected clip/time keys; ignored assets must be
restored or regenerated before rerunning inference.

Use the existing isolated runtimes, without changing the base environment:

- CELLECT: `/kaggle/envs/cell-tracking-notebooks/bin/python`.
- OrganoidTracker: `/kaggle/envs/detector-screen-organoid/bin/python`.
- NucVerse, Xenopus and AnyStar: `/kaggle/envs/detector-screen-tf2162/bin/python`.
- Spotiflow: `/kaggle/envs/detector-screen-torch/bin/python`.
- PAC-MAP: `/kaggle/envs/detector-screen-pacmap/bin/python`.
- Cellpose: `/kaggle/envs/detector-screen-cellpose/bin/python`.

All adapters accept `--role pilot` or `--role assessment`. NucVerse assessment
also uses `--persistent-session`; Xenopus assessment uses `--default-only`;
OrganoidTracker uses `--minimum-free-gpu-gib 12`. PAC-MAP requires its selected
checkpoint via `--checkpoint`. AnyStar uses `--gpu-memory-mb 3500`. Exact commands
and package receipts are in each model's ignored work directory and exported
provenance.

Set `PYTHONNOUSERSITE=1` and numerical thread limits to four for inference.
The adapters share
`/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock`.
TensorFlow has explicit allocator caps; PyTorch adapters release allocations
between frames. CPU reconstruction should run outside the GPU lock. A shared
VRAM interruption was recovered by resuming completed artifacts, with no change
to learned weights or inference thresholds.

The original Cellpose process retained OpenCV's separate default thread pool;
CPU affinity was bounded after 197 assessment frames. The exact boundary and
loaded-source hash are recorded in `cellpose/cpu-limit-receipt.json`. Timings are
descriptive local measurements, not a controlled speed comparison. Kaggle
notebook integration is outside this local screen.

Evaluation and compact export use the notebook runtime:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.verify_geometry
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.evaluate
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.analyze_failures
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.validate_outputs
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.record_runtimes
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.publish --required incumbent cellect organoid nucverse spotiflow xenopus xenopus_centroid pacmap cellpose_cpdino_vitb
PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.detector_screen.refresh_canvas
```

`watch.py` refreshes evidence from already-running jobs and exits when all
required assessment outputs exist. It does not start inference, train models,
or predict links. Export refuses incomplete required panels and omits partial
cohorts from comparison tables.

The evaluator rounds native coordinates with `np.rint`, matching the official
exporter's integer convention, and applies common image-bound clipping.
Original float predictions are retained. It calls the installed official
distance matcher and asserts unique GT matches and the requested physical gate.
These assertions guard a verified pathological dense-fallback bug in upstream
tracksdata; the independent audit found no activation among the measured
settings it checked. Caches bind the panel, GT, prediction and timing receipts;
paired comparisons require a current incumbent result for every frame.

Source-parity probes, resampling ramp fixtures, label-ID-to-confidence checks,
raw-GEFF comparisons and integer-exporter checks are preserved with provenance.
The final artifact validator checks complete coverage, finite native-coordinate
arrays, receipt counts, fixed resampling metadata and the shared Spotiflow
threshold subsets. It hashes every realized prediction and timing receipt.
Both-endpoint coverage is a detector diagnostic; it is not edge Jaccard,
division accuracy, or a competition score. Candidate counts precede temporal
pruning. No training, assignment model, submission, or active detector
configuration is changed by this experiment.
