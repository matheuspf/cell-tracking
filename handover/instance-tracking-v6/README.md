# Instance-mask tracking v6

**Execution handover, not measured v6 results.** Start with [CODEX_PROMPT.md](CODEX_PROMPT.md)
and read the final [parent refresh](PARENT_REFRESH.md).

The task now is to keep a cell's **image-supported 3D instance mask, bounding box,
shape and appearance through tracking**, rather than immediately reducing it to
one point. Export points only at the final Kaggle adapter. Confirm whether the
fluorescence delineates nuclei, cytoplasm or membranes: a nuclear mask is not a
whole-cell membrane reconstruction.

Branch: `handover/instance-tracking-v6`, based on `handover/image-native-tracking-v5`
at `fe00326ea6ca79a5b5c54910de03925d353e0629`. The parent advanced during authoring;
the initial review used `0a3105a`, and PARENT_REFRESH.md records the three added
complete comparisons. V5 remains explicitly in progress, with eleven complete
score sets including two oracles. Its retained C0 is 0.934802374260586 on 199 clips.

## New direction

1. Run actual learned instance segmentation: FOCUS-3D when locally authorized;
   independently run Cellpose-SAM 3D. StarDist3D is a bounded third comparator or
   fallback, not another mandatory installation project.
2. Compare point-only, bounding-box, and full-mask linking on identical detections,
   then compare mask-derived detections. Measure mask overlap after image-derived
   motion, morphology and parent-versus-daughter-union evidence.
3. Run Ultrack's segmentation hierarchy and temporal selection on these masks;
   test a single segmenter before combining foreground/contour evidence.
4. Preserve C0 and test a conservative mask-feature upgrade separately from a
   standalone segmentation-first tracker. Do not inherit v5's regressing J decoder
   as the only integration path.

The target remains **>=0.95 official local score**, not a guarantee. The measured
reference and target gap are in [baseline.json](baseline.json). Independent latent
cell boundaries and hidden-test performance are not established by sparse points.

## Read and execute

- [REVIEW.md](REVIEW.md): current evidence and why this differs from v5.
- [MODEL_DECISIONS.md](MODEL_DECISIONS.md): model choice, access, units and APIs.
- [EXPERIMENTS.md](EXPERIMENTS.md): M600-M690 executable study contract.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): modules, masks, adapters and tests.
- [SOURCES.md](SOURCES.md), [config.json](config.json), [model_registry.json](model_registry.json).
- `mask_contracts.py`, `test_mask_contracts.py`: working NumPy reference helpers;
  these are NOT detector inference or a replacement for the official evaluator.
- `preflight.py`: read-only discovery and manifest validation; no installation,
  network call, model loading or scientific rerun.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover   -s handover/instance-tracking-v6 -p 'test_mask_contracts.py' -v
python handover/instance-tracking-v6/preflight.py --repo .
```

No activation patch or ZIP copy is required. Codex builds the missing integration
modules and executes the study in a separate namespace. All v1-v5 results, running
jobs, source trees and raw inputs must be preserved. Git contains code and sanitized
summaries only; masks, images, checkpoints and detailed GT remain local.
