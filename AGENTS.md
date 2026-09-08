# Biohub — Cell Tracking During Development

This workspace is for Kaggle competition
`biohub-cell-tracking-during-development`.

## Competition knowledge

- Use the relevant `.agents/skills/competition-*` skill for competition questions.
- Read `reference/overview/` for official pages, then relevant organizer threads
  under `reference/forum/topics/`. Search `reference/forum/INDEX.md` for other topics.
- Inspect the actual files in `data/` when determining schemas or array layout.
- `reference/manifest.json` records reference sync state. Refresh with
  `PYTHONPATH=tools python -m kaggle_extract sync`; do not hand-edit the manifest.
- Reference snapshots can become stale. Refresh time-sensitive facts and distinguish
  organizer statements from community suggestions. Preserve extractor throttles.

## Runtime and paths

- Conda environment/kernel: `cell-tracking` / `Python (cell-tracking)`, Python 3.12.
- Downloaded notebook runtime: `/kaggle/envs/cell-tracking-notebooks`, kernel
  `Python (cell-tracking notebooks)`. See `docs/notebooks.md` for selection,
  input artifacts, and local execution paths.
- Set `PYTHONNOUSERSITE=1` for scripted checks.
- Canonical data: `/kaggle/input/competitions/biohub-cell-tracking-during-development`.
- Repo `data` and `/kaggle/input/biohub-cell-tracking-during-development` are aliases.
  Restore them with `python tools/setup_paths.py`.
- Raw data is an input: keep it unchanged. Outputs belong in
  `/kaggle/working/cell-tracking` or ignored `work/`.
- Keep credentials, downloaded inputs, raw references, model weights, and generated
  submissions outside Git. Track code, configs, environment pins, and concise results.

## Data and submission contract

The task detects and tracks cells in 3D microscopy over time. Images are Zarr v3
arrays; training annotations are sparse GEFF graphs. Read actual metadata before
assuming dimensions or interpreting coordinates. Consider embryo identity when
designing validation; visible example test clips are not independent validation.

Submissions are notebooks producing `submission.csv` with node and edge rows.
Read `reference/overview/evaluation.md` and `code-requirements.md` for current
requirements. The development environment is a data-inspection scaffold; validate
future training and inference dependencies against the Kaggle runtime.
