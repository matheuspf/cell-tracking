# Biohub — Cell Tracking During Development

Workspace for [the Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).
The repository scaffolding and reference extractor are adapted from the local
`~/code/kaggle/ptcg` template. Competition knowledge and paths are specific to Biohub.

## Start working

```sh
conda activate cell-tracking
export PYTHONNOUSERSITE=1
python tools/inspect_data.py
```

VS Code uses the `cell-tracking` environment; Jupyter provides a
`Python (cell-tracking)` kernel. The locked runtime includes Zarr v3, NumPy,
pandas, SciPy, scikit-image, image readers, plotting, and notebook tooling.
Add training dependencies when selecting a model.

## Layout

| Path | Purpose |
|---|---|
| `data/` | Alias to `/kaggle/input/competitions/biohub-cell-tracking-during-development` |
| `/kaggle/input/biohub-cell-tracking-during-development` | Flat alias for Kaggle notebook compatibility |
| `/kaggle/working/cell-tracking` | Experiment outputs, derived data, and submissions |
| `.agents/skills/competition-*` | Five base competition skills for Codex |
| `.claude/skills/competition-*` | The same five skills for Claude Code |
| `reference/overview/` | Regenerable official overview, data, evaluation, rules, and timeline |
| `reference/forum/` | Forum metadata index and selected cached threads |
| `tools/` | Data download, verification, inspection, path setup, and reference sync |
| `configs/`, `notebooks/`, `docs/` | Configuration, analysis, and maintained project notes |
| `work/` | Local manifests, logs, and scratch files |

The downloaded data, references, credentials, and generated artifacts are ignored
by Git. See [competition notes](docs/competition.md) for the task and data schema.

## Recreate this setup

Requires Conda and configured Kaggle credentials. The Kaggle account must have
access to the competition data. From the repo root:

```sh
bash scripts/bootstrap.sh
```

The script creates the environment, installs `requirements-local.txt`, registers
the kernel, installs Chromium for forum retrieval, downloads and extracts the
competition archive, creates path aliases, and refreshes the reference mirror.
The archive is approximately 81.4 GiB; allow space for both the ZIP and extracted
inputs during setup. The ZIP is removed only after successful extraction and verification.

```sh
# Resume a download, or check an existing completed download without fetching again
python tools/download_data.py

# Verify all expected paths and sizes; add --checksums to reread every file
python tools/download_data.py --verify-only

# Restore aliases and refresh reference pages, forum metadata, and skills
python tools/setup_paths.py
PYTHONPATH=tools python -m kaggle_extract sync
PYTHONPATH=tools python -m kaggle_extract status
```

`work/data-manifest.json` records archive identity, extracted member sizes and
CRCs, and completion time. Extraction verifies each ZIP member's CRC. A partial
download or extraction is not recorded as complete.

## Skills and reference maintenance

The base skills are `competition-overview`, `competition-data`,
`competition-rules`, `competition-forum`, and `competition-status`. Their source
is `tools/kaggle_extract/skills.py`; regenerate both agent layouts after changes:

```sh
PYTHONPATH=tools python -m kaggle_extract skills
PYTHONPATH=tools python -m kaggle_extract topic 727154
```

Sync uses the template's throttled Kaggle metadata client. Topic bodies are
fetched on demand through Playwright. Optional Anthropic summaries require
`requirements-distill.txt` and an explicitly invoked `distill` or `--distill`;
ordinary setup and sync do not call a model API.

See [development notes](docs/development.md) for environment updates and checks.

## Public notebook mirrors

Three notebooks with the highest verified current scores from the Code page are
set up under `/kaggle`: Harmonic Fusion (0.946), biohub-942tta (0.946), and
biohub-942tta-repro-20260907 (0.945). Their Python 3.12/CUDA runtime is
`/kaggle/envs/cell-tracking-notebooks`; use the `Python (cell-tracking notebooks)`
kernel. See [notebook setup](docs/notebooks.md) for clickable files, exact
versions, input datasets, and the Code page's stale-score caveat.
