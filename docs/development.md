# Local development

```sh
conda activate cell-tracking
export PYTHONNOUSERSITE=1
python -m pip check
python tools/setup_paths.py
python tools/inspect_data.py
```

Use `tools/competition_paths.py` for the canonical competition path and output
root. `CELL_TRACKING_DATA_ROOT` and `CELL_TRACKING_WORK_ROOT` can override them;
`.env.example` documents the defaults. Export overrides in the shell; tools do
not implicitly load `.env` files.

## Dependencies

`requirements.txt` lists direct dependencies; `requirements-local.txt` pins the
resolved Python 3.12 runtime. `environment.yml` recreates the Conda environment.
To intentionally update dependencies after editing the direct list:

```sh
uv pip compile requirements.txt --python "$(command -v python)" --output-file requirements-local.txt
python -m pip install -r requirements-local.txt
python -m pip check
```

This environment supports data inspection and analysis. Choose and pin GPU/model
dependencies when implementing training, then verify Kaggle inference compatibility.

## Checks

```sh
python -m pytest
python -m ruff check tools tests
bash -n scripts/bootstrap.sh
python tools/download_data.py --verify-only
```

The downloader checks member CRCs while extracting and only writes a completion
manifest after the full archive succeeds. `--verify-only --checksums` performs a
fresh CRC pass over all extracted files; ordinary verification checks paths and
sizes and is much faster.

## Artifact conventions

Track code, configs, environment pins, original notebooks, and concise experiment
records. Keep downloaded input, reference-page copies, notebook snapshots,
checkpoints, caches, and submissions in ignored paths. Record seeds, dataset
versions, configuration, and artifact hashes alongside results when experiments
are added. The bootstrap never submits a notebook or pushes a Git remote.
