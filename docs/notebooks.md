# Downloaded public notebooks

Selected on 2026-09-08 from the competition Code page, ordered by public score,
then verified on each notebook's published-version page:

| Notebook | Author | Version | Verified public score | Script version ID |
|---|---|---:|---:|---:|
| [Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion?scriptVersionId=347965685) | flexonafft | 29 | 0.946 | 347965685 |
| [biohub-942tta](https://www.kaggle.com/code/redoctopusk/biohub-942tta?scriptVersionId=347821442) | redoctopusk | 1 | 0.946 | 347821442 |
| [biohub-942tta-repro-20260907](https://www.kaggle.com/code/brucezheng666/biohub-942tta-repro-20260907?scriptVersionId=347995146) | brucezheng666 | 1 | 0.945 | 347995146 |

The Code list retains stale pre-rescore scores: its first three entries displayed
0.966/0.966/0.965, while their published pages showed 0.885/0.885/0.883. This
selection uses the current verified scores. Organizer metric-patch and rescore
threads are cached in `reference/forum/topics/727154-*.md` and `728324-*.md`.
The exact selection is tracked in [`configs/notebooks.json`](../configs/notebooks.json).

## Open or run locally

| Notebook | Jupyter copy | Python export |
|---|---|---|
| Harmonic Fusion | [Notebook](/kaggle/working/biohub-harmonic-fusion.ipynb) | [Script](/kaggle/working/biohub-harmonic-fusion.py) |
| 942 TTA | [Notebook](/kaggle/working/biohub-942tta.ipynb) | [Script](/kaggle/working/biohub-942tta.py) |
| 942 TTA reproduction | [Notebook](/kaggle/working/biohub-942tta-repro-20260907.ipynb) | [Script](/kaggle/working/biohub-942tta-repro-20260907.py) |

```sh
conda activate /kaggle/envs/cell-tracking-notebooks
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd /kaggle/working
# When you want to run the full pipeline:
python biohub-harmonic-fusion.py
```

For Jupyter, choose `Python (cell-tracking notebooks)`. Configuration/import
cells were executed in that registered kernel for all three notebooks. Full
inference was not launched. The upstream notebooks share output locations such
as `/kaggle/working/submission.csv` and `tracking_repo`; run them one at a time
and preserve any prior experiment outputs you want to keep.

Archived originals, metadata, Python exports, `LOCAL_SETUP.md`, and
`download-report.json` are under:

- `/kaggle/notebooks/biohub-cell-tracking-during-development/flexonafft/biohub-harmonic-fusion/`
- `/kaggle/notebooks/biohub-cell-tracking-during-development/redoctopusk/biohub-942tta/`
- `/kaggle/notebooks/biohub-cell-tracking-during-development/brucezheng666/biohub-942tta-repro-20260907/`

Repo `notebooks/downloaded` is an ignored alias to this archive root.

## Inputs and dependencies

All three notebooks use the same four sources. The mirror compared every
declared file name and byte size with Kaggle and reused shared data:

| Source | Files | Bytes |
|---|---:|---:|
| Main competition | 24,886 | 87,609,892,618 |
| `pilkwang/biohub-tracking-support-pack-50ep-v1` | 87 | 355,646,564 |
| `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` | 16 | 79,512,645 |
| `pilkwang/biohub-temporal-unet3d-seed314159-v1` | 91 | 355,747,040 |

Dataset inputs live under `/kaggle/input/datasets/pilkwang/<slug>` with flat
`/kaggle/input/<slug>` aliases for hard-coded notebook paths. Model artifacts and
offline wheels are included in those mirrors.

The dedicated environment uses Python 3.12.14 and PyTorch 2.8.0+cu128. CUDA
initialization and a 3D convolution passed on the local RTX 4090. Direct package
pins are in [`requirements-notebooks.txt`](../requirements-notebooks.txt), also
copied to `/kaggle/envs/cell-tracking-notebooks/requirements-local.txt`.

The bundled `tracksdata` wheel is the notebook's exact development build. Its
93 implementation files match upstream commit
`980c2d30aeca76b86eddef0aeadb4d10dee8530d`; the remaining Python file is generated
version metadata. Other packages were installed from package indexes. All six
`.pt`/`.pth` checkpoints loaded with `weights_only=True`, including the required
DeepCenter epoch-2 checkpoint. Provenance, hashes, and checks are copied into
each archive's `local-verification.json`.

Recreate the runtime after restoring those input datasets:

```sh
bash scripts/setup_notebook_env.sh
python tools/prepare_notebook_exports.py
```

Notebook originals remain unchanged. Runnable notebook metadata selects the local
kernel. The Python exports relocate later-cell `from __future__` imports to the
module preamble so they compile in a terminal; the inference code is preserved.
