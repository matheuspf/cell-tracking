"""Shared paths for the Biohub cell-tracking workspace."""

import os
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = Path("/kaggle")
CANONICAL_DATA_ROOT = KAGGLE_ROOT / "input" / "competitions" / COMPETITION
DATA_ROOT = Path(os.environ.get("CELL_TRACKING_DATA_ROOT", CANONICAL_DATA_ROOT))
WORK_ROOT = Path(os.environ.get("CELL_TRACKING_WORK_ROOT", KAGGLE_ROOT / "working/cell-tracking"))


def require_data_root() -> Path:
    if not (DATA_ROOT / "sample_submission.csv").is_file():
        raise FileNotFoundError(f"Competition data missing at {DATA_ROOT}; run the data downloader.")
    return DATA_ROOT
