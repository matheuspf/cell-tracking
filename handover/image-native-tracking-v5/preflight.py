#!/usr/bin/env python3
"""Read-only v5 path/budget check. Does not verify data, import GPU code or install."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def inspect(repo: Path, study_root: Path, data: Path) -> dict:
    repo = repo.resolve(); study_root = study_root.resolve(); data = data.resolve()
    required_repo = ["AGENTS.md", "handover/multidata-training-v4/CONTINUATION.md",
                     "results/multidata-training-v4/final_report.md",
                     "results/multidata-training-v4/candidate_gt_coverage.csv",
                     "tools/multidata_training_v4/models.py"]
    old_names = ["annotation-selection-v1", "strong-tracker-v2", "strong-tracker-v3", "multidata-training-v4"]
    required = {"repo:" + name: (repo / name).is_file() for name in required_repo}
    required.update({"old_root:" + name: (study_root / name).is_dir() for name in old_names})
    required["competition_train"] = (data / "train").is_dir()
    anchor = study_root
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    free_gib = shutil.disk_usage(anchor).free / 2**30
    budget_gib = max(0.0, min(24.0, 0.6 * free_gib, free_gib - 8.0))
    return {"status": "paths_present_not_a_data_audit" if all(required.values()) else "missing_local_dependencies",
            "repo": str(repo), "study_root": str(study_root), "data": str(data),
            "checks": required, "missing": [k for k, ok in required.items() if not ok],
            "free_gib": free_gib, "proposed_new_output_cap_gib": budget_gib,
            "optional_hoct_weights_must_be_checked_later": True,
            "gpu_checked": False, "hashes_verified": False, "writes_performed": False,
            "note": "A fresh clone has code/results, not old microscopy/checkpoints/caches. Do not rerun v4 to repair missing storage."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--study-root", type=Path, default=Path("/kaggle/working/cell-tracking"))
    p.add_argument("--data", type=Path, default=Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"))
    args = p.parse_args()
    print(json.dumps(inspect(args.repo, args.study_root, args.data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
