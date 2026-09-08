#!/usr/bin/env python3
"""Create Kaggle-compatible aliases without duplicating downloaded data."""

import json
from pathlib import Path

from competition_paths import COMPETITION, DATA_ROOT, KAGGLE_ROOT, REPO_ROOT, WORK_ROOT


def ensure_alias(alias: Path, target: Path) -> str:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if alias.is_symlink() and alias.resolve() == target:
        return "already-correct"
    if alias.exists() and not alias.is_symlink():
        raise RuntimeError(f"Refusing to replace real path: {alias}")
    if alias.is_symlink():
        raise RuntimeError(f"Alias points elsewhere: {alias}. Resolve the conflict explicitly.")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target, target_is_directory=True)
    return "created"


def main() -> int:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for alias in (REPO_ROOT / "data", KAGGLE_ROOT / "input" / COMPETITION):
        status = ensure_alias(alias, DATA_ROOT)
        statuses[str(alias)] = {"target": str(DATA_ROOT.resolve()), "status": status}
        print(f"{status}: {alias} -> {DATA_ROOT.resolve()}")
    out = REPO_ROOT / "work/path-aliases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(statuses, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
