"""Validate v11's local handover only; no downloads, training or data reads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contracts import validate_registry


def check(directory: Path) -> dict:
    registry = json.loads((directory / 'study.json').read_text())
    validate_registry(registry)
    files = registry['handover_files']
    result = {}
    for filename in files:
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Handover file escapes its directory')
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty handover file: {filename}')
        result[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = directory.parents[1] / 'DIVISION_RELIABILITY_V11_START_HERE.md'
    if not root.is_file():
        raise ValueError('Missing root entry point')
    result[root.name] = hashlib.sha256(root.read_bytes()).hexdigest()
    return {'status': 'planning_contracts_passed', 'files': result,
            'model_training_executed': False, 'official_scores_recomputed': False,
            'runtime_or_data_isolation_validated': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = check(Path(__file__).resolve().parent)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
