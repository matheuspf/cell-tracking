#!/usr/bin/env python3
"""Read-only v6 handover and local-artifact discovery. No model/data loading."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def inspect(repo: Path, output_root: Path) -> dict:
    repo = repo.resolve()
    handover = repo / 'handover/instance-tracking-v6'
    errors = []
    manifest = handover / 'MANIFEST.json'
    if manifest.exists():
        obj = json.loads(manifest.read_text())
        for row in obj['files']:
            path = (handover / row['path']).resolve()
            if not path.is_relative_to(handover.resolve()):
                errors.append('unsafe manifest path')
                continue
            if not path.is_file():
                errors.append(f"missing payload {row['path']}")
                continue
            b = path.read_bytes()
            if len(b) != row['bytes'] or hashlib.sha256(b).hexdigest() != row['sha256']:
                errors.append(f"payload mismatch {row['path']}")
    else:
        errors.append('manifest absent')
    paths = {
        'raw_competition': Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
        'v1': Path('/kaggle/working/cell-tracking/annotation-selection-v1'),
        'v2': Path('/kaggle/working/cell-tracking/strong-tracker-v2'),
        'v3': Path('/kaggle/working/cell-tracking/strong-tracker-v3'),
        'v4': Path('/kaggle/working/cell-tracking/multidata-training-v4'),
        'v5': Path('/kaggle/working/cell-tracking/image-native-tracking-v5'),
        'external_prepared': repo / 'work/biohub-data-guide/prepared',
        'official_metric': repo / 'work/annotation-selection-v1/official',
    }
    out = output_root.resolve()
    for name, path in paths.items():
        protected = path.resolve()
        if out == protected or out.is_relative_to(protected):
            errors.append(f'output is inside protected input/old-study path: {name}')
    parent = out
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    free = shutil.disk_usage(parent).free / 2**30
    present = {name: {'path': str(p), 'present': p.exists()} for name, p in paths.items()}
    return {'handover_valid': not errors, 'errors': errors, 'paths': present,
            'output_path': str(out), 'free_gib_at_nearest_existing_parent': free,
            'disk_reserve_8gib_met': free >= 8,
            'focus_authorization_verified': False,
            'active_jobs_checked': False,
            'models_or_images_loaded': False,
            'note': 'Missing local artifacts are discovery, not evidence of failed experiments. '
                    'Codex must inspect running jobs, GPU, source terms, actual inputs and model weights.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output-root', type=Path, default=Path('/kaggle/working/cell-tracking/instance-tracking-v6'))
    a = p.parse_args()
    result = inspect(a.repo, a.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['handover_valid'] else 2)

if __name__ == '__main__':
    main()
