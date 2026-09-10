#!/usr/bin/env python3
"""Read-only dependency discovery; no data load, downloads, installs or experiments."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path('/kaggle/working/cell-tracking')
DEFAULT_DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development')
OLD_NAMES = ('annotation-selection-v1', 'strong-tracker-v2', 'strong-tracker-v3', 'multidata-training-v4')


def inspect_paths(repo, study_parent, data_root):
    repo, study_parent, data_root = (Path(p).resolve() for p in (repo, study_parent, data_root))
    required = {'competition_train': data_root/'train',
                'v1': study_parent/OLD_NAMES[0], 'v2': study_parent/OLD_NAMES[1],
                'v3': study_parent/OLD_NAMES[2], 'v4': study_parent/OLD_NAMES[3],
                'official_metric': repo/'work/annotation-selection-v1/official'}
    optional = {'external_originals': repo/'work/biohub-forum-archive',
                'external_prepared': repo/'work/biohub-data-guide',
                'original_deepcenter': Path('/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')}
    anchor = study_parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free / 2**30
    missing = [k for k, v in required.items() if not v.exists()]
    return {'status': 'paths_present_not_validated' if not missing else 'dependencies_missing',
            'python': sys.executable, 'python_version': sys.version.split()[0],
            'required': {k: {'path': str(v), 'exists': v.exists()} for k,v in required.items()},
            'optional': {k: {'path': str(v), 'exists': v.exists()} for k,v in optional.items()},
            'missing': missing, 'free_disk_gib': free,
            'suggested_max_new_cache_gib': max(0., min(12., free-5.)),
            'raw_data_read': False, 'model_hashes_verified': False, 'official_scoring_run': False,
            'expired_v4_deadline_reused': False,
            'note': 'Presence is not integrity or restored-artifact completeness; implement P500 checks locally.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--study-parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    args = p.parse_args()
    result = inspect_paths(args.repo, args.study_parent, args.data_root)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(2 if result['missing'] else 0)


if __name__ == '__main__':
    main()
