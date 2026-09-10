#!/usr/bin/env python3
"""Read-only v6 path/capacity report; does not start jobs or install packages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def inspect(repo: Path, artifacts: Path, data: Path) -> dict:
    checks = {
        'repo_agents': repo / 'AGENTS.md',
        'v5_continuation': repo / 'handover/image-native-tracking-v5/CONTINUATION.md',
        'competition_data': data / 'train',
        'v1_artifacts': artifacts / 'annotation-selection-v1',
        'v2_artifacts': artifacts / 'strong-tracker-v2',
        'v3_artifacts': artifacts / 'strong-tracker-v3',
        'v5_artifacts': artifacts / 'image-native-tracking-v5',
    }
    rows = {name: {'path': str(p), 'exists': p.exists()} for name,p in checks.items()}
    disk_path = artifacts
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    free = shutil.disk_usage(disk_path).free / 2**30
    core = all(rows[k]['exists'] for k in ('repo_agents','competition_data','v1_artifacts','v2_artifacts','v3_artifacts'))
    return {
        'status': 'paths_present_not_validated' if core else 'missing_required_paths',
        'checks': rows, 'disk_checked_at': str(disk_path), 'free_gib': free,
        'reserve_gib': 8, 'available_above_reserve_gib': max(0.,free-8),
        'v6_output_exists': (artifacts/'segmentation-tracking-v6').exists(),
        'models_verified': False, 'image_schema_verified': False,
        'gpu_or_active_jobs_checked': False,
        'note': 'Existence is not integrity. Inspect active v5 jobs and current result receipts before allocating compute.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--artifact-root',type=Path,default=Path('/kaggle/working/cell-tracking'))
    parser.add_argument('--data-root',type=Path,default=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'))
    args = parser.parse_args()
    result = inspect(args.repo_root.resolve(),args.artifact_root.resolve(),args.data_root.resolve())
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status']=='paths_present_not_validated' else 2


if __name__=='__main__':
    raise SystemExit(main())
