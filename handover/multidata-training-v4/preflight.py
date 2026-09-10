#!/usr/bin/env python3
"""Read-only path/count preflight. Does not download, train or certify data integrity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def inspect(repo: Path, archive: Path, prepared: Path, prior: Path) -> dict:
    synth = archive / 'downloads/kaggle/biohub_synthetic'
    checks = []
    for name, directory, pattern, expected, required in [
        ('static_images', synth/'static', 'vol_*.npz', 1539, True),
        ('sequence_images', synth/'sequences', 'seq_*.npz', 2174, True),
        ('static_labels', prepared/'synthetic', 'vol_*_labels.npz', 1539, True),
        ('sequence_labels', prepared/'synthetic', 'seq_*_labels.npz', 2174, True),
        ('zoo_graphs', prepared/'zoo', '*_graph.npz', 6, False),
        ('riken_archives', archive/'downloads/ssbd/bdml', '*.zip', 7, False),
    ]:
        found = len(list(directory.glob(pattern))) if directory.is_dir() else 0
        checks.append(dict(name=name,path=str(directory),expected=expected,found=found,
                           required=required,passed=found==expected))
    for name, path, required in [
        ('synthetic_manifest', prepared/'synthetic_manifest.json', True),
        ('inventory', repo/'docs/external-data-guide/dataset_inventory.json', True),
        ('zebrafish_graph', prepared/'zoo/zebrafish_graph.npz', False),
        ('v3_results', repo/'results/strong-tracker-v3/winning_config.json', True),
        ('v3_local_store', prior, True),
    ]:
        ok = path.exists()
        checks.append(dict(name=name,path=str(path),required=required,passed=ok,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest() if ok and path.is_file() else None))
    return dict(status='paths_ready' if all(c['passed'] for c in checks if c['required']) else 'missing_required_paths',
                checks=checks,training_performed=False,
                caveat='Counts/existence only; W400 must verify hashes, sample alignment, supervision, permissions and provenance.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--archive-root',type=Path)
    p.add_argument('--prepared-root',type=Path,help='Directory containing synthetic_manifest.json and synthetic/ and zoo/')
    p.add_argument('--v3-root',type=Path,default=Path('/kaggle/working/cell-tracking/strong-tracker-v3'))
    args=p.parse_args();repo=args.repo_root.resolve()
    result=inspect(repo,args.archive_root or repo/'work/biohub-forum-archive',
                   args.prepared_root or repo/'work/biohub-data-guide/prepared',args.v3_root)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['status']=='paths_ready' else 2)


if __name__=='__main__':main()
