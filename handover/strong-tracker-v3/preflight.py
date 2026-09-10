#!/usr/bin/env python3
"""Read-only v2 selected-cache audit; no images, models, network or installation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_safe_output(output: Path, protected: list[Path]) -> Path:
    output = output.resolve()
    for root in protected:
        root = root.resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError(f'output overlaps protected input: {root}')
    return output


def inspect_cache(v2: Path, expected_count: int=199) -> dict:
    v2 = v2.resolve(strict=True)
    lock_path = v2/'selected_prediction_lock.json'
    lock = json.loads(lock_path.read_text())
    if lock.get('variant') != 'bypass_motion_bounds':
        raise ValueError('wrong incumbent variant')
    if lock.get('annotation_reads') != 0 or not lock.get('byte_identical_to_scored_graphs'):
        raise ValueError('incumbent validation receipt incomplete')
    hashes = lock.get('hashes',{})
    if not isinstance(hashes,dict) or len(hashes) != expected_count:
        raise ValueError('unexpected number of incumbent graphs')
    actual = {p.stem:p for p in (v2/'selected_predictions').glob('*.npz')}
    if set(actual) != set(hashes):
        raise ValueError('missing or extra selected graph')
    for name,digest in hashes.items():
        if not isinstance(name,str) or Path(name).name != name or name in ('.','..'):
            raise ValueError('invalid dataset identifier')
        path = actual[name].resolve(strict=True)
        if v2 not in path.parents:
            raise ValueError('selected graph resolves outside cache')
        if sha256(path) != digest:
            raise ValueError(f'changed selected graph: {name}')
    return dict(status='hashes_verified_not_rescored',samples=len(hashes),
                variant=lock['variant'],lock_sha256=sha256(lock_path),
                note='Local official graph scoring and runtime checks still required.')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2-root',type=Path,default=Path('/kaggle/working/cell-tracking/strong-tracker-v2'))
    p.add_argument('--v1-root',type=Path,default=Path('/kaggle/working/cell-tracking/annotation-selection-v1'))
    p.add_argument('--data-root',type=Path,default=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'))
    p.add_argument('--output',type=Path,help='Optional new JSON receipt; existing file is never overwritten')
    args=p.parse_args()
    out = require_safe_output(args.output,[args.v2_root,args.v1_root,args.data_root]) if args.output else None
    report=inspect_cache(args.v2_root)
    report['v1_available']=args.v1_root.is_dir()
    report['data_available']=args.data_root.is_dir()
    if not report['v1_available'] or not report['data_available']:
        report['status']='selected_hashes_verified_inputs_missing'
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if out:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:
            f.write(text)
    print(text,end='')
    if report['status'] != 'hashes_verified_not_rescored':
        raise SystemExit(2)


if __name__=='__main__':
    main()
