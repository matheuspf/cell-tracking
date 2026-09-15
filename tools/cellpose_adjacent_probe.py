"""Extend the image-only Cellpose pilot with t+1 for valid association pairs.

Reuses the exact frozen detector implementation with separate output roots.
Both this driver and that implementation are bound in the extension manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tools import cellpose_models_probe as detector

BASE_WORK, BASE_OUT = detector.WORK, detector.OUT
WORK, OUT = BASE_WORK / 'adjacent', BASE_OUT / 'adjacent'


def prepare():
    detector.deny_labels()
    parent = detector.read(BASE_OUT / 'pilot-plan.json')
    plan = dict(parent)
    frames = []
    for row in parent['frames']:
        t = row['time'] + 1
        frames.append({**row, 'key': f"{row['dataset']}-t{t:03}", 'time': t,
                       'role': 'pilot_adjacent_extension', 'image_path': row['next_path'],
                       'previous_path': row['image_path'],
                       'next_path': str(Path(row['image_path']).with_name(f't{t+1:03}.npy'))})
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(), frames=frames,
                image_sha256={r['key']: detector.sha(r['image_path']) for r in frames},
                parent_plan_sha256=detector.sha(BASE_OUT / 'pilot-plan.json'),
                extension_driver_sha256=detector.sha(__file__),
                extension_reason='Original pilot t25/t75 frames are nonadjacent. Add t26/t76 for twelve consecutive-frame pairs without changing or rescoring the original frozen detector recipe.',
                evaluation='Combine original and extension into 24 frames, twelve adjacent pairs. Freeze all predictions and native scores before opening GT. No complete-clip competition score.')
    dest = OUT / 'pilot-plan.json'
    if dest.exists():
        old = detector.read(dest)
        assert {k:v for k,v in old.items() if k!='created_utc'} == {k:v for k,v in plan.items() if k!='created_utc'}
    else:
        detector.write(dest, plan)
    print(dest, flush=True)


def detect(name):
    plan = detector.read(OUT / 'pilot-plan.json')
    assert plan['extension_driver_sha256'] == detector.sha(__file__)
    assert plan['parent_plan_sha256'] == detector.sha(BASE_OUT / 'pilot-plan.json')
    detector.WORK, detector.OUT = WORK, OUT
    detector.detect(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','detect'))
    parser.add_argument('--model', choices=('cpdino','cpsam_v2'))
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        if args.model is None: parser.error('--model required')
        detect(args.model)
