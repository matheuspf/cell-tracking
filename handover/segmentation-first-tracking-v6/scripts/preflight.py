#!/usr/bin/env python3
"""Read-only plan/source/disk inventory. No model, GT, network or GPU execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = '03ab55727c5248b292b248113eaa54123791ef03'


def validate_plan(p):
    if p['kind'] != 'execution_plan_not_results' or p['parent_commit'] != BASE:
        raise ValueError('incorrect plan kind or source pin')
    stages = p['stages']
    if [r['id'] for r in stages] != [f'X{x}' for x in range(600,691,10)]:
        raise ValueError('expected X600-X690 exactly once')
    if any(r['status'] != 'planned' or r['result'] is not None for r in stages):
        raise ValueError('author registry contains claimed execution results')
    budget = p['budget']
    if budget['primary_complete_configs']+budget['replica_complete_configs'] != 16:
        raise ValueError('all full config executions must fit the registered 16 slots')
    if p['labels']['unannotated'] != 'unknown' or p['labels']['dense_masks_downloaded'] is not False:
        raise ValueError('sparse-label contract violated')
    return {'stages':len(stages),'execution_results_claimed':False}


def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def audit(root=ROOT, here=HERE):
    p = json.loads((here/'experiments.json').read_text())
    report = validate_plan(p)
    anchors=[]
    for relative, expected in p['source_git_blobs'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('source anchor escapes repository')
        actual = blob(path.read_bytes()) if path.is_file() else None
        anchors.append({'path':relative,'expected':expected,'actual':actual,'matches':actual==expected})
    report.update(kind='read_only_handover_audit_not_detector_validation',
                  root=str(root),parent_commit=BASE,anchors=anchors,
                  source_pins_match=all(r['matches'] for r in anchors),
                  filesystem_free_gib=shutil.disk_usage(root).free/2**30,
                  local_live_jobs_checked=False,weights_access_checked=False,
                  microscopy_or_gt_opened=False,gpu_experiments_executed=False)
    return report


if __name__ == '__main__':
    try:
        print(json.dumps(audit(),indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError) as e:
        print(f'preflight: {e}',file=sys.stderr)
        raise SystemExit(1)
