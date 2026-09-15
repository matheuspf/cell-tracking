from __future__ import annotations

import os
if os.environ.get('PUBLIC946_INFERENCE') == '1':
    from .isolation import install
    install()

import argparse
from pathlib import Path
from .common import DEFAULT_ARCHIVE, DEFAULT_ARTIFACTS, DEFAULT_DATA, DEFAULT_OUT, DEFAULT_RUNTIME


def main():
    p = argparse.ArgumentParser(description='Expanded registered public946 study; no training or uploads')
    p.add_argument('command', choices=['preflight', 'audit', 'controls', 'pilot', 'singles', 'combinations',
                                      'transfers', 'robustness', 'package', 'report', 'run', 'worker', 'evaluate'])
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    p.add_argument('--arm')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gpu-hours', type=float, default=96)
    p.add_argument('--gpu-gib', type=float, default=22)
    p.add_argument('--scratch-gib', type=float, default=48)
    p.add_argument('--workers', type=int, default=2, help='Bounded full-cohort subprocess concurrency on the one GPU')
    p.add_argument('--job', type=Path)
    args = p.parse_args()
    if args.command in ('preflight', 'audit'):
        from . import preflight
        return getattr(preflight, 'run' if args.command == 'preflight' else 'audit')(args)
    if args.command == 'worker':
        from .worker import run
        return run(args)
    if args.command == 'evaluate':
        from .evaluation import run
        return run(args)
    from .study import run
    return run(args)


if __name__ == '__main__':
    main()
