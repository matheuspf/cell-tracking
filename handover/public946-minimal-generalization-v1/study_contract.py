"""Validate/freeze the expanded study; this does not run or certify inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SINGLES = tuple(f'E{i:02d}' for i in range(1, 9))
COMBINATIONS = {'C01': ['E02', 'E03'], 'C02': ['E04', 'E05'], 'C03': ['E05', 'E06']}
LOCKED_FILES = ('START_HERE.md', 'PLAN.md', 'EXPERIMENTS.md', 'VALIDATION.md',
                'CODEX_PROMPT.md', 'BASELINE_AND_EVIDENCE.md', 'experiment_lock.json',
                'study_contract.py', 'baseline_contract.py')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(lock: dict) -> None:
    require(lock.get('schema_version') == 2, 'Expected expanded schema 2')
    require(lock.get('branch') == 'handover/public946-minimal-generalization-v1', 'Wrong branch')
    require(lock.get('max_total_configurations') == 15, 'Expected 15 bounded configurations')
    require(lock.get('new_training_runs') == 0, 'Training is outside this study')
    require(lock.get('parameter_sweeps_allowed') is False, 'Sweeps are forbidden')
    require(lock.get('kaggle_submission_authorized') is False, 'Uploads not authorized')
    arms = lock.get('arms', [])
    require(isinstance(arms, list), 'arms must be a list')
    ids = [a.get('id') for a in arms]
    expected = ['B0', 'B1', *SINGLES, *COMBINATIONS, 'X01', 'X02']
    require(ids == expected, 'Unexpected, missing, duplicated, or reordered arm IDs')
    by_id = {a['id']: a for a in arms}
    require(by_id['B0'].get('parent') is None, 'B0 must be the public source')
    require(by_id['B0'].get('modules') == [], 'B0 cannot carry scientific changes')
    require(by_id['B1'].get('parent') == 'B0', 'B1 must derive from B0')
    require(by_id['B1'].get('modules') == ['no_motion'], 'B1 must only disable motion')
    for arm_id in SINGLES:
        arm = by_id[arm_id]
        require(arm.get('parent') == 'B0', f'{arm_id} must be compared to B0')
        require(arm.get('modules') == [arm_id], f'{arm_id} must isolate its module')
    for arm_id, components in COMBINATIONS.items():
        arm = by_id[arm_id]
        require(arm.get('parent') == 'B0', f'{arm_id} wrong parent')
        require(arm.get('modules') == components, f'{arm_id} unregistered combination')
    for arm_id in ('X01', 'X02'):
        require(by_id[arm_id].get('parent') == 'B1', f'{arm_id} wrong transfer parent')
        require(by_id[arm_id].get('modules') == ['locked_finalist'], f'{arm_id} must resolve once')
    for arm in arms:
        require(arm.get('status') == 'planned_not_executed', 'Registry is a plan, not a result ledger')
        require(bool(arm.get('name')), 'Missing arm name')
    require(lock.get('max_finalists') == 2, 'Only two finalists are allowed')
    require(lock.get('leaderboard_adaptive_revision_allowed') is False, 'LB adaptation forbidden')


def validate_resolution(lock: dict, resolution: dict) -> None:
    """Check structure only. Scientific eligibility requires the measured receipts."""
    validate(lock)
    require(set(resolution) == {'X01', 'X02'}, 'Resolution requires both transfer slots')
    selected = [v for v in resolution.values() if v is not None]
    eligible = set(SINGLES) | set(COMBINATIONS)
    eligible.remove('E01')  # Guarded relinking is incompatible with disabling it.
    require(all(isinstance(v, str) and v in eligible for v in selected), 'Invalid transfer recipe')
    require(len(selected) == len(set(selected)), 'Duplicate transfer recipes')
    require(resolution['X01'] is not None or resolution['X02'] is None, 'Fill X01 before X02')


def freeze(root: Path, output: Path) -> dict:
    lock = json.loads((root / 'experiment_lock.json').read_text())
    validate(lock)
    payload = {'schema_version': 1, 'status': 'plan_frozen_not_executed',
               'files': {name: digest(root / name) for name in LOCKED_FILES}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return payload


def verify(root: Path, frozen: Path) -> None:
    receipt = json.loads(frozen.read_text())
    require(receipt.get('status') == 'plan_frozen_not_executed', 'Not a plan freeze')
    require(set(receipt.get('files', {})) == set(LOCKED_FILES), 'Incomplete freeze')
    for name, expected in receipt['files'].items():
        require(digest(root / name) == expected, f'Plan drift: {name}')
    validate(json.loads((root / 'experiment_lock.json').read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'matrix', 'freeze', 'verify', 'resolve'])
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--file', type=Path, help='Freeze receipt or transfer-resolution JSON')
    args = parser.parse_args()
    try:
        lock = json.loads((args.root / 'experiment_lock.json').read_text())
        validate(lock)
        if args.command == 'matrix':
            print('id\tparent\tmodules\tname')
            for a in lock['arms']:
                print(f"{a['id']}\t{a['parent'] or '-'}\t{','.join(a['modules']) or '-'}\t{a['name']}")
        elif args.command in ('freeze', 'verify', 'resolve'):
            if args.file is None:
                parser.error('--file is required for this command')
            if args.command == 'freeze':
                freeze(args.root, args.file)
            elif args.command == 'verify':
                verify(args.root, args.file)
            else:
                validate_resolution(lock, json.loads(args.file.read_text()))
            print('OK: structural plan contract only; no inference or metric certification')
        else:
            print('OK: 2 anchors, 8 single mechanisms, 3 fixed combinations, 2 transfer slots')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
