#!/usr/bin/env python3
"""Validate the v10 planning contract; never read microscopy or run training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent


def validate(spec: dict) -> list[str]:
    errors: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    try:
        require(spec['status_at_creation'] == 'planned_not_executed', 'Planning status must not claim execution')
        require(spec['measured_results_claimed'] is False, 'No v10 results exist at plan creation')
        require(re.fullmatch(r'[0-9a-f]{40}', spec['base_sha']) is not None, 'Invalid base commit')
        require(re.fullmatch(r'[0-9a-f]{40}', spec['metric_revision']) is not None, 'Invalid metric revision')
        require(spec['primary']['candidate'] == 'C10' and spec['primary']['control'] == 'C00', 'Primary must remain C10 versus C00')
        require(spec['primary']['selected_before_target_scoring'] is True, 'Primary must be frozen before targets')
        require(spec['seeds'] == [20260918, 314159], 'Both registered seeds are required')
        directions = {(d['source'], d['target']) for d in spec['directions']}
        require(directions == {('44b6', '6bba'), ('6bba', '44b6')} and len(spec['directions']) == 2, 'Both complete embryo directions are required')
        require(spec['expected_clips'] == {'44b6': 71, '6bba': 128}, 'Expected cohort must contain all 199 clips')
        require(spec['read_actual_metadata'] is True, 'Actual metadata must be inspected')
        for field in ['clean_exposed_dependencies_allowed', 'target_fitting_allowed', 'target_based_selection_allowed']:
            require(spec[field] is False, f'{field} must be false')
        require(spec['clean_initialization'] == 'random_all_neural_components', 'Clean initialization must exclude inherited weights')
        split = spec['source_split']
        require(abs(split['fit_fraction'] + split['calibration_fraction'] - 1.0) < 1e-12, 'Source fractions must sum to one')
        require(split['acquisition_independence_certified'] is False, 'Unknown acquisition independence cannot be certified')
        jobs = []
        mandatory = 0
        for group in spec['fit_groups']:
            require(set(group['sources']) == {'44b6', '6bba'}, 'Fit groups must include both source embryos')
            if group['lane'] == 'clean':
                require(group['mandatory'] and group['seeds'] == spec['seeds'], 'Clean fits and both seeds are mandatory')
            for variant in group['variants']:
                for source in group['sources']:
                    for seed in group['seeds']:
                        jobs.append((group['id'], variant, source, seed))
                        mandatory += int(group['mandatory'])
        require(len(set(jobs)) == len(jobs), 'Duplicate neural jobs')
        require(len(jobs) == spec['maximum_neural_fits'] == 12, 'Maximum matrix must contain 12 neural fits')
        require(mandatory == spec['mandatory_neural_fits'] == 8, 'Clean core must contain eight neural fits')
        require(spec['calibration_variants_share_neural_weights'] is True, 'Calibration is not an extra neural fit')
        for family, floor in [('upstream', 8000), ('events', 2000)]:
            cfg = spec[family]
            require(cfg['minimum_updates'] >= floor and min(cfg['update_choices']) >= floor, f'{family} cannot silently undercut its training floor')
            require(cfg['warmup_fraction'] <= 0.1 and cfg['warmup_cap'] < cfg['minimum_updates'], f'{family} warmup must leave a real training/decay interval')
        event = spec['events']
        require(event['schedule'] == 'sequential_linear_warmup_then_cosine', 'Warmup and decay must be sequential')
        require(event['identity_prefix_fraction'] <= 0.1, 'The identity prefix must not consume event training')
        require(sum(event['R_joint_groups'].values()) == sum(event['N_joint_groups'].values()) == event['effective_batch_groups'] == 32, 'Matched group budgets must total 32')
        require(event['N_joint_groups']['uniform_supported_negative_event'] > 0 and event['N_joint_groups']['hard_supported_negative_event'] > 0, 'N must train on supported negative-only events')
        require(spec['calibration']['ordinary_anchor_temporal_subsampling'] is False, 'No deterministic 1/9 calibration substitute')
        r = spec['resources']
        require(sum(r['initial_allocations_hours'].values()) == r['gpu_lease_hours_cap'] == 72, 'Resource allocations must sum to the 72-hour cap')
        require(r['minimum_inference_and_cold_reserve_hours'] >= 15, 'Reserve full inference and cold proof')
        require(r['optional_operational_may_reduce_clean_horizon'] is False and r['paid_compute_allowed'] is False, 'Preserve clean priority and local-only compute')
        a = spec['adoption']
        require(a['target_score'] == 0.95 and a['target_required_each_direction_each_seed'] is True, 'The milestone requires every direction and seed')
        require(a['replace_P0_automatically'] is False, 'P0 must not be automatically replaced')
        require(spec['stages'] == ['S00', 'S10', 'S20', 'S30', 'S40', 'S50', 'S60'], 'Complete execution stages required')
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f'Malformed registry: {exc}')
    return errors


def check_documents(spec: dict, root: Path) -> list[str]:
    errors = []
    for name in spec['required_documents']:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f'Missing or invalid required document: {name}')
            continue
        if path.suffix != '.md':
            continue
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            link = (path.parent / target.split('#', 1)[0]).resolve()
            if not link.is_relative_to(root.resolve()) or not link.exists():
                errors.append(f'Broken relative link: {name} -> {target}')
    return errors


def self_test(spec: dict) -> bool:
    class Contracts(unittest.TestCase):
        def test_registered_plan(self):
            self.assertEqual(validate(spec), [])

        def test_rejects_broken_contracts(self):
            mutations = [
                ('target access', lambda s: s.update(target_fitting_allowed=True)),
                ('inherited weights', lambda s: s.update(clean_exposed_dependencies_allowed=True)),
                ('target selection', lambda s: s.update(target_based_selection_allowed=True)),
                ('one seed', lambda s: s.update(seeds=[20260918])),
                ('one direction', lambda s: s['directions'].pop()),
                ('missing clips', lambda s: s['expected_clips'].update({'6bba': 127})),
                ('undertraining', lambda s: s['events'].update(update_choices=[178])),
                ('concurrent decay', lambda s: s['events'].update(schedule='concurrent')),
                ('positive-only N', lambda s: s['events']['N_joint_groups'].update(uniform_supported_negative_event=0)),
                ('no evaluation reserve', lambda s: s['resources'].update(minimum_inference_and_cold_reserve_hours=0)),
                ('unbounded cost', lambda s: s['resources'].update(gpu_lease_hours_cap=999)),
                ('automatic promotion', lambda s: s['adoption'].update(replace_P0_automatically=True)),
                ('fabricated result', lambda s: s.update(measured_results_claimed=True)),
                ('incomplete stages', lambda s: s['stages'].pop()),
                ('malformed registry', lambda s: s.pop('primary')),
            ]
            for label, mutate in mutations:
                with self.subTest(contract=label):
                    broken = copy.deepcopy(spec)
                    mutate(broken)
                    self.assertTrue(validate(broken))

    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)).wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        spec = json.loads((HERE / 'study.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f'Cannot read registry: {exc}', file=sys.stderr)
        return 2
    if args.self_test:
        return 0 if self_test(spec) else 1
    root = HERE.parents[1]
    errors = validate(spec)
    if not errors:
        errors.extend(check_documents(spec, root))
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print('Plan valid: 8 mandatory / 12 maximum neural fits; all handover links resolve.')
    print('Planning validation only. No data, dependency, training or score claim.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
