"""Synthetic plan-contract tests; no microscopy or model inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from study_contract import (BASE, LOCKED_FILES, freeze, validate, validate_resolution, verify)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads((BASE / 'experiment_lock.json').read_text())

    def test_valid_registry(self):
        validate(self.lock)

    def test_old_single_arm_lock_rejected(self):
        self.lock['schema_version'] = 1
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_missing_arm_rejected(self):
        self.lock['arms'].pop()
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_duplicate_arm_rejected(self):
        self.lock['arms'][3]['id'] = 'E01'
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_baseline_contamination_rejected(self):
        self.lock['arms'][0]['modules'] = ['no_motion']
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_singles_must_start_from_public(self):
        self.lock['arms'][2]['parent'] = 'B1'
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_arbitrary_stack_rejected(self):
        self.lock['arms'][10]['modules'].append('E08')
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_sweep_rejected(self):
        self.lock['parameter_sweeps_allowed'] = True
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_upload_permission_rejected(self):
        self.lock['kaggle_submission_authorized'] = True
        with self.assertRaises(ValueError):
            validate(self.lock)

    def test_valid_transfers(self):
        validate_resolution(self.lock, {'X01': 'E02', 'X02': 'C02'})
        validate_resolution(self.lock, {'X01': None, 'X02': None})

    def test_motion_transfer_rejected(self):
        with self.assertRaises(ValueError):
            validate_resolution(self.lock, {'X01': 'E01', 'X02': None})

    def test_duplicate_transfer_rejected(self):
        with self.assertRaises(ValueError):
            validate_resolution(self.lock, {'X01': 'E02', 'X02': 'E02'})

    def test_transfer_order_rejected(self):
        with self.assertRaises(ValueError):
            validate_resolution(self.lock, {'X01': None, 'X02': 'E02'})

    def test_freeze_verify_drift_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in LOCKED_FILES:
                (root / name).write_bytes((BASE / name).read_bytes())
            receipt = root / 'frozen.json'
            freeze(root, receipt)
            verify(root, receipt)
            with self.assertRaises(FileExistsError):
                freeze(root, receipt)
            (root / 'EXPERIMENTS.md').write_text('changed after registration')
            with self.assertRaises(ValueError):
                verify(root, receipt)


if __name__ == '__main__':
    unittest.main()
