"""Synthetic source-contract tests only; no microscopy/model claims."""
import ast
import unittest
from baseline_contract import inspect_source, make_candidate


class ContractTests(unittest.TestCase):
    def test_one_change(self):
        source = 'import os\nOUTPUT_MOTION_RELINK = os.getenv("BIOHUB_OUTPUT_MOTION_RELINK", "1") == "1"\nOTHER = 4\n'
        candidate, receipt = make_candidate(source)
        self.assertEqual(candidate, 'import os\nOUTPUT_MOTION_RELINK = False\nOTHER = 4\n')
        self.assertFalse(receipt['runtime_parity_tested'])
        self.assertFalse(receipt['leaderboard_validated'])
        self.assertEqual(len(receipt['scientific_changes']), 1)

    def test_multiline(self):
        candidate, _ = make_candidate('OUTPUT_MOTION_RELINK = (\n    True\n)\n')
        self.assertIn('False', candidate)
        ast.parse(candidate)

    def test_unicode_and_crlf(self):
        source = '# célula\r\nOUTPUT_MOTION_RELINK: bool = True  # intact\r\n'
        candidate, _ = make_candidate(source)
        self.assertEqual(candidate, source.replace('= True', '= False'))

    def test_duplicate_write(self):
        with self.assertRaises(ValueError):
            make_candidate('OUTPUT_MOTION_RELINK=True\nOUTPUT_MOTION_RELINK=True\n')

    def test_nested_write(self):
        with self.assertRaises(ValueError):
            make_candidate('OUTPUT_MOTION_RELINK=True\ndef f():\n global OUTPUT_MOTION_RELINK\n OUTPUT_MOTION_RELINK=True\n')

    def test_already_disabled(self):
        with self.assertRaises(ValueError):
            make_candidate('OUTPUT_MOTION_RELINK=False\n')

    def test_missing_flag(self):
        with self.assertRaises(ValueError):
            make_candidate('OTHER=True\n')

    def test_no_execution(self):
        receipt = inspect_source('raise RuntimeError("MUST NOT EXECUTE")\nOUTPUT_MOTION_RELINK=True\n')
        self.assertFalse(receipt['notebook_executed'])

    def test_other_flags_preserved(self):
        source = 'OUTPUT_GAP_CLOSE=True\nOUTPUT_MOTION_RELINK=True\nOUTPUT_SAFE_DIVISIONS=True\n'
        candidate, _ = make_candidate(source)
        self.assertIn('OUTPUT_GAP_CLOSE=True', candidate)
        self.assertIn('OUTPUT_SAFE_DIVISIONS=True', candidate)

    def test_delete_rejected(self):
        with self.assertRaises(ValueError):
            make_candidate('OUTPUT_MOTION_RELINK=True\ndel OUTPUT_MOTION_RELINK\n')


if __name__ == '__main__':
    unittest.main()
