"""Ensure the evaluation-only precision correction cannot alter fitting."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from . import execution_precision as precision
from .common import Blocked


class PrecisionTests(unittest.TestCase):
    def test_only_compact_evaluation_selects_override(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);repair=root/'repair.json';repair.write_text('{"status":"passed"}')
            with patch.object(precision,'WORK',root),patch.object(precision,'REPAIR',repair),patch.dict(os.environ):
                for stage,arm in [('train-upstream','C00'),('train-compact','C11'),('predict','C00'),('predict','C01'),('calibrate','C01'),('run','C00')]:
                    os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
                    self.assertIsNone(precision.configure(stage,arm))
                    self.assertNotIn('NVIDIA_TF32_OVERRIDE',os.environ)
                for stage in ('mine','predict','calibrate','calibration-predict'):
                    os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
                    r=precision.configure(stage,'C11')
                    self.assertEqual(os.environ['NVIDIA_TF32_OVERRIDE'],'0')
                    self.assertTrue(r['applied_before_numerical_import'])

    def test_explicit_package_selects_cold_precision(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);repair=root/'repair.json';repair.write_text('{"status":"passed"}')
            package=root/'package';package.mkdir()
            with patch.object(precision,'WORK',root),patch.object(precision,'REPAIR',repair),patch.dict(os.environ):
                os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
                (package/'manifest.json').write_text('{"arm":"C01"}')
                self.assertIsNone(precision.configure('infer','C00',package))
                self.assertNotIn('NVIDIA_TF32_OVERRIDE',os.environ)
                (package/'manifest.json').write_text('{"arm":"C11"}')
                self.assertIsNotNone(precision.configure('infer','C00',package))
                self.assertEqual(os.environ['NVIDIA_TF32_OVERRIDE'],'0')

    def test_late_configuration_is_rejected(self):
        with patch.dict(sys.modules,{'torch':object()}):
            with self.assertRaisesRegex(Blocked,'before importing torch'):precision.configure('mine','C11')


if __name__=='__main__':unittest.main()
