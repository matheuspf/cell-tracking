import copy
import dis
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from public946_minimal.common import DEFAULT_ARCHIVE, DEFAULT_OUT, REPO
from public946_minimal.selection import eligible, ranking
from public946_minimal.source import repair_namespace
from public946_minimal.neural import predictor_source


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(completed=list(range(199)), failed=[], metric_revision='same',
            pooled=dict(score=.9, division_jaccard=.2),
            per_embryo={'a': dict(score=.9), 'b': dict(score=.9)})
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate['pooled']['score'] += .001
        for v in self.candidate['per_embryo'].values():
            v['score'] += .001
        self.noise = dict(score=0., division_jaccard=0.)

    def test_full_positive_gate(self):
        self.assertTrue(eligible(self.candidate, self.baseline, self.noise)['eligible'])

    def test_pilot_and_one_embryo_gain_cannot_qualify(self):
        pilot = copy.deepcopy(self.candidate)
        pilot['completed'] = list(range(4))
        self.assertFalse(eligible(pilot, self.baseline, self.noise)['eligible'])
        self.candidate['per_embryo']['b']['score'] = .899
        self.assertFalse(eligible(self.candidate, self.baseline, self.noise)['eligible'])

    def test_division_loss_and_numerical_tie_rejected(self):
        self.candidate['pooled']['division_jaccard'] -= .01
        self.assertFalse(eligible(self.candidate, self.baseline, self.noise)['eligible'])
        self.assertFalse(eligible(self.baseline, self.baseline, self.noise)['eligible'])

    def test_rank_worst_embryo_then_pooled(self):
        uneven = copy.deepcopy(self.candidate)
        uneven['pooled']['score'] += .01
        uneven['per_embryo']['b']['score'] = .9001
        self.assertLess(ranking('E02', self.candidate, self.baseline, 10), ranking('E07', uneven, self.baseline, 5))


class IsolationTests(unittest.TestCase):
    def test_guard_in_fresh_process_and_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'truth.geff'
            target.write_text('must not read')
            env = os.environ.copy()
            env.update(PUBLIC946_OUT=tmp, PUBLIC946_INFERENCE='1',
                PYTHONPATH=str(REPO/'tools/public946_minimal/guard') + ':' + str(REPO/'tools'),
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
            code = 'from public946_minimal.isolation import probe; assert probe(' + repr(str(target)) + ')'
            parent = 'import subprocess,sys; ' + code + '; subprocess.run([sys.executable,"-c",' + repr(code) + '],check=True)'
            subprocess.run([sys.executable, '-c', parent], env=env, check=True, capture_output=True)


@unittest.skipUnless((DEFAULT_ARCHIVE/'biohub-harmonic-fusion.py').exists(), 'Pinned ignored source archive unavailable')
class ArchivedSourceTests(unittest.TestCase):
    def test_all_repair_global_names_resolve(self):
        import builtins
        ns = repair_namespace(DEFAULT_ARCHIVE/'biohub-harmonic-fusion.py', DEFAULT_OUT, DEFAULT_OUT, {})
        for name, value in ns.items():
            if isinstance(value, types.FunctionType) and value.__globals__ is ns:
                used = {i.argval for i in dis.get_instructions(value) if i.opname == 'LOAD_GLOBAL'}
                self.assertEqual(used - set(ns) - set(vars(builtins)), set(), name)
        self.assertTrue(ns['OUTPUT_MOTION_RELINK'])
        self.assertEqual(ns['GAP2_MAX_LINKS_ABS'], 180)

    def test_every_registered_predictor_hook_compiles(self):
        p = DEFAULT_OUT/'public_source/tracking_repo/scripts/predict_unet_transformer.py'
        if not p.exists():
            self.skipTest('Materialized public source unavailable')
        for modules in ([], ['E03'], ['E04'], ['E05'], ['E08'], ['E04', 'E05']):
            compile(predictor_source(p.read_text(), modules), str(p), 'exec')


if __name__ == '__main__':
    unittest.main()
