"""Synthetic reference tests only; no claim of native/HOCT/official integration."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contracts import (canonical_fork, division_budget, division_scenario,
    equivalent_support_score, raw_fork_advantage, supported_edge_labels, validate_lineage)
from preflight import inspect


class SparseLabels(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(supported_edge_labels([(10, 20)], {10: 1, 20: 2}, [(1, 2)]), [1])
    def test_competing_parent(self):
        self.assertEqual(supported_edge_labels([(10, 20), (11, 20)], {10: 1, 20: 2}, [(1, 2)]), [1, 0])
    def test_unknown_second_daughter(self):
        self.assertEqual(supported_edge_labels([(10, 20), (10, 21)], {10: 1, 20: 2}, [(1, 2)]), [1, -1])
    def test_missing_parent_is_censored(self):
        self.assertEqual(supported_edge_labels([(11, 20)], {20: 2}, [(1, 2)]), [-1])
    def test_parent_outside_candidate_bank(self):
        self.assertEqual(supported_edge_labels([(11, 20)], {10: 1, 20: 2}, [(1, 2)]), [-1])
    def test_two_daughters_are_both_positive(self):
        self.assertEqual(supported_edge_labels([(10, 20), (10, 21)], {10: 1, 20: 2, 21: 3}, [(1, 2), (1, 3)]), [1, 1])
    def test_empty(self):
        self.assertEqual(supported_edge_labels([], {}, []), [])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): supported_edge_labels([(1, 2), (1, 2)], {}, [])
    def test_nonunique_match_rejected(self):
        with self.assertRaises(ValueError): supported_edge_labels([], {1: 3, 2: 3}, [])
    def test_gt_merge_rejected(self):
        with self.assertRaises(ValueError): supported_edge_labels([], {}, [(1, 3), (2, 3)])
    def test_self_edge_rejected(self):
        with self.assertRaises(ValueError): supported_edge_labels([(1, 1)], {}, [])
    def test_negative_id_rejected(self):
        with self.assertRaises(ValueError): supported_edge_labels([(-1, 2)], {}, [])


class TemporalContracts(unittest.TestCase):
    def test_daughter_permutation(self):
        self.assertEqual(canonical_fork(1, [3, 2]), canonical_fork(1, [2, 3]))
    def test_bad_fork(self):
        for children in ([1, 2], [2, 2], [2], [2, 3, 4]):
            with self.subTest(children=children), self.assertRaises(ValueError): canonical_fork(1, children)
    def test_duplicate_support_invariance(self):
        a = [("pathA", 2.0), ("pathB", 3.0)]
        self.assertEqual(equivalent_support_score(a), equivalent_support_score(a * 100))
    def test_order_invariance(self):
        a = [("A", 1), ("B", -1)]
        self.assertEqual(equivalent_support_score(a), equivalent_support_score(a[::-1]))
    def test_stable_large_logits(self):
        self.assertAlmostEqual(equivalent_support_score([("A", 1000), ("B", 1000)]), 1000)
    def test_inconsistent_duplicate_rejected(self):
        with self.assertRaises(ValueError): equivalent_support_score([("A", 1), ("A", 2)])
    def test_empty_support_rejected(self):
        with self.assertRaises(ValueError): equivalent_support_score([])
    def test_nan_support_rejected(self):
        with self.assertRaises(ValueError): equivalent_support_score([("A", float("nan"))])
    def test_old_objective_dominance(self):
        for reward in (0.0, .48, .9, 1.0): self.assertLess(raw_fork_advantage(reward, 1.2, 0), 0)
    def test_fork_can_win(self):
        self.assertGreater(raw_fork_advantage(.9, .5, .1), 0)
    def test_independent_birth_can_win(self):
        self.assertLess(raw_fork_advantage(.1, .8, .1), 0)
    def test_valid_fork_lineage(self):
        validate_lineage({1: 0, 2: 1, 3: 1, 4: 2}, [(1, 2), (1, 3), (3, 4)])
    def test_noop_graph(self):
        validate_lineage({1: 0}, [])
    def test_skip_edge_rejected(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0, 2: 2}, [(1, 2)])
    def test_merge_rejected(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0, 2: 0, 3: 1}, [(1, 3), (2, 3)])
    def test_three_daughters_rejected(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0, 2: 1, 3: 1, 4: 1}, [(1, 2), (1, 3), (1, 4)])
    def test_dangling_rejected(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0}, [(1, 2)])
    def test_duplicate_edge_rejected(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0, 2: 1}, [(1, 2), (1, 2)])
    def test_exclusive_observations(self):
        with self.assertRaises(ValueError): validate_lineage({1: 0, 2: 0}, [], [(1, 2)])
    def test_unselected_exclusion_member(self):
        validate_lineage({1: 0}, [], [(1, 2)])


class BudgetsAndPreflight(unittest.TestCase):
    def test_baseline_budget(self):
        b = division_budget(.934802374260586, 29, 92, 122)
        self.assertAlmostEqual(b['delta_required'], .015197625739414)
        self.assertAlmostEqual(b['adjusted_edge_contribution'], .9228682178819851)
        self.assertAlmostEqual(b['required_division_jaccard_if_edges_unchanged'], .2713178211801481)
    def test_scenario_crosses_target(self):
        b = division_budget(.934802374260586, 29, 92, 122)
        self.assertAlmostEqual(division_scenario(b, 66, 92)['conditional_score'], .9500287117091456)
    def test_baseline_scenario_identity(self):
        b = division_budget(.934802374260586, 29, 92, 122)
        self.assertAlmostEqual(division_scenario(b, 29, 92)['delta'], 0)
    def test_bad_counts(self):
        for tp in (-1, True, 1.5):
            with self.subTest(tp=tp), self.assertRaises(ValueError): division_budget(.9, tp, 1, 1)
    def test_no_gt_divisions(self):
        with self.assertRaises(ValueError): division_budget(.9, 0, 1, 0)
    def test_too_many_tp(self):
        b = division_budget(.9, 1, 1, 1)
        with self.assertRaises(ValueError): division_scenario(b, 3, 1)
    def test_preflight_is_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); before = list(root.rglob('*'))
            r = inspect(root, root / 'absent_studies', root / 'absent_data')
            self.assertEqual(before, list(root.rglob('*')))
            self.assertFalse(r['writes_performed'])
            self.assertEqual(r['status'], 'missing_local_dependencies')
            self.assertLessEqual(r['proposed_new_output_cap_gib'], max(0., r['free_gib'] - 8.))
    def test_foreign_cwd_help(self):
        script = Path(__file__).resolve().with_name('preflight.py')
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, '-B', str(script), '--help'], cwd=tmp, capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr)
    def test_recorded_baseline_consistent(self):
        root = Path(__file__).resolve().parent
        baseline = json.loads((root / 'baseline.json').read_text())
        cfg = json.loads((root / 'config.json').read_text())
        b = division_budget(baseline['score'], **baseline['division_counts'], target=cfg['target_score'])
        self.assertEqual(baseline['edge_counts']['tp'] + baseline['edge_counts']['fn'], 128883)
        self.assertAlmostEqual(b['adjusted_edge_contribution'], baseline['adjusted_edge_contribution'])
        self.assertEqual(cfg['base_commit'], baseline['reviewed_commit'])


if __name__ == '__main__': unittest.main()
