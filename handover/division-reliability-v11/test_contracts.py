"""Planning/reference tests; no real images or GPU claims."""
import copy
import json
import math
from pathlib import Path
import unittest

from contracts import (aggregate, audit_model_ancestry, canonical_logits, fork_gain,
                       learning_rate, logsumexp, masked_pair_loss, occurrence_target,
                       validate_registry)


class SparseTests(unittest.TestCase):
    def test_supported_positive(self):
        self.assertEqual(occurrence_target([1, -1, 0]), 1)

    def test_supported_negative(self):
        self.assertEqual(occurrence_target([0, 0]), 0)

    def test_unknown_prevents_negative(self):
        self.assertEqual(occurrence_target([0, -1]), -1)

    def test_incomplete_groups_unknown(self):
        self.assertEqual(occurrence_target([0, 0], complete=False), -1)
        self.assertEqual(occurrence_target([1, 0], complete=False), -1)

    def test_empty_unknown(self):
        self.assertEqual(occurrence_target([]), -1)

    def test_invalid_labels(self):
        for labels in ([2], [True], [0.0]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                occurrence_target(labels)

    def test_unknown_loss_invariant(self):
        self.assertEqual(masked_pair_loss([1, 2, 1e4], [1, 0, -1]),
                         masked_pair_loss([1, 2, -1e4], [1, 0, -1]))

    def test_negative_only_ranking_zero(self):
        self.assertEqual(masked_pair_loss([3, 4], [0, 0]), 0)

    def test_all_unknown_loss_zero(self):
        self.assertEqual(masked_pair_loss([5, 4], [-1, -1]), 0)

    def test_compatible_positives_are_set(self):
        self.assertAlmostEqual(masked_pair_loss([0, 0, 0], [1, 1, 0]), math.log(1.5))


class AlgebraTests(unittest.TestCase):
    def test_stable_logsumexp(self):
        self.assertAlmostEqual(logsumexp([1000, 1000]), 1000 + math.log(2))

    def test_finite_required(self):
        with self.assertRaises(ValueError):
            logsumexp([float('nan')])

    def test_empty_actions_rejected(self):
        with self.assertRaises(ValueError):
            logsumexp([])

    def test_duplicate_invariance(self):
        self.assertEqual(canonical_logits([('a', 1), ('a', 1), ('b', 0)]),
                         canonical_logits([('b', 0), ('a', 1)]))

    def test_duplicate_conflict(self):
        with self.assertRaises(ValueError):
            canonical_logits([('a', 1), ('a', 2)])

    def test_order_invariance(self):
        self.assertEqual(fork_gain(4, {'a': 2, 'b': 1}, 'a', 1, [0], 2),
                         fork_gain(4, {'b': 1, 'a': 2}, 'a', 1, [0], 2))

    def test_action_competition(self):
        one = fork_gain(4, {'a': 2}, 'a', 0, [0], 2)
        two = fork_gain(4, {'a': 2, 'b': 2}, 'a', 0, [0], 2)
        self.assertAlmostEqual(one - two, math.log(2))

    def test_no_fork_competitor(self):
        self.assertAlmostEqual(fork_gain(4, {'a': 2}, 'a', 1, [5], 2), -2)

    def test_keep_zero_is_competitor(self):
        self.assertAlmostEqual(fork_gain(4, {'a': 2}, 'a', 1, [-9], 2), 3)

    def test_missing_action(self):
        with self.assertRaises(ValueError):
            fork_gain(1, {'b': 2}, 'a', 0, [], 2)


class ScheduleAndMetricTests(unittest.TestCase):
    def test_schedule_peak(self):
        self.assertAlmostEqual(learning_rate(100, 2000), 3e-4)

    def test_schedule_endpoints(self):
        self.assertAlmostEqual(learning_rate(1, 2000), 3e-6)
        self.assertAlmostEqual(learning_rate(2000, 2000), 1e-6)

    def test_sequential_decay(self):
        values = [learning_rate(i, 2000) for i in range(100, 2001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_invalid_step(self):
        with self.assertRaises(ValueError):
            learning_rate(0, 2000)

    def test_official_weighting_not_clip_mean(self):
        rows = [dict(edge_tp=9, edge_fp=1, edge_fn=0, adjusted_edge_jaccard=.9,
                     division_tp=1, division_fp=0, division_fn=0),
                dict(edge_tp=0, edge_fp=1, edge_fn=0, adjusted_edge_jaccard=0.,
                     division_tp=0, division_fp=2, division_fn=1)]
        self.assertAlmostEqual(aggregate(rows, empty_edge=0, empty_division=0), 9/11 + .025)

    def test_explicit_empty_conventions(self):
        self.assertAlmostEqual(aggregate([], empty_edge=.5, empty_division=.25), .525)

    def test_invalid_count(self):
        row = dict(edge_tp=-1, edge_fp=0, edge_fn=0, adjusted_edge_jaccard=0,
                   division_tp=0, division_fp=0, division_fn=0)
        with self.assertRaises(ValueError):
            aggregate([row], empty_edge=0, empty_division=0)


class AncestryTests(unittest.TestCase):
    def graph(self):
        return {'model': dict(exposure='source_only', embryos_read=['44b6'],
                              parents=['code'], sha256='a'*64),
                'code': dict(exposure='weight_free_code', embryos_read=[],
                             parents=[], sha256='b'*64)}

    def test_valid_source_chain(self):
        self.assertEqual(audit_model_ancestry(self.graph(), 'model', '44b6'), {'model', 'code'})

    def test_exposed_parent(self):
        g = self.graph(); g['code']['exposure'] = 'known_target_exposure'
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')

    def test_hidden_target_read(self):
        g = self.graph(); g['model']['embryos_read'].append('6bba')
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')

    def test_missing_parent(self):
        g = self.graph(); del g['code']
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')

    def test_cycle(self):
        g = self.graph(); g['code']['parents'] = ['model']
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')

    def test_missing_access_manifest(self):
        g = self.graph(); del g['model']['embryos_read']
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')

    def test_missing_hash(self):
        g = self.graph(); g['model']['sha256'] = ''
        with self.assertRaises(ValueError):
            audit_model_ancestry(g, 'model', '44b6')


class RegistryTests(unittest.TestCase):
    def study(self):
        return json.loads((Path(__file__).parent / 'study.json').read_text())

    def test_registry(self):
        validate_registry(self.study())

    def test_reject_weakened_plan(self):
        mutations = [('seeds', [20260918]), ('arms', ['C00', 'C11']),
                     ('target_selection', True), ('v10_results_observed_remotely', True),
                     ('event_update_options', [178]), ('max_new_neural_fits', 12),
                     ('status', 'complete'), ('operational_neural_matrix', True),
                     ('primary_comparison', ['C11', 'C00']),
                     ('raw_data_or_weights_committed', True)]
        for key, value in mutations:
            g = self.study(); g[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_registry(g)

    def test_reserve(self):
        g = copy.deepcopy(self.study()); g['budget']['inference_reserve_hours'] = 1
        with self.assertRaises(ValueError):
            validate_registry(g)


if __name__ == '__main__':
    unittest.main()
