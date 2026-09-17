"""Run directly with Python; no scientific runtime or local microscopy required."""
import math
import unittest

from contracts import P0, P0_EDGE, lr_factor, required_divisions, training_receipt_errors


class PlanningContracts(unittest.TestCase):
    def test_baseline_decomposition(self):
        self.assertAlmostEqual(P0_EDGE + .1*29/243, P0, places=14)

    def test_target_counts(self):
        for fp, tp in [(92, 66), (102, 69), (112, 72), (132, 77)]:
            row = required_divisions(false_positives=fp)
            self.assertEqual(row['required_tp'], tp)
            self.assertGreaterEqual(row['score_at_required_tp'], .95)
            self.assertLess(P0_EDGE + .1*(tp-1)/(151+fp), .95)

    def test_infeasible_not_fabricated(self):
        row = required_divisions(target=2.)
        self.assertFalse(row['feasible_by_counts'])
        self.assertIsNone(row['score_at_required_tp'])

    def test_invalid_counts(self):
        for fp in [-1, True, 2.5]:
            with self.assertRaises(ValueError):
                required_divisions(false_positives=fp)

    def test_nonfinite_score(self):
        with self.assertRaises(ValueError):
            required_divisions(target=math.nan)

    def test_schedule_warmup(self):
        self.assertEqual(lr_factor(0), 1/128)
        self.assertEqual(lr_factor(127), 1.)

    def test_schedule_plateau(self):
        self.assertEqual(lr_factor(128), 1.)
        self.assertEqual(lr_factor(2048), 1.)
        self.assertEqual(lr_factor(3276), 1.)

    def test_schedule_decay(self):
        values = [lr_factor(i) for i in range(3277, 4096)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], .1)

    def test_bad_schedule(self):
        for step in [-1, 4096, True]:
            with self.assertRaises(ValueError):
                lr_factor(step)
        with self.assertRaises(ValueError):
            lr_factor(0, total=178)

    def test_178_update_run_fails_receipt(self):
        errors = training_receipt_errors({'joint_optimizer_updates': 178})
        self.assertTrue(any('joint_optimizer_updates' in e for e in errors))
        self.assertTrue(any('negative_only_groups_seen' in e for e in errors))

    def test_complete_schema_not_execution_claim(self):
        row = dict(joint_optimizer_updates=4096, positive_groups_seen=15,
                   negative_only_groups_seen=100, unknown_as_negative_count=0,
                   new_target_used_for_training=False)
        for key in ['full_source_screen', 'literal_zero_path_test',
                    'counterfactual_label_parity', 'finite_gradient_checks',
                    'random_background_stream', 'checkpoint_resume_parity']:
            row[key] = True
        self.assertEqual(training_receipt_errors(row), [])
        row['unknown_as_negative_count'] = 1
        self.assertTrue(training_receipt_errors(row))

    def test_bypass_not_zero_path(self):
        errors = training_receipt_errors({'literal_zero_path_test': False})
        self.assertTrue(any('literal_zero_path_test' in e for e in errors))


if __name__ == '__main__':
    unittest.main(verbosity=2)
