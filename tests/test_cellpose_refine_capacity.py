"""Physical-grid and candidate-assignment checks for the diagnostic bound."""
import unittest

import numpy as np

from tools.analyze_cellpose_refine_capacity import capacity, feasible_distances


class IntegerCapacity(unittest.TestCase):
    def test_two_axial_planes_exceed_three_um_movement(self):
        base = np.array([[20., 100., 100.]])
        target = np.array([[22., 100., 100.]])
        d = feasible_distances(base, target, [64, 256, 256])
        self.assertEqual(d[0, 0], 1.625)
        self.assertEqual(capacity(d, 1), 0)
        self.assertEqual(capacity(d, 2), 1)

    def test_lateral_movement_and_one_candidate_for_two_targets(self):
        base = np.array([[20., 100., 100.]])
        targets = np.array([[20., 100., 108.], [20., 100., 92.]])
        d = feasible_distances(base, targets, [64, 256, 256])
        np.testing.assert_array_equal(d[:, 0], [.40625, .40625])
        self.assertEqual(capacity(d, 1), 1)

    def test_zero_movement_and_empty_inputs(self):
        d = feasible_distances(np.array([[0., 0., 0.]]), np.array([[0., 0., 1.]]), [64, 256, 256], movement=0)
        self.assertEqual(d[0, 0], .40625)
        self.assertEqual(capacity(np.empty((2, 0)), 7), 0)


if __name__ == "__main__":
    unittest.main()
