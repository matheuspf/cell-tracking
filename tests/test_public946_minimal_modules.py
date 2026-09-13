import collections
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from public946_minimal.modules import (context_windows, fork_smooth, gated_relink, inverse_phase,
    localize_originals, median_views, parabola_offset, peak_offsets, trilinear_features)
from public946_minimal.neural import Evidence, forward_view, inverse_view, predictor_source
from public946_minimal.worker import validate


def edge(a, b):
    return dict(source_id=a, target_id=b, edge_prob=.6)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {i: {} for i in range(4)}
        self.before = [edge(0, 2), edge(1, 3)]
        self.after = [edge(0, 3), edge(1, 2)]

    def test_atomic_swap(self):
        p = {(0, 2): .2, (1, 3): .3, (0, 3): .7, (1, 2): .8}
        result, records = gated_relink(self.nodes, self.before, self.after, p)
        self.assertEqual(result, self.after)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['accepted'])

    def test_tie_missing_and_degree_change_abstain(self):
        for p, after in [({(0, 2): .5, (1, 3): .5, (0, 3): .5, (1, 2): .5}, self.after),
                         ({(0, 2): .1}, self.after),
                         ({(0, 2): .1, (1, 3): .1, (0, 3): .9}, [edge(0, 3)])]:
            result, records = gated_relink(self.nodes, self.before, after, p)
            self.assertEqual(result, self.before)
            self.assertFalse(any(r['accepted'] for r in records))

    def test_fork_is_rejected(self):
        nodes = {i: {} for i in range(6)}
        before = self.before + [edge(0, 4), edge(1, 5)]
        after = self.after + [edge(0, 4), edge(1, 5)]
        result, records = gated_relink(nodes, before, after, {(0, 2): .1, (1, 3): .1, (0, 3): .9, (1, 2): .9})
        self.assertEqual(result, before)
        self.assertFalse(records[0]['no_fork'])

    def test_order_invariant_components(self):
        p = {(0, 2): .2, (1, 3): .3, (0, 3): .7, (1, 2): .8}
        _, a = gated_relink(self.nodes, self.before, self.after, p)
        _, b = gated_relink(self.nodes, self.before[::-1], self.after[::-1], p)
        self.assertEqual(a, b)


class LocalizationTests(unittest.TestCase):
    def test_shifted_parabola_and_half_tie(self):
        for displacement in (-.5, -.25, 0., .2, .5):
            h = [4 - (q - displacement)**2 for q in (-1, 0, 1)]
            self.assertAlmostEqual(float(parabola_offset(*h)), displacement)

    def test_flat_convex_nonmaximum_nonfinite(self):
        for h in [(1, 1, 1), (1, 0, 1), (0, 1, 2), (0, float('nan'), 0)]:
            self.assertEqual(float(parabola_offset(*h)), 0.)

    def test_border_anisotropy_and_relocated_nodes(self):
        z, y, x = np.meshgrid(*[np.arange(5)]*3, indexing='ij')
        response = 30 - (z - 2.25)**2 - (y - 1.8)**2 - (x - 2.5)**2
        offsets, supported = peak_offsets(response, [[2, 2, 2], [0, 0, 0]])
        np.testing.assert_allclose(offsets[0], [.25, -.2, .5])
        np.testing.assert_array_equal(offsets[1], 0)
        original = {0: dict(z=2., y=8., x=8.), 1: dict(z=2., y=8., x=8.)}
        nodes = copy.deepcopy(original)
        nodes[1]['x'] = 9.
        nodes[2] = dict(z=1., y=1., x=1.)
        r = localize_originals(nodes, original, {0: offsets[0], 1: offsets[0]}, [1, 4, 4])
        self.assertEqual(r['eligible_originals'], 1)
        np.testing.assert_allclose([nodes[0][a] for a in 'zyx'], [2.25, 7.2, 10.])
        self.assertEqual(nodes[1]['x'], 9.)
        self.assertEqual(nodes[2]['x'], 1.)


class FeatureTests(unittest.TestCase):
    def test_affine_field_and_channels(self):
        z, y, x = torch.meshgrid(torch.arange(4.), torch.arange(5.), torch.arange(6.), indexing='ij')
        f = torch.stack([2*z + 3*y + 5*x, torch.ones_like(x)*7])[None]
        q = torch.tensor([[[1.25, 2.5, 3.1], [0, 0, 0], [3, 4, 5]]])
        mask = torch.tensor([[True, True, False]])
        actual = trilinear_features(f, q, mask)
        torch.testing.assert_close(actual, torch.tensor([[[25.5, 7], [0, 7], [0, 0]]]))

    def test_integer_corners_singleton_and_masks(self):
        f = torch.arange(12.).reshape(1, 2, 1, 2, 3)
        q = torch.tensor([[[0., 0, 0], [0, 1, 2], [0, 0, 1]]])
        mask = torch.tensor([[True, True, False]])
        actual = trilinear_features(f, q, mask)
        torch.testing.assert_close(actual, torch.tensor([[[0., 6], [5, 11], [0, 0]]]), rtol=0, atol=0)

    def test_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):
            trilinear_features(torch.ones(1, 1, 2, 2, 2), torch.tensor([[[1.5, 0, 0]]]), torch.ones(1, 1, dtype=torch.bool))


class ViewTests(unittest.TestCase):
    def test_all_view_inverses_and_reorder(self):
        x = torch.arange(2*3*5.).reshape(2, 3, 5)
        for view in range(8):
            torch.testing.assert_close(inverse_view(forward_view(x, view), view), x, rtol=0, atol=0)

    def test_even_median_outlier_and_order(self):
        views = [torch.full((2, 3), float(x)) for x in (1, 1, 2, 3, 4, 5, 6, 100)]
        torch.testing.assert_close(median_views(views), torch.full((2, 3), 3.5))
        torch.testing.assert_close(median_views(views[::-1]), median_views(views))
        torch.testing.assert_close(median_views([views[0]]*8), views[0])

    def test_masked_median(self):
        views = [torch.tensor([1., 10.]), torch.tensor([3., 2.]), torch.tensor([100., 4.])]
        masks = [torch.tensor([1, 0], dtype=torch.bool), torch.tensor([1, 1], dtype=torch.bool), torch.tensor([0, 1], dtype=torch.bool)]
        torch.testing.assert_close(median_views(views, masks), torch.tensor([2., 3.]))
        with self.assertRaises(ValueError):
            median_views(views, [torch.zeros(2, dtype=torch.bool)]*3)

    def test_phase_constant_odd_shapes_and_support(self):
        a, valid = inverse_phase(torch.ones(1, 1, 3, 5, 7)*4, [0, .5, .5])
        torch.testing.assert_close(a, torch.ones_like(a)*4)
        self.assertEqual(int(valid.sum()), 3*4*6)
        self.assertFalse(bool(valid[:, -1, :].any()))

    def test_phase_inverse_sign_and_peak(self):
        z, y, x = torch.meshgrid(torch.arange(3.), torch.arange(7.), torch.arange(9.), indexing='ij')
        response = 5*y + 2*x
        actual, valid = inverse_phase(response, [0, .5, .5])
        torch.testing.assert_close(actual[valid], (response + 3.5)[valid])
        impulse = torch.zeros(1, 1, 3, 7, 9)
        impulse[0, 0, 1, 3, 4] = 1
        moved, _ = inverse_phase(impulse, [0, 1, 1])
        self.assertEqual(float(moved[0, 0, 1, 2, 3]), 1.)

    def test_phase_image_padding_no_wrap(self):
        raw = np.arange(1*1*5*7, dtype=np.float32).reshape(1, 1, 5, 7)
        result = Evidence([]).load_frame(None, raw, 0, [1, 3, 4], [1, 2, 2], phase=[0, 1, 1])
        self.assertEqual(float(result[0, 0, 0]), float(raw[0, 0, 1, 1]))

    def test_reflection_raw_coordinates_double_inverse(self):
        for size in (255, 256, 257):
            q = np.array([0., 12., 63.])
            transformed = (size - 1)/4 - q
            np.testing.assert_allclose((size - 1)/4 - transformed, q)


class ContextTests(unittest.TestCase):
    def test_actual_two_frame_context_identity(self):
        for length in (2, 3, 7, 100):
            for t in range(length - 1):
                self.assertEqual(context_windows(length, 2, t), [t])

    def test_coverage_and_boundaries(self):
        self.assertEqual(context_windows(7, 5, 3), [0, 1, 2])
        self.assertEqual(context_windows(7, 5, 0), [0])
        self.assertEqual(context_windows(7, 5, 5), [2])
        self.assertEqual(context_windows(1, 2, 0), [])
        p = np.array([[.2, .8], [.8, .2]])
        np.testing.assert_array_equal(np.stack([p]*4).mean(0), p)


class SmoothTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {i: dict(t=i, z=float(i), y=float(i*2), x=1.) for i in range(7)}
        self.edges = [edge(i, i+1) for i in range(6)]

    def test_translating_straight_track(self):
        original = copy.deepcopy(self.nodes)
        result = fork_smooth(self.nodes, self.edges, collections.defaultdict(int))
        for i in original:
            np.testing.assert_allclose([result[i][a] for a in 'zyx'], [original[i][a] for a in 'zyx'], atol=1e-12)

    def test_y_branch_short_daughter_anchors_and_edges(self):
        nodes = copy.deepcopy(self.nodes)
        nodes[7] = dict(t=4, z=30., y=1., x=1.)
        nodes[3]['z'] = 20.
        nodes[2]['z'] = 10.
        edges = self.edges + [edge(3, 7)]
        previous_edges = copy.deepcopy(edges)
        original = copy.deepcopy(nodes)
        stats = collections.defaultdict(int)
        result = fork_smooth(nodes, edges, stats)
        for i in (2, 3, 4, 7):
            self.assertEqual(result[i], original[i])
        self.assertEqual(edges, previous_edges)
        self.assertEqual(stats['E07_anchors'], 4)

    def test_consecutive_forks_and_inserted_node(self):
        self.nodes[7] = dict(t=4, z=10., y=1., x=1.)
        self.nodes[8] = dict(t=5, z=12., y=1., x=1.)
        self.edges += [edge(3, 7), edge(4, 8)]
        original = copy.deepcopy(self.nodes)
        result = fork_smooth(self.nodes, self.edges, collections.defaultdict(int))
        for i in (2, 3, 4, 5, 7, 8):
            self.assertEqual(result[i], original[i])


class ValidationTests(unittest.TestCase):
    def test_graph_checks(self):
        nodes = np.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.assertEqual(validate(nodes, np.array([[0, 1]]), [2, 3, 3, 3])['edges'], 1)
        for edges in (np.array([[0, 2]]), np.array([[1, 0]]), np.array([[0, 1], [0, 1]])):
            with self.assertRaises(ValueError):
                validate(nodes, edges, [2, 3, 3, 3])


if __name__ == '__main__':
    unittest.main()
