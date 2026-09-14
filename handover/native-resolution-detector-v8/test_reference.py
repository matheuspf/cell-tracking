"""Synthetic tests only; no Biohub images, checkpoint or official score is loaded."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from reference import (Grid, collision_groups, support_precedence, masked_bce,
                       masked_offsets, assert_source_isolated, budget_indices)


def asset(provenance="verified_source_only", train=("A",), calibration=(), parents=()):
    return dict(provenance=provenance, evidence="fixture-only", biohub_training_embryos=list(train),
                biohub_calibration_embryos=list(calibration), parents=list(parents))


class GridTests(unittest.TestCase):
    def test_native_roundtrip(self):
        p = np.array([[0, 0, 0], [63, 255, 255], [1.2, 2.3, 4.5]])
        grid = Grid(); i, d = grid.encode(p)
        np.testing.assert_allclose(grid.decode(i, d), p)
    def test_stride_roundtrip_keeps_fraction(self):
        grid = Grid(stride_native=(1, 4, 4)); p = np.array([[4, 27, 13]])
        i, d = grid.encode(p)
        self.assertEqual(i.tolist(), [[4, 7, 3]])
        np.testing.assert_allclose(grid.decode(i, d), p)
    def test_pool_has_different_origin(self):
        stride = Grid(stride_native=(1, 4, 4)); pool = Grid((1, 4, 4), (0, 1.5, 1.5))
        np.testing.assert_allclose(pool.to_native([[1, 2, 3]]) - stride.to_native([[1, 2, 3]]), [[0, 1.5, 1.5]])
    def test_pool_boundary_roundtrip(self):
        grid = Grid((1, 4, 4), (0, 1.5, 1.5)); p = np.array([[0, 0, 0], [63, 255, 255]])
        i, d = grid.encode(p); self.assertTrue((i >= 0).all())
        np.testing.assert_allclose(grid.decode(i, d), p)
    def test_micrometer_offsets(self):
        i, d = Grid().encode(np.array([[1.25, 2.25, 3.25]]))
        np.testing.assert_allclose(d, [[.40625, .1015625, .1015625]])
    def test_collision_exposed_not_averaged(self):
        i, d = Grid((1, 4, 4)).encode(np.array([[1, 1, 1], [1, 1.5, 1.5], [3, 16, 16]]))
        self.assertEqual(collision_groups(i), [[0, 1]])
        self.assertFalse(np.array_equal(d[0], d[1]))
    def test_empty_points(self):
        i, d = Grid().encode(np.empty((0, 3))); self.assertEqual(i.shape, (0, 3))
        self.assertEqual(collision_groups(i), [])
    def test_invalid_grid(self):
        for stride in ((0, 1, 1), (-1, 1, 1), (float('nan'), 1, 1), (1, 1)):
            with self.subTest(stride=stride), self.assertRaises(ValueError): Grid(stride_native=stride)
    def test_invalid_points(self):
        with self.assertRaises(ValueError): Grid().encode(np.array([[0, np.nan, 1]]))
        with self.assertRaises(ValueError): Grid().encode(np.ones((3, 4)))
    def test_decode_shape_mismatch(self):
        with self.assertRaises(ValueError): Grid().decode(np.zeros((1, 3)), np.zeros((2, 3)))


class MaskedLossTests(unittest.TestCase):
    def test_unknown_bce_gradient_zero(self):
        p = torch.zeros(4, requires_grad=True); y = torch.tensor([1., float('nan'), 0., float('nan')])
        loss = masked_bce(p, y, torch.tensor([True, False, True, False])); loss.backward()
        torch.testing.assert_close(p.grad, torch.tensor([-.25, 0., .25, 0.]))
    def test_empty_bce_support(self):
        p = torch.full((3,), float('nan'), requires_grad=True)
        loss = masked_bce(p, p.detach(), torch.zeros(3, dtype=torch.bool)); loss.backward()
        self.assertEqual(loss.item(), 0.); torch.testing.assert_close(p.grad, torch.zeros(3))
    def test_empty_bce_weight(self):
        p = torch.zeros(2, requires_grad=True)
        loss = masked_bce(p, torch.ones(2), torch.ones(2, dtype=torch.bool), torch.zeros(2)); loss.backward()
        self.assertEqual(loss.item(), 0.); torch.testing.assert_close(p.grad, torch.zeros(2))
    def test_invalid_supported_bce(self):
        with self.assertRaises(ValueError): masked_bce(torch.zeros(1), torch.tensor([float('nan')]), torch.tensor([True]))
        with self.assertRaises(ValueError): masked_bce(torch.zeros(1), torch.tensor([2.]), torch.tensor([True]))
    def test_invalid_bce_mask_shape(self):
        with self.assertRaises(ValueError): masked_bce(torch.zeros(2), torch.zeros(2), torch.tensor([True]))
        with self.assertRaises(ValueError): masked_bce(torch.zeros(2), torch.zeros(2), torch.ones(2))
    def test_negative_weight(self):
        with self.assertRaises(ValueError): masked_bce(torch.zeros(1), torch.ones(1), torch.tensor([True]), torch.tensor([-1.]))
    def test_offset_unknown_gradient_zero(self):
        p = torch.zeros((2, 3), requires_grad=True)
        y = torch.tensor([[1., 1., 1.], [float('nan')]*3])
        loss = masked_offsets(p, y, torch.tensor([True, False])); loss.backward()
        torch.testing.assert_close(p.grad[1], torch.zeros(3)); self.assertTrue(torch.isfinite(loss))
    def test_empty_offset_support(self):
        p = torch.full((2, 3), float('nan'), requires_grad=True)
        loss = masked_offsets(p, p.detach(), torch.zeros(2, dtype=torch.bool)); loss.backward()
        self.assertEqual(loss.item(), 0.); torch.testing.assert_close(p.grad, torch.zeros((2, 3)))
    def test_invalid_offset_shape(self):
        with self.assertRaises(ValueError): masked_offsets(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError): masked_offsets(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2))
    def test_support_precedence_and_unknown(self):
        result = support_precedence(np.array([1, 0, 0, 0], bool), np.array([1, 1, 0, 0], bool), np.array([1, 1, 1, 0], bool))
        np.testing.assert_array_equal(result['human'], [1, 0, 0, 0])
        np.testing.assert_array_equal(result['pseudo'], [0, 1, 0, 0])
        np.testing.assert_array_equal(result['background'], [0, 0, 1, 0])
        np.testing.assert_array_equal(result['unknown'], [0, 0, 0, 1])
    def test_support_types(self):
        with self.assertRaises(ValueError): support_precedence(np.ones(1), np.ones(1, bool), np.ones(1, bool))


class ProvenanceTests(unittest.TestCase):
    def test_clean_source_and_random_ancestor(self):
        a = {'student': asset(parents=('init',)), 'init': asset('random_initialization', train=())}
        self.assertEqual(assert_source_isolated('student', a, 'A', 'B'), {'student', 'init'})
    def test_unknown_weight_rejected(self):
        with self.assertRaises(ValueError): assert_source_isolated('m', {'m': asset('unknown')}, 'A', 'B')
    def test_target_in_teacher_rejected(self):
        a = {'m': asset(parents=('teacher',)), 'teacher': asset(train=('B',))}
        with self.assertRaises(ValueError): assert_source_isolated('m', a, 'A', 'B')
    def test_target_in_calibration_rejected(self):
        with self.assertRaises(ValueError): assert_source_isolated('m', {'m': asset(calibration=('B',))}, 'A', 'B')
    def test_missing_dependency(self):
        with self.assertRaises(ValueError): assert_source_isolated('m', {'m': asset(parents=('absent',))}, 'A', 'B')
    def test_provenance_cycle(self):
        a = {'m': asset(parents=('teacher',)), 'teacher': asset(parents=('m',))}
        with self.assertRaises(ValueError): assert_source_isolated('m', a, 'A', 'B')
    def test_empty_evidence(self):
        a = asset(); a['evidence'] = ''
        with self.assertRaises(ValueError): assert_source_isolated('m', {'m': a}, 'A', 'B')
    def test_missing_manifest_field(self):
        a = asset(); del a['parents']
        with self.assertRaises(ValueError): assert_source_isolated('m', {'m': a}, 'A', 'B')
    def test_external_verified_without_target(self):
        a = asset('verified_external_no_target_exposure', train=())
        self.assertEqual(assert_source_isolated('m', {'m': a}, 'A', 'B'), {'m'})


class BudgetAndRegistryTests(unittest.TestCase):
    def test_budget_is_per_frame(self):
        rows = budget_indices(np.array([.9, .8, .2, .7]), np.array([0, 0, 1, 1]), np.arange(4), 1)
        np.testing.assert_array_equal(rows, [0, 3])
    def test_no_fabricated_points(self):
        rows = budget_indices(np.array([.9]), np.array([0]), np.array([1]), 100)
        np.testing.assert_array_equal(rows, [0])
    def test_stable_score_ties(self):
        scores = np.array([.9, .9]); ids = np.array([10, 2]); frames = np.array([0, 0])
        selected = budget_indices(scores, frames, ids, 1); self.assertEqual(ids[selected][0], 2)
    def test_duplicate_ids_fail(self):
        with self.assertRaises(ValueError): budget_indices(np.ones(2), np.zeros(2, int), np.ones(2, int), 1)
    def test_empty_candidates(self):
        self.assertEqual(len(budget_indices(np.empty(0), np.empty(0, int), np.empty(0, int), 5)), 0)
    def test_nonfinite_scores(self):
        with self.assertRaises(ValueError): budget_indices(np.array([np.nan]), np.array([0]), np.array([1]), 1)
    def test_registry_mandatory_no_external_weights(self):
        d = json.loads((Path(__file__).parent/'study.json').read_text())
        for recipe in d['recipes']:
            if recipe['required']:
                self.assertEqual(recipe['initialization'], 'random_source_only')
                self.assertIsNone(recipe['external_dependency'])
    def test_registry_has_native_primary(self):
        d = json.loads((Path(__file__).parent/'study.json').read_text())
        recipe = next(r for r in d['recipes'] if r['id'] == d['fallback_primary'])
        self.assertEqual(recipe['input_zyx'], [64, 256, 256]); self.assertEqual(recipe['regime'], 'R_ema')
        self.assertEqual(len([r for r in d['recipes'] if r['required']]), 9)


if __name__ == '__main__':
    unittest.main()
