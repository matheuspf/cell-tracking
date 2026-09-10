"""Synthetic reference tests only; no Biohub microscopy/model inference is claimed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preflight import inspect_paths
from reference_contracts import (grid_to_native, serialize_positions, validate_selected_graph,
    validate_region_mapping, sparse_edge_targets, fuse_window_logits, fork_cost_difference)
from target_budget import scenarios

HERE = Path(__file__).resolve().parent


class GridTests(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_array_equal(grid_to_native([[1,2,3]]), [[1,2,3]])
    def test_anisotropic_stride(self):
        np.testing.assert_array_equal(grid_to_native([[2,3,4]], [1,4,4]), [[2,12,16]])
    def test_crop_origin(self):
        np.testing.assert_array_equal(grid_to_native([[2,3,4]], [1,2,2], [5,7,9]), [[7,13,17]])
    def test_fractional_roundtrip(self):
        p=np.array([[2.25,3.7,4.125]])
        native=grid_to_native(p, [1,4,4], [3,2,1])
        np.testing.assert_allclose((native-[3,2,1])/[1,4,4],p)
    def test_empty(self):
        self.assertEqual(grid_to_native(np.empty((0,3))).shape,(0,3))
    def test_bad_shape(self):
        with self.assertRaises(ValueError):grid_to_native([1,2,3])
    def test_bad_stride(self):
        with self.assertRaises(ValueError):grid_to_native([[1,2,3]], [1,0,4])
    def test_nan(self):
        with self.assertRaises(ValueError):grid_to_native([[1,float('nan'),3]])
    def test_rounding_clipping_recorded(self):
        p,n=serialize_positions([[63.8,255.8,20.1]], [64,256,256])
        np.testing.assert_array_equal(p,[[63,255,20]]);self.assertEqual(n,1)
    def test_outside_rejected(self):
        with self.assertRaises(ValueError):serialize_positions([[64,0,0]], [64,256,256])
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):serialize_positions([[-.1,0,0]], [64,256,256])


class GraphTests(unittest.TestCase):
    def test_true_fork(self):
        self.assertEqual(validate_selected_graph({0:0,1:1,2:1},[(0,1),(0,2)])['forks'],1)
    def test_empty_graph(self):
        self.assertEqual(validate_selected_graph({},[])['nodes'],0)
    def test_isolated_nodes(self):
        self.assertEqual(validate_selected_graph({1:2},[])['edges'],0)
    def test_merge_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({0:0,1:0,2:1},[(0,2),(1,2)])
    def test_three_children_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({0:0,1:1,2:1,3:1},[(0,1),(0,2),(0,3)])
    def test_skip_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({0:0,1:2},[(0,1)])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({0:0,1:1},[(0,1),(0,1)])
    def test_dangling_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({0:0},[(0,1)])
    def test_boolean_id_rejected(self):
        with self.assertRaises(ValueError):validate_selected_graph({False:0},[])
    def test_mask_label_reused_across_frames(self):
        self.assertEqual(validate_region_mapping({10:0,11:1},[(0,1,10),(1,1,11)])['represented'],2)
    def test_mask_collision_rejected(self):
        with self.assertRaises(ValueError):validate_region_mapping({10:0,11:0},[(0,1,10),(0,1,11)])
    def test_unrepresented_fixed_node_rejected(self):
        with self.assertRaises(ValueError):validate_region_mapping({10:0,11:1},[(0,1,10)])
    def test_partial_mapping_named(self):
        self.assertEqual(validate_region_mapping({10:0,11:1},[(0,1,10)],False)['unrepresented'],1)
    def test_wrong_frame_rejected(self):
        with self.assertRaises(ValueError):validate_region_mapping({10:0},[(1,1,10)])


class SparseTests(unittest.TestCase):
    def test_two_children_positive(self):
        y=sparse_edge_targets({0:0,1:1,2:1},{0:10,1:11,2:12},[(0,1),(0,2)],[(10,11),(10,12)])
        np.testing.assert_array_equal(y,[1,1])
    def test_single_child_does_not_make_unmatched_second_negative(self):
        y=sparse_edge_targets({0:0,1:1,2:1},{0:10,1:11,2:-1},[(0,1),(0,2)],[(10,11)])
        np.testing.assert_array_equal(y,[1,-1])
    def test_wrong_predecessor_negative(self):
        y=sparse_edge_targets({0:0,1:0,2:1},{0:10,1:-1,2:12},[(0,2),(1,2)],[(10,12)])
        np.testing.assert_array_equal(y,[1,0])
    def test_two_children_exhaust_binary_capacity(self):
        y=sparse_edge_targets({0:0,1:1},{0:10,1:-1},[(0,1)],[(10,12),(10,13)])
        np.testing.assert_array_equal(y,[0])
    def test_unknown_root_not_known_birth(self):
        y=sparse_edge_targets({0:0,1:1},{0:-1,1:12},[(0,1)],[])
        np.testing.assert_array_equal(y,[-1])
    def test_ambiguity_masks(self):
        y=sparse_edge_targets({0:0,1:1},{0:10,1:11},[(0,1)],[(10,11)],{1})
        np.testing.assert_array_equal(y,[-1])
    def test_candidate_degree_not_output_degree(self):
        t={0:0,1:1,2:1,3:1};m={0:10,1:11,2:-1,3:-1}
        np.testing.assert_array_equal(sparse_edge_targets(t,m,[(0,1),(0,2),(0,3)],[(10,11)]),[1,-1,-1])
    def test_match_collision_rejected(self):
        with self.assertRaises(ValueError):sparse_edge_targets({0:0,1:1},{0:10,1:10},[(0,1)],[])
    def test_missing_match_rejected(self):
        with self.assertRaises(ValueError):sparse_edge_targets({0:0,1:1},{0:10},[(0,1)],[])
    def test_ground_truth_merge_rejected(self):
        with self.assertRaises(ValueError):sparse_edge_targets({0:0,1:1},{0:10,1:12},[(0,1)],[(10,12),(11,12)])


class EvidenceTests(unittest.TestCase):
    def test_identical_duplicate_view_deduplicated(self):
        a=(1,2,'w1',3.,1.)
        self.assertEqual(fuse_window_logits([a,a]),{(1,2):3.})
    def test_overlapping_equal_scores_do_not_inflate(self):
        self.assertEqual(fuse_window_logits([(1,2,'w1',3.,1.),(1,2,'w2',3.,1.)]),{(1,2):3.})
    def test_weighted_mean(self):
        self.assertEqual(fuse_window_logits([(1,2,'w1',2.,1.),(1,2,'w2',4.,3.)]),{(1,2):3.5})
    def test_conflicting_duplicate_fails(self):
        with self.assertRaises(ValueError):fuse_window_logits([(1,2,'w1',2.,1.),(1,2,'w1',3.,1.)])
    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):fuse_window_logits([(1,2,'w1',float('nan'),1.)])
    def test_zero_weight_fails(self):
        with self.assertRaises(ValueError):fuse_window_logits([(1,2,'w1',2.,0.)])
    def test_order_invariance(self):
        a=[(1,2,'b',3.,2.),(1,2,'a',1.,1.)]
        self.assertEqual(fuse_window_logits(a),fuse_window_logits(reversed(a)))
    def test_legacy_fork_strictly_dominated(self):
        for p in [0.,.5,1.]:self.assertGreater(fork_cost_difference(p,1.2),0.)
    def test_new_event_evidence_can_select_fork(self):
        self.assertLess(fork_cost_difference(.9,1.2,event_gain=.5),0.)
    def test_nondivision_still_feasible(self):
        self.assertGreater(fork_cost_difference(.1,1.2,event_gain=.2),0.)


class BudgetAndPreflightTests(unittest.TestCase):
    def setUp(self):self.b=json.loads((HERE/'baseline.json').read_text())
    def test_target_gap(self):
        self.assertAlmostEqual(scenarios(self.b)['gap'],.015197625739414)
    def test_divisions_required(self):
        r=scenarios(self.b);self.assertEqual(r['required_tp_at_unchanged_fp'],66)
        self.assertEqual(r['additional_true_divisions_at_unchanged_fp'],37)
    def test_conditional_score(self):
        self.assertAlmostEqual(scenarios(self.b)['cases'][1]['calculated_score'],.9500287117091456)
    def test_components_checked(self):
        b=copy.deepcopy(self.b);b['adjusted_edge_contribution']+=.01
        with self.assertRaises(ValueError):scenarios(b)
    def test_impossible_target_marked(self):
        self.assertFalse(scenarios(self.b,1.2)['feasible_at_unchanged_fp'])
    def test_nonfinite_target_rejected(self):
        with self.assertRaises(ValueError):scenarios(self.b,float('nan'))
    def test_preflight_missing_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            r=inspect_paths(d,Path(d)/'out',Path(d)/'data')
            self.assertEqual(r['status'],'dependencies_missing');self.assertFalse(r['raw_data_read'])
    def test_preflight_is_not_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/train').mkdir(parents=True)
            for n in ['annotation-selection-v1','strong-tracker-v2','strong-tracker-v3','multidata-training-v4']:(root/'out'/n).mkdir(parents=True)
            (root/'work/annotation-selection-v1/official').mkdir(parents=True)
            r=inspect_paths(root,root/'out',root/'data')
            self.assertEqual(r['status'],'paths_present_not_validated');self.assertFalse(r['official_scoring_run'])
    def test_variant_cap_and_target(self):
        cfg=json.loads((HERE/'experiments.json').read_text())
        self.assertEqual(cfg['caps']['complete_new_graph_configurations'],24)
        self.assertEqual(cfg['target_score'],.95);self.assertFalse(cfg['expired_v4_deadline_inherited'])
    def test_correct_3d_model_route(self):
        m=json.loads((HERE/'model_registry.json').read_text())
        self.assertEqual(m['trackastra']['primary_model'],'ctc')
        self.assertEqual(m['hoct']['forbidden_stub'],'hoct.create_graph_from_points')


if __name__ == '__main__':unittest.main()
