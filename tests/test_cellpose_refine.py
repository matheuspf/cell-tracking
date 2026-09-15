"""Critical sparse-supervision, ancestry and geometric contracts for refinement."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.cellpose_refine.common import SPACING, bounded_integer, config, validate_ancestry
from tools.cellpose_refine.model import Refiner, sparse_offset_loss
from tools.cellpose_refine.prepare import jitter_queries, natural_targets
from tools.cellpose_refine.diagnostics import make_strata, signed_and_stratified


class RefineContracts(unittest.TestCase):
    def test_unannotated_proposals_are_unknown(self):
        cfg = config()
        queries = np.array([[10., 10., 10.], [10., 150., 150.]])
        gt = np.array([[1, 9, 10, 10, 10]])
        targets = natural_targets(queries, gt, cfg)
        np.testing.assert_array_equal(targets["weight"], [2., 0.])
        pred = torch.ones(2, 3, requires_grad=True)
        sparse_offset_loss(pred, torch.from_numpy(targets["target_um"]), torch.from_numpy(targets["weight"])).backward()
        self.assertEqual(torch.count_nonzero(pred.grad[1]).item(), 0)
        self.assertGreater(torch.count_nonzero(pred.grad[0]).item(), 0)

    def test_close_annotated_neighbors_are_not_forced_training_correspondences(self):
        gt = np.array([[1, 9, 10, 10, 10], [2, 9, 10, 10, 11]])
        t = natural_targets(gt[:, 2:].astype(float), gt, config())
        self.assertEqual(int((t["weight"] > 0).sum()), 0)

    def test_jitters_replay_and_have_physical_targets(self):
        gt = np.array([[1, 9, 0, 128, 128], [2, 9, 20, 100, 100]])
        a = jitter_queries(gt, [64, 256, 256], "44b6_fixture-t009", config())
        b = jitter_queries(gt, [64, 256, 256], "44b6_fixture-t009", config())
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        mapping = {int(r[0]): r[2:] for r in gt}
        expected = (np.array([mapping[int(i)] for i in a["gt_ids"]]) - a["queries"]) * SPACING
        np.testing.assert_allclose(a["target_um"], expected, atol=1e-6)
        self.assertTrue((a["queries"] >= 0).all())
        self.assertTrue((np.linalg.norm(a["target_um"], axis=1) <= 2.5 + 1e-6).all())

    def test_integer_projection_handles_rounding_exceeding_limit(self):
        q = np.array([[20.49, 100., 100.]])
        refined = q + np.array([[3. / 1.625, 0., 0.]])
        raw = np.rint(refined)
        self.assertGreater(np.linalg.norm((raw - np.rint(q)) * SPACING), 3.)
        actual, projected = bounded_integer(q, refined, [64, 256, 256])
        self.assertTrue(projected[0])
        self.assertLessEqual(np.linalg.norm((actual - np.rint(q)) * SPACING), 3.)

    def test_zero_head_is_identity_and_receives_learning_gradients(self):
        torch.set_num_threads(2)
        model = Refiner(config()).eval()
        out = model(torch.randn(2, 3, 768), torch.randn(2, 7, 25, 25), torch.randn(2, 3))
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        sparse_offset_loss(out, torch.ones_like(out), torch.ones(2)).backward()
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)

    def test_ancestry_rejects_target_calibration_and_nested_teachers(self):
        fields = ("labels", "unlabeled_images", "calibration", "teacher_construction", "normalization_fit")
        clean = {"adaptation_exposure": {f: [] for f in fields}}
        validate_ancestry("44b6", clean)
        for f in fields:
            bad = {"adaptation_exposure": {k: ["6bba"] if k == f else [] for k in fields}}
            with self.assertRaises(ValueError):
                validate_ancestry("44b6", {**clean, "parents": [bad]})
        with self.assertRaises(ValueError):
            validate_ancestry("44b6", {"adaptation_exposure": {"labels": []}})

    def test_source_and_inference_read_guards(self):
        code = '''
from tools.cellpose_refine.train import install_source_guard
from tools.cellpose_refine.infer import install_inference_guard
install_source_guard("44b6")
try:
 open("/tmp/cellpose-refine-v1/features/6bba/probe.npz", "rb")
except PermissionError:
 pass
else:
 raise AssertionError("target read accepted")
install_inference_guard(raw=True)
for path in ["/tmp/probe.geff/nodes/ids", "/tmp/cellpose-refine-v1/features/44b6/probe.npz", "/tmp/detector-screen-20260914/predictions/44b6_probe.npz"]:
 try:
  open(path, "rb")
 except PermissionError:
  pass
 else:
  raise AssertionError(path)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_evaluation_strata_and_conditional_error_denominators(self):
        nodes = [[1, 1, 1, 16, 16], [2, 1, 20, 16, 16], [3, 1, 60, 16, 16]]
        gt = {"clips": {"fixture": {"nodes": [[0, 0, 10, 16, 16], *nodes],
                                    "edges": [[0, 1], [0, 2]]}},
              "frames": {"fixture-t001": {"gt_nodes": nodes}}}
        volume = np.full((64, 32, 32), 10, dtype=np.uint16)
        volume[1, 16, 16] = 100
        volume[60, 16, 16] = 100
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "volume.npy"
            np.save(path, volume)
            labels = make_strata([{"key": "fixture-t001", "dataset": "fixture",
                                   "image_path": str(path)}], gt)
        self.assertIn("within_3um_center_bound", labels["fixture-t001"][1])
        self.assertIn("faint_peak", labels["fixture-t001"][2])
        self.assertIn("z_upper_quarter", labels["fixture-t001"][3])
        self.assertIn("annotated_division_neighborhood", labels["fixture-t001"][2])
        self.assertNotIn("annotated_division_neighborhood", labels["fixture-t001"][3])
        rows = [{"key": "fixture-t001", "embryo": "44b6",
                 "_found_ids": {"3.0": [1, 3], "7.0": [1, 2, 3]},
                 "_records": {"1": {"predicted_um": [2., 1., 4.5], "gt_um": [1., 3., 4.]}}}]
        d = signed_and_stratified(rows, labels, "pooled")
        self.assertEqual(d["strata"]["annotated_division_neighborhood"]["gt"], 2)
        self.assertEqual(d["strata"]["annotated_division_neighborhood"]["recall"]["3.0"], .5)
        self.assertEqual(d["strata"]["faint_peak"]["recall"]["7.0"], 1.)
        self.assertEqual(d["signed_error_on_matched_at_7"]["n"], 1)
        np.testing.assert_array_equal(d["signed_error_on_matched_at_7"]["mean"], [1., -2., .5])
        self.assertIsNone(signed_and_stratified(rows, labels, "6bba")["signed_error_on_matched_at_7"]["mean"])


if __name__ == "__main__":
    unittest.main()
