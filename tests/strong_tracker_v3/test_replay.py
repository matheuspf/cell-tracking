import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from strong_tracker_v3.inference import forbidden_path
from strong_tracker_v3.replay import VARIANTS, as_arrays, notebook_environment


class ReplayContracts(unittest.TestCase):
    def test_conditional_grid_has_all_four_division_smoothing_states(self):
        states = {(True, True)}
        for name, config in VARIANTS.items():
            if 'pruning' not in name and 'gaps' not in name:
                states.add((config.get('OUTPUT_SAFE_DIVISIONS', True), config.get('OUTPUT_LINEFIT_SMOOTH', True)))
        self.assertEqual(states, {(True, True), (False, True), (True, False), (False, False)})

    def test_removal_controls_disable_entire_stage(self):
        self.assertEqual(VARIANTS['no_motion_no_pruning'], {
            'OUTPUT_PRUNE_ISOLATED': False, 'OUTPUT_FILTER_SHORT_TRACKS': False})
        self.assertEqual(VARIANTS['no_motion_no_gaps'], {
            'OUTPUT_GAP_CLOSE': False, 'OUTPUT_GAP2_RECOVERY': False})

    def test_environment_extraction_does_not_execute_notebook(self):
        source = 'import os\nos.environ["A"]="1"\nraise RuntimeError("not executable")\nos.environ["A"]="2"\nos.environ["B"]=unknown()\n'
        self.assertEqual(notebook_environment(source), {'A': '2'})

    def test_arrays_keep_native_ids_and_missing_scores(self):
        nodes = {9: dict(t=1, z=3.5, y=4., x=5.), 2: dict(t=0, z=1., y=2., x=3.)}
        result = as_arrays(nodes, [dict(source_id=2, target_id=9, edge_prob=None)])
        np.testing.assert_array_equal(result['nodes'][:, 0], [2, 9])
        self.assertEqual(result['nodes'][1, 2], 3.5)
        self.assertTrue(np.isnan(result['edge_prob'][0]))

    def test_annotation_guard_resolves_symlink_and_allows_only_new_geff(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hidden = root / 'train' / 'hidden.geff'
            hidden.mkdir(parents=True)
            alias = root / 'disguised'
            alias.symlink_to(hidden, target_is_directory=True)
            self.assertTrue(forbidden_path(alias / 'zarr.json', root / 'fresh'))
            self.assertFalse(forbidden_path(root / 'fresh' / 'prediction.geff' / 'zarr.json', root / 'fresh'))
            self.assertTrue(forbidden_path(root / 'inventory.json', root / 'fresh'))
            self.assertTrue(forbidden_path(root / 'evaluation' / 'matches.npz', root / 'fresh'))
            self.assertTrue(forbidden_path(root / 'score_rows.csv', root / 'fresh'))

    def test_guard_precedes_numpy_and_blocks_real_open(self):
        code = ('import sys; from strong_tracker_v3.inference import deny_annotations; '
                'assert "numpy" not in sys.modules; deny_annotations(); '
                'open("/tmp/forbidden.geff/zarr.json")')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PermissionError: Annotations unavailable', result.stderr)

    def test_fresh_checkpoint_requires_exact_pipeline_and_graph(self):
        from strong_tracker_v3.common import save_arrays, sha, write_json
        from strong_tracker_v3.fresh import fresh_can_resume
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', {
                'V3_RAW_CHECKPOINTS': td, 'V3_PIPELINE_FINGERPRINT': 'frozen-pipeline'}):
            root = Path(td)
            self.assertFalse(fresh_can_resume('sample'))
            path = root / 'sample.npz'
            save_arrays(path, nodes=np.empty((0, 5)), edges=np.empty((0, 2)))
            write_json(root / 'sample.json', dict(sha256=sha(path), pipeline_fingerprint='frozen-pipeline'))
            self.assertTrue(fresh_can_resume('sample'))
            with patch.dict('os.environ', {'V3_PIPELINE_FINGERPRINT': 'changed-pipeline'}):
                with self.assertRaisesRegex(AssertionError, 'Stale fresh neural checkpoint'):
                    fresh_can_resume('sample')
            path.write_bytes(b'changed graph')
            with self.assertRaisesRegex(AssertionError, 'Fresh neural checkpoint changed'):
                fresh_can_resume('sample')

    def test_selected_noop_preserves_arrays_and_rejects_unknown_policy(self):
        from strong_tracker_v3.inference import apply_policy
        n = np.array([[8, 0, 1, 1, 1], [19, 1, 1, 2, 1]], np.int64)
        e = np.array([[8, 19]], np.int64)
        sample = dict(image_shape=[2, 3, 3, 3])
        nn, ee, receipt = apply_policy(None, sample, n, e, None, None, {}, source_model='44b6')
        self.assertIs(nn, n)
        self.assertIs(ee, e)
        self.assertTrue(receipt['no_op'])
        with self.assertRaisesRegex(ValueError, 'Unknown selected policy'):
            apply_policy(None, sample, n, e, None, None, {'misspelled_arm': True}, source_model='44b6')

    def test_export_rejects_changed_scored_source_before_copying(self):
        from types import SimpleNamespace
        from strong_tracker_v3.common import sha, write_json
        from strong_tracker_v3.inference import export_selected
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'candidate_graphs' / 'measured' / 'sample.npz'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'officially scored bytes')
            write_json(root / 'export_scoring_manifest.json', dict(
                variant='measured', metric_revision='pinned',
                graph_file_sha256={'sample': sha(source)}))
            source.write_bytes(b'changed graph')
            ctx = SimpleNamespace(out=root, metric_revision='pinned',
                samples=lambda: [dict(dataset='sample')])
            with patch('strong_tracker_v3.inference.deny_annotations', return_value={}):
                with self.assertRaisesRegex(AssertionError, 'Scored export source changed'):
                    export_selected(ctx, 'measured', configuration={'policy': {'association': {'margin': 3.0}}})
            self.assertFalse((root / 'selected_predictions').exists())


if __name__ == '__main__':
    unittest.main()
