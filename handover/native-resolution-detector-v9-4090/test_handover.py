"""CPU synthetic tests; no claims about actual images, CUDA fit or model performance."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contracts import ContractError, load_study, require_complete_frames, require_source_ancestry, validate_study
from preflight import inspect_metadata


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(HERE / "study.json")

    def test_valid_registry(self):
        validate_study(self.study)

    def test_fit_count(self):
        self.assertEqual(sum(2 * len(r["seeds"]) for r in self.study["recipes"]), 32)

    def test_target_training_rejected(self):
        self.study["validation"]["target_image_training"] = True
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_fold_reuse_rejected(self):
        self.study["validation"]["directions"][0]["target"] = "44b6"
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_native_resize_rejected(self):
        self.study["model"]["primary_input_resizing_allowed"] = True
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_core_seed_mismatch_rejected(self):
        self.study["recipes"][0]["seeds"] = [1]
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_budget_overrun_rejected(self):
        self.study["resources"]["total_active_gpu_hours"] = 1
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_duplicate_recipe_rejected(self):
        self.study["recipes"].append(copy.deepcopy(self.study["recipes"][0]))
        with self.assertRaises(ContractError):
            validate_study(self.study)

    def test_external_permission_rejected(self):
        self.study["permissions"]["external_downloads"] = True
        with self.assertRaises(ContractError):
            validate_study(self.study)


class CoverageTests(unittest.TestCase):
    def test_complete_including_zero_detection_frame(self):
        require_complete_frames({"a": 3}, {"a": [0, 1, 2]})

    def test_missing_clip(self):
        with self.assertRaises(ContractError):
            require_complete_frames({"a": 2, "b": 2}, {"a": [0, 1]})

    def test_extra_clip(self):
        with self.assertRaises(ContractError):
            require_complete_frames({"a": 1}, {"a": [0], "b": [0]})

    def test_duplicate_frame(self):
        with self.assertRaises(ContractError):
            require_complete_frames({"a": 3}, {"a": [0, 1, 1]})

    def test_missing_final_frame(self):
        with self.assertRaises(ContractError):
            require_complete_frames({"a": 3}, {"a": [0, 1]})

    def test_boolean_frame_is_not_integer_id(self):
        with self.assertRaises(ContractError):
            require_complete_frames({"a": 2}, {"a": [False, True]})


class AncestryTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            "init": {"kind": "random_initialization", "provenance": "verified_source_only", "fit_embryos": [], "parents": [], "evidence": "seed/config receipt"},
            "teacher": {"kind": "teacher", "provenance": "verified_source_only", "fit_embryos": ["44b6"], "parents": ["init"], "evidence": "source manifest/hash"},
            "student": {"kind": "checkpoint", "provenance": "verified_source_only", "fit_embryos": ["44b6"], "parents": ["teacher"], "evidence": "training run receipt"},
        }

    def test_clean_transitive_chain(self):
        require_source_ancestry(self.items, "student", "44b6")

    def test_teacher_target_exposure(self):
        self.items["teacher"]["fit_embryos"].append("6bba")
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")

    def test_unknown_teacher(self):
        self.items["teacher"]["provenance"] = "unknown"
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")

    def test_cycle(self):
        self.items["teacher"]["parents"] = ["student"]
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")

    def test_missing_ancestor(self):
        del self.items["teacher"]
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")

    def test_no_evidence(self):
        del self.items["teacher"]["evidence"]
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")

    def test_undeclared_population(self):
        self.items["teacher"]["fit_embryos"] = []
        with self.assertRaises(ContractError):
            require_source_ancestry(self.items, "student", "44b6")


class PreflightTests(unittest.TestCase):
    def test_help_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(HERE / "preflight.py"), "--help"], cwd=tmp, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-root", result.stdout)

    def test_missing_data_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = inspect_metadata(Path(tmp), load_study(HERE / "study.json"))
        self.assertEqual(result["clip_count"], 0)
        self.assertTrue(result["errors"])

    def test_metadata_without_image_or_label_reads(self):
        study = load_study(HERE / "study.json")
        study["data"]["expected_clips"] = 1
        study["data"]["expected_embryo_clips"] = {"44b6": 1}
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train"
            arr = train / "44b6_fixture.zarr" / "0"
            arr.mkdir(parents=True)
            (arr / "zarr.json").write_text(json.dumps({"shape": [3, 64, 256, 256], "data_type": "uint16"}))
            (train / "44b6_fixture.geff").mkdir()
            result = inspect_metadata(Path(tmp), study)
        self.assertFalse(result["errors"])
        self.assertFalse(result["image_chunks_read"])
        self.assertFalse(result["geff_labels_read"])
        self.assertEqual(result["clips"][0]["shape_tzyx"][0], 3)

    def test_bad_native_shape(self):
        study = load_study(HERE / "study.json")
        with tempfile.TemporaryDirectory() as tmp:
            arr = Path(tmp) / "train" / "44b6_fixture.zarr" / "0"
            arr.mkdir(parents=True)
            (arr / "zarr.json").write_text(json.dumps({"shape": [3, 64, 64, 64], "data_type": "uint16"}))
            result = inspect_metadata(Path(tmp), study)
        self.assertTrue(any("Unexpected native" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
