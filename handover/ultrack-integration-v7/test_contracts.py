"""Authoring tests only: no Ultrack installation or microscopy is implied."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from contracts import integer, validate_observations
from preflight import admitted, blob_sha, existing_parent, GIB

SHAPE = (4, 12, 32, 40)


def movie():
    return [dict(id=101, parent_id=-1, t=0, z=4, y=14, x=12, track_id=999),
            dict(id=505, parent_id=101, t=1, z=4, y=14, x=13, track_id=999),
            dict(id=777, parent_id=505, t=2, z=4, y=14, x=14, track_id=999)]


class ObservationTests(unittest.TestCase):
    def test_uses_observation_ids_not_track_ids(self):
        self.assertEqual(validate_observations(movie(), SHAPE)["edges"], [(101, 505), (505, 777)])

    def test_binary_division(self):
        rows = movie() + [dict(id=888, parent_id=505, t=2, z=4, y=18, x=14)]
        self.assertEqual(validate_observations(rows, SHAPE)["divisions"], 1)

    def test_missing_parent(self):
        rows = movie(); rows[1]["parent_id"] = 432
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_missing_parent_column(self):
        rows = movie(); del rows[1]["parent_id"]
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_duplicate_id(self):
        rows = movie(); rows[1]["id"] = 101
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_temporal_jump(self):
        rows = movie(); rows[2]["t"] = 3
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_backward_edge(self):
        rows = movie(); rows[0]["parent_id"] = 777
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_three_daughters(self):
        rows = movie() + [dict(id=i, parent_id=505, t=2, z=4, y=18, x=14) for i in (888, 889)]
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_out_of_bounds(self):
        for name, value in (("x", 40), ("z", -1), ("t", 4), ("y", float("nan"))):
            rows = movie(); rows[1][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError): validate_observations(rows, SHAPE)

    def test_integer_export_required(self):
        rows = movie(); rows[1]["x"] = 13.5
        with self.assertRaises(ValueError): validate_observations(rows, SHAPE)
        self.assertEqual(validate_observations(rows, SHAPE, integer_coordinates=False)["nodes"], 3)

    def test_bad_identifier(self):
        for value in (True, 1.1, float("inf"), "101", float(2**53)):
            with self.subTest(value=value), self.assertRaises(ValueError): integer(value, "id")

    def test_exact_large_integer(self):
        self.assertEqual(integer(2**53 + 1, "id"), 2**53 + 1)

    def test_empty_output_is_structurally_valid_not_a_score(self):
        self.assertEqual(validate_observations([], SHAPE), {"nodes": 0, "edges": [], "divisions": 0})

    def test_invalid_shape(self):
        for shape in ((4, 12, 32), (4, 0, 32, 40), (4, 12.5, 32, 40)):
            with self.subTest(shape=shape), self.assertRaises(ValueError): validate_observations([], shape)

    def test_does_not_mutate_rows(self):
        rows = movie(); before = copy.deepcopy(rows)
        validate_observations(rows, SHAPE)
        self.assertEqual(rows, before)


class PreflightTests(unittest.TestCase):
    def test_reserve_plus_allocation(self):
        self.assertFalse(admitted(int(6.838*GIB), 0))
        self.assertFalse(admitted(10*GIB, 3))
        self.assertTrue(admitted(11*GIB, 3))

    def test_invalid_allocation(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError): admitted(100*GIB, value)

    def test_existing_parent_does_not_create_paths(self):
        with tempfile.TemporaryDirectory() as root:
            wanted = Path(root)/"new"/"nested"
            self.assertEqual(existing_parent(wanted), Path(root).resolve())
            self.assertFalse(wanted.exists())

    def test_git_blob_hash(self):
        data = b"example\n"
        self.assertEqual(blob_sha(data), hashlib.sha1(b"blob 8\0example\n").hexdigest())

    def test_study_registry(self):
        spec = json.loads((Path(__file__).parent/"study.json").read_text())
        self.assertEqual([a["id"] for a in spec["arms"]], ["C0", "P0", "U0", "U1", "H1", "U2"])
        self.assertEqual(sum(spec["evaluation"]["embryo_clip_counts"].values()), 199)
        self.assertEqual(spec["upstream"]["native_cost_link_function"], "identity")
        self.assertEqual(spec["upstream"]["solver"], "CBC")
        self.assertEqual(spec["evaluation"]["promotion_reference"], "P0")
        self.assertFalse(spec["authoring"]["real_ultrack_executed"])
        self.assertIsNone(spec["authoring"]["new_scores"])


if __name__ == "__main__":
    unittest.main()
