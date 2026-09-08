"""Synthetic checks only. Official matcher parity and real data tests run locally."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis import membership_curve, membership_stats, required_tp_retention, strict_summary
from audit_metadata import audit, locate_estimates


def row(dataset="A", tp=90, fp=5, fn=5, nodes=100, estimate=100, dtp=3, dfp=1, dfn=1):
    return dict(dataset=dataset, edge_tp=tp, edge_fp=fp, edge_fn=fn,
                num_pred_nodes=nodes, estimated_total=estimate,
                division_tp=dtp, division_fp=dfp, division_fn=dfn)


class ArithmeticTests(unittest.TestCase):
    def test_identity_count_ratio(self):
        self.assertAlmostEqual(strict_summary([row()], ["A"])["score"], .96)

    def test_half_count(self):
        self.assertAlmostEqual(strict_summary([row(nodes=50)], ["A"])["score"], 1.005)

    def test_weighted_not_macro(self):
        rows = [row(tp=9, fp=0, fn=1, nodes=50, dtp=0, dfp=0, dfn=0),
                row("B", tp=50, fp=25, fn=25, nodes=200, dtp=0, dfp=0, dfn=0)]
        expected = (10 * .9 * 1.05 + 100 * .5 * .9) / 110
        self.assertAlmostEqual(strict_summary(rows, ["A", "B"])["score"], expected)

    def test_micro_divisions(self):
        rows = [row(dtp=1, dfp=0, dfn=0), row("B", dtp=0, dfp=0, dfn=9)]
        self.assertAlmostEqual(strict_summary(rows, ["A", "B"])["division_jaccard"], .1)

    def test_no_divisions(self):
        result = strict_summary([row(dtp=0, dfp=0, dfn=0)], ["A"])
        self.assertIsNone(result["division_jaccard"])
        self.assertAlmostEqual(result["score"], .9)

    def test_missing_sample(self):
        with self.assertRaises(ValueError):
            strict_summary([row()], ["A", "B"])

    def test_duplicate_sample(self):
        with self.assertRaises(ValueError):
            strict_summary([row(), row()], ["A"])

    def test_empty_expected(self):
        with self.assertRaises(ValueError):
            strict_summary([], [])

    def test_invalid_counts(self):
        for value in (-1, .5, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_summary([row(tp=value)], ["A"])

    def test_invalid_estimates(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_summary([row(estimate=value)], ["A"])

    def test_clipped_adjustment(self):
        self.assertAlmostEqual(strict_summary([row(nodes=2000)], ["A"])["score"], .06)

    def test_zero_edge_sample(self):
        rows = [row(), row("B", tp=0, fp=0, fn=0, dtp=0, dfp=0, dfn=0)]
        result = strict_summary(rows, ["A", "B"])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_adj"], 1)
        self.assertAlmostEqual(result["score"], .96)

    def test_all_zero_edges(self):
        with self.assertRaises(ValueError):
            strict_summary([row(tp=0, fp=0, fn=0)], ["A"])

    def test_previous_example_threshold(self):
        result = required_tp_retention(.886, .6, 1, .5, .970)
        self.assertAlmostEqual(result["required_tp_retention"], .91 / (.886 * 1.05))
        self.assertAlmostEqual(result["independent_node_recall"], math.sqrt(.91 / (.886 * 1.05)))
        self.assertAlmostEqual(.886 * .99 ** 2 * 1.05 + .06, .97178703)

    def test_impossible_small_deletion(self):
        self.assertFalse(required_tp_retention(.886, .6, 1, .8, .970)["feasible_by_tp_retention_only"])

    def test_fp_removal_lowers_requirement(self):
        a = required_tp_retention(.886, .6, 1, .5, .970)
        b = required_tp_retention(.886, .6, 1, .5, .970, fp_to_gt=.05, fp_retained=.2)
        self.assertLess(b["required_tp_retention"], a["required_tp_retention"])

    def test_confusion_example(self):
        labels = [1] * 100 + [0] * 900
        keep = [True] * 99 + [False] + [True] * 401 + [False] * 499
        result = membership_stats(labels, keep)
        self.assertAlmostEqual(result["accuracy"], .598)
        self.assertAlmostEqual(result["phi"], 49 / 150)
        self.assertAlmostEqual(result["deleted_annotation_rate"], .002)

    def test_membership_degenerate(self):
        result = membership_stats([0, 0], [True, False])
        self.assertIsNone(result["annotation_recall"])
        self.assertIsNone(result["phi"])

    def test_bad_membership(self):
        for labels, keep in (([], []), ([2], [True]), ([1], [1]), ([1], [])):
            with self.subTest(labels=labels, keep=keep), self.assertRaises(ValueError):
                membership_stats(labels, keep)

    def test_per_dataset_budget(self):
        rows = [dict(dataset=g, candidate_id=str(i), annotation_label=i % 2, score=i)
                for g in ("A", "B") for i in range(4)]
        result = membership_curve(rows, [.5])[0]
        self.assertEqual(result["retained_fraction"], .5)
        self.assertTrue(all(x["retained_fraction"] == .5 for x in result["per_dataset"].values()))

    def test_ties_do_not_use_labels_or_row_order(self):
        rows = [dict(dataset="A", candidate_id=str(i), annotation_label=i % 2, score=0) for i in range(20)]
        self.assertEqual(membership_curve(rows, [.5]), membership_curve(list(reversed(rows)), [.5]))
        flipped = copy.deepcopy(rows)
        for r in flipped:
            r["annotation_label"] = 1 - r["annotation_label"]
        a, b = membership_curve(rows, [.5])[0], membership_curve(flipped, [.5])[0]
        self.assertEqual(a["tp"] + b["tp"], 10)

    def test_duplicate_candidate_rejected(self):
        r = dict(dataset="A", candidate_id="1", annotation_label=1, score=.5)
        with self.assertRaises(ValueError):
            membership_curve([r, r], [.5])

    def test_nonfinite_score_rejected(self):
        r = dict(dataset="A", candidate_id="1", annotation_label=1, score=float("nan"))
        with self.assertRaises(ValueError):
            membership_curve([r], [.5])


class MetadataTests(unittest.TestCase):
    def make_data(self, root, duplicate=False, v2=False):
        train = root / "train"
        for path, shape in (("emb1_clip.zarr/0", [100, 64, 256, 256]),
                            ("emb1_clip.geff/nodes/ids", [100]),
                            ("emb1_clip.geff/edges/ids", [90, 2])):
            folder = train / path
            folder.mkdir(parents=True)
            (folder / (".zarray" if v2 else "zarr.json")).write_text(json.dumps({"shape": shape}))
        meta = {"attributes": {"geff": {"extra": {"estimated_number_of_nodes": 1000}}}}
        if duplicate:
            meta["other"] = {"estimated_number_of_nodes": 999}
        (train / "emb1_clip.geff" / (".zattrs" if v2 else "zarr.json")).write_text(json.dumps(meta))

    def test_complete_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root)
            result = audit(root)
            self.assertEqual(result["status"], "metadata_complete")
            self.assertAlmostEqual(result["reference_coverage_estimate_pooled"], .1)
            self.assertIsNone(result["unique_biological_cell_coverage"])

    def test_complete_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root, v2=True)
            self.assertEqual(audit(root)["status"], "metadata_complete")

    def test_ambiguous_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root, duplicate=True)
            result = audit(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["reference_coverage_estimate_pooled"])

    def test_empty_data_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(audit(Path(tmp))["status"], "incomplete")

    def test_unpaired_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_data(root)
            (root / "train/extra.zarr").mkdir()
            self.assertEqual(audit(root)["status"], "incomplete")

    def test_recursive_estimates(self):
        self.assertEqual(locate_estimates({"x": [{"estimated_number_of_nodes": 5}]}),
                         [("x[0].estimated_number_of_nodes", 5)])


if __name__ == "__main__":
    unittest.main()
