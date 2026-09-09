"""Final evidence export must preserve identity and limit causal attribution."""
import json
from pathlib import Path
import tempfile
import unittest

from strong_tracker_v3.context import RunContext
from strong_tracker_v3.ledger_delivery import action_effect, normalize_action, preflight, regret


class FinalLedgerTests(unittest.TestCase):
    def action(self, variant="A_test"):
        original=dict(value=2.5, removed=[[1,2]], added=[[1,3]], affected_sources=[1],
                      native_support=[dict(source_id=1,target_id=3,probability=.9)],solver_status="optimal")
        step=dict(component="A",component_variant="A_test",before="incumbent",after="A_test")
        return normalize_action(original,dataset="sample",embryo="44b6",variant=variant,
            step=step,step_index=0,action_index=0,source_hash="sourcehash")

    def matches(self):
        mapping={1:10,2:20,3:30,4:40,5:50}
        before=dict(mapping=mapping,sets=dict(tp_pairs={(1,2)},fp_pairs={(4,5)},recovered_gt_edges={(10,20)}))
        after=dict(mapping=mapping,sets=dict(tp_pairs={(1,3)},fp_pairs={(4,5)},recovered_gt_edges={(10,30)}))
        return before,after

    def test_prediction_evidence_preserved_and_variant_identity_distinct(self):
        a=self.action()
        self.assertEqual(json.loads(a["original_prediction_evidence_json"])["native_support"][0]["probability"],.9)
        self.assertNotEqual(a["action_id"],self.action("AD_primary")["action_id"])
        self.assertFalse(any("gt_" in key for key in a))

    def test_fixed_match_edge_identity_changes_have_count_only_attribution(self):
        b,a=self.matches()
        effect=action_effect(self.action(),b,a,True)
        self.assertEqual((effect["removed_tp"],effect["added_tp"],effect["lost_gt_edges"],effect["gained_gt_edges"]),(1,1,1,1))
        self.assertIsNone(effect["per_action_score_delta"])
        self.assertIsNone(effect["per_action_division_tp_delta"])

    def test_coordinate_or_matching_interaction_has_no_invented_action_counts(self):
        b,a=self.matches()
        effect=action_effect(self.action(),b,a,False)
        for key in ("removed_tp","added_tp","removed_fp","added_fp","lost_gt_edges","gained_gt_edges"):
            self.assertIsNone(effect[key])

    def test_whole_graph_regret_separates_pair_identity_from_net_counts(self):
        b,a=self.matches()
        score=dict(edge_tp=1,edge_fp=1,edge_fn=2,division_tp=1,division_fp=0,division_fn=0,num_pred_nodes=5)
        result=regret(b,a,score,dict(score,division_tp=0,division_fp=1,division_fn=1))
        self.assertEqual(result["delta_edge_tp"],0)
        self.assertEqual(result["recovered_gt_edges_new"],1)
        self.assertEqual(result["recovered_gt_edges_lost"],1)
        self.assertEqual(result["delta_division_tp"],-1)

    def test_incomplete_round_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"inputs.json").write_text(json.dumps([dict(dataset=f"sample_{k}") for k in range(199)]))
            (root/"score_rows.csv").write_text("variant,dataset\nincumbent,sample_0\n")
            (root/"operating_points.csv").write_text("variant,embryo,score\nincumbent,pooled,0.9342063149703403\n")
            with self.assertRaisesRegex(ValueError,"32 variants"):
                preflight(RunContext.default(out=root))
            self.assertFalse((root/"edit_ledger.parquet").exists())


if __name__ == "__main__":
    unittest.main()
