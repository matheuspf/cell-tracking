"""Final contract manifests and unified action evidence, without new inference.

Run only after all 32 configurations have complete fresh official score/match
artifacts. The top-level edit table contains prediction evidence only. Observed
GT effects stay under evaluation/; division attribution remains whole-graph.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .common import digest, graph_hash, load_graph, now, read_json, sha, write_json
from .context import RunContext

BASELINE = 0.9342063149703403
COUNTS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes")
RESCUES = ("R_native_restore", "R_image_local", "R_heatmap_local", "R_image_persistent")
TEACHERS = tuple(f"C_{t}_canonical" for t in ("raw_neural", "old_final", "v2_E_native", "v2_E_hgb"))
MATCH_KEYS = ("tp_pairs", "fp_pairs", "recovered_gt_edges")


def expected_variants():
    from .association import VARIANTS as associations
    from .event_pipeline import variants as events
    from .replay import VARIANTS as repairs
    result = {"incumbent", "historical_v1", "AD_primary", "ADR_primary", *associations,
              *repairs, *RESCUES, *TEACHERS, *(v["name"] for v in events())}
    if len(result) != 32:
        raise ValueError("Final ledger contract requires the frozen 32-configuration grid")
    return result


def stage_name(variant):
    if variant in ("incumbent", "historical_v1"):
        return "V300"
    if variant.startswith("A_"):
        return "V320"
    if variant.startswith("D_"):
        return "V340"
    if variant in RESCUES:
        return "V350"
    if variant in ("AD_primary", "ADR_primary"):
        return "V360"
    return "V310"


def graph_path(ctx, variant, name):
    if variant == "incumbent":
        return ctx.incumbent(name)
    if variant == "historical_v1":
        return ctx.v1 / "baseline/public" / f"{name}.npz"
    return ctx.out / "candidate_graphs" / variant / f"{name}.npz"


def load_match(path):
    arrays = load_graph(path)
    return dict(sets={k: set(map(tuple, arrays[k])) for k in MATCH_KEYS},
                mapping=dict(map(tuple, arrays["matched_ids"])))


def regret(before, after, before_score, after_score):
    row = {"fresh_matching_changed": before["mapping"] != after["mapping"]}
    for key, prefix in (("tp_pairs", "predicted_tp_pairs"), ("fp_pairs", "fp_pairs"),
                        ("recovered_gt_edges", "recovered_gt_edges")):
        old, new = before["sets"][key], after["sets"][key]
        row.update({prefix + "_new": len(new - old), prefix + "_lost": len(old - new),
                    prefix + "_retained": len(old & new)})
    for key in COUNTS:
        row["delta_" + key] = int(after_score[key]) - int(before_score[key])
    if row["predicted_tp_pairs_new"] - row["predicted_tp_pairs_lost"] != row["delta_edge_tp"]:
        raise ValueError("Fresh TP-pair census disagrees with official counts")
    if row["recovered_gt_edges_new"] - row["recovered_gt_edges_lost"] != row["delta_edge_tp"]:
        raise ValueError("Fresh GT-edge census disagrees with official counts")
    if row["fp_pairs_new"] - row["fp_pairs_lost"] != row["delta_edge_fp"]:
        raise ValueError("Fresh FP-pair census disagrees with official counts")
    return row


def read_steps(ctx, variant, name):
    """Return complete stage actions and their original prediction receipts."""
    def association(v):
        path = ctx.out / "association_ledgers" / v / f"{name}.json"
        return dict(component="A", component_variant=v, before="incumbent", after=v,
                    path=path, actions=read_json(path)["edits"])
    if variant.startswith("A_"):
        return [association(variant)]
    if variant.startswith("D_"):
        path = ctx.out / "edit_ledgers" / variant / f"{name}.json"
        return [dict(component="D", component_variant=variant, before="incumbent", after=variant,
                     path=path, actions=read_json(path)["actions"])]
    if variant in RESCUES:
        if variant == "R_image_persistent":
            path = ctx.out / "rescue_fixed" / f"{name}.json"
            actions = read_json(path)["ledger"]
        elif variant == "R_heatmap_local":
            path = ctx.out / "heatmap_rescue" / f"{name}.json"
            actions = read_json(path)["actions"]
        else:
            path = ctx.out / "rescue" / f"{name}.json"
            actions = read_json(path)["variants"][variant]["ledger"]
        return [dict(component="R", component_variant=variant, before="incumbent", after=variant,
                     path=path, actions=actions)]
    if variant in ("AD_primary", "ADR_primary"):
        path = ctx.out / "combinations" / f"{name}.json"
        record = read_json(path)
        steps = [association("A_residual_m1.5"),
                 dict(component="D", component_variant="D_image_p020", before="A_residual_m1.5",
                      after="AD_primary", path=path, actions=record["division"]["ledger"])]
        if variant == "ADR_primary":
            steps.append(dict(component="R", component_variant="R_image_persistent", before="AD_primary",
                              after="ADR_primary", path=path, actions=record["rescue_ledger"]))
        return steps
    return []


def normalize_action(action, *, dataset, embryo, variant, step, step_index, action_index, source_hash):
    removed = [list(map(int, e)) for e in action.get("removed_edges", action.get("removed", []))]
    added = [list(map(int, e)) for e in action.get("added_edges", action.get("added", []))]
    coordinate = action.get("kind") == "local_image_centroid"
    inserted = list(map(int, action.get("node_ids", [])))
    affected = {int(v) for e in removed + added for v in e}
    affected.update(inserted)
    affected.update(int(v) for v in action.get("affected_sources", []))
    if "node_id" in action:
        affected.add(int(action["node_id"]))
    if "anchor_id" in action:
        affected.add(int(action["anchor_id"]))
    identity = dict(dataset=dataset, variant=variant, step_index=step_index,
                    action_index=action_index, source_sha256=source_hash)
    return dict(**identity, action_id=digest(identity), embryo=embryo, stage=stage_name(variant),
        component=step["component"], component_variant=step["component_variant"],
        before_variant=step["before"], after_variant=step["after"],
        kind=action.get("kind", "association_rewire"),
        objective_value=action.get("value"), event_probability=action.get("event_probability"),
        solver_status=action.get("solver_status", "deterministic_image_evidence"),
        coordinate_change=coordinate, node_insertion=bool(inserted),
        removed_edges_json=json.dumps(removed, separators=(",", ":")),
        added_edges_json=json.dumps(added, separators=(",", ":")),
        affected_node_ids_json=json.dumps(sorted(affected), separators=(",", ":")),
        inserted_node_ids_json=json.dumps(inserted, separators=(",", ":")),
        owner_alternatives_json=json.dumps(action.get("owner_alternatives", []), separators=(",", ":")),
        original_prediction_evidence_json=json.dumps(action, sort_keys=True, separators=(",", ":")))


def action_effect(action, before, after, identified):
    """Counts, never score/division credit; null when matching effects interact."""
    result = dict(action_id=action["action_id"], dataset=action["dataset"], variant=action["variant"],
        step_index=action["step_index"], action_index=action["action_index"], component=action["component"],
        before_variant=action["before_variant"], after_variant=action["after_variant"],
        additive_edge_count_attribution=identified,
        scope="fixed-node fresh-matching edge decomposition" if identified else "whole-stage effects only; coordinate/node/matching interactions",
        removed_tp=None, added_tp=None, removed_fp=None, added_fp=None,
        removed_unknown=None, added_unknown=None, lost_gt_edges=None, gained_gt_edges=None,
        per_action_score_delta=None, per_action_division_tp_delta=None)
    if not identified:
        return result
    removed = set(map(tuple, json.loads(action["removed_edges_json"])))
    added = set(map(tuple, json.loads(action["added_edges_json"])))
    result.update(removed_tp=len(removed & before["sets"]["tp_pairs"]),
        added_tp=len(added & after["sets"]["tp_pairs"]),
        removed_fp=len(removed & before["sets"]["fp_pairs"]),
        added_fp=len(added & after["sets"]["fp_pairs"]),
        removed_unknown=len(removed - before["sets"]["tp_pairs"] - before["sets"]["fp_pairs"]),
        added_unknown=len(added - after["sets"]["tp_pairs"] - after["sets"]["fp_pairs"]))
    old_truth = {(before["mapping"][a], before["mapping"][b]) for a, b in removed & before["sets"]["tp_pairs"]}
    new_truth = {(after["mapping"][a], after["mapping"][b]) for a, b in added & after["sets"]["tp_pairs"]}
    result.update(lost_gt_edges=len(old_truth - new_truth), gained_gt_edges=len(new_truth - old_truth))
    return result


class ParquetStream:
    def __init__(self, path, schema):
        import pyarrow.parquet as pq
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix(".tmp.parquet")
        self.schema = schema
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self.rows = 0
    def write(self, rows):
        import pyarrow as pa
        if rows:
            self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
            self.rows += len(rows)
    def close(self, success=True):
        self.writer.close()
        if success:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)


def schemas():
    import pyarrow as pa
    strings = ["dataset", "embryo", "variant", "stage", "component", "component_variant", "before_variant", "after_variant",
        "action_id", "source_sha256", "source_ledger", "before_graph_hash", "after_graph_hash", "kind", "solver_status",
        "removed_edges_json", "added_edges_json", "affected_node_ids_json", "inserted_node_ids_json",
        "owner_alternatives_json", "original_prediction_evidence_json"]
    action = pa.schema([(k, pa.string()) for k in strings] + [(k, pa.int64()) for k in ["step_index", "action_index"]]
        + [(k, pa.float64()) for k in ["objective_value", "event_probability"]]
        + [(k, pa.bool_()) for k in ["coordinate_change", "node_insertion"]])
    effect = pa.schema([(k, pa.string()) for k in ["action_id", "dataset", "variant", "component", "before_variant", "after_variant", "scope"]]
        + [(k, pa.int64()) for k in ["step_index", "action_index", "removed_tp", "added_tp", "removed_fp", "added_fp",
                                    "removed_unknown", "added_unknown", "lost_gt_edges", "gained_gt_edges", "per_action_division_tp_delta"]]
        + [("additive_edge_count_attribution", pa.bool_()), ("per_action_score_delta", pa.float64())])
    return action, effect


def preflight(ctx):
    import pandas as pd
    rows = pd.read_csv(ctx.out / "score_rows.csv")
    points = pd.read_csv(ctx.out / "operating_points.csv")
    expected = expected_variants()
    names = {s["dataset"] for s in ctx.samples()}
    if len(names) != 199 or set(rows.variant) != expected or len(rows) != 32 * 199:
        raise ValueError("Ledger delivery waits for all 32 variants × 199 official score rows")
    if rows.duplicated(["variant", "dataset"]).any():
        raise ValueError("Duplicate score rows")
    if set(points.variant) != expected or len(points) != 32 * 3 or points.duplicated(["variant", "embryo"]).any():
        raise ValueError("Pooled and both embryo operating points are required for all 32 variants")
    for variant, group in rows.groupby("variant"):
        if set(group.dataset) != names:
            raise ValueError(f"Incomplete exact clip set: {variant}")
        for folder in ("scores", "matches"):
            path = ctx.out / "evaluation" / folder / variant
            suffix = "*.json" if folder == "scores" else "*.npz"
            if {p.stem for p in path.glob(suffix)} != names:
                raise ValueError(f"Incomplete fresh {folder}: {variant}")
    baseline = points[(points.variant == "incumbent") & (points.embryo == "pooled")]
    if len(baseline) != 1 or abs(float(baseline.iloc[0].score) - BASELINE) > 1e-12:
        raise ValueError("Ledger comparisons must use the preserved v2 incumbent")
    return rows, points


def aggregate_regret(rows, points, *, stages=False):
    point = {(r["variant"], r["embryo"]): r for r in points.to_dict("records")}
    groups = defaultdict(list)
    identity = ["variant", "stage"] + (["step_index", "component", "component_variant", "before_variant", "after_variant"] if stages else [])
    for row in rows:
        for embryo in (row["embryo"], "pooled"):
            groups[tuple(row[k] for k in identity) + (embryo,)].append(row)
    numeric = ["fresh_matching_changed", *[p + "_" + s for p in ("predicted_tp_pairs", "fp_pairs", "recovered_gt_edges")
        for s in ("new", "lost", "retained")], *["delta_" + k for k in COUNTS]]
    result = []
    for key, members in sorted(groups.items()):
        row = dict(zip([*identity, "embryo"], key))
        embryo = row["embryo"]
        row.update(samples=len(members), **{k: int(sum(m[k] for m in members)) for k in numeric})
        final = point[row["variant"], embryo]
        row.update(variant_score=float(final["score"]), delta_v2=float(final["score"] - point["incumbent", embryo]["score"]))
        if stages:
            before, after = point[row["before_variant"], embryo], point[row["after_variant"], embryo]
            row.update(stage_after_score=float(after["score"]),
                stage_after_delta_v2=float(after["score"] - point["incumbent", embryo]["score"]),
                incremental_stage_score_delta=float(after["score"] - before["score"]),
                count_delta_reference="before_variant; final variant and stage-after score deltas both explicitly versus v2",
                additive_edge_count_clips=sum(m["additive_edge_count_attribution"] for m in members),
                coordinate_or_node_changing_clips=sum(m["node_arrays_changed"] for m in members),
                accepted_actions=sum(m["accepted_actions"] for m in members))
        else:
            row["count_delta_reference"] = "v2 incumbent"
        result.append(row)
    return result


def run(ctx=None, args=None):
    """Generate final contracts after scoring; all detailed outputs remain local."""
    import pandas as pd
    ctx = ctx or RunContext.default()
    ctx.check_outputs()
    score_rows, points = preflight(ctx)
    variants = sorted(expected_variants())
    scores = {(r["variant"], r["dataset"]): r for r in score_rows.to_dict("records")}
    action_schema, effect_schema = schemas()
    actions_out = ParquetStream(ctx.out / "edit_ledger.parquet", action_schema)
    effects_out = ParquetStream(ctx.out / "evaluation/edit_effects.parquet", effect_schema)
    whole, stages, source_files = [], [], {}
    action_counts = Counter()
    succeeded = False
    try:
        for sample in ctx.samples():
            name, embryo = sample["dataset"], sample["embryo"]
            matches, graphs = {}, {}
            def match(v):
                if v not in matches:
                    receipt=read_json(ctx.out/"evaluation/scores"/v/f"{name}.json")
                    cached=scores[v,name]
                    if (receipt["result"]["graph_hash"]!=cached["graph_hash"] or
                        receipt["inputs"]["graph_file_sha256"]!=cached["graph_file_sha256"] or
                        any(int(receipt["result"][k])!=int(cached[k]) for k in COUNTS)):
                        raise ValueError("Collected score rows differ from fresh per-graph receipts")
                    item=load_match(ctx.out / "evaluation/matches" / v / f"{name}.npz")
                    if (len(item["sets"]["tp_pairs"])!=int(cached["edge_tp"]) or
                        len(item["sets"]["recovered_gt_edges"])!=int(cached["edge_tp"]) or
                        len(item["sets"]["fp_pairs"])!=int(cached["edge_fp"]) or
                        len(set(item["mapping"].values()))!=len(item["mapping"])):
                        raise ValueError("Fresh matching arrays disagree with their official score receipt")
                    matches[v] = item
                return matches[v]
            def graph(v):
                if v not in graphs:
                    path = graph_path(ctx, v, name)
                    if sha(path) != scores[v, name]["graph_file_sha256"]:
                        raise ValueError(f"Scored graph changed: {v}/{name}")
                    g = load_graph(path)
                    if graph_hash(g["nodes"], g["edges"]) != scores[v, name]["graph_hash"]:
                        raise ValueError("Scored graph array fingerprint differs")
                    graphs[v] = g
                return graphs[v]
            base = match("incumbent")
            for variant in variants:
                whole.append(dict(dataset=name, embryo=embryo, variant=variant, stage=stage_name(variant),
                    reference_variant="incumbent", **regret(base, match(variant), scores["incumbent", name], scores[variant, name])))
                for step_index, step in enumerate(read_steps(ctx, variant, name)):
                    before, after = step["before"], step["after"]
                    bg, ag = graph(before), graph(after)
                    bm, am = match(before), match(after)
                    node_changed = not np.array_equal(bg["nodes"], ag["nodes"])
                    identified = not node_changed and bm["mapping"] == am["mapping"]
                    source_path = str(step["path"].relative_to(ctx.out))
                    source_hash = source_files.setdefault(source_path, sha(step["path"]))
                    edge_state = set(map(tuple, bg["edges"]))
                    action_rows, effect_rows = [], []
                    totals = Counter()
                    for action_index, original in enumerate(step["actions"]):
                        action = normalize_action(original, dataset=name, embryo=embryo, variant=variant, step=step,
                            step_index=step_index, action_index=action_index, source_hash=source_hash)
                        action.update(source_ledger=source_path, before_graph_hash=scores[before, name]["graph_hash"],
                                      after_graph_hash=scores[after, name]["graph_hash"])
                        removed = set(map(tuple, json.loads(action["removed_edges_json"])))
                        added = set(map(tuple, json.loads(action["added_edges_json"])))
                        if not removed <= edge_state or (added - removed) & edge_state:
                            raise ValueError(f"Action ledger is not a valid atomic edge replay: {variant}/{name}/{action_index}")
                        edge_state.difference_update(removed)
                        edge_state.update(added)
                        effect = action_effect(action, bm, am, identified)
                        if identified:
                            totals["tp"] += effect["added_tp"] - effect["removed_tp"]
                            totals["fp"] += effect["added_fp"] - effect["removed_fp"]
                        action_rows.append(action)
                        effect_rows.append(effect)
                        action_counts[(variant, step["component"], action["kind"])] += 1
                    if edge_state != set(map(tuple, ag["edges"])):
                        raise ValueError(f"Complete action ledger does not reconstruct stage edges: {variant}/{name}/{step_index}")
                    change = regret(bm, am, scores[before, name], scores[after, name])
                    if identified and (totals["tp"] != change["delta_edge_tp"] or totals["fp"] != change["delta_edge_fp"]):
                        raise ValueError("Additive action edge counts do not reconcile with fresh whole-stage counts")
                    stages.append(dict(dataset=name, embryo=embryo, variant=variant, stage=stage_name(variant), step_index=step_index,
                        component=step["component"], component_variant=step["component_variant"], before_variant=before, after_variant=after,
                        node_arrays_changed=node_changed, additive_edge_count_attribution=identified,
                        accepted_actions=len(action_rows), full_edge_replay_verified=True, **change))
                    actions_out.write(action_rows)
                    effects_out.write(effect_rows)
            print(f"ledger_delivery {name}: {actions_out.rows:,} actions", flush=True)
        succeeded = True
    finally:
        actions_out.close(succeeded)
        effects_out.close(succeeded)
    pd.DataFrame(whole).to_parquet(ctx.out / "evaluation/variant_clip_regret.parquet", index=False)
    pd.DataFrame(stages).to_parquet(ctx.out / "evaluation/stage_clip_regret.parquet", index=False)
    summary = dict(created=now(),schema_version="v3_final_action_contract_1",samples=199,variants=32,score_rows=6368,
        baseline_v2=BASELINE,actions=actions_out.rows,action_effect_rows=effects_out.rows,
        action_counts=[dict(variant=v,component=c,kind=k,actions=n) for (v,c,k),n in sorted(action_counts.items())],
        variant_regret=aggregate_regret(whole,points),stage_regret=aggregate_regret(stages,points,stages=True),
        action_ledger_has_gt_fields=False,gt_effects_separate=True,all_stage_edge_replays_exact=True,
        additive_counts_only_when_nodes_and_fresh_matching_unchanged=True,
        no_per_action_division_or_score_attribution=True,
        scope="Final exported evidence only; no predictions, fits, thresholds or official scores were changed",
        caveats=["A/D/R actions in combined policies are intentionally repeated under their final variant.",
                 "Per-action GT effects are null when node coordinates, node sets or fresh matching change.",
                 "Whole-stage and whole-variant scores come from complete official graph evaluation; stage deltas do not imply independent gains.",
                 "Division assignment is global within each clip; no per-action division credit is fabricated.",
                 "R_image_local remains an invalid-for-promotion historical control."])
    summary["outputs"]={str(p.relative_to(ctx.out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in
        [ctx.out/"edit_ledger.parquet",ctx.out/"evaluation/edit_effects.parquet",ctx.out/"evaluation/variant_clip_regret.parquet",ctx.out/"evaluation/stage_clip_regret.parquet"]}
    write_json(ctx.out/"ledger_delivery_summary.json",summary)
    manifests(ctx,points,source_files,summary)
    return summary


def manifests(ctx,points,source_files,summary):
    """Hash-only public-safe indexes; detailed source tables stay outside Git."""
    def record(path):
        path=Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return dict(path=str(path.relative_to(ctx.out)),sha256=sha(path),bytes=path.stat().st_size)
    association=read_json(ctx.out/"association_summary.json")
    coverage=read_json(ctx.out/"event_candidate_coverage.json")
    inputs=["incumbent_manifest.json","label_free_input_hash_manifest.json","association_feature_lock.json",
        "association_summary.json","teacher_reconciliation_summary.json","event_candidate_coverage.json",
        "event_table_manifest.json","event_inference_config.json","rescue_config_lock.json","rescue_fixed_config_lock.json",
        "rescue_fixed_prediction_lock.json","heatmap_rescue_config.json","combination_config_lock.json"]
    candidate=dict(created=now(),schema_version="v3_final_candidate_index_1",samples=199,baseline_v2=BASELINE,
        canonical_graph="Preserved v2 selected_predictions; per-stage graph/hash provenance remains in action and source receipts",
        image_features="Rebuilt from current centers and adjacency; raw/pre-ILP native IDs require proven correspondence",
        inference_tables_have_no_gt_fields=True,
        source_coverage_is_evaluation_metadata_not_inference_input=True,
        association_candidate_counts=association["feature_totals"],event_pool_coverage=coverage,
        provenance=[record(ctx.out/name) for name in inputs],
        prediction_tables=[record(ctx.out/name) for name in ["disagreements.parquet","event_features.parquet","edit_ledger.parquet"]],
        training_only_table=record(ctx.out/"evaluation/training_labels.parquet"),
        source_prediction_receipts={name:digest({p:h for p,h in source_files.items() if p.split('/')[0]==name})
                                    for name in sorted({p.split('/')[0] for p in source_files})},
        detailed_receipt_hashes=source_files,
        local_only=["edit_ledger.parquet","evaluation/training_labels.parquet","evaluation/edit_effects.parquet",
                    "evaluation/variant_clip_regret.parquet","evaluation/stage_clip_regret.parquet"],
        notes=["This manifest indexes existing proposals and evidence; it creates no candidates.",
               "Gate counts, coordinate transforms and native provenance are preserved in per-clip source receipts.",
               "Table negative-sampling metadata is parent_group_sampling_fraction, not exact row propensity."])
    write_json(ctx.out/"candidate_manifest.json",candidate)
    rounds={p.name:record(p) for p in sorted((ctx.out/"rounds").glob('*.json'))}
    variants={}
    for filename in rounds:
        for variant,hashes in read_json(ctx.out/"rounds"/filename)["variants"].items():
            if variant in variants and variants[variant]!=hashes:
                raise ValueError("Same variant has incompatible round fingerprints")
            variants[variant]=hashes
    if set(variants)!=expected_variants() or any(len(v)!=199 for v in variants.values()):
        raise ValueError("Final round index lacks exact 32×199 frozen graph hashes")
    locks=["association_model_lock.json","association_round_lock.json","event_model_lock.json","event_probability_lock.json",
           "event_grid_lock.json","event_round_lock.json","combination_config_lock.json","combination_inference_receipt.json"]
    payload=dict(schema_version="v3_final_round_index_1",baseline_v2=BASELINE,variants=variants,
        rounds=rounds,source_and_policy_locks=[record(ctx.out/name) for name in locks],
        inputs=record(ctx.out/"inputs.json"),operating_points=record(ctx.out/"operating_points.csv"),
        score_rows=record(ctx.out/"score_rows.csv"),metric_revision=ctx.metric_revision,
        code_sha256=sha(__file__),both_direction_models_frozen_before_corresponding_round_scores=True,
        exposure="Sequential stage outcomes were examined; source policies and each complete comparison round retain their original exposure receipts",
        evidence_only=True,graph_configurations=32,expected_clips=199,
        evaluation_ledgers_excluded_from_inference=True)
    path=ctx.out/"round_lock.json"
    if path.exists():
        old=read_json(path)
        if old!=payload:
            raise ValueError("Final round index changed; preserve the prior delivery snapshot")
    else:
        write_json(path,payload,immutable=True)


if __name__ == "__main__":
    run()
