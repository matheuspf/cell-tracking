"""Verify and summarize the two complete learned-division tracking arms."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from tools.annotation_selection.common import now, write_json
from tools.cellpose_ultrack.event_costs import bank_fingerprint
from tools.cellpose_ultrack.learned_divisions import ARMS, OUT, WORK, paths
from tools.cellpose_ultrack.track import REPO, ROOT, validate_graph
from tools.detector_screen.cellpose_adapter import sha256


def collect(out=OUT, arms=ARMS, path_fn=paths):
    plan = json.loads((out / "plan.json").read_text())
    panel = json.loads((ROOT / "panel.json").read_text())
    if sha256(ROOT / "panel.json") != plan["panel_sha256"]:
        raise ValueError("Pilot cohort changed")
    result, checks = {}, {}
    for arm in arms:
        root, config = path_fn(arm)
        recipe = json.loads(config.read_text())
        summary = json.loads((root / "evaluation/summary.json").read_text())
        if not summary["complete_panel"] or summary["config_sha256"] != plan["configs"][arm]["sha256"]:
            raise ValueError("Incomplete or changed configuration")
        if sha256(config) != plan["configs"][arm]["sha256"]:
            raise ValueError("Declared settings changed")
        if summary["scored_datasets"] != [c["dataset"] for c in panel["clips"]]:
            raise ValueError("Changed scored cohort")
        statuses, all_gaps, clip_checks = [], [], []
        for clip, row in zip(panel["clips"], summary["rows"], strict=True):
            name = clip["dataset"]
            directory = root / "tracking" / name
            receipt = json.loads((directory / "result.json").read_text())
            evidence_path = WORK / "division-evidence" / f"{name}.npz"
            evidence = json.loads(evidence_path.with_suffix(".json").read_text())
            if receipt["division_evidence_sha256"] != sha256(evidence_path) or evidence["sha256"] != sha256(evidence_path):
                raise ValueError("Optical evidence changed")
            if sha256(directory / "graph.npz") != row["inputs"]["graph_sha256"] or receipt["graph_sha256"] != row["inputs"]["graph_sha256"]:
                raise ValueError("Scored graph differs from the saved prediction")
            if sha256(directory / "submission.csv") != receipt["csv_sha256"]:
                raise ValueError("Exported CSV changed")
            if bank_fingerprint(directory / "data.db") != evidence["inputs"]["source_bank"]:
                raise ValueError("Tracking changed region/candidate/exclusion data")
            if bank_fingerprint(ROOT / "tracking" / name / "data.db") != evidence["inputs"]["source_bank"]:
                raise ValueError("Original input bank changed")
            if sha256(directory / "metadata.toml") != sha256(ROOT / "tracking" / name / "metadata.toml"):
                raise ValueError("Image geometry metadata changed")
            for source_model in plan["models"].values():
                if sha256(Path(source_model["path"])) != source_model["sha256"]:
                    raise ValueError("Frozen optical checkpoint changed")
            with np.load(evidence_path,allow_pickle=False) as d:
                allowed = {tuple(map(int,p)):float(b) for p,b in zip(d["pairs"],d["bonus"],strict=True)}
                if len(allowed) != len(d["pairs"]):
                    raise ValueError("Duplicate division pair evidence")
            with np.load(directory / "graph.npz",allow_pickle=False) as d:
                nodes, edges = d["nodes"],d["edges"]
            validate_graph(nodes,edges,clip["shape"])
            if recipe.get("division_persistence"):
                from tools.cellpose_ultrack.persistent_divisions import validate_persistence
                measured = validate_persistence(nodes, edges, clip["shape"][0] - 1)
                if measured != receipt["division_persistence_check"]:
                    raise ValueError("Daughter persistence receipt changed")
                if (receipt["configuration_plan_sha256"] != sha256(out / "plan.json")
                        or receipt["division_persistence_adapter_sha256"] != sha256(REPO / "tools/cellpose_ultrack/persistent_divisions.py")):
                    raise ValueError("Persistence plan or solver adapter changed")
            children = {}
            for p,c in edges:
                children.setdefault(int(p),[]).append(int(c))
            forks = [(p,*sorted(cs)) for p,cs in children.items() if len(cs)==2]
            if any(p not in allowed for p in forks):
                raise ValueError("An exported fork bypassed explicit daughter-pair evidence")
            if len(forks) != receipt["predicted_divisions"]:
                raise ValueError("Fork count changed")
            if not receipt["graph_checks_passed"] or not receipt["csv_round_trip_passed"]:
                raise ValueError("Graph checks failed")
            if any(s["solutions"] < 1 for s in receipt["solver"]):
                raise ValueError("An unsolved window reached evaluation")
            statuses.extend(s["status"] for s in receipt["solver"])
            all_gaps.extend(s["gap"] for s in receipt["solver"])
            clip_checks.append(dict(dataset=name, selected_forks=len(forks),
                                    forks_with_positive_optical_evidence=sum(allowed[p]>0 for p in forks),
                                    selected_pair_bonus_sum=sum(allowed[p] for p in forks) if arm=="pair-image" else 0,
                                    graph_sha256=receipt["graph_sha256"],
                                    all_selected_forks_have_exact_pair_evidence=True,
                                    total_pair_variables=sum(s["division_evidence"]["pair_variables"] for s in receipt["solver"]),
                                    solve_seconds=receipt["stage_seconds"]["solve_seconds"]))
            if recipe.get("division_persistence"):
                clip_checks[-1]["daughter_persistence"] = receipt["division_persistence_check"]
                clip_checks[-1]["persistence_constraints"] = {
                    field:sum(s["division_persistence"][field] for s in receipt["solver"])
                    for field in ("local_constraints", "right_boundary_constraints", "inherited_daughter_constraints",
                                  "clip_final_exemptions", "uncommitted_overlap_deferrals")}
        result[arm] = dict(summaries=summary["summaries"], rows=[{k:v for k,v in r.items() if k!="inputs"} for r in summary["rows"]],
                           predicted_divisions=sum(r["predicted_divisions"] for r in summary["rows"]),
                           config_sha256=summary["config_sha256"])
        checks[arm] = dict(complete_clips=len(clip_checks), source_banks_and_metadata_unchanged=True,
                           graph_csv_and_actual_pair_identity_checks_passed=True,
                           solver_statuses=dict(Counter(statuses)), maximum_solver_relative_gap=max(all_gaps),
                           clips=clip_checks)
    for group in ("pooled","44b6","6bba"):
        a,b = [result[arm]["summaries"][group] for arm in arms]
        print(group,json.dumps(dict(control=a,image=b,score_change=b["score"]-a["score"])),flush=True)
    write_json(out / "comparison.json",dict(created_utc=now(),arms=result,validation=checks,
               plan_sha256=sha256(out / "plan.json"),
               optical_receipts=[json.loads((WORK / "division-evidence" / f"{c['dataset']}.json").read_text()) for c in panel["clips"]],
               paired_score_changes={g: result["pair-image"]["summaries"][g]["score"]-result["pair-control"]["summaries"][g]["score"] for g in ("pooled","44b6","6bba")},
               scope="Exploratory source-routed frozen optical evidence on the six reused pilot clips; inherited exposure and generator calibration persist. No new fitting, unseen-embryo validation or baseline adoption."))


if __name__ == "__main__":
    collect()
