"""Bounded event-cost controls on the frozen six-clip Cellpose hypothesis bank.

Prepare all configurations before any new scoring. Run ``track`` and ``evaluate``
with the printed roots/configs, then collect the matched results with ``collect``.
The input masks, hierarchy, candidate links and their IoU weights stay fixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sqlite3

from tools.annotation_selection.common import now, write_json
from tools.cellpose_ultrack.track import REPO, ROOT
from tools.detector_screen.cellpose_adapter import sha256

STUDY = "cellpose-ultrack-event-costs-20260914"
OUT = REPO / "results" / STUDY
BASE_CONFIG = REPO / "configs/cellpose-ultrack-windowed-v1.json"
SUPPLEMENT_REV = "14d24aa3cada922d2ff7ac5b8adf39df85658b44"
ARMS = {
    "division-surcharge": {
        "tracking": {"division_weight": -0.011},
        "purpose": "A 0.01 extra cost for a fork relative to a new track; an interior second link must supply IoU^4 > 0.01 (IoU > 0.3162) in the fixed-node local comparison. This is an objective-scale control, not a fitted biological prior.",
    },
    "reference-events": {
        "tracking": {"appear_weight": -0.1, "disappear_weight": -0.1, "division_weight": -0.1},
        "purpose": "Copy only the three signed event weights from the pinned sparse_zebrafish supplementary configuration. Equal birth and division weights still give a zero local second-link threshold; this tests the accompanying continuity pressure. It is not a replication of the paper's dataset or pipeline.",
    },
    "no-division-control": {
        "tracking": {"division_weight": -1.01},
        "purpose": "Diagnostic boundary: with IoU^4 in [0,1] and birth weight -0.001, replacing a fork's extra edge by a birth strictly improves the objective. Require zero exported forks. This is not a division model or a deployment candidate.",
    },
}


def paths(arm):
    return (REPO / "work" / f"{STUDY}-{arm}",
            REPO / "configs" / f"{STUDY}-{arm}.json")


def bank_fingerprint(path):
    """Hash all candidate data, omitting only the solver's selected/parent fields."""
    result = {}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        for table in ("nodes", "links", "overlaps"):
            columns = [r[1] for r in db.execute(f"PRAGMA table_info({table})")
                       if table != "nodes" or r[1] not in ("selected", "parent_id")]
            h = hashlib.sha256()
            h.update(json.dumps(columns).encode())
            count = 0
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY id"):
                value = pickle.dumps(row, protocol=4)
                h.update(len(value).to_bytes(8, "little"))
                h.update(value)
                count += 1
            result[table] = {"rows": count, "sha256": h.hexdigest()}
        lo, hi = db.execute("SELECT min(weight),max(weight) FROM links").fetchone()
        if not (0 <= lo <= hi <= 1):
            raise ValueError("The no-division objective bound requires IoU weights in [0,1]")
        result["raw_link_weight_range"] = [lo, hi]
    return result


def prepare():
    recipe = json.loads(BASE_CONFIG.read_text())
    panel = json.loads((ROOT / "panel.json").read_text())
    configs = {}
    for arm, spec in ARMS.items():
        root, config = paths(arm)
        new = copy.deepcopy(recipe)
        new["study"] = f"{STUDY}-{arm}"
        new["tracking"].update(spec["tracking"])
        new["selection"] = ("Declared before this bounded cost study was scored; previous six-clip stock results and error audit were already inspected. "
                            + spec["purpose"] + " No further search or best-run replacement is part of this study.")
        new["exposure"] += " Event-cost controls are development ablations on the already inspected pilot panel."
        write_json(config, new, immutable=True)
        configs[arm] = dict(root=str(root), config=str(config), config_sha256=sha256(config), **spec)
    plan_path = OUT / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["arms"] != configs or plan["source_config_sha256"] != sha256(BASE_CONFIG):
            raise ValueError("The declared study changed; create a new study")
    else:
        plan = dict(schema=1, created_utc=now(), study=STUDY, arms=configs,
                    source_root=str(ROOT), source_config_sha256=sha256(BASE_CONFIG),
                    source_summary_sha256=sha256(REPO / "results" / ROOT.name / "summary.json"),
                    panel_sha256=sha256(ROOT / "panel.json"),
                    forum="https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573",
                    supplementary_revision=SUPPLEMENT_REV,
                    supplementary_config=f"https://github.com/royerlab/ultrack_supplementary/blob/{SUPPLEMENT_REV}/configuration/sparse_zebrafish/config.toml",
                    endpoints="Fresh full official graph metric on all six clips, embryo summaries, edge and division TP/FP/FN, all predicted forks, selected nodes, feasible solver status and unchanged candidate bank.",
                    selection="Report every declared arm; no further threshold sweep or baseline adoption. All masks, hypotheses, links, windows and compute limits remain fixed.")
        write_json(plan_path, plan, immutable=True)
    source_banks = {c["dataset"]: bank_fingerprint(ROOT / "tracking" / c["dataset"] / "data.db")
                    for c in panel["clips"]}
    write_json(OUT / "source-banks.json", source_banks, immutable=True)
    for arm in ARMS:
        root, config = paths(arm)
        root.mkdir(parents=True, exist_ok=True)
        for name in ("images", "predictions", "cellpose"):
            link = root / name
            if not link.exists():
                link.symlink_to(ROOT / name, target_is_directory=True)
            if link.resolve() != (ROOT / name).resolve():
                raise ValueError(f"Unexpected reused input: {link}")
        write_json(root / "panel.json", panel, immutable=True)
        for clip in panel["clips"]:
            name = clip["dataset"]
            source, target = ROOT / "tracking" / name, root / "tracking" / name
            target.mkdir(parents=True, exist_ok=True)
            metadata = target / "metadata.toml"
            if not metadata.exists():
                shutil.copy2(source / "metadata.toml", metadata)
            if sha256(metadata) != sha256(source / "metadata.toml"):
                raise ValueError("Reused image geometry metadata changed")
            reuse_path = target / "reuse.json"
            if not reuse_path.exists():
                if (target / "data.db").exists() or (target / "inputs.json").exists():
                    raise ValueError(f"Refusing to overwrite an unrecorded tracking database: {target}")
                shutil.copy2(source / "data.db", target / "data.db")
                write_json(reuse_path, dict(source=str(source),
                           source_inputs=json.loads((source / "inputs.json").read_text()),
                           initial_database_sha256=sha256(target / "data.db")), immutable=True)
                stages = json.loads((source / "stages.json").read_text())
                write_json(target / "stages.json", {k: v for k, v in stages.items() if k != "solve_seconds"})
                shutil.copy2(source / "centroid-audit.json", target / "centroid-audit.json")
                for array in ("foreground.zarr", "contours.zarr"):
                    (target / array).symlink_to(source / array, target_is_directory=True)
            if bank_fingerprint(target / "data.db") != source_banks[name]:
                raise ValueError(f"Candidate bank changed for {arm}/{name}")
        print(json.dumps({"arm": arm, "root": str(root), "config": str(config)}), flush=True)


def collect():
    plan = json.loads((OUT / "plan.json").read_text())
    source_banks = json.loads((OUT / "source-banks.json").read_text())
    panel = json.loads((ROOT / "panel.json").read_text())
    if sha256(ROOT / "panel.json") != plan["panel_sha256"]:
        raise ValueError("Panel changed")
    original = REPO / "results" / ROOT.name / "summary.json"
    if sha256(original) != plan["source_summary_sha256"]:
        raise ValueError("Original scored control changed")
    sources = {"stock": json.loads(original.read_text())}
    for row in sources["stock"]["rows"]:
        directory = ROOT / "tracking" / row["dataset"]
        receipt = json.loads((directory / "result.json").read_text())
        if sha256(directory / "graph.npz") != row["inputs"]["graph_sha256"]:
            raise ValueError("Original control graph changed")
        if sha256(directory / "submission.csv") != receipt["csv_sha256"]:
            raise ValueError("Original control CSV changed")
    validation = {}
    for arm in ARMS:
        root, config = paths(arm)
        if sha256(config) != plan["arms"][arm]["config_sha256"]:
            raise ValueError(f"Declared cost configuration changed: {arm}")
        summary = json.loads((root / "evaluation/summary.json").read_text())
        if not summary["complete_panel"] or summary["config_sha256"] != sha256(config):
            raise ValueError(f"Incomplete or mismatched evaluation: {arm}")
        if summary["scored_datasets"] != [c["dataset"] for c in panel["clips"]]:
            raise ValueError("The scored cohort changed")
        scored = {r["dataset"]: r for r in summary["rows"]}
        statuses, gaps = [], []
        for clip in panel["clips"]:
            name = clip["dataset"]
            directory = root / "tracking" / name
            receipt = json.loads((directory / "result.json").read_text())
            if sha256(directory / "graph.npz") != receipt["graph_sha256"]:
                raise ValueError("Scored graph changed")
            if receipt["graph_sha256"] != scored[name]["inputs"]["graph_sha256"]:
                raise ValueError("Evaluation belongs to a different graph")
            if sha256(directory / "submission.csv") != receipt["csv_sha256"]:
                raise ValueError("Exported CSV changed after round-trip validation")
            if not receipt["graph_checks_passed"] or not receipt["csv_round_trip_passed"]:
                raise ValueError("Graph/serialization validation failed")
            if bank_fingerprint(directory / "data.db") != source_banks[name]:
                raise ValueError(f"Solver mutated its candidate bank: {arm}/{name}")
            if bank_fingerprint(ROOT / "tracking" / name / "data.db") != source_banks[name]:
                raise ValueError("Source candidate bank changed")
            if sha256(directory / "metadata.toml") != sha256(ROOT / "tracking" / name / "metadata.toml"):
                raise ValueError("Reused image geometry metadata changed")
            statuses.extend(r["status"] for r in receipt["solver"])
            gaps.extend(r["gap"] for r in receipt["solver"])
            if any(r["solutions"] < 1 for r in receipt["solver"]):
                raise ValueError("No feasible solution in a tracking window")
            if arm == "no-division-control" and receipt["predicted_divisions"] != 0:
                raise ValueError("No-division objective control exported a fork")
        validation[arm] = dict(candidate_banks_unchanged=True, original_control_unchanged=True,
                               graph_and_csv_checks_passed=True, solver_statuses=statuses,
                               maximum_solver_relative_gap=max(gaps))
        sources[arm] = summary
    results = {}
    for arm, summary in sources.items():
        results[arm] = dict(summaries=summary["summaries"],
                           predicted_divisions=sum(r["predicted_divisions"] for r in summary["rows"]),
                           selected_edges=sum(r["selected_edges"] for r in summary["rows"]),
                           gt_nodes=sum(r["gt_nodes"] for r in summary["rows"]),
                           matched_nodes=sum(r["matched_nodes"] for r in summary["rows"]),
                           config_sha256=summary["config_sha256"],
                           rows=[{k: v for k, v in r.items() if k != "inputs"} for r in summary["rows"]])
        results[arm]["score_change_from_stock"] = (summary["summaries"]["pooled"]["score"]
                                                   - sources["stock"]["summaries"]["pooled"]["score"])
    write_json(OUT / "comparison.json", dict(schema=1, created_utc=now(), plan_sha256=sha256(OUT / "plan.json"),
               arms=results, validation=validation,
               baseline_error_analysis_sha256=sha256(REPO / "results" / ROOT.name / "error-analysis.json"),
               scope="Development controls on the same six already inspected clips. Sparse divisions are not a biological base-rate estimate. No baseline promotion or hidden-test claim."))
    for arm, result in results.items():
        print(arm, json.dumps(result["summaries"]["pooled"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "collect"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else collect()


if __name__ == "__main__":
    main()
