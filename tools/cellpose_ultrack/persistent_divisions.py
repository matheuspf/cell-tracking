"""One declared follow-up: require both daughters to continue for one frame."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import sqlite3

from tools.annotation_selection.common import now, write_json
from tools.cellpose_ultrack import learned_divisions as optical
from tools.cellpose_ultrack.track import REPO, ROOT
from tools.detector_screen.cellpose_adapter import sha256

STUDY = "cellpose-ultrack-persistent-divisions-20260914"
OUT = REPO / "results" / STUDY
ARMS = optical.ARMS


def paths(arm):
    return REPO / "work" / f"{STUDY}-{arm}", REPO / "configs" / f"{STUDY}-{arm}.json"


def prepare():
    old_plan = json.loads((optical.OUT / "plan.json").read_text())
    audit_path = optical.OUT / "branch-audit.json"
    audit = json.loads(audit_path.read_text())
    panel = json.loads((ROOT / "panel.json").read_text())
    configs = {}
    for arm in ARMS:
        root, config = paths(arm)
        recipe = copy.deepcopy(json.loads(optical.paths(arm)[1].read_text()))
        recipe.update(study=f"{STUDY}-{arm}",
                      configuration_plan=str(OUT / "plan.json"),
                      division_persistence=dict(continuation_steps=1),
                      model_description=recipe["model_description"] + " + one-frame daughter persistence",
                      selection="Follow-up declared after optical-arm audit: one structural constraint, both daughters must have a selected outgoing edge when a following frame exists. Same frozen image scores and pair bank, zero-image matched control. No scalar or duration sweep. Reused pilot evidence.")
        write_json(config, recipe, immutable=True)
        configs[arm] = dict(root=str(root), config=str(config), sha256=sha256(config))
    payload = dict(study=STUDY, configs=configs, models=old_plan["models"],
                   panel_sha256=sha256(ROOT / "panel.json"),
                   evidence_plan_sha256=sha256(optical.OUT / "plan.json"),
                   evidence_scale=old_plan["evidence_scale"], logit_clip=old_plan["logit_clip"],
                   audit_sha256=sha256(audit_path), audit_summary=audit["summary"],
                   hypothesis="Nine of 29 optical-arm evaluable false divisions have a daughter with no continuation; the recovered true division has both daughters continue once. This post-prediction observation motivates an exploratory structural follow-up, not an independent validation.",
                   rule="Each chosen daughter must have at least one selected outgoing edge in the following frame. Clip-final daughters are exempt. Window seams use committed outgoing edges on the right and enforce persistence for inherited forks on the left. Unknown biological events are never used as negative labels.",
                   comparison="Both complete six-clip arms with the identical pair bank, one with zero optical bonus and one with the frozen optical bonus. Report both, including failures. No automatic baseline adoption.",
                   inherited_exposure=old_plan["inherited_exposure"])
    destination = OUT / "plan.json"
    if destination.exists():
        if {k:v for k,v in json.loads(destination.read_text()).items() if k != "created_utc"} != payload:
            raise ValueError("Declared persistence study changed")
    else:
        write_json(destination, dict(created_utc=now(), **payload), immutable=True)
    for arm in ARMS:
        root, _ = paths(arm)
        root.mkdir(parents=True, exist_ok=True)
        for name, source in [("images", ROOT / "images"), ("predictions", ROOT / "predictions"),
                             ("cellpose", ROOT / "cellpose"),
                             ("division-evidence", optical.WORK / "division-evidence")]:
            target = root / name
            if not target.is_symlink():
                target.symlink_to(source, target_is_directory=True)
            if target.resolve() != source.resolve():
                raise ValueError("Unexpected input alias")
        write_json(root / "panel.json", panel, immutable=True)
        for clip in panel["clips"]:
            source = ROOT / "tracking" / clip["dataset"]
            target = root / "tracking" / clip["dataset"]
            target.mkdir(parents=True, exist_ok=True)
            if (target / "reuse.json").exists():
                continue
            if (target / "data.db").exists():
                raise ValueError("Unrecorded database already exists")
            for name in ("data.db", "metadata.toml", "centroid-audit.json"):
                shutil.copy2(source / name, target / name)
            stages = json.loads((source / "stages.json").read_text())
            write_json(target / "stages.json", {k:v for k,v in stages.items() if k != "solve_seconds"})
            for name in ("foreground.zarr", "contours.zarr"):
                (target / name).symlink_to(source / name, target_is_directory=True)
            write_json(target / "reuse.json", dict(source=str(source),
                       source_inputs=json.loads((source / "inputs.json").read_text()),
                       initial_database_sha256=sha256(target / "data.db")), immutable=True)
        print(json.dumps(configs[arm]), flush=True)


def attach_persistence(solver, node_times, last_time, committed_nodes):
    """Constrain actual selected edges, including forks crossing window seams.

    committed_nodes is a read-only snapshot of (id, time, parent_id) for nodes
    selected by earlier windows. At uncommitted overlap ends the check is
    deferred; every exported interior fork is verified after all windows.
    """
    import mip
    native = solver._backward_map
    present = {int(n) for n in native}
    start, end = min(node_times[n] for n in present), max(node_times[n] for n in present)
    outgoing = defaultdict(list)
    for k, (a, _) in enumerate(solver._edges_df[["sources", "targets"]].itertuples(index=False, name=None)):
        outgoing[int(native[a])].append(solver._edges[k])
    committed_children = defaultdict(list)
    for child, _, parent in committed_nodes:
        if int(parent) >= 0:
            committed_children[int(parent)].append(int(child))
    right_anchored = any(int(t) == end + 1 for _, t, _ in committed_nodes)
    counts = dict(local_constraints=0, right_boundary_constraints=0, inherited_daughter_constraints=0,
                  clip_final_exemptions=0, uncommitted_overlap_deferrals=0)
    for event in list(solver._model.vars):
        if not event.name.startswith("pair_"):
            continue
        _, parent, first, second = event.name.split("_")
        for child in (int(first), int(second)):
            t = node_times[child]
            if t == last_time:
                counts["clip_final_exemptions"] += 1
            elif t == end and right_anchored:
                solver._model.add_constr(event <= int(bool(committed_children[child])))
                counts["right_boundary_constraints"] += 1
            elif t == end:
                counts["uncommitted_overlap_deferrals"] += 1
            else:
                solver._model.add_constr(event <= mip.xsum(outgoing[child]))
                counts["local_constraints"] += 1
    for parent, children in list(committed_children.items()):
        if len(children) == 2 and node_times[parent] < start:
            for child in children:
                if child in present and node_times[child] == start and start < last_time:
                    solver._model.add_constr(mip.xsum(outgoing[child]) >= 1)
                    counts["inherited_daughter_constraints"] += 1
    return dict(window_start=start, window_end=end, right_anchored=right_anchored, **counts)


def attach_from_database(solver, database, last_time):
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        times = {int(n):int(t) for n,t in db.execute("SELECT id,t FROM nodes")}
        committed = db.execute("SELECT id,t,parent_id FROM nodes WHERE selected").fetchall()
    return attach_persistence(solver, times, last_time, committed)


def validate_persistence(nodes, edges, last_time):
    times = {int(row[0]):int(row[1]) for row in nodes}
    children = defaultdict(list)
    for p,c in edges:
        children[int(p)].append(int(c))
    forks = [cs for cs in children.values() if len(cs) == 2]
    missing = [c for cs in forks for c in cs if times[c] < last_time and not children[c]]
    if missing:
        raise ValueError(f"Exported interior daughters lack continuation: {missing[:10]}")
    return dict(all_interior_daughters_persist=True, forks=len(forks),
                clip_final_daughters=sum(times[c] == last_time for cs in forks for c in cs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "collect"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        from tools.cellpose_ultrack.learned_results import collect
        collect(out=OUT, arms=ARMS, path_fn=paths)


if __name__ == "__main__":
    main()
