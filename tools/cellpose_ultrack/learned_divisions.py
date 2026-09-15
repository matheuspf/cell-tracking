"""Add frozen temporal image evidence for explicit ultrack daughter pairs.

The optical tower is the existing v4 C4 ImageEvent model fitted on the opposite
embryo. It encodes actual pixels at each current region centroid. No borrowed
native-node features, geometric mitosis gate, annotations or new fitting enter
prediction. Pair logits are an evidence score, not biological probabilities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from itertools import combinations
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from tools.annotation_selection.common import now, write_json
from tools.cellpose_ultrack.event_costs import bank_fingerprint
from tools.cellpose_ultrack.track import REPO, ROOT
from tools.detector_screen.cellpose_adapter import sha256

STUDY = "cellpose-ultrack-learned-divisions-20260914"
OUT = REPO / "results" / STUDY
WORK = REPO / "work" / STUDY
MODELS = Path("/kaggle/working/cell-tracking/multidata-training-v4/models")
CALIBRATION = REPO / "results/multidata-training-v4/calibration.json"
ARMS = ("pair-control", "pair-image")
EVIDENCE_SCALE = .01
LOGIT_CLIP = 8.
BASE_CONFIG = REPO / "configs/cellpose-ultrack-event-costs-20260914-division-surcharge.json"


def paths(arm):
    return REPO / "work" / f"{STUDY}-{arm}", REPO / "configs" / f"{STUDY}-{arm}.json"


def prepare():
    panel = json.loads((ROOT / "panel.json").read_text())
    calibration = json.loads(CALIBRATION.read_text())
    models = {}
    for source in ("44b6", "6bba"):
        name = f"I_C4_{source}"
        receipt = json.loads((MODELS / f"{name}.json").read_text())
        h = sha256(MODELS / f"{name}.pt")
        if h != calibration["models"][name]["model_sha256"] or h != receipt["checkpoint_sha256"]:
            raise ValueError("Existing optical model differs from its calibration or training receipt")
        if receipt["config"]["source"] != source or receipt["config"]["domains"] != [source]:
            raise ValueError("Wrong direct source-embryo fit")
        models[source] = dict(name=name, path=str(MODELS / f"{name}.pt"), sha256=h,
                              temperature=calibration["models"][name]["temperature"],
                              training_receipt_sha256=sha256(MODELS / f"{name}.json"),
                              direct_sources=receipt["direct_sources"],
                              initialization=receipt["config"]["init"])
    configs = {}
    for arm in ARMS:
        root, config = paths(arm)
        recipe = copy.deepcopy(json.loads(BASE_CONFIG.read_text()))
        recipe.update(study=f"{STUDY}-{arm}",
                      model_description="Cellpose cpDINO-ViT-B + ultrack explicit daughter pairs" +
                                        (" + frozen v4 C4 temporal optical evidence" if arm == "pair-image" else " (zero-image control)"),
                      division_evidence=dict(directory="division-evidence", arm=arm,
                                             scale=EVIDENCE_SCALE, logit_clip=LOGIT_CLIP),
                      selection="Two arms declared before new optical predictions or scores. Same region/link bank and explicit pair constraints; optical bonus zero in pair-control. No parameter or threshold sweep.",
                      exposure="Reused pilot embryos. Existing opposite-embryo direct optical fit, synthetic initialization/replay and original generator temperature. Inherited Biohub-calibrated simulator/incumbent exposure persists. No independent validation or calibrated biological posterior.")
        write_json(config, recipe, immutable=True)
        configs[arm] = dict(root=str(root), config=str(config), sha256=sha256(config))
    plan_path = OUT / "plan.json"
    payload = dict(study=STUDY, models=models, configs=configs, base_config_sha256=sha256(BASE_CONFIG),
                   panel_sha256=sha256(ROOT / "panel.json"), calibration_sha256=sha256(CALIBRATION),
                   evidence_scale=EVIDENCE_SCALE, logit_clip=LOGIT_CLIP,
                   pairs="All unordered nonoverlapping daughter pairs in the unchanged candidate-link graph; no top-k pair cap. Reuse the same pair bank in both arms.",
                   pruning="Only discard pairs dominated in both arms: min(IoU1^4,IoU2^4) + division_weight - appear_weight + max(0,optical_bonus) < -1e-9. Replacing the worse edge by a birth keeps all selected nodes and downstream edges.",
                   optical="Parent and each daughter sampled at native centers across -1,0,+1 frames using the original v4 patch adapter; per-node encoding; symmetric daughter-pair logit minus parent no-fork logit, divided by the original model temperature, clipped to +/-8 and scaled by 0.01.",
                   comparison="Full official six-clip graph score, both embryos, TP/FP/FN, candidate coverage, and actual solver status. Report both arms; no replacement of v3.",
                   inherited_exposure="Synthetic generator calibrated on Biohub 44b6; optical real-image fits use legacy incumbent points with historical upstream exposure. Opposite-embryo direct fitting does not erase that exposure.")
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k:v for k,v in old.items() if k != "created_utc"} != payload:
            raise ValueError("Declared learned-division study changed")
    else:
        write_json(plan_path, dict(created_utc=now(), **payload), immutable=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        root, _ = paths(arm)
        root.mkdir(parents=True, exist_ok=True)
        for name, source in [("images", ROOT / "images"), ("predictions", ROOT / "predictions"),
                             ("cellpose", ROOT / "cellpose"), ("division-evidence", WORK / "division-evidence")]:
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
            if not (target / "reuse.json").exists():
                if (target / "data.db").exists():
                    raise ValueError("Unrecorded database already exists")
                shutil.copy2(source / "data.db", target / "data.db")
                shutil.copy2(source / "metadata.toml", target / "metadata.toml")
                shutil.copy2(source / "centroid-audit.json", target / "centroid-audit.json")
                stages = json.loads((source / "stages.json").read_text())
                write_json(target / "stages.json", {k:v for k,v in stages.items() if k != "solve_seconds"})
                for name in ("foreground.zarr", "contours.zarr"):
                    (target / name).symlink_to(source / name, target_is_directory=True)
                write_json(target / "reuse.json", dict(source=str(source),
                           source_inputs=json.loads((source / "inputs.json").read_text()),
                           initial_database_sha256=sha256(target / "data.db")), immutable=True)
        print(json.dumps(configs[arm]), flush=True)


class ImageMovie:
    def __init__(self, name, shape):
        self.name, self.shape = name, tuple(shape)

    def __getitem__(self, t):
        return np.load(ROOT / "images" / self.name / f"t{int(t):03}.npy", mmap_mode="r", allow_pickle=False)


def candidate_pairs(database):
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        nodes = np.asarray(db.execute("SELECT id,t,z,y,x FROM nodes ORDER BY id").fetchall(), dtype=np.int64)
        links = db.execute("SELECT source_id,target_id,weight FROM links ORDER BY source_id,target_id").fetchall()
        exclusions = {tuple(sorted((int(a),int(b)))) for a,b in db.execute("SELECT node_id,ancestor_id FROM overlaps")}
    by_parent = defaultdict(list)
    for p, c, w in links:
        by_parent[int(p)].append((int(c), float(w)))
    ids, weak = [], []
    for p, children in by_parent.items():
        for (a, wa), (b, wb) in combinations(children, 2):
            if (min(a,b), max(a,b)) in exclusions:
                continue
            ids.append((p, min(a,b), max(a,b)))
            weak.append(min(wa**4, wb**4))
    return nodes, np.asarray(ids, np.int64).reshape(-1,3), np.asarray(weak, np.float64)


def optical_logits(model, embedding, indices, temperature, batch=8192):
    import torch
    result = np.empty(len(indices), np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            ix = torch.as_tensor(indices[start:start+batch], device="cuda")
            p, a, b = embedding[ix[:,0]], embedding[ix[:,1]], embedding[ix[:,2]]
            score = model.pair(torch.cat([p, a+b, (a-b).abs()], -1)).squeeze(-1)
            no_fork = model.null(p).squeeze(-1)
            result[start:start+len(ix)] = ((score-no_fork)/temperature).cpu().numpy()
    return result


def infer():
    import os
    import sys
    def deny_annotation_reads(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(args[0])).resolve())
            if ".geff" in path or "/evaluation/gt/" in path or "/evaluation/event_labels/" in path:
                raise PermissionError("Annotation inputs are unavailable during optical inference")
    sys.addaudithook(deny_annotation_reads)
    import torch
    from tools.multidata_training_v4.adapters import temporal_patches
    from tools.multidata_training_v4.models import ImageEvent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the frozen optical model")
    torch.set_num_threads(2)
    plan = json.loads((OUT / "plan.json").read_text())
    panel = json.loads((ROOT / "panel.json").read_text())
    for clip in panel["clips"]:
        name = clip["dataset"]
        destination = WORK / "division-evidence" / f"{name}.npz"
        receipt_path = destination.with_suffix(".json")
        source = "6bba" if clip["embryo"] == "44b6" else "44b6"
        record = plan["models"][source]
        database = ROOT / "tracking" / name / "data.db"
        bank = bank_fingerprint(database)
        inputs = dict(plan_sha256=sha256(OUT / "plan.json"), source_bank=bank,
                      source_model=record, clip=clip,
                      code_sha256=sha256(Path(__file__)),
                      patch_adapter_sha256=sha256(REPO / "tools/multidata_training_v4/adapters.py"),
                      model_code_sha256=sha256(REPO / "tools/multidata_training_v4/models.py"))
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old["inputs"] != inputs or old["sha256"] != sha256(destination):
                raise ValueError("Optical evidence cache changed")
            continue
        start = time.perf_counter()
        nodes, pairs, weak = candidate_pairs(database)
        # Exact (time, ZYX) duplicates share only their pixel encoder. Region
        # identities, link edges and exclusions stay separate in the ILP.
        unique, inverse = np.unique(nodes[:,1:], axis=0, return_inverse=True)
        node_index = {int(r[0]):i for i,r in enumerate(nodes)}
        indices = np.asarray([[inverse[node_index[int(i)]] for i in row] for row in pairs], np.int64)
        patches, valid = temporal_patches(ImageMovie(name, clip["shape"]), unique[:,0], unique[:,1:], clip["spacing_um"])
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Optical checkpoint changed")
        model = ImageEvent().cuda().eval()
        model.load_state_dict(torch.load(record["path"], map_location="cuda", weights_only=False)["model"])
        encoded = []
        with torch.inference_mode():
            for start_ix in range(0, len(unique), 256):
                p = torch.as_tensor(patches[start_ix:start_ix+256], device="cuda").float()/255
                v = torch.as_tensor(valid[start_ix:start_ix+256], device="cuda")
                encoded.append(model.encode(p[:,None], v[:,None])[:,0])
            embedding = torch.cat(encoded)
        logits = optical_logits(model, embedding, indices, record["temperature"])
        # A real model parity check verifies the arbitrary-pair adapter against
        # its original six-candidate score_encoded API, and daughter symmetry.
        check = min(16, len(indices))
        with torch.inference_mode():
            q = torch.as_tensor(indices[:check], device="cuda")
            z = torch.zeros((check,7,128), device="cuda")
            z[:,:3] = embedding[q]
            x = torch.zeros((check,6,16), device="cuda")
            x[:,:2,11] = 1
            original = model.score_encoded(z,x)[0]
            expected = ((original[:,1]-original[:,0])/record["temperature"]).cpu().numpy()
        swapped = optical_logits(model, embedding, indices[:check, [0,2,1]], record["temperature"])
        if not np.allclose(logits[:check], expected, atol=1e-5, rtol=1e-5) or not np.allclose(logits[:check], swapped, atol=1e-5, rtol=1e-5):
            raise ValueError("Optical pair adapter changed the trained model")
        if not np.isfinite(logits).all():
            raise ValueError("Invalid optical evidence")
        bonus = EVIDENCE_SCALE * np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
        # A conservative upper bound uses the interior appearance penalty;
        # border appearances are free and cannot invalidate this pruning.
        retain = weak - .01 + np.maximum(bonus, 0) >= -1e-9
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, pairs=pairs[retain], bonus=bonus[retain],
                            optical_logit=logits[retain], weaker_link_reward=weak[retain])
        receipt = dict(inputs=inputs, sha256=sha256(destination), dataset=name,
                       source_embryo=source, target_embryo=clip["embryo"],
                       annotations_used=False, existing_source_model_reused=True,
                       annotation_read_guard_installed=True,
                       all_compatible_pairs=len(pairs), retained_pairs=int(retain.sum()),
                       retained_parents=len(np.unique(pairs[retain,0])), unique_pixel_centers=len(unique),
                       positive_optical_pairs=int(np.sum(logits>0)), model_adapter_parity_passed=True,
                       daughter_permutation_invariance_passed=True,
                       seconds=time.perf_counter()-start)
        write_json(receipt_path, receipt)
        print("OPTICAL " + json.dumps({k:v for k,v in receipt.items() if k != "inputs"}), flush=True)
        del model, embedding, encoded, patches, valid
        torch.cuda.empty_cache()


def attach_pair_objective(solver, pairs, bonus):
    """Require the chosen two edges to match one explicit scored daughter pair."""
    import mip
    native = solver._backward_map
    index = {int(i):k for k,i in enumerate(native)}
    frame = solver._edges_df
    edges = {(int(native[a]), int(native[b])):solver._edges[k]
             for k,(a,b) in enumerate(frame[["sources","targets"]].itertuples(index=False, name=None))}
    by_parent = defaultdict(list)
    terms = []
    for (p,a,b), value in zip(pairs, bonus):
        e1, e2 = edges.get((int(p),int(a))), edges.get((int(p),int(b)))
        if e1 is None or e2 is None:
            continue
        event = solver._model.add_var(name="pair_" + "_".join(map(str,(p,a,b))), var_type=mip.BINARY)
        solver._model.add_constr(event <= e1)
        solver._model.add_constr(event <= e2)
        by_parent[int(p)].append(event)
        terms.append(float(value) * event)
    for p, i in index.items():
        solver._model.add_constr(mip.xsum(by_parent[p]) == solver._divisions[i])
    solver._model.objective += mip.xsum(terms)
    return dict(pair_variables=sum(map(len,by_parent.values())), parents_with_pairs=len([p for p in by_parent if by_parent[p]]),
                restricted_division_variables=len(index), minimum_bonus=float(np.min(bonus)) if len(bonus) else 0,
                maximum_bonus=float(np.max(bonus)) if len(bonus) else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else infer()


if __name__ == "__main__":
    main()
